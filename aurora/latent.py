"""Copyright (c) RWE Supply & Trading GmbH. Licensed under the MIT license.

Expose Aurora's backbone latent vectors at run time.

The latent representation of interest is the output of :class:`Swin3DTransformerBackbone`, the
tensor that is handed to the decoder. Rather than threading a `return_latent` flag through
:meth:`Aurora.forward`, we attach a PyTorch forward hook to the backbone module. This keeps
:meth:`Aurora.forward` byte-identical to upstream, so the capture mechanism is unaffected by
changes to the model's calling convention (for example the move from `lead_time` to `lead_times`
in Aurora 1.5), and it composes with the stock :func:`aurora.rollout.rollout`, including its
sub-stepping, noise accumulation, and roll-out input clipping.
"""

from contextlib import contextmanager
from typing import Generator, Iterator, Optional, Sequence

import torch

from aurora.batch import Batch
from aurora.model.aurora import Aurora
from aurora.rollout import rollout

__all__ = [
    "LatentCapture",
    "capture_latents",
    "latents_to_grid",
    "patch_res_for",
    "rollout_with_latents",
]


class LatentCapture:
    """Capture the backbone output of an :class:`aurora.Aurora` model.

    The most recent latent is available as :attr:`latent`, and every latent seen since the last
    :meth:`reset` is available as :attr:`latents`.

    Attributes:
        latents (list[torch.Tensor]): Latents captured so far, in the order they were produced.
    """

    def __init__(self, model: Aurora, detach: bool = True, to_cpu: bool = False) -> None:
        """Construct a capture.

        Args:
            model (:class:`aurora.Aurora`): The model to capture latents from.
            detach (bool, optional): Detach the latent from the autograd graph. Set to `False` if
                you need to backpropagate through the latent. Defaults to `True`.
            to_cpu (bool, optional): Move captured latents to the CPU. This frees accelerator
                memory when accumulating many roll-out steps. Defaults to `False`.
        """
        self.model = model
        self.detach = detach
        self.to_cpu = to_cpu
        self.latents: list[torch.Tensor] = []
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _hook(self, module: torch.nn.Module, args: tuple, output: torch.Tensor) -> None:
        latent = output
        if self.detach:
            latent = latent.detach()
        if self.to_cpu:
            latent = latent.to("cpu")
        self.latents.append(latent)

    @property
    def latent(self) -> torch.Tensor:
        """torch.Tensor: The most recently captured latent of shape `(B, L, D)`."""
        if not self.latents:
            raise RuntimeError(
                "No latent has been captured yet. Run the model inside the capture first."
            )
        return self.latents[-1]

    def reset(self) -> None:
        """Discard all captured latents."""
        self.latents.clear()

    def __enter__(self) -> "LatentCapture":
        if self._handle is not None:
            raise RuntimeError("This capture is already active.")
        self._handle = self.model.backbone.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._handle is not None
        self._handle.remove()
        self._handle = None


@contextmanager
def capture_latents(
    model: Aurora, detach: bool = True, to_cpu: bool = False
) -> Iterator[LatentCapture]:
    """Context manager yielding a :class:`LatentCapture` attached to `model`.

    Args:
        model (:class:`aurora.Aurora`): The model to capture latents from.
        detach (bool, optional): Detach latents from the autograd graph. Defaults to `True`.
        to_cpu (bool, optional): Move captured latents to the CPU. Defaults to `False`.

    Yields:
        :class:`LatentCapture`: The active capture.
    """
    with LatentCapture(model, detach=detach, to_cpu=to_cpu) as capture:
        yield capture


def rollout_with_latents(
    model: Aurora,
    batch: Batch,
    steps: int,
    fine_lead_times: Optional[Sequence[float]] = None,
    use_noise_accumulation: bool = True,
    apply_rollout_input_clipping: bool = True,
    detach: bool = True,
    to_cpu: bool = False,
) -> Generator[tuple[Batch, torch.Tensor], None, None]:
    """Roll out Aurora, yielding the prediction and the backbone latent at every step.

    This wraps :func:`aurora.rollout.rollout`, so it inherits that function's behaviour exactly,
    including sub-stepping via `fine_lead_times`, noise accumulation, and roll-out input clipping.
    One latent is yielded per prediction, including for each sub-step.

    Args:
        model (:class:`aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.Batch`): The batch to start the roll-out from.
        steps (int): The number of main roll-out steps.
        fine_lead_times (sequence of float, optional): See :func:`aurora.rollout.rollout`.
        use_noise_accumulation (bool, optional): See :func:`aurora.rollout.rollout`.
        apply_rollout_input_clipping (bool, optional): See :func:`aurora.rollout.rollout`.
        detach (bool, optional): Detach latents from the autograd graph. Defaults to `True`.
        to_cpu (bool, optional): Move captured latents to the CPU. Defaults to `False`.

    Yields:
        tuple[:class:`aurora.Batch`, torch.Tensor]: The prediction and the corresponding backbone
            latent of shape `(B, L, D)`.
    """
    with LatentCapture(model, detach=detach, to_cpu=to_cpu) as capture:
        for pred in rollout(
            model,
            batch,
            steps,
            fine_lead_times=fine_lead_times,
            use_noise_accumulation=use_noise_accumulation,
            apply_rollout_input_clipping=apply_rollout_input_clipping,
        ):
            if not capture.latents:
                raise RuntimeError(
                    "The backbone did not run for this roll-out step, so no latent was captured."
                )
            latent = capture.latents[-1]
            capture.reset()
            yield pred, latent


def latents_to_grid(latent: torch.Tensor, patch_res: tuple[int, int, int]) -> torch.Tensor:
    """Reshape a flat backbone latent into its spatial grid.

    Args:
        latent (torch.Tensor): Latent of shape `(B, L, D)` as produced by the backbone.
        patch_res (tuple[int, int, int]): The patch resolution `(C, H, W)` that the latent
            corresponds to, as computed in :meth:`Aurora.forward`.

    Returns:
        torch.Tensor: Latent of shape `(B, C, H, W, D)`.
    """
    B, L, D = latent.shape
    C, H, W = patch_res
    if L != C * H * W:
        raise ValueError(
            f"Latent has {L} tokens, but the patch resolution {patch_res} implies {C * H * W}."
        )
    return latent.view(B, C, H, W, D)


def patch_res_for(model: Aurora, batch: Batch) -> tuple[int, int, int]:
    """Compute the backbone patch resolution `(C, H, W)` for `batch`.

    This mirrors the calculation in :meth:`Aurora.forward` and is useful for interpreting a
    captured latent with :func:`latents_to_grid`.

    Args:
        model (:class:`aurora.Aurora`): The model the latent came from.
        batch (:class:`aurora.Batch`): The batch the latent was produced for.

    Returns:
        tuple[int, int, int]: The patch resolution `(C, H, W)`.
    """
    batch = batch.crop(patch_size=model.patch_size)
    H, W = batch.spatial_shape
    return (
        model.encoder.latent_levels,
        H // model.encoder.patch_size,
        W // model.encoder.patch_size,
    )
