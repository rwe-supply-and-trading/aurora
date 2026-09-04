"""Copyright (c) RWE Supply & Trading GmbH. Licensed under the MIT license.

Tests for run-time exposure of Aurora's backbone latent vectors.

These tests pin the properties the latent-extraction pipeline depends on, so that a future
upstream sync that changes the model's calling convention fails loudly here rather than silently
producing wrong latents.
"""

import pytest
import torch

from ..v1p5._helpers import _OUTPUT_ONLY_SURF, _SURF_VARS, _make_batch, _make_small_v1p5
from aurora.latent import (
    LatentCapture,
    capture_latents,
    latents_to_grid,
    patch_res_for,
)


def _input_batch():
    """A batch without the output-only variables, as real input data would be."""
    return _make_batch(surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF))


def _lead_times():
    return torch.tensor([6.0])


def test_capture_records_one_latent_per_forward():
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()

    with capture_latents(model) as capture:
        assert capture.latents == []
        with torch.inference_mode():
            model.forward(batch, lead_times=_lead_times())
        assert len(capture.latents) == 1
        with torch.inference_mode():
            model.forward(batch, lead_times=_lead_times())
        assert len(capture.latents) == 2


def test_latent_has_backbone_output_shape():
    """The latent must be the `(B, L, D)` tensor handed to the decoder."""
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()

    with capture_latents(model) as capture, torch.inference_mode():
        model.forward(batch, lead_times=_lead_times())

    latent = capture.latent
    assert latent.ndim == 3, f"Expected (B, L, D), got {tuple(latent.shape)}"

    B = next(iter(batch.surf_vars.values())).shape[0]
    C, H, W = patch_res_for(model, batch)
    assert latent.shape[0] == B
    assert (
        latent.shape[1] == C * H * W
    ), f"Latent token count {latent.shape[1]} does not match patch resolution {(C, H, W)}"
    # The latent width must be what the decoder consumes. Note this is *not* `backbone.embed_dim`:
    # the backbone is a U-Net that changes width across its encoder/decoder path.
    assert latent.shape[2] == model.decoder.embed_dim


def test_latent_matches_decoder_input():
    """The captured latent must be exactly the tensor the decoder consumes.

    This is the core correctness property: it is what makes the captured vector *the* latent
    representation rather than some other intermediate activation.
    """
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()

    seen: list[torch.Tensor] = []

    def record_decoder_input(module, args, kwargs):
        seen.append(args[0] if args else kwargs["x"])

    handle = model.decoder.register_forward_pre_hook(record_decoder_input, with_kwargs=True)
    try:
        with capture_latents(model) as capture, torch.inference_mode():
            model.forward(batch, lead_times=_lead_times())
    finally:
        handle.remove()

    assert len(seen) == 1
    torch.testing.assert_close(capture.latent, seen[0])


def test_latent_grid_reshape_roundtrip():
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()

    with capture_latents(model) as capture, torch.inference_mode():
        model.forward(batch, lead_times=_lead_times())

    patch_res = patch_res_for(model, batch)
    grid = latents_to_grid(capture.latent, patch_res)

    C, H, W = patch_res
    B, _, D = capture.latent.shape
    assert tuple(grid.shape) == (B, C, H, W, D)
    # Reshaping must not reorder or alter any value.
    torch.testing.assert_close(grid.reshape(B, C * H * W, D), capture.latent)


def test_latent_grid_rejects_mismatched_patch_res():
    latent = torch.randn(1, 24, 8)
    with pytest.raises(ValueError, match="tokens"):
        latents_to_grid(latent, (2, 3, 5))  # 30 != 24


def test_hook_is_removed_on_exit():
    """The capture must not leave a hook behind that would leak memory across runs."""
    model = _make_small_v1p5()
    before = len(model.backbone._forward_hooks)

    with capture_latents(model):
        assert len(model.backbone._forward_hooks) == before + 1

    assert len(model.backbone._forward_hooks) == before


def test_hook_is_removed_on_exception():
    model = _make_small_v1p5()
    before = len(model.backbone._forward_hooks)

    with pytest.raises(RuntimeError, match="boom"), capture_latents(model):
        raise RuntimeError("boom")

    assert len(model.backbone._forward_hooks) == before


def test_capture_does_not_change_predictions():
    """Capturing latents must be observationally pure with respect to the forecast."""
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()

    with torch.inference_mode():
        expected = model.forward(batch, lead_times=_lead_times())

    with capture_latents(model), torch.inference_mode():
        actual = model.forward(batch, lead_times=_lead_times())

    for k, v in expected.surf_vars.items():
        torch.testing.assert_close(actual.surf_vars[k], v)
    for k, v in expected.atmos_vars.items():
        torch.testing.assert_close(actual.atmos_vars[k], v)


def test_latent_is_detached_by_default():
    model = _make_small_v1p5()
    batch = _input_batch()

    with capture_latents(model) as capture:
        model.forward(batch, lead_times=_lead_times())

    assert not capture.latent.requires_grad


def test_latent_can_retain_grad():
    """`detach=False` must keep the latent connected to the autograd graph."""
    model = _make_small_v1p5()
    batch = _input_batch()

    with capture_latents(model, detach=False) as capture:
        model.forward(batch, lead_times=_lead_times())

    assert capture.latent.grad_fn is not None


def test_latent_before_any_forward_raises():
    model = _make_small_v1p5()
    with capture_latents(model) as capture, pytest.raises(RuntimeError, match="No latent"):
        _ = capture.latent


def test_reset_clears_latents():
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()

    with capture_latents(model) as capture, torch.inference_mode():
        model.forward(batch, lead_times=_lead_times())
        assert len(capture.latents) == 1
        capture.reset()
        assert capture.latents == []


def test_nested_capture_is_rejected():
    model = _make_small_v1p5()
    capture = LatentCapture(model)
    with capture:  # noqa: SIM117 - the nesting is the behaviour under test
        with pytest.raises(RuntimeError, match="already active"):
            with capture:
                pass


def test_capture_works_with_activation_checkpointing():
    """Activation checkpointing wraps modules, which must not swallow the hook.

    This matters for fine-tuning, where checkpointing is normally enabled.
    """
    model = _make_small_v1p5()
    model.configure_activation_checkpointing()
    model.train()
    batch = _input_batch()

    with capture_latents(model, detach=False) as capture:
        model.forward(batch, lead_times=_lead_times())

    assert len(capture.latents) == 1
    assert capture.latent.ndim == 3


def test_capture_moves_latent_to_cpu_when_requested():
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()

    with capture_latents(model, to_cpu=True) as capture, torch.inference_mode():
        model.forward(batch, lead_times=_lead_times())

    assert capture.latent.device.type == "cpu"
