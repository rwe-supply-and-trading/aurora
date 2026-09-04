"""Copyright (c) RWE Supply & Trading GmbH. Licensed under the MIT license.

Tests for `rollout_with_latents` against the Aurora 1.5 roll-out.

Aurora 1.5's roll-out gained sub-stepping (`fine_lead_times`), noise accumulation, and roll-out
input clipping. `rollout_with_latents` delegates to the stock roll-out precisely so it inherits
those behaviours; these tests pin that delegation.
"""

import torch

from ..v1p5._helpers import _OUTPUT_ONLY_SURF, _SURF_VARS, _make_batch, _make_small_v1p5
from aurora import AuroraV1p5Ensemble
from aurora.latent import capture_latents, rollout_with_latents
from aurora.rollout import rollout


def _input_batch():
    return _make_batch(surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF))


def _make_small_v1p5_ensemble():
    """A small stochastic Aurora 1.5 ensemble model, mirroring `_make_small_v1p5`."""
    return AuroraV1p5Ensemble(
        surf_vars=_SURF_VARS,
        static_vars=("lsm", "z"),
        atmos_vars=("z", "u", "v", "t", "q"),
        output_only_surf_vars=_OUTPUT_ONLY_SURF,
        encoder_depths=(2, 2),
        encoder_num_heads=(4, 8),
        decoder_depths=(2, 2),
        decoder_num_heads=(8, 4),
        embed_dim=64,
        num_heads=4,
        use_lora=False,
        autocast=False,
    )


def test_yields_one_latent_per_step():
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()

    with torch.inference_mode():
        out = list(rollout_with_latents(model, batch, steps=3))

    assert len(out) == 3
    for pred, latent in out:
        assert latent.ndim == 3
        assert pred.metadata.rollout_step >= 1


def test_latents_differ_across_steps():
    """A constant latent across steps would mean the capture is stale."""
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()

    with torch.inference_mode():
        latents = [latent for _, latent in rollout_with_latents(model, batch, steps=3)]

    assert not torch.allclose(latents[0], latents[1])
    assert not torch.allclose(latents[1], latents[2])


def test_predictions_match_plain_rollout():
    """Adding latent capture must not perturb the forecast produced by `rollout`."""
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()

    torch.manual_seed(0)
    with torch.inference_mode():
        expected = list(rollout(model, batch, steps=2))

    torch.manual_seed(0)
    with torch.inference_mode():
        actual = [pred for pred, _ in rollout_with_latents(model, batch, steps=2)]

    assert len(expected) == len(actual)
    for e, a in zip(expected, actual):
        assert e.metadata.time == a.metadata.time
        for k, v in e.surf_vars.items():
            torch.testing.assert_close(a.surf_vars[k], v)
        for k, v in e.atmos_vars.items():
            torch.testing.assert_close(a.atmos_vars[k], v)


def test_substepping_yields_a_latent_per_substep():
    """With `fine_lead_times`, 1.5 emits one prediction per sub-step; latents must track them."""
    model = _make_small_v1p5(variable_lead_time=True)
    model.eval()
    batch = _input_batch()

    fine_lead_times = [3.0, 6.0]
    with torch.inference_mode():
        out = list(rollout_with_latents(model, batch, steps=2, fine_lead_times=fine_lead_times))

    # Two main steps, two sub-steps each.
    assert len(out) == 4
    for _, latent in out:
        assert latent.ndim == 3


def test_substepping_predictions_match_plain_rollout():
    model = _make_small_v1p5(variable_lead_time=True)
    model.eval()
    batch = _input_batch()

    fine_lead_times = [3.0, 6.0]

    with torch.inference_mode():
        expected = list(rollout(model, batch, steps=2, fine_lead_times=fine_lead_times))
    with torch.inference_mode():
        actual = [
            pred
            for pred, _ in rollout_with_latents(
                model, batch, steps=2, fine_lead_times=fine_lead_times
            )
        ]

    assert [p.metadata.time for p in expected] == [p.metadata.time for p in actual]
    for e, a in zip(expected, actual):
        for k, v in e.surf_vars.items():
            torch.testing.assert_close(a.surf_vars[k], v)


def test_hook_removed_after_rollout():
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()
    before = len(model.backbone._forward_hooks)

    with torch.inference_mode():
        list(rollout_with_latents(model, batch, steps=2))

    assert len(model.backbone._forward_hooks) == before


def test_hook_removed_if_generator_abandoned():
    """Abandoning the generator early must still clean up the hook."""
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()
    before = len(model.backbone._forward_hooks)

    with torch.inference_mode():
        gen = rollout_with_latents(model, batch, steps=5)
        next(gen)
        gen.close()

    assert len(model.backbone._forward_hooks) == before


def test_latent_corresponds_to_its_own_prediction():
    """Each yielded latent must be the one that produced the yielded prediction."""
    model = _make_small_v1p5()
    model.eval()
    batch = _input_batch()

    # Capture every latent independently, in order, over an equivalent plain roll-out.
    with capture_latents(model) as capture:
        with torch.inference_mode():
            list(rollout(model, batch, steps=3))
        reference = list(capture.latents)

    with torch.inference_mode():
        paired = [latent for _, latent in rollout_with_latents(model, batch, steps=3)]

    assert len(reference) == len(paired) == 3
    for r, p in zip(reference, paired):
        torch.testing.assert_close(r, p)


def test_capture_works_on_ensemble_model():
    """`AuroraV1p5Ensemble` reuses the same backbone, so capture must work there too."""
    model = _make_small_v1p5_ensemble()
    model.eval()
    batch = _input_batch()

    with capture_latents(model) as capture, torch.inference_mode():
        model.forward(batch, lead_times=torch.tensor([6.0]))

    assert capture.latent.ndim == 3
    assert torch.isfinite(capture.latent).all()
