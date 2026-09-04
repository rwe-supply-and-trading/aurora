# Latent Vectors

```{note}
This page documents a feature of the RWE Supply & Trading fork of Aurora. It is not part of
upstream `microsoft/aurora`.
```

Aurora's backbone produces a latent representation of the atmospheric state that is then decoded
into predictions. That latent tensor is often more useful than the predictions themselves for
downstream modelling, since it is a learned, compact summary of the state. This fork exposes it at
run time.

## What is captured

The latent is the output of the Swin3D backbone, which is exactly the tensor handed to the decoder.
It has shape `(B, L, D)`:

- `B` is the batch size.
- `L` is the number of tokens, equal to `C * H * W` for the patch resolution `(C, H, W)`.
- `D` is the decoder's embedding dimension (`model.decoder.embed_dim`).

Note that `D` is *not* `model.backbone.embed_dim`. The backbone is a U-Net whose width changes
across its encoder and decoder path, so the output width differs from the nominal embedding
dimension.

## Capturing a single forward pass

Use {py:func}`aurora.latent.capture_latents`, which attaches a forward hook for the duration of the
context:

```python
import torch

from aurora import AuroraV1p5
from aurora.latent import capture_latents

model = AuroraV1p5()
model.load_checkpoint()
model.eval()

with capture_latents(model) as capture, torch.inference_mode():
    pred = model.forward(batch)

latent = capture.latent  # (B, L, D)
```

`capture.latents` holds every latent captured since the last `capture.reset()`, in order.

## Capturing during a roll-out

{py:func}`aurora.latent.rollout_with_latents` wraps the standard
{py:func}`aurora.rollout.rollout` and yields the prediction alongside its latent:

```python
from aurora.latent import rollout_with_latents

with torch.inference_mode():
    for pred, latent in rollout_with_latents(model, batch, steps=4):
        ...
```

Because it delegates to the stock roll-out, it inherits that function's behaviour exactly,
including Aurora 1.5's sub-stepping, noise accumulation, and roll-out input clipping. All of
`rollout`'s keyword arguments are forwarded. When sub-stepping with `fine_lead_times`, one latent
is yielded per sub-step:

```python
for pred, latent in rollout_with_latents(
    model, batch, steps=2, fine_lead_times=[3.0, 6.0]
):
    ...  # Four pairs in total: two sub-steps for each of two main steps.
```

## Interpreting the latent spatially

The flat token dimension can be folded back into its grid:

```python
from aurora.latent import latents_to_grid, patch_res_for

patch_res = patch_res_for(model, batch)   # (C, H, W)
grid = latents_to_grid(latent, patch_res)  # (B, C, H, W, D)
```

The grid is at *patch* resolution, not the native grid: each cell covers `model.patch_size` grid
points in latitude and longitude.

## Memory and gradients

By default latents are detached from the autograd graph. To backpropagate through the latent, pass
`detach=False`. Accumulating many roll-out steps on an accelerator can be expensive, since each
latent is a full `(B, L, D)` tensor; pass `to_cpu=True` to move each latent to host memory as it is
captured.

```python
with capture_latents(model, detach=False) as capture:
    pred = model.forward(batch)
    loss = some_objective(capture.latent)
    loss.backward()
```

## Why a hook rather than a flag

The capture is implemented with a PyTorch forward hook on the backbone, so
{py:meth}`aurora.Aurora.forward` remains identical to upstream. This matters for maintenance: the
Aurora 1.5 upgrade changed `forward`'s signature (`lead_time` became a `lead_times` tensor), which
would have collided with a `return_latent` flag threaded through that signature. Keeping the
capture out of the call signature means future upstream syncs do not conflict with it.
