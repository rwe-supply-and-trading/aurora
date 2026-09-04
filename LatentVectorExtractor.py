#!/usr/bin/env python
"""Copyright (c) RWE Supply & Trading GmbH. Licensed under the MIT license."""

import datetime
import os

import kafou_arraylake as arraylake
import numpy as np
import torch
import xarray as xr

from aurora import AuroraPretrained
from aurora.data import ERA5DataLoaderFOAM
from aurora.rollout import rollout, rollout_with_latents

os.environ["PYTHONUNBUFFERED"] = "1"

for var in [
    "CURL_CA_BUNDLE",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
]:
    os.environ.pop(var, None)


def batch_to_tensor(batch):
    tensors = []

    B, T = None, None

    # -------------------------
    # 1. Surface vars
    # -------------------------
    for k in sorted(batch.surf_vars.keys()):
        v = batch.surf_vars[k]  # (B, T, H, W)

        if B is None:
            B, T = v.shape[:2]

        v = v.unsqueeze(-1)  # (B, T, H, W, 1)
        tensors.append(v)

    # -------------------------
    # 2. Atmospheric vars
    # -------------------------
    for k in sorted(batch.atmos_vars.keys()):
        v = batch.atmos_vars[k]  # (B, T, L, H, W)

        # ✅ enforce exact expected shape
        assert v.ndim == 5, f"{k} has wrong shape: {v.shape}"

        # move level → feature dim
        v = v.permute(0, 1, 3, 4, 2)  # (B, T, H, W, L)

        tensors.append(v)

    # -------------------------
    # 3. Static vars
    # -------------------------
    for k in sorted(batch.static_vars.keys()):
        v = batch.static_vars[k]  # (B, H, W) or (H, W)

        if v.ndim == 2:
            v = v.unsqueeze(0)  # (1, H, W)

        # expand to (B, T, H, W)
        v = v.unsqueeze(1)  # (B, 1, H, W)
        v = v.expand(B, T, *v.shape[2:])  # (B, T, H, W)

        v = v.unsqueeze(-1)  # (B, T, H, W, 1)

        tensors.append(v)

    # -------------------------
    # 🔥 FINAL CHECK
    # -------------------------
    for i, t in enumerate(tensors):
        assert t.ndim == 5, f"Tensor {i} has wrong ndim: {t.shape}"

    return torch.cat(tensors, dim=-1)


# ------------------------------------------------------
# Aurora inference wrapper
# ------------------------------------------------------
class LatentVectorExtractor:
    """
    Aurora inference wrapper for ERA5 latent-vector extraction.

    Responsibilities:
      - Load sample and invariant datasets from a source repository
      - Run the Aurora model in GPU inference mode
      - Return latent vectors as an xarray.Dataset

    """

    def __init__(
        self,
        *,
        source_repo: str | None = None,
        client: arraylake.Client | None = None,
        source_branch: str = "main",
        device: str = "cuda",
    ):
        if client is None:
            client = arraylake.Client()

        print(f"\n[LVE] Opening source repo={source_repo} branch={source_branch}")
        repo = client.get_repo(source_repo)
        session = repo.readonly_session(source_branch)

        sample_ds = xr.open_zarr(
            session.store, group="samples", zarr_format=3, consolidated=False, chunks=None
        )

        print(sample_ds)

        inv_ds = xr.open_zarr(
            session.store, group="invariant", zarr_format=3, consolidated=False, chunks=None
        )

        self.data_loader = ERA5DataLoaderFOAM(sample_ds=sample_ds, invariant_ds=inv_ds)

        print("[LVE] Loading Aurora model checkpoint...")
        self.model = AuroraPretrained()
        self.model.load_checkpoint()
        self.model.eval()
        self.model.to(device)
        self.device = device
        print("[LVE] Model ready.")

    def rollout_lvs(
        self,
        item: datetime.datetime,
        steps: int,
    ) -> xr.Dataset:
        """
        Run Aurora rollout and return full-grid latent vectors.

        Returns
        -------
        xr.Dataset with dims:
            (lead_time, spatial_location, feature)
        """

        if not isinstance(item, datetime.datetime):
            raise TypeError("item must be a datetime.datetime")

        print(f"\n[LVE.rollout_lvs] item={item} steps={steps}")

        batch = self.data_loader[item]

        lvs = []

        with torch.inference_mode():
            for step, (_pred, latent) in enumerate(rollout_with_latents(self.model, batch, steps)):
                # latent: (1, S, F) -> (S, F)
                latent_np = latent.detach().to("cpu").numpy().squeeze(0)
                print(f"[LVE.rollout_lvs] step={step} latent_np.shape={latent_np.shape}")
                lvs.append(latent_np)

        lv_arr = np.stack(lvs, axis=0).astype("float32", copy=False)

        lead_time = np.arange(1, steps + 1, dtype="int64") * 6

        out = xr.Dataset(
            data_vars={
                "lv": (("lead_time", "spatial_location", "feature"), lv_arr),
            },
            coords={
                "lead_time": ("lead_time", lead_time),
                # spatial_location is implicit and positional here
            },
            attrs={"init_time": np.datetime64(item, "ns")},
        )

        print("[LVE.rollout_lvs] out.lv shape:", out["lv"].shape)
        print("[LVE.rollout_lvs] out.lead_time:", out["lead_time"].values)

        return out

    def rollout(
        self,
        item: datetime.datetime,
        steps: int,
    ) -> xr.Dataset:
        """
        Run Aurora rollout and return full-grid latent vectors.

        Returns
        -------
        xr.Dataset with dims:
            (lead_time, lat, lon, feature)
        """

        if not isinstance(item, datetime.datetime):
            raise TypeError("item must be a datetime.datetime")

        print(f"\n[LVE.rollout] item={item} steps={steps}")

        batch = self.data_loader[item]

        lvs = []

        with torch.inference_mode():
            for step, preds in enumerate(rollout(self.model, batch, steps)):
                preds = batch_to_tensor(preds)  # (B, T, H, W, F)
                preds = preds[:, -1]  # (B, H, W, F)

                preds = preds.squeeze(0)  # (H, W, F)

                print(f"[LVE.rollout] step={step} shape={preds.shape}")

                lvs.append(preds.cpu())

        lv_arr = torch.stack(lvs).numpy().astype("float32", copy=False)

        lead_time = np.arange(1, steps + 1, dtype="int64") * 6

        # Aurora's native coordinates
        lat = np.linspace(90, -89.75, 720, dtype="float32")  # descending
        lon = np.linspace(0, 359.75, 1440, dtype="float32")  # 0-360 range

        out = xr.Dataset(
            data_vars={
                "normalized_sample": (("lead_time", "lat", "lon", "feature"), lv_arr),
            },
            coords={
                "lead_time": ("lead_time", lead_time),
                "lat": ("lat", lat),
                "lon": ("lon", lon),
            },
            attrs={"init_time": np.datetime64(item, "ns")},
        )

        print("[LVE.rollout] out.lv shape:", out["normalized_sample"].shape)
        print("[LVE.rollout] out.lead_time:", out["lead_time"].values)
        print(f"[LVE.rollout] lat[0]={out.lat.values[0]}, lat[-1]={out.lat.values[-1]}")
        print(f"[LVE.rollout] lon[0]={out.lon.values[0]}, lon[-1]={out.lon.values[-1]}")

        return out
