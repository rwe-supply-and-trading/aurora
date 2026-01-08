#!/usr/bin/env python

"""
Aurora Latent Forecast Writer (ERA5 → Zarr / Icechunk)

This script generates Aurora *forecast* latent vectors from ERA5 inputs and
stores them in a Zarr v3 dataset backed by Icechunk. It supports both single-job
execution and distributed GPU execution via SLURM, with safe concurrent region
writes and a final atomic merge + commit.

For each 6-hourly ERA5 initialization time `t`, the Aurora model performs a
multi-step rollout and writes latent forecasts at:

    t + 6h, t + 12h, ..., t + 6h * rollout_steps

These are stored in a 4D array:

    latent_forecast(init_time, lead_time, spatial_location, feature)

The temporal offset semantics (`init_time → valid_time = init_time + lead_time`)
are intentional and centralized in `LatentVectorExtractor`.

The store is initialized *once* with coordinates and metadata only; the dense
latent array is created empty and incrementally filled via region writes.

Main commands
-------------
init
  Initialize a new latent-forecast repository with coordinates and an empty
  `latent_forecast` array. Must be run exactly once per destination repo/branch.

save-lvs
  Run Aurora inference for a contiguous range of `init_time` values and write
  forecast cubes directly into the store using region writes.

submit-jobs
  Split a large time range into disjoint SLURM jobs, run `save-lvs` in parallel
  on GPUs, merge all write sessions, and commit once atomically.

Dataset invariants
------------------
- Writes are indexed by `init_time`; each job writes disjoint slices.
- `valid_time = init_time + lead_time` is stored explicitly.
- The root attribute `valid_time_range` is monotonic and tracks temporal
  coverage of successfully written data.

Intended use
------------
- Large-scale historical backfills
- Incremental operational updates
- HPC / SLURM environments with shared object storage



To Run
------
tmux

conda activate aurora


sbatch   --ntasks=1   --cpus-per-task=32   --mem=300G   --job-name=lv-submit   --wrap='
    python latent_forecast_writer.py submit-jobs \
      2025-07-01T00:00:00 \
      2025-07-31T18:00:00 \
      --src-repo kafou/aurora-era5-samples \
      --src-branch extend-2025 \
      --dest-repo kafou/aurora-era5-forecast-latent-vectors-july-v2 \
      --dest-branch main \
      --aws-profile kafou \
      --timesteps-per-job 8 \
      --coordination-location s3://icechunk-write-coordination'

"""

import os

os.environ["PYTHONUNBUFFERED"] = "1"


import datetime
import os
import pickle
import random
import subprocess
import sys
import time

import click
import fsspec
import kafou_arraylake as arraylake
import numpy as np
import torch
import xarray as xr
import zarr
from icechunk.distributed import merge_sessions

from aurora import AuroraPretrained
from aurora.data import ERA5DataLoaderFOAM
from aurora.rollout import rollout_with_latents

SOURCE_REPO = "kafou/aurora-era5-samples"
SOURCE_BRANCH = "extend-2025"

DESTINATION_REPO = "kafou/aurora-era5-forecast-latent-vectors"
DESTINATION_BRANCH = "main"


def random_job_string(length: int = 8) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(chars) for _ in range(length))


def get_job_count(job_prefix: str) -> int:
    """
    Count remaining SLURM jobs whose name starts with job_prefix.
    """
    res = subprocess.run(
        ["squeue", "-h", "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sum(1 for name in res.stdout.splitlines() if name.startswith(job_prefix))


def get_spatial_indices_from_bounds(
    *,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    n_lev: int = 4,
    n_lat: int = 180,
    n_lon: int = 360,
) -> np.ndarray:
    """
    Return flat spatial_location indices for a lat/lon bounding box.

    Latitude is assumed to be ERA-style (north → south).
    Longitude bounds must be Greenwich-centered in [-180, 180].

    The returned indices always refer to the ORIGINAL flattened
    (lev, lat, lon) ordering:
        spatial_location = lev * (n_lat * n_lon) + lat * n_lon + lon
    """

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    # ------------------------------------------------------------------
    # Coordinate grids (centers)
    # ------------------------------------------------------------------

    lat_vals = np.linspace(89.5, -89.5, n_lat)  # north → south
    lon_vals_0360 = np.linspace(0.5, 359.5, n_lon)  # 0–360
    lon_vals = ((lon_vals_0360 + 180) % 360) - 180  # −180…180

    # ------------------------------------------------------------------
    # Latitude mask
    # ------------------------------------------------------------------

    lat_mask = (lat_vals >= lat_min) & (lat_vals <= lat_max)
    lat_idx = np.where(lat_mask)[0]

    if lat_idx.size == 0:
        raise ValueError("Latitude range selects no grid points.")

    # ------------------------------------------------------------------
    # Longitude mask (wrap-aware)
    # ------------------------------------------------------------------

    if lon_min <= lon_max:
        lon_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)
    else:
        # wrapped interval across prime meridian
        lon_mask = (lon_vals >= lon_min) | (lon_vals <= lon_max)

    lon_idx = np.where(lon_mask)[0]

    if lon_idx.size == 0:
        raise ValueError("Longitude range selects no grid points.")

    # Sort longitude indices by actual longitude value
    # (fixes visual seam issues when crossing 0°)
    lon_idx = lon_idx[np.argsort(lon_vals[lon_idx])]

    # ------------------------------------------------------------------
    # Build flat spatial_location indices
    # lev-major, then lat, then lon
    # ------------------------------------------------------------------

    spatial_indices = np.concatenate(
        [lev * n_lat * n_lon + lat * n_lon + lon_idx for lev in range(n_lev) for lat in lat_idx]
    ).astype(np.int64)

    return spatial_indices


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
        source_repo: str = SOURCE_REPO,
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
        # shape: (lead_time, spatial_location, feature)

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


@click.group()
def cli():
    pass


def init_forecast_zarr_store(
    *,
    store,
    init_times: np.ndarray,
    rollout_steps: int,
    lat_range: tuple | None = None,
    lon_range: tuple | None = None,
) -> None:
    """
    Initialize forecast store with coords + empty `lv`.

    Assumes init_times is already validated, monotonic,
    and dtype datetime64[ns].
    """

    n_feature = 1024

    # --------------------------------------------------
    # Spatial handling
    # --------------------------------------------------

    if lat_range and lon_range:
        spatial_coord = get_spatial_indices_from_bounds(
            lat_range=lat_range,
            lon_range=lon_range,
        )
        n_spatial = len(spatial_coord)
    else:
        spatial_coord = np.arange(259200, dtype="int64")
        n_spatial = 259200

    # --------------------------------------------------
    # Lead / valid time
    # --------------------------------------------------

    lead_times = (np.arange(1, rollout_steps + 1) * 6).astype("int64")
    valid_times = init_times[:, None] + lead_times[None, :] * np.timedelta64(1, "h")

    # --------------------------------------------------
    # Coordinate dataset
    # --------------------------------------------------

    coord_ds = xr.Dataset(
        coords={
            "init_time": ("init_time", init_times),
            "lead_time": ("lead_time", lead_times),
            "spatial_location": ("spatial_location", spatial_coord),
            "feature": ("feature", np.arange(n_feature, dtype="int64")),
            "valid_time": (("init_time", "lead_time"), valid_times),
        },
        attrs={
            "description": "Aurora latent forecast dataset",
            "rollout_steps": int(rollout_steps),
            "valid_time_range": (
                str(init_times.min()),
                str(init_times.max()),
            ),
            # ---- spatial intent ----
            "spatial_subset_type": ("bounding_box" if lat_range and lon_range else "global"),
            "lat_range_requested": lat_range,
            "lon_range_requested": lon_range,
        },
    )

    coord_ds.to_zarr(
        store,
        zarr_format=3,
        mode="w",
        consolidated=False,
        write_empty_chunks=False,
        encoding={
            "init_time": {"chunks": (len(init_times),)},
            "lead_time": {"chunks": (len(lead_times),)},
            "valid_time": {"chunks": (len(init_times), len(lead_times))},
            "spatial_location": {"chunks": (n_spatial,)},
            "feature": {"chunks": (n_feature,)},
        },
    )

    zarr.create_array(
        store,
        name="lv",
        shape=(len(init_times), len(lead_times), n_spatial, n_feature),
        chunks=(1, len(lead_times), 1024, 128),
        dtype="float32",
        fill_value=np.nan,
        compressors=[],
        dimension_names=("init_time", "lead_time", "spatial_location", "feature"),
    )

    print("[INIT] wrote coords + empty lv")


def build_init_times(
    *,
    start_time,
    end_time,
    step_hours: int = 6,
    init_hour: int | None = None,
    init_time_range: tuple | None = None,
) -> np.ndarray:
    """
    Build a canonical, validated init_time axis.

    This is the ONLY place where init_times are constructed or filtered.
    """

    # ---------------------------
    # Normalize inputs
    # ---------------------------

    start_time = np.datetime64(start_time, "ns")
    end_time = np.datetime64(end_time, "ns")

    if start_time > end_time:
        raise ValueError("start_time must be <= end_time")

    if init_hour is not None and init_hour not in (0, 6, 12, 18):
        raise ValueError("init_hour must be one of 0, 6, 12, 18")

    # ---------------------------
    # Build full grid
    # ---------------------------

    times = np.arange(
        start_time,
        end_time + np.timedelta64(step_hours, "h"),
        np.timedelta64(step_hours, "h"),
        dtype="datetime64[ns]",
    )

    # ---------------------------
    # Optional hour filter
    # ---------------------------

    if init_hour is not None:
        hours = times.astype("datetime64[h]").astype(int) % 24
        times = times[hours == init_hour]

    # ---------------------------
    # Optional bounds filter
    # ---------------------------

    if init_time_range is not None:
        t0, t1 = init_time_range
        t0 = np.datetime64(t0, "ns")
        t1 = np.datetime64(t1, "ns")

        times = times[(times >= t0) & (times <= t1)]

    # ---------------------------
    # Final validation
    # ---------------------------

    if times.size == 0:
        raise ValueError("No init_times remain after filtering.")

    if not np.all(np.diff(times) >= np.timedelta64(0, "ns")):
        raise ValueError("init_times must be monotonic")

    return times


@cli.command("init")
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--dest-repo", required=True, show_default=True)
@click.option("--dest-branch", default="main", show_default=True)
@click.option("--src-repo", required=True, show_default=True)
@click.option("--src-branch", default="main", show_default=True)
@click.option("--rollout-steps", type=int, default=10, show_default=True)
@click.option("--lat-min", type=float, default=None)
@click.option("--lat-max", type=float, default=None)
@click.option("--lon-min", type=float, default=None)
@click.option("--lon-max", type=float, default=None)
@click.option(
    "--init-hour",
    type=int,
    default=None,
    help="If provided, keep only init_times at this UTC hour (e.g. 6 for 06Z).",
)
def init(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    dest_repo: str,
    dest_branch: str,
    src_repo: str,
    src_branch: str,
    rollout_steps: int,
    lat_min: float | None,
    lat_max: float | None,
    lon_min: float | None,
    lon_max: float | None,
    init_hour: int | None,
):
    """
    Initialize latent forecast Zarr store:

    Dimensions:
    init_time:          N_init
    lead_time:          rollout_steps
    spatial_location:   N_spatial
    feature:            N_feature

    Coordinates:
    * init_time         (init_time) datetime64[ns]
    * lead_time         (lead_time) int64
    * spatial_location  (spatial_location) int64
    * feature           (feature) int64
        valid_time        (init_time, lead_time) datetime64[ns]

    Data variables:
        latent_forecast   (init_time, lead_time, spatial_location, feature) float32

    """
    init_times = build_init_times(
        start_time=start_time,
        end_time=end_time,
        init_hour=init_hour,
    )
    client = arraylake.Client()

    # Get repository or create it if it doesn't exist
    try:
        dest_repo_obj = client.get_repo(dest_repo)
        print(f"[INIT] Using existing repo {dest_repo}")
    except Exception:
        dest_repo_obj = client.create_repo(dest_repo)
        print(f"[INIT] Created repo {dest_repo}")

    # Create a writable session
    session = dest_repo_obj.writable_session(dest_branch)

    lat_range = (lat_min, lat_max)
    lon_range = (lon_min, lon_max)

    # Delegate all real work to helper function
    init_forecast_zarr_store(
        store=session.store,
        init_times=init_times,
        rollout_steps=rollout_steps,
        lat_range=lat_range,
        lon_range=lon_range,
    )

    # Commit
    commit_id = session.commit("Initialized latent vector store.")
    print(f"[INIT] Committed: {commit_id} to {dest_repo}:{dest_branch}")

    # print what data loks like
    print(xr.open_zarr(session.store, zarr_format=3, consolidated=False, chunks=None))


@cli.command("save-lvs")
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--src-repo", default=SOURCE_REPO, show_default=True)
@click.option("--src-branch", default=SOURCE_BRANCH, show_default=True)
@click.option("--dest-repo", default=DESTINATION_REPO, show_default=True)
@click.option("--dest-branch", default=DESTINATION_BRANCH, show_default=True)
@click.option("--write-session-location", type=str, default=None)
@click.option("--aws-profile", type=str, default="kafou", show_default=True)
@click.option("--rollout-steps", type=int, default=10, show_default=True)
@click.option("--lat-min", type=float, default=None)
@click.option("--lat-max", type=float, default=None)
@click.option("--lon-min", type=float, default=None)
@click.option("--lon-max", type=float, default=None)
@click.option("--init-hour", type=int, default=None)
def save_lvs(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_repo: str,
    src_branch: str,
    dest_repo: str,
    dest_branch: str,
    write_session_location: str | None,
    aws_profile: str,
    rollout_steps: int,
    lat_min: float | None,
    lat_max: float | None,
    lon_min: float | None,
    lon_max: float | None,
    init_hour: int | None,
):
    # --------------------------------------------------
    # Canonical init_times (single source of truth)
    # --------------------------------------------------

    init_times = build_init_times(
        start_time=start_time,
        end_time=end_time,
        init_hour=init_hour,
    )

    init_times64 = init_times.astype("datetime64[ns]")
    print(f"[SAVE_LVS] init_times to process: {init_times64}")
    print(f"[SAVE_LVS] lead_times to process: {np.arange(1, rollout_steps + 1) * 6}")

    # --------------------------------------------------
    # Session handling
    # --------------------------------------------------

    client = arraylake.Client()

    if write_session_location is not None:
        fs = fsspec.filesystem("s3", profile=aws_profile)
        with fs.open(os.path.join(write_session_location, "session.pickle"), "rb") as f:
            dest_session = pickle.load(f)
    else:
        repo = client.get_repo(dest_repo)
        dest_session = repo.writable_session(dest_branch)

    # --------------------------------------------------
    # Open store + validate coverage
    # --------------------------------------------------

    ds_store = xr.open_zarr(
        dest_session.store,
        zarr_format=3,
        consolidated=False,
        chunks=None,
    )
    root = zarr.open_group(dest_session.store, zarr_format=3)
    saved_root_attrs = dict(root.attrs)

    store_init = ds_store["init_time"].values.astype("datetime64[ns]")
    store_lead = ds_store["lead_time"].values.astype("int64")

    # Coverage check (vectorized, once)
    if init_times64.min() < store_init.min() or init_times64.max() > store_init.max():
        raise RuntimeError(
            f"Store init_time coverage {store_init.min()}..{store_init.max()} "
            f"does not cover requested {init_times64.min()}..{init_times64.max()}. "
            "Did you run `init` with the same bounds?"
        )

    # --------------------------------------------------
    # Latent extractor
    # --------------------------------------------------

    lve = LatentVectorExtractor(
        source_repo=src_repo,
        source_branch=src_branch,
        client=client,
    )

    # --------------------------------------------------
    # Canonical spatial_indices (for post-inference slicing)
    # --------------------------------------------------

    if any(v is not None for v in (lat_min, lat_max, lon_min, lon_max)):
        if None in (lat_min, lat_max, lon_min, lon_max):
            raise click.ClickException("lat/lon bounds must be complete pairs")

        spatial_indices = get_spatial_indices_from_bounds(
            lat_range=(lat_min, lat_max),
            lon_range=(lon_min, lon_max),
        )
    else:
        spatial_indices = None

    store_spatial = ds_store["spatial_location"].values.astype("int64")

    if spatial_indices is not None:
        if not np.array_equal(store_spatial, spatial_indices):
            raise RuntimeError(
                "Spatial indices derived from lat/lon bounds do not match "
                "the destination store.\n"
                "Did you run `init` with the same spatial bounds?"
            )

    # --------------------------------------------------
    # Write loop
    # --------------------------------------------------

    for init_time64 in init_times64:
        # ---- full inference (positional spatial axis) ----
        lv_ds = lve.rollout_lvs(
            item=init_time64.astype("datetime64[s]").item(),
            steps=rollout_steps,
        )

        # ---- enforce lead_time compatibility ----
        # TODO: This is where we would subset leadtime
        if not np.array_equal(lv_ds["lead_time"].values.astype("int64"), store_lead):
            lv_ds = lv_ds.reindex({"lead_time": ds_store["lead_time"]})
            if lv_ds["lv"].isnull().any():
                raise ValueError("After reindex, lv contains NaNs — lead_time mismatch.")

        # ---- spatial subsetting  ----
        lv_arr = lv_ds["lv"].values  # (lead_time, spatial_location, feature)

        if spatial_indices is not None:
            lv_arr = lv_arr[:, spatial_indices, :]

        # ---- build labeled cube that matches the store exactly ----
        lv_da = xr.DataArray(
            lv_arr[None, ...],
            dims=("init_time", "lead_time", "spatial_location", "feature"),
            coords={
                "init_time": ("init_time", np.array([init_time64], dtype="datetime64[ns]")),
                "lead_time": ds_store["lead_time"],
                "spatial_location": ds_store["spatial_location"],
                "feature": ds_store["feature"],
            },
            name="lv",
        )

        lv_da.to_zarr(
            dest_session.store,
            zarr_format=3,
            consolidated=False,
            mode="a",
            region="auto",
        )

    # --------------------------------------------------
    # Finalize
    # --------------------------------------------------

    root = zarr.open_group(dest_session.store, zarr_format=3)
    root.attrs.clear()
    root.attrs.update(saved_root_attrs)

    if write_session_location is None:
        commit_id = dest_session.commit(f"Added {init_times64.min()} to {init_times64.max()}")
        print(f"[SAVE_LVS] Committed: {commit_id}")
    else:
        outpath = os.path.join(
            write_session_location,
            f"lv_{str(init_times64.min()).replace(':', '').replace('-', '').split('.')[0]}_"
            f"{str(init_times64.max()).replace(':', '').replace('-', '').split('.')[0]}.pickle",
        )

        fs = fsspec.filesystem("s3", profile=aws_profile)
        with fs.open(outpath, "wb") as f:
            pickle.dump(dest_session, f)
        print(f"[SAVE_LVS] Wrote session pickle: {outpath}")


@cli.command("submit-jobs")
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--src-repo", default=SOURCE_REPO, show_default=True)
@click.option("--src-branch", default=SOURCE_BRANCH, show_default=True)
@click.option("--dest-repo", default=DESTINATION_REPO, show_default=True)
@click.option("--dest-branch", default=DESTINATION_BRANCH, show_default=True)
@click.option("--aws-profile", type=str, default="kafou", show_default=True)
@click.option("--rollout-steps", type=int, default=10, show_default=True)
@click.option("--lat-min", type=float, default=None)
@click.option("--lat-max", type=float, default=None)
@click.option("--lon-min", type=float, default=None)
@click.option("--lon-max", type=float, default=None)
@click.option("--init-hour", type=int, default=None)
@click.option(
    "--coordination-location",
    type=str,
    default="s3://icechunk-write-coordination",
    show_default=True,
)
@click.option(
    "--timesteps-per-job",
    type=int,
    default=8,
    show_default=True,
    help="Number of 6h init_times per SLURM job",
)
def submit_jobs(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_repo: str,
    src_branch: str,
    dest_repo: str,
    dest_branch: str,
    aws_profile: str,
    coordination_location: str,
    timesteps_per_job: int,
    rollout_steps: int,
    lat_min: float | None,
    lat_max: float | None,
    lon_min: float | None,
    lon_max: float | None,
    init_hour: int | None,
):
    """
    Distributed latent-forecast generation via SLURM.

    Contract:
      - `init` MUST already have been run
      - Each job runs `save-lvs` on a disjoint init_time span
      - All jobs write into the same Icechunk session
      - Final merge + commit is atomic
    """

    # ------------------------------------------------------------
    # Validate times
    # ------------------------------------------------------------
    for t, name in [(start_time, "start_time"), (end_time, "end_time")]:
        if t.hour not in (0, 6, 12, 18) or t.minute or t.second or t.microsecond:
            raise click.ClickException(f"{name} must be 6-hour aligned")

    if start_time > end_time:
        raise click.ClickException("start_time must be <= end_time")

    # ------------------------------------------------------------
    # Build init_time chunks
    # ------------------------------------------------------------
    init_times: list[datetime.datetime] = []
    t = start_time
    while t <= end_time:
        init_times.append(t)
        t += datetime.timedelta(hours=6)

    chunks = [
        init_times[i : i + timesteps_per_job] for i in range(0, len(init_times), timesteps_per_job)
    ]

    time_spans: list[tuple[datetime.datetime, datetime.datetime]] = [
        (chunk[0], chunk[-1]) for chunk in chunks
    ]

    # ------------------------------------------------------------
    # Create shared Icechunk session
    # ------------------------------------------------------------
    lv_job_id = random_job_string()
    session_location = os.path.join(coordination_location, lv_job_id)
    session_pickle = os.path.join(session_location, "session.pickle")

    client = arraylake.Client()
    repo = client.get_repo(dest_repo)
    base_session = repo.writable_session(dest_branch)

    fs = fsspec.filesystem("s3", profile=aws_profile)
    fs.makedirs(session_location, exist_ok=True)

    print(f"[SUBMIT] Saving base session → {session_pickle}")
    with fs.open(session_pickle, "wb") as f:
        with base_session.allow_pickling():
            pickle.dump(base_session, f)

    # ------------------------------------------------------------
    # Submit SLURM jobs
    # ------------------------------------------------------------
    job_prefix = f"lv-{lv_job_id}"
    expected_spans: set[tuple[str, str]] = set()

    for start, end in time_spans:
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%S")

        expected_spans.add(
            (
                start.strftime("%Y%m%dT%H%M%S"),
                end.strftime("%Y%m%dT%H%M%S"),
            )
        )

        cmd = [
            "sbatch",
            "--ntasks=1",
            "--cpus-per-task=32",
            "--gpus=1",
            f"--job-name={job_prefix}_{start_str}_{end_str}",
            sys.argv[0],
            "save-lvs",
            start_str,
            end_str,
            f"--src-repo={src_repo}",
            f"--src-branch={src_branch}",
            f"--dest-repo={dest_repo}",
            f"--dest-branch={dest_branch}",
            f"--rollout-steps={rollout_steps}",
            f"--aws-profile={aws_profile}",
            f"--write-session-location={session_location}",
        ]

        # ---- spatial bounds (must be complete pairs) ----
        if any(v is not None for v in (lat_min, lat_max, lon_min, lon_max)):
            if None in (lat_min, lat_max, lon_min, lon_max):
                raise RuntimeError("lat/lon bounds must be complete pairs")

            cmd.append(f"--lat-min={lat_min}")
            cmd.append(f"--lat-max={lat_max}")
            cmd.append(f"--lon-min={lon_min}")
            cmd.append(f"--lon-max={lon_max}")

        # ---- optional init hour ----
        if init_hour is not None:
            cmd.append(f"--init-hour={init_hour}")

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"sbatch failed:\n{res.stderr}")

    # ------------------------------------------------------------
    # Wait for jobs to finish
    # ------------------------------------------------------------
    print("[SUBMIT] Waiting for SLURM jobs to finish...")
    time.sleep(10)

    while True:
        remaining = get_job_count(job_prefix)
        if remaining == 0:
            break
        print(f"[SUBMIT] {remaining} jobs remaining...")
        time.sleep(60)

    # ------------------------------------------------------------
    # Merge completed sessions
    # ------------------------------------------------------------
    print("[SUBMIT] All jobs complete. Merging sessions...")

    sessions = []
    for path in fs.ls(session_location):
        fname = os.path.basename(path)
        if fname.startswith("lv_") and fname.endswith(".pickle"):
            _, s, e = fname.replace(".pickle", "").split("_")
            expected_spans.discard((s, e))
            with fs.open(path, "rb") as f:
                sessions.append(pickle.load(f))
            fs.rm(path)

    merged = merge_sessions(base_session, *sessions)

    # ------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------
    # if expected_spans:
    #     commit_msg = (
    #         f"PARTIAL save-lvs {start_time.isoformat()}..{end_time.isoformat()} "
    #         f"(missing {len(expected_spans)} spans)"
    #     )
    # else:
    #     commit_msg = f"save-lvs {start_time.isoformat()}..{end_time.isoformat()}"

    commit_msg = f"save-lvs {start_time.isoformat()}..{end_time.isoformat()}"

    commit_id = merged.commit(commit_msg)
    print(f"[SUBMIT] Commit complete: {commit_id}")

    # if expected_spans:
    #     print("[SUBMIT] Missing spans:")
    #     for s, e in sorted(expected_spans):
    #         print(f"  {s} → {e}")


if __name__ == "__main__":
    cli()
