#!/usr/bin/env python

import datetime
import os
import pickle
import random
import subprocess
import sys
import time

import click
import fsspec
import icechunk
import kafou_arraylake as arraylake
import numpy as np
import torch
import xarray as xr
import zarr
from dataset_io import ensure_time_in_arrays

from aurora import AuroraPretrained
from aurora.data import ERA5DataLoaderFOAM
from aurora.rollout import rollout_with_latents

os.environ["PYTHONUNBUFFERED"] = "1"


import numpy as np


def _to_iso(dt_like):
    if isinstance(dt_like, np.datetime64):
        return np.datetime_as_string(dt_like, unit="s")
    elif isinstance(dt_like, datetime.datetime):
        return dt_like.isoformat()
    else:
        raise TypeError(f"Unsupported datetime type: {type(dt_like)}")


def get_batches(times: np.ndarray, n: int):
    """
    Generate batches of timestamps from ds between start_time and end_time, n at a time.
    """
    for i in range(0, len(times), n):
        yield times[i : i + n]


def get_clamped_time_range(
    ds: xr.Dataset,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> tuple[datetime.datetime, datetime.datetime]:
    """
    Clamp (start_time, end_time) to the available time range in ds.
    Raises if start_time is entirely outside the dataset.
    """
    # time_coord = ds[next(c for c in ds.coords if np.issubdtype(ds[c].dtype, np.datetime64))]

    # src_end = (
    #     time_coord.isel({time_coord.dims[0]: -1})
    #     .item()
    #     .astype("datetime64[ns]")
    #     .astype(datetime.datetime)
    # )

    # print(start_time)
    # print(f"What data type is start_time? {type(start_time)}")
    # print(src_end)
    # print(f"What data type is src_end? {type(src_end)}")
    # if start_time > src_end:
    #     raise RuntimeError(
    #         f"start_time={start_time.isoformat()} is after last available "
    #         f"source timestamp={src_end.isoformat()}"
    #     )

    # clamped_end = min(end_time, src_end)

    # if clamped_end < end_time:
    #     print(
    #         f"[SUBMIT] Clamping end_time → {clamped_end.isoformat()} "
    #         f"(source max={src_end.isoformat()})"
    #     )

    return start_time, end_time


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


def commit_with_retries(session, msg: str, max_attempts: int = 6):
    for attempt in range(1, max_attempts + 1):
        try:
            return session.commit(msg)
        except (RuntimeError, icechunk.IcechunkError):
            if attempt == max_attempts:
                raise
            time.sleep(2**attempt)


def set_metadata(
    *,
    session,
    start_time: datetime.datetime | np.datetime64 | None = None,
    end_time: datetime.datetime | np.datetime64 | None = None,
    extra: dict | None = None,
) -> None:
    # Canonicalize to ISO STRINGS ONCE
    start_s = _to_iso(start_time) if start_time is not None else None
    end_s = _to_iso(end_time) if end_time is not None else None

    root = zarr.open_group(session.store, zarr_format=3)

    attrs = dict(root.attrs)

    old = attrs.get("valid_times")

    if old:
        attrs["valid_times"] = [
            min(old[0], start_s) if start_s is not None else old[0],
            max(old[1], end_s) if end_s is not None else old[1],
        ]
    elif start_s is None and end_s is None:
        attrs["valid_times"] = [None, None]
    else:
        attrs["valid_times"] = [start_s, end_s]

    attrs["last_updated"] = datetime.datetime.now().isoformat()

    if extra:
        attrs.update(extra)

    root.attrs.clear()
    root.attrs.update(attrs)


def get_spatial_indices_from_bounds(
    *,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    n_lev: int = 4,
    n_lat: int = 180,
    n_lon: int = 360,
    expected_spatial: np.ndarray | None = None,
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

    # Coordinate grids (centers)
    lat_vals = np.linspace(89.5, -89.5, n_lat)  # north → south
    lon_vals_0360 = np.linspace(0.5, 359.5, n_lon)  # 0–360
    lon_vals = ((lon_vals_0360 + 180) % 360) - 180  # −180…180

    # Latitude mask
    lat_mask = (lat_vals >= lat_min) & (lat_vals <= lat_max)
    lat_idx = np.where(lat_mask)[0]

    if lat_idx.size == 0:
        raise ValueError("Latitude range selects no grid points.")

    # Longitude mask (wrap-aware)
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

    # Build flat spatial_location indices
    # lev-major, then lat, then lon
    spatial_indices = np.concatenate(
        [lev * n_lat * n_lon + lat * n_lon + lon_idx for lev in range(n_lev) for lat in lat_idx]
    ).astype(np.int64)

    spatial_indices = spatial_indices.astype(np.int64)

    if expected_spatial is not None:
        expected_spatial = expected_spatial.astype(np.int64)
        if not np.array_equal(expected_spatial, spatial_indices):
            raise RuntimeError(
                "Spatial indices derived from lat/lon bounds do not match "
                "the destination store.\n"
                "Did you run `init` with the same spatial bounds?"
            )

    return spatial_indices


def get_or_create_repo_branch(
    *,
    client: arraylake.Client,
    repo_name: str,
    branch_name: str,
    base_branch: str = "main",
):
    """
    Open an Arraylake repo and branch, creating them if they do not exist.
    """
    # Repo
    try:
        repo = client.get_repo(repo_name)
        print(f"✓ Opened existing repo: {repo_name}")
    except Exception:
        repo = client.create_repo(repo_name)
        print(f"✓ Created new repo: {repo_name}")

    # Branch
    branches = repo.list_branches()
    if branch_name not in branches:
        print(f"Branch '{branch_name}' does not exist — creating from '{base_branch}'")
        base_commit = repo.lookup_branch(base_branch)
        repo.create_branch(branch_name, base_commit)
    else:
        print(f"✓ Branch '{branch_name}' already exists")

    # Writable session
    session = repo.writable_session(branch_name)
    return repo, session


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


def build_init_times(
    start_time,
    end_time,
    init_hour: int | None = None,
    step_hours: int = 6,
) -> np.ndarray:
    """
    Build a canonical, validated init_time axis.

    This is the ONLY place where init_times are constructed or filtered.
    """
    # Normalize inputs
    start_time = np.datetime64(start_time, "ns")
    end_time = np.datetime64(end_time, "ns")

    if start_time > end_time:
        raise ValueError("start_time must be <= end_time")

    if init_hour is not None and init_hour not in (0, 6, 12, 18):
        raise ValueError("init_hour must be one of 0, 6, 12, 18")

    # Build full grid
    times = np.arange(
        start_time,
        end_time + np.timedelta64(step_hours, "h"),
        np.timedelta64(step_hours, "h"),
        dtype="datetime64[ns]",
    )

    # Optional hour filter
    if init_hour is not None:
        hours = times.astype("datetime64[h]").astype(int) % 24
        times = times[hours == init_hour]

    # Final validation
    if times.size == 0:
        raise ValueError("No init_times remain after filtering.")

    if not np.all(np.diff(times) >= np.timedelta64(0, "ns")):
        raise ValueError("init_times must be monotonic")

    return times


def init_store(
    *,
    store,
    init_times: np.ndarray,
    rollout_steps: int,
    lat_range: tuple | None = None,
    lon_range: tuple | None = None,
) -> None:
    """
    Initialize forecast store with coords + empty `lv`.

    Creates:
        Dimensions:         (feature, init_time, lead_time, spatial_location)
        Coordinates:
            * feature       (feature) int64
            * init_time     (init_time) datetime64[ns]
            * lead_time     (lead_time) int64
            * spatial_location  (spatial_location) int64
        Data variables:
            lv              (init_time, lead_time, spatial_location, feature) float32
    """
    print("[INIT] Initializing destination store...")

    n_feature = 1024

    # Spatial handling
    if (
        lat_range is not None
        and lon_range is not None
        and lat_range[0] is not None
        and lat_range[1] is not None
        and lon_range[0] is not None
        and lon_range[1] is not None
    ):
        print(f"[INIT] Subsetting coordinates to lat=({lat_range}), lon=({lon_range})")
        spatial_coord = get_spatial_indices_from_bounds(
            lat_range=lat_range,
            lon_range=lon_range,
        )
        n_spatial = len(spatial_coord)
    else:
        spatial_coord = np.arange(259200, dtype="int64")
        n_spatial = 259200

    # Lead / valid time
    lead_times = (np.arange(1, rollout_steps + 1) * 6).astype("int64")
    valid_times = init_times[:, None] + lead_times[None, :] * np.timedelta64(1, "h")

    # Coordinate dataset
    print("[INIT] Writing coordinates...")
    coord_ds = xr.Dataset(
        coords={
            "init_time": ("init_time", init_times),
            "lead_time": ("lead_time", lead_times),
            "spatial_location": ("spatial_location", spatial_coord),
            "feature": ("feature", np.arange(n_feature, dtype="int64")),
        },
        attrs={
            "description": "Aurora latent vector forecast dataset.",
            "rollout_steps": int(rollout_steps),
            "lead_times": lead_times,
            "spatial_subset_type": ("bounding_box" if lat_range and lon_range else "global"),
            "lat_range": (lat_range if lat_range and lon_range else None),
            "lon_range": (lon_range if lat_range and lon_range else None),
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
            "spatial_location": {"chunks": (n_spatial,)},
            "feature": {"chunks": (n_feature,)},
        },
    )

    # TODO: tune chunking strategy ??
    # we want to store in 1 MB chunks to optimize for read performance.
    print("[INIT] Createing lv DataArray...")
    print(f"      shape  = ({len(init_times)}, {len(lead_times)}, {n_spatial}, {n_feature})")
    print(f"      chunks = (1, {len(lead_times)}, 1024, 128)")

    compressors = [zarr.codecs.BloscCodec(clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)]
    zarr.create_array(
        store,
        name="lv",
        shape=(len(init_times), len(lead_times), n_spatial, n_feature),
        chunks=(1, len(lead_times), 1024, 128),
        dtype="float32",
        fill_value=np.nan,
        compressors=compressors,
        dimension_names=("init_time", "lead_time", "spatial_location", "feature"),
    )

    print("[INIT] Initialization complete.")


@click.group()
def cli():
    pass


@cli.command("init")
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--dst-repo", required=True, show_default=True)
@click.option("--dst-branch", default="main", show_default=True)
@click.option("--src-repo", required=True, show_default=True)
@click.option("--src-branch", default="main", show_default=True)
@click.option("--rollout-steps", type=int, default=10, show_default=True)
@click.option("--lat-min", type=float, default=None)
@click.option("--lat-max", type=float, default=None)
@click.option("--lon-min", type=float, default=None)
@click.option("--lon-max", type=float, default=None)
@click.option("--force-init", is_flag=True, default=False)
@click.option(
    "--init-hour",
    type=int,
    default=None,
    help="If provided, keep only init_times at this UTC hour (e.g. 6 for 06Z).",
)
def init(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    dst_repo: str,
    dst_branch: str,
    src_repo: str,
    src_branch: str,
    rollout_steps: int,
    lat_min: float | None,
    lat_max: float | None,
    lon_min: float | None,
    lon_max: float | None,
    init_hour: int | None,
    force_init: bool,
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

    Data variables:
        latent_forecast   (init_time, lead_time, spatial_location, feature) float32

    """

    client = arraylake.Client()

    # Get or create destination repo + branch
    print(f"[INIT] Opening/creating destination repo={dst_repo} branch={dst_branch}")
    repo, session = get_or_create_repo_branch(
        client=client,
        repo_name=dst_repo,
        branch_name=dst_branch,
        base_branch="main",
    )

    # Check for existing schema, Delete if forced
    root = zarr.open_group(session.store, mode="a")
    if force_init:
        print("⚠️  --force-init enabled: deleting existing schema")
        for key in ("samples", "invariant"):
            if key in root:
                del root[key]

    init_times = build_init_times(
        start_time=start_time,
        end_time=end_time,
        init_hour=init_hour,
    )

    # Delegate all real work to helper function
    init_store(
        store=session.store,
        init_times=init_times,
        rollout_steps=rollout_steps,
        lat_range=(lat_min, lat_max),
        lon_range=(lon_min, lon_max),
    )

    # Add Metadata
    set_metadata(session=session, start_time=start_time, end_time=end_time)

    # Commit
    commit_id = session.commit("Initialized latent vector store.")
    print(f"[INIT] Committed: {commit_id} to {dst_repo}:{dst_branch}")

    # print what data loks like
    print(xr.open_zarr(session.store, zarr_format=3, consolidated=False, chunks=None))


@cli.command("lv-worker")
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--src-repo", required=True)
@click.option("--src-branch", required=True)
@click.option("--dst-repo", required=True)
@click.option("--dst-branch", required=True)
@click.option("--fork-pickle", type=str, default=None)
@click.option("--aws-profile", type=str, default="kafou", show_default=True)
def lv_worker(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_repo: str,
    src_branch: str,
    dst_repo: str,
    dst_branch: str,
    fork_pickle: str | None,
    aws_profile: str,
):
    print("\n" + "=" * 80)
    print(f"[BATCH] {start_time:%Y-%m-%d} → {end_time:%Y-%m-%d}")
    print("=" * 80)

    # --------------------------------------------------
    # Session handling
    # --------------------------------------------------
    print("▶ [SESSION] initializing client + destination session")
    client = arraylake.Client()

    if fork_pickle is not None:
        print("    ↳ loading fork session pickle")
        fs = fsspec.filesystem("s3", profile="kafou")
        with fs.open(fork_pickle, "rb") as fobj:
            dst_session = pickle.load(fobj)
        print("    ✔ fork session loaded")
    else:
        print("    ↳ opening writable destination session")
        repo = client.get_repo(dst_repo)
        dst_session = repo.writable_session(dst_branch)
        print("    ✔ destination session opened")

    # --------------------------------------------------
    # Get Destination Session Store
    # --------------------------------------------------
    ds_store = xr.open_zarr(
        dst_session.store,
        zarr_format=3,
        consolidated=False,
        chunks=None,
    )

    spatial_indices = ds_store.spatial_location.values.tolist()

    # --------------------------------------------------
    # Get Timestamps ?
    # --------------------------------------------------
    lead_times = ds_store["lead_time"].values.astype("int64")
    init_times = ds_store["init_time"].sel(init_time=slice(start_time, end_time)).values

    rollout_steps = len(lead_times)
    print(f"[lv_worker] init_times to process: {init_times}")
    print(f"[lv_worker] lead_times to process: {np.arange(1, rollout_steps + 1) * 6}")

    # --------------------------------------------------
    # Forecast
    # --------------------------------------------------
    lve = LatentVectorExtractor(
        source_repo=src_repo,
        source_branch=src_branch,
        client=client,
    )

    for init_time in init_times:
        # ---- full inference (positional spatial axis) ----
        lv_ds = lve.rollout_lvs(
            item=init_time.astype("datetime64[s]").item(),
            steps=rollout_steps,
        )

        # ---- enforce lead_time compatibility ----
        # TODO: This is where we would subset leadtime
        if not np.array_equal(lv_ds["lead_time"].values.astype("int64"), lead_times):
            lv_ds = lv_ds.reindex({"lead_time": ds_store["lead_time"]})
            if lv_ds["lv"].isnull().any():
                raise ValueError("After reindex, lv contains NaNs — lead_time mismatch.")

        # ---- spatial subsetting  ----
        lv_arr = lv_ds["lv"].values  # (lead_time, spatial_location, feature)
        if len(spatial_indices) != (4 * 180 * 360):
            lv_arr = lv_arr[:, spatial_indices, :]

        # ---- build labeled cube that matches the store exactly ----
        lv_da = xr.DataArray(
            lv_arr[None, ...],
            dims=("init_time", "lead_time", "spatial_location", "feature"),
            coords={
                "init_time": ("init_time", np.array([init_time], dtype="datetime64[ns]")),
                "lead_time": ds_store["lead_time"],
                "spatial_location": ds_store["spatial_location"],
                "feature": ds_store["feature"],
            },
            name="lv",
        )

        lv_da.to_zarr(
            dst_session.store,
            zarr_format=3,
            consolidated=False,
            mode="a",
            region="auto",
        )

    # --------------------------------------------------
    # Write out session pickle
    # --------------------------------------------------
    if fork_pickle is None:
        commit_id = dst_session.commit(f"Added {init_times.min()} to {init_times.max()}")
        print(f"[lv_worker] Committed session: {commit_id}")

    else:
        outpath = os.path.join(
            fork_pickle,
            f"lv_{start_time:%Y%m%dT%H%M%S}_{end_time:%Y%m%dT%H%M%S}_.pickle",
        )
        fs = fsspec.filesystem("s3", profile=aws_profile)
        print(f"Writing {outpath}")
        with fs.open(outpath, "wb") as fobj:
            pickle.dump(dst_session, fobj)
        print(f"[lv_worker] wrote session {outpath}")


@cli.command("submit-jobs")
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--src-repo", required=True)
@click.option("--src-branch", required=True)
@click.option("--dst-repo", required=True)
@click.option("--dst-branch", required=True)
@click.option("--aws-profile", type=str, default="kafou", show_default=True)
@click.option("--init-hour", type=int, default=None)
@click.option(
    "--coordination-location",
    type=str,
    default="s3://icechunk-write-coordination",
    show_default=True,
)
@click.option(
    "--times-at-once",
    type=int,
    default=8,
    show_default=True,
)
def submit_jobs(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_repo: str,
    src_branch: str,
    dst_repo: str,
    dst_branch: str,
    init_hour: int | None,
    aws_profile: str,
    coordination_location: str,
    times_at_once: int,
):
    """
    Distributed latent-forecast generation via SLURM.

    Contract:
      - `init` MUST already have been run
      - Each job runs `lv-worker` on a disjoint init_time span
      - All jobs write into the same Icechunk session
      - Final merge + commit is atomic
    """
    print("\n" + "=" * 80)
    print(f"[SUBMIT] {start_time:%Y-%m-%d %H:%M} → {end_time:%Y-%m-%d %H:%M}")
    print("=" * 80)

    # --------------------------------------------------
    # Client + repo setup
    # --------------------------------------------------
    # Validate times
    client = arraylake.Client()
    repo = client.get_repo(dst_repo)
    base_session = repo.writable_session(dst_branch)
    ds = xr.open_zarr(base_session.store, zarr_format=3, consolidated=False, chunks=None)

    # --------------------------------------------------
    # Clamp time range
    # --------------------------------------------------
    print("▶ [TIME] clamping requested time range to source availability")
    orig_start, orig_end = start_time, end_time
    start_time, end_time = get_clamped_time_range(ds=ds, start_time=start_time, end_time=end_time)

    # --------------------------------------------------
    # Extend time axis
    # --------------------------------------------------
    print(f"▶ [DEST] ensuring time axis through {end_time}")
    base_session = repo.writable_session(dst_branch)
    msg = ensure_time_in_arrays(
        store=base_session.store,
        timestamp=end_time,
        time_dim="init_time",
        time_frequency="6h",
        init_hour=init_hour,
    )

    if msg is not None:
        commit_with_retries(session=base_session, msg=str(msg))

    # --------------------------------------------------
    # Fork base session
    # --------------------------------------------------
    print("▶ [SESSION] creating base fork for workers")
    base_session = repo.writable_session(dst_branch)
    fork = base_session.fork()
    print("  ✔ base session forked")

    # --------------------------------------------------
    # Batch planning
    # --------------------------------------------------
    # Build batches
    ds = xr.open_zarr(base_session.store, zarr_format=3, consolidated=False, chunks=None)
    init_times = ds["init_time"].sel(init_time=slice(start_time, end_time)).values
    batches = list(get_batches(init_times, times_at_once))
    print("▶ [BATCH] planning batches")
    print(f"  ↳ total batches: {len(batches)}")
    print(f"  ↳ timesteps per batch: {times_at_once}")

    if len(batches) > 0:
        print(f"  ↳ first batch: {batches[0][0]} → {batches[0][-1]}")
        print(f"  ↳ last batch:  {batches[-1][0]} → {batches[-1][-1]}")

    # --------------------------------------------------
    # Save base fork session
    # --------------------------------------------------
    job_id = random_job_string(10)
    session_location = os.path.join(coordination_location, job_id)
    fork_pickle = os.path.join(session_location, "fork.pickle")

    print("▶ [SESSION] saving base fork session")
    print(f"  ↳ job_id: {job_id}")
    print(f"  ↳ coordination dir: {session_location}")
    print(f"  ↳ fork pickle: {fork_pickle}")

    fs = fsspec.filesystem("s3", profile="kafou")
    with fs.open(fork_pickle, "wb") as fobj:
        pickle.dump(fork, fobj)

    print("  ✔ base fork pickle saved")

    # --------------------------------------------------
    # Submit SLURM jobs
    # --------------------------------------------------
    print("▶ [SLURM] submitting worker jobs")
    print(f"  ↳ total jobs: {len(batches)}")

    # Submit SLURM jobs
    print(f"[SUBMIT] Submitting {len(batches)} SLURM jobs ({times_at_once} timesteps per job)")
    job_prefix = f"lv_{job_id}"
    expected = set()

    for batch in batches:
        start, end = batch[0], batch[-1]  # datetime.datetime, datetime.datetime
        start_s, end_s = _to_iso(start), _to_iso(end)
        expected.add((start_s, end_s))

        cmd = [
            "sbatch",
            "--ntasks=1",
            "--cpus-per-task=32",
            "--gpus=1",
            f"--job-name={job_prefix}_{start_s}_{end_s}",
            sys.argv[0],
            "lv-worker",
            start_s,
            end_s,
            f"--src-repo={src_repo}",
            f"--src-branch={src_branch}",
            f"--dst-repo={dst_repo}",
            f"--dst-branch={dst_branch}",
            f"--aws-profile={aws_profile}",
            f"--fork-pickle={fork_pickle}",
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"sbatch failed:\n{res.stderr}")

    print("  ✔ all SLURM jobs submitted")

    print("▶ [SLURM] waiting for workers to finish")
    time.sleep(60)

    # Wait for jobs to finish
    print("[SUBMIT] Waiting for workers...")
    job_count = get_job_count(job_prefix)
    while job_count != 0:
        job_count = get_job_count(job_prefix)
        print(f"[SUBMIT] {job_count} jobs remaining ...")
        time.sleep(60)

    print("▶ [MERGE] collecting + merging worker sessions")

    time.sleep(10)

    worker_forks = []
    for fspath in fs.ls(session_location):
        if fspath.endswith(".worker.pickle"):
            print(f"  ↳ loading worker fork: {fspath}")
            with fs.open(fspath, "rb") as fobj:
                worker_forks.append(pickle.load(fobj))
        fs.rm(fspath)

    print(f"  ↳ loaded {len(worker_forks)} worker forks")

    print("  ▶ merging forks into base session")
    base_session.merge(*worker_forks)
    print("  ✔ forks merged")

    # Add Metadata
    valid_start = ds.init_time.values.min()
    set_metadata(session=base_session, start_time=valid_start, end_time=end_time)

    # Commit
    msg = f"latent-vectorized {start_s} → {end_s}"
    commit_id = commit_with_retries(session=base_session, msg=msg)
    print(f"  ✔ commit complete: {commit_id}")
    print("▶ [SUBMIT DONE]")


if __name__ == "__main__":
    cli()
