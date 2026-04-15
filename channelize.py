#!/usr/bin/env python

"""

conda activate aurora

# 14 days worth of 6h timesteps
python -u channelize.py submit-jobs 2025-11-30T18:00:00 2026-03-31T18:00:00 \
    --src-repo rwe/era5-0p25-6h-nonprod-ohio \
    --src-branch main \
    --dst-repo kafou/aurora-era5-samples \
    --dst-branch extend-2025 \
    --token ema_538dadd713b94095ad24386973e7b109_fbc2a0d2fae4df6553e6d6abbfb3f66130139de2707f3021e07e6dbb39890ef3 \
    --times-at-once 56 \
    --cpus 4 \
    --coordination-location s3://icechunk-write-coordination/extend-era5-samples

"""

import datetime
import logging
import os
import pickle
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import click
import fsspec
import icechunk
import kafou_arraylake as arraylake
import numpy as np
import xarray as xr
import zarr
from dataset_io import ensure_time_in_arrays
from zarr.core.config import config as zarr_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

xr.set_options(keep_attrs=True)

zarr_config.set({"async.concurrency": 16})

# Repo-level config: concurrency tuning for icechunk storage backend.
# Passed as `config=` to client.get_repo().
REPO_CONFIG = icechunk.RepositoryConfig.default()

# Storage-level config: network timeout tuning.
# minimum_throughput_bytes_per_second=0 disables the throughput floor check.
STORAGE_OPTIONS: dict = {
    "network_stream_timeout_seconds": 600,
}

INVARIANT_REPO = "rwe/era5-0p25-6h-nonprod-ohio"
S3_PROFILE = "kafou"

NAME_MAP = {
    "sfc": {
        "VAR_2T": "2t",
        "MSL": "msl",
        "VAR_10U": "10u",
        "VAR_10V": "10v",
    },
    "pl": {
        "Z": "z",
        "U": "u",
        "V": "v",
        "T": "t",
        "Q": "q",
    },
    "inv": {
        "LSM": "lsm",
        "Z": "z",
        "SLT": "slt",
    },
}


# --------#
# HELPERS #
# --------#
def random_job_string(length: int = 8) -> str:
    return secrets.token_hex(length // 2 + 1)[:length]


def s3_filesystem() -> fsspec.AbstractFileSystem:
    return fsspec.filesystem("s3", profile=S3_PROFILE)


def arraylake_client(token: str | None = None) -> arraylake.Client:
    return arraylake.Client(token=token) if token else arraylake.Client()


def open_src_session(
    *,
    src_repo: str,
    src_branch: str = "main",
    token: str | None = None,
) -> "arraylake.Session":
    """Open a readonly Icechunk session on the source repo.

    Separated from open_src_datasets to avoid the broken return-type
    ambiguity that session_only=True caused.
    """
    client = arraylake_client(token)
    src_repo_obj = client.get_repo(src_repo, config=REPO_CONFIG, storage_options=STORAGE_OPTIONS)
    return src_repo_obj.readonly_session(src_branch)


def open_src_datasets(
    *,
    src_repo: str,
    src_branch: str = "main",
    token: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, "arraylake.Session"]:
    """Returns: (sfc_ds, pl_ds, inv_ds, src_session)."""
    client = arraylake_client(token)

    repo_lower = src_repo.lower()
    if "era5" in repo_lower:
        dataset_type = "era5"
    elif "ecmwf" in repo_lower:
        dataset_type = "ecmwf"
    else:
        raise ValueError(f"Source repo must contain 'era5' or 'ecmwf': {src_repo}")

    src_repo_obj = client.get_repo(src_repo, config=REPO_CONFIG, storage_options=STORAGE_OPTIONS)
    src_session = src_repo_obj.readonly_session(src_branch)

    if dataset_type == "ecmwf":
        sfc_ds = xr.open_zarr(
            src_session.store,
            group="ENFO-T0",
            zarr_format=3,
            consolidated=False,
            chunks=None,
        )
        sfc_ds = sfc_ds.isel(ensemble_member=0, drop=True)
        if "lat" in sfc_ds.coords and "lon" in sfc_ds.coords:
            sfc_ds = sfc_ds.rename(
                {"lat": "latitude", "lon": "longitude", "pressure_level": "level"}
            )
        if "t2m" in sfc_ds.data_vars:
            sfc_ds = sfc_ds.rename(
                {"t2m": "VAR_2T", "slp": "MSL", "u10m": "VAR_10U", "v10m": "VAR_10V"}
            )
        pl_ds = sfc_ds.copy(deep=True)
        if "z" in pl_ds.data_vars:
            pl_ds = pl_ds.rename({"z": "Z", "u": "U", "v": "V", "t": "T", "q": "Q"})
        pl_ds = pl_ds.sortby("level")

    elif dataset_type == "era5":
        sfc_ds = xr.open_zarr(
            src_session.store,
            group="surface",
            zarr_format=3,
            consolidated=False,
            chunks=None,
        )
        pl_ds = xr.open_zarr(
            src_session.store,
            group="pressure_level",
            zarr_format=3,
            consolidated=False,
            chunks=None,
        )
        pl_ds = pl_ds.sortby("level")

    # Invariant is always from ERA5
    inv_repo_obj = client.get_repo(
        INVARIANT_REPO, config=REPO_CONFIG, storage_options=STORAGE_OPTIONS
    )
    inv_session = inv_repo_obj.readonly_session("main")
    inv_ds = xr.open_zarr(
        inv_session.store,
        group="invariant",
        zarr_format=3,
        consolidated=False,
        chunks=None,
    )

    return sfc_ds, pl_ds, inv_ds, src_session


def get_variable_locations(n_pressure_levels: int = 13) -> tuple[dict, dict]:
    """Return (in_locs, out_locs) channel index maps.

    Each maps {"sfc": {var: (start_idx, n_channels)}, "pl": {...}}.
    """
    in_locs: dict[str, dict[str, tuple[int, int]]] = {"sfc": {}, "pl": {}}
    out_locs: dict[str, dict[str, tuple[int, int]]] = {"sfc": {}, "pl": {}}

    i = 0
    for src_name, dst_name in NAME_MAP["sfc"].items():
        loc = (i, 1)
        in_locs["sfc"][src_name] = loc
        out_locs["sfc"][dst_name] = loc
        i += 1

    for src_name, dst_name in NAME_MAP["pl"].items():
        loc = (i, n_pressure_levels)
        in_locs["pl"][src_name] = loc
        out_locs["pl"][dst_name] = loc
        i += n_pressure_levels

    return in_locs, out_locs


def decode_zarr_time(grp: zarr.Group, time_var: str = "time") -> np.ndarray:
    """Decode zarr time array, handling CF conventions.

    Returns datetime64[ns] array.

    NOTE: This exists because xarray's CF decoder does not work reliably
    when reading raw zarr groups (outside of open_zarr). If the source
    stores switch to xarray-decodable time, this can be replaced with
    direct xarray reads.
    """
    import re

    time_arr = grp[time_var][:]
    time_attrs = dict(grp[time_var].attrs)
    units = time_attrs.get("units")
    calendar = time_attrs.get("calendar", "standard")

    logger.info(f"      time attrs: units={units}, calendar={calendar}")

    if units is not None:
        match = re.match(r"(\w+)\s+since\s+(.+)", units)
        if not match:
            raise ValueError(f"Cannot parse time units: {units}")

        unit, ref_str = match.groups()
        ref_date = np.datetime64(ref_str.replace(" ", "T"))

        multipliers = {
            "seconds": 1,
            "minutes": 60,
            "hours": 3600,
            "days": 86400,
        }
        if unit not in multipliers:
            raise ValueError(f"Unsupported time unit: {unit}")

        ns_offsets = (time_arr * multipliers[unit] * 1_000_000_000).astype("timedelta64[ns]")
        return (ref_date + ns_offsets).astype("datetime64[ns]")

    if np.issubdtype(time_arr.dtype, np.integer):
        return time_arr.view("datetime64[ns]")
    if np.issubdtype(time_arr.dtype, np.datetime64):
        return time_arr.astype("datetime64[ns]")

    raise ValueError(f"Cannot decode time with dtype {time_arr.dtype} and no units attr")


def estimate_manifest_size(grp: zarr.Group, vars: list[str]) -> tuple[int, float]:
    """Estimate chunk reference count and manifest size in MB for a set of variables.

    Uses ~12 bytes/ref based on observed icechunk manifest sizes, not the
    commonly cited 110 bytes/ref which applies to other formats.

    Returns:
        (total_chunk_refs, estimated_mb)
    """
    import math

    total = 0
    for v in vars:
        arr = grp[v]
        n = math.prod(math.ceil(s / c) for s, c in zip(arr.shape, arr.chunks))
        total += n
    return total, total * 12 / 1e6


def init_store(
    store: zarr.abc.store.Store,
    *,
    sfc_ds: xr.Dataset,
    pl_ds: xr.Dataset,
    inv_ds: xr.Dataset,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
) -> None:
    """Initialize the destination Zarr layout.

    Creates:
      - invariant/        : static invariant fields
      - samples/          : coordinates + empty sample_data array
        * coords          : time, channel, latitude, longitude, atmos_levels
        * attrs           : var_locs (channel layout)
        * sample_data     : (time, channel, latitude, longitude)
    """
    logger.info("Initializing destination store...")

    # -- Invariant fields --
    logger.info("  Writing 'invariant' group...")
    inv_ds = inv_ds[list(NAME_MAP["inv"])].rename(NAME_MAP["inv"])
    inv_ds.to_zarr(store, group="invariant", zarr_format=3, consolidated=False, mode="a")

    # -- Schema derivation --
    time_coord = sfc_ds.time.sel(
        time=slice(
            np.datetime64(start_time) if start_time is not None else None,
            np.datetime64(end_time) if end_time is not None else None,
        )
    )
    lat_coord, lon_coord = sfc_ds.latitude, sfc_ds.longitude

    n_levels = int(pl_ds.level.size)
    levels = pl_ds.level.values.astype("int32")
    _, out_var_locs = get_variable_locations(n_levels)

    n_channels = len(NAME_MAP["sfc"]) + len(NAME_MAP["pl"]) * n_levels
    channel = np.arange(n_channels, dtype="int32")

    time_len, lat_len, lon_len = len(time_coord), len(lat_coord), len(lon_coord)
    time_chunk, lat_chunk, lon_chunk = 1, 103, 72

    # -- Coordinates --
    logger.info("  Writing 'samples' coordinates...")
    coords_ds = xr.Dataset(
        data_vars={"atmos_levels": ("atmos_levels", levels)},
        coords={
            "time": time_coord,
            "latitude": lat_coord,
            "longitude": lon_coord,
            "channel": ("channel", channel),
        },
        attrs={
            "var_locs": out_var_locs,
            "shape": f"({time_len}, {n_channels}, {lat_len}, {lon_len})",
            "chunks": f"({time_chunk}, {n_channels}, {lat_chunk}, {lon_chunk})",
        },
    )

    coords_ds.to_zarr(
        store,
        group="samples",
        zarr_format=3,
        consolidated=False,
        mode="w",
        encoding={
            "time": {"chunks": (time_len,)},
            "latitude": {"chunks": (lat_len,)},
            "longitude": {"chunks": (lon_len,)},
            "atmos_levels": {"chunks": (len(levels),)},
            "channel": {"chunks": (n_channels,)},
        },
    )

    # -- Empty sample_data array --
    logger.info(f"  Creating sample_data: shape=({time_len}, {n_channels}, {lat_len}, {lon_len})")
    logger.info(f"    chunks=({time_chunk}, {n_channels}, {lat_chunk}, {lon_chunk})")

    group = zarr.open_group(store, path="samples", mode="a")
    group.create_array(
        "sample_data",
        shape=(time_len, n_channels, lat_len, lon_len),
        chunks=(time_chunk, n_channels, lat_chunk, lon_chunk),
        dtype="float32",
        dimension_names=("time", "channel", "latitude", "longitude"),
        fill_value=np.nan,
        compressors=[zarr.codecs.BloscCodec(clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)],
        overwrite=True,
    )

    logger.info("Initialization complete.")


def get_batches(
    start: datetime.datetime,
    end: datetime.datetime,
    n: int = 1,
) -> list[list[datetime.datetime]]:
    """Generate batches of 6-hourly timestamps between start and end, n per batch."""
    step = datetime.timedelta(hours=6)
    batches = []
    cur = start
    while cur <= end:
        batch = []
        for i in range(n):
            t = cur + i * step
            if t > end:
                break
            batch.append(t)
        if batch:
            batches.append(batch)
        cur += n * step
    return batches


def get_or_create_repo_branch(
    *,
    client: arraylake.Client,
    repo_name: str,
    branch_name: str,
    base_branch: str = "main",
) -> tuple["arraylake.Repo", "arraylake.Session"]:
    """Open an Arraylake repo and branch, creating them if they do not exist."""
    try:
        repo = client.get_repo(repo_name, config=REPO_CONFIG, storage_options=STORAGE_OPTIONS)
        logger.info(f"  Opened existing repo: {repo_name}")
    except Exception:
        repo = client.create_repo(repo_name, config=REPO_CONFIG, storage_options=STORAGE_OPTIONS)
        logger.info(f"  Created new repo: {repo_name}")

    branches = repo.list_branches()
    if branch_name not in branches:
        logger.info(f"  Branch '{branch_name}' does not exist, creating from '{base_branch}'")
        base_commit = repo.lookup_branch(base_branch)
        repo.create_branch(branch_name, base_commit)
    else:
        logger.info(f"  Branch '{branch_name}' already exists")

    session = repo.writable_session(branch_name)
    return repo, session


def commit_with_retries(
    session: "arraylake.Session",
    msg: str,
    max_attempts: int = 6,
) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            return session.commit(msg)
        except (RuntimeError, icechunk.IcechunkError):
            if attempt == max_attempts:
                raise
            time.sleep(2**attempt)


def get_job_count(job_prefix: str) -> int:
    """Count remaining SLURM jobs whose name starts with job_prefix."""
    res = subprocess.run(
        ["squeue", "-h", "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sum(1 for name in res.stdout.splitlines() if name.startswith(job_prefix))


def set_metadata(
    *,
    session: "arraylake.Session",
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
    extra: dict | None = None,
) -> None:
    root = zarr.open_group(session.store, path="samples", zarr_format=3)
    attrs = dict(root.attrs)

    attrs["last_updated"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S")

    if extra:
        attrs.update(extra)

    root.attrs.put(attrs)


# ----#
# CLI #
# ----#


@click.group()
def cli():
    pass


@cli.command("channelize-worker")
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--src-repo", required=True)
@click.option("--src-branch", default="main")
@click.option("--dst-repo", required=True)
@click.option("--dst-branch", default="main")
@click.option("--fork-pickle", required=False, default=None)
@click.option("--token", default=None)
def channelize_worker(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_repo: str,
    src_branch: str,
    dst_repo: str,
    dst_branch: str,
    fork_pickle: str | None,
    token: str | None,
) -> None:
    logger.info(f"\n{'=' * 80}")
    logger.info(f"[BATCH] {start_time:%Y-%m-%d} -> {end_time:%Y-%m-%d}")
    logger.info("=" * 80)

    # -- Session --
    logger.info("[SESSION] initializing")

    client = arraylake_client(token)
    if token is None and src_repo.startswith("rwe/"):
        logger.info("  logging in to Arraylake")
        client.login()

    fs = s3_filesystem()

    if fork_pickle is not None:
        logger.info(f"  loading fork session from {fork_pickle}")
        with fs.open(fork_pickle, "rb") as fobj:
            dst_session = pickle.load(fobj)
    else:
        logger.info(f"  opening writable session on {dst_repo}/{dst_branch}")
        repo = client.get_repo(dst_repo, config=REPO_CONFIG, storage_options=STORAGE_OPTIONS)
        dst_session = repo.writable_session(dst_branch)

    # -- Destination store --
    logger.info("[DEST] reading destination timestamps")

    ds_store = xr.open_zarr(
        dst_session.store,
        zarr_format=3,
        consolidated=False,
        group="samples",
        chunks=None,
    )

    timestamps = (
        ds_store["time"].sel(time=slice(start_time, end_time)).values.astype("datetime64[ns]")
    )

    logger.info(f"  {len(timestamps)} timestamps to process")
    for t in timestamps:
        logger.info(f"    {t}")

    # -- Source session (readonly, no dataset metadata needed) --
    logger.info("[SRC] opening source session")
    src_session = open_src_session(src_repo=src_repo, src_branch=src_branch, token=token)

    # -- Load data --
    # Icechunk handles transient network errors internally (retries with
    # exponential backoff). Permanent errors (missing data) fail immediately.
    logger.info("[LOAD] loading surface + pressure-level data")

    sfc_grp = zarr.open_group(src_session.store, path="surface", mode="r")
    pl_grp = zarr.open_group(src_session.store, path="pressure_level", mode="r")

    # Verify level ordering matches what Aurora expects (ascending pressure).
    # Raw zarr does not apply sortby — must confirm here.
    src_levels = pl_grp["level"][:]
    logger.info(f"  source level order: {src_levels}")
    if not np.all(src_levels == np.sort(src_levels)):
        raise click.ClickException(
            f"Source pressure levels are not sorted ascending: {src_levels}. "
            "Channel assignment will be wrong — re-check level ordering."
        )

    # Manifest size estimate.
    # Uses ~12 bytes/ref based on observed icechunk manifest sizes.
    sfc_refs, sfc_mb = estimate_manifest_size(sfc_grp, list(NAME_MAP["sfc"]))
    pl_refs, pl_mb = estimate_manifest_size(pl_grp, list(NAME_MAP["pl"]))
    logger.info(
        f"[MANIFEST] estimated refs: {sfc_refs + pl_refs:,} "
        f"(sfc={sfc_refs:,} ~{sfc_mb:.1f}MB, pl={pl_refs:,} ~{pl_mb:.1f}MB) "
        f"→ total ~{sfc_mb + pl_mb:.1f} MB"
    )

    time_ns = decode_zarr_time(sfc_grp)
    if len(time_ns) == 0:
        raise click.ClickException("Source time coordinate is empty")

    start_ns = np.datetime64(start_time).astype("datetime64[ns]")
    end_ns = np.datetime64(end_time).astype("datetime64[ns]")
    logger.info(f"  source range: {time_ns[0]} to {time_ns[-1]}")

    start_idx = int(np.searchsorted(time_ns, start_ns))
    end_idx = int(np.searchsorted(time_ns, end_ns, side="right"))

    if start_idx >= end_idx:
        raise click.ClickException(
            f"No timestamps in [{start_time}, {end_time}]. "
            f"Source range: [{time_ns[0]}, {time_ns[-1]}]"
        )

    # Align the time slice to source chunk boundaries to avoid partial-chunk
    # reads. A partial chunk still costs one full GET — aligning means every
    # GET returns useful data with no wasted overfetch.
    chunk_t = int(sfc_grp[next(iter(NAME_MAP["sfc"]))].chunks[0])
    aligned_start = (start_idx // chunk_t) * chunk_t
    aligned_end = ((end_idx + chunk_t - 1) // chunk_t) * chunk_t
    t_slice_aligned = slice(aligned_start, aligned_end)

    # Offsets to trim the aligned data back to the requested window.
    trim_start = start_idx - aligned_start
    trim_end = trim_start + (end_idx - start_idx)

    n_steps = end_idx - start_idx
    n_steps_aligned = aligned_end - aligned_start
    sfc_gets = (n_steps_aligned // chunk_t) * 3 * 6
    pl_gets = (n_steps_aligned // chunk_t) * 13 * 3 * 6
    logger.info(
        f"  time slice: [{start_idx}:{end_idx}] ({n_steps} steps) "
        f"aligned to [{aligned_start}:{aligned_end}] ({n_steps_aligned} steps, "
        f"chunk_t={chunk_t}) "
        f"trim=[{trim_start}:{trim_end}]"
    )
    logger.info(f"  estimated GETs: sfc={sfc_gets:,} pl={pl_gets:,} total={sfc_gets + pl_gets:,}")

    def _load(args: tuple) -> tuple[str, np.ndarray]:
        grp, v, sl, ts, te = args
        return v, grp[v][sl][ts:te]

    # 9 workers — one per variable (4 sfc + 5 pl).
    # zarr releases the GIL on IO so threads genuinely overlap.
    # Each worker fetches the chunk-aligned slice then trims to the exact window.
    with ThreadPoolExecutor(max_workers=9) as ex:
        sfc_data = dict(
            ex.map(
                _load,
                [(sfc_grp, v, t_slice_aligned, trim_start, trim_end) for v in NAME_MAP["sfc"]],
            )
        )
        pl_data = dict(
            ex.map(
                _load,
                [(pl_grp, v, t_slice_aligned, trim_start, trim_end) for v in NAME_MAP["pl"]],
            )
        )
        time_vals = time_ns[start_idx:end_idx]

    lat_vals = sfc_grp["latitude"][:]
    lon_vals = sfc_grp["longitude"][:]

    # -- Build channelized block --
    logger.info("[BUILD] stacking channels")

    n_levels = int(pl_grp["level"].shape[0])
    in_locs, _ = get_variable_locations(n_levels)

    first_sfc = next(iter(sfc_data.values()))
    T, Y, X = len(time_vals), first_sfc.shape[1], first_sfc.shape[2]
    n_channels = len(NAME_MAP["sfc"]) + len(NAME_MAP["pl"]) * n_levels

    block = np.empty((T, n_channels, Y, X), dtype=np.float32)

    for v, (cidx, _) in in_locs["sfc"].items():
        block[:, cidx, :, :] = sfc_data[v]

    for v, (cidx, size) in in_locs["pl"].items():
        # pl_data[v] shape: (T, n_levels, Y, X) — level dim already in source order.
        # Verified above that source levels are sorted ascending.
        block[:, cidx : cidx + size, :, :] = pl_data[v]

    logger.info(f"  block shape: ({T}, {n_channels}, {Y}, {X})")

    # -- Infer spatial bounds from destination store --
    dst_lats = ds_store["latitude"].values
    dst_lons = ds_store["longitude"].values
    lat_min, lat_max = float(dst_lats.min()), float(dst_lats.max())
    lon_min, lon_max = float(dst_lons.min()), float(dst_lons.max())
    logger.info(
        f"  dst bounds: lat [{lat_min:.2f}, {lat_max:.2f}], lon [{lon_min:.2f}, {lon_max:.2f}]"
    )

    # -- Write --
    logger.info("[WRITE] writing to zarr")

    write_ds = xr.Dataset(
        {"sample_data": (("time", "channel", "latitude", "longitude"), block)},
        coords={
            "time": time_vals,
            "latitude": lat_vals,
            "longitude": lon_vals,
        },
    )

    write_ds.to_zarr(
        dst_session.store,
        group="samples",
        region="auto",
        mode="r+",
        consolidated=False,
    )

    logger.info("  write complete")

    # -- Finalize --
    logger.info("[SESSION] finalizing")

    if fork_pickle is None:
        commit_id = dst_session.commit(f"Added {timestamps.min()} to {timestamps.max()}")
        logger.info(f"  committed: {commit_id}")
    else:
        time_str = f"{start_time:%Y%m%d_%H%M%S}_{end_time:%Y%m%d_%H%M%S}"
        out_pickle = fork_pickle.replace(".pickle", f".{time_str}.worker.pickle")
        logger.info(f"  writing worker pickle: {out_pickle}")
        with fs.open(out_pickle, "wb") as fobj:
            pickle.dump(dst_session, fobj)

    logger.info(f"[DONE] {start_time.isoformat()} -> {end_time.isoformat()}")


@cli.command("submit-jobs")
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--src-repo", required=True)
@click.option("--src-branch", default="main")
@click.option("--dst-repo", required=True)
@click.option("--dst-branch", default="main")
@click.option("--token", default=None)
@click.option("--times-at-once", type=int, default=56)
@click.option("--coordination-location", required=True)
@click.option("--cpus", type=int, default=16)
def submit_jobs(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_repo: str,
    src_branch: str,
    dst_repo: str,
    dst_branch: str,
    token: str | None,
    times_at_once: int,
    coordination_location: str,
    cpus: int,
) -> None:
    """Distributed channelization using SLURM + Icechunk sessions."""

    logger.info(f"\n{'=' * 80}")
    logger.info(f"[SUBMIT] {start_time:%Y-%m-%d %H:%M} -> {end_time:%Y-%m-%d %H:%M}")
    logger.info("=" * 80)

    # -- Setup --
    logger.info("[SETUP] initializing")

    client = arraylake_client(token)
    if token is None and (src_repo.startswith("rwe/") or dst_repo.startswith("rwe/")):
        client.login()

    repo = client.get_repo(dst_repo, config=REPO_CONFIG, storage_options=STORAGE_OPTIONS)

    # -- Extend time axis --
    logger.info(f"[DEST] ensuring time axis through {end_time}")

    base_session = repo.writable_session(dst_branch)
    msg = ensure_time_in_arrays(
        store=base_session.store,
        timestamp=end_time,
        time_frequency="6h",
        group="samples",
    )

    if msg is not None:
        logger.info(f"  extending: {msg}")
        commit_with_retries(session=base_session, msg=str(msg))
    else:
        logger.info("  time axis already sufficient")

    # -- Fork --
    logger.info("[SESSION] creating base fork")

    base_session = repo.writable_session(dst_branch)
    fork = base_session.fork()

    # -- Batch planning --
    batches = get_batches(start_time, end_time, times_at_once)

    logger.info(f"[BATCH] {len(batches)} batches, {times_at_once} timesteps each")
    if batches:
        logger.info(f"  first: {batches[0][0]} -> {batches[0][-1]}")
        logger.info(f"  last:  {batches[-1][0]} -> {batches[-1][-1]}")

    # -- Save fork pickle --
    job_id = random_job_string(10)
    session_location = os.path.join(coordination_location, job_id)
    fork_pickle = os.path.join(session_location, "fork.pickle")

    logger.info(f"[SESSION] saving fork pickle (job_id={job_id})")

    fs = s3_filesystem()
    with fs.open(fork_pickle, "wb") as fobj:
        pickle.dump(fork, fobj)

    # -- Submit SLURM jobs --
    logger.info(f"[SLURM] submitting {len(batches)} jobs (cpus={cpus})")

    job_prefix = f"chan_{job_id}"

    for i, batch in enumerate(batches, start=1):
        batch_start, batch_end = batch[0], batch[-1]
        start_s, end_s = batch_start.isoformat(), batch_end.isoformat()

        logger.info(f"  [{i}/{len(batches)}] {start_s} -> {end_s}")

        cmd = [
            "sbatch",
            "--ntasks=1",
            f"--cpus-per-task={cpus}",
            "--mem=50G",
            f"--job-name={job_prefix}_{start_s}_{end_s}",
            sys.argv[0],
            "channelize-worker",
            start_s,
            end_s,
            f"--src-repo={src_repo}",
            f"--src-branch={src_branch}",
            f"--dst-repo={dst_repo}",
            f"--dst-branch={dst_branch}",
            f"--fork-pickle={fork_pickle}",
            *([] if token is None else [f"--token={token}"]),
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"sbatch failed:\n{res.stderr}")

        # Stagger submissions to spread manifest fetches across time and
        # avoid thundering-herd on S3 from all workers starting simultaneously.
        time.sleep(10)

    logger.info("  all jobs submitted")

    # -- Wait --
    logger.info("[SLURM] waiting for workers")

    time.sleep(60)
    while (remaining := get_job_count(job_prefix)) > 0:
        logger.info(f"  {remaining} jobs remaining...")
        time.sleep(60)

    logger.info("  all jobs complete")

    # -- Merge --
    logger.info("[MERGE] collecting worker sessions")

    time.sleep(10)

    worker_forks = []
    for fspath in fs.ls(session_location):
        if fspath.endswith(".worker.pickle"):
            logger.info(f"  loading {fspath}")
            with fs.open(fspath, "rb") as fobj:
                worker_forks.append(pickle.load(fobj))
        fs.rm(fspath)

    logger.info(f"  merging {len(worker_forks)} forks")
    base_session.merge(*worker_forks)

    # -- Commit --
    logger.info("[COMMIT] setting metadata")

    set_metadata(session=base_session, start_time=start_time, end_time=end_time)

    commit_msg = f"channelize {start_time.isoformat()} -> {end_time.isoformat()}"
    commit_id = commit_with_retries(session=base_session, msg=commit_msg)

    logger.info(f"  committed: {commit_id}")
    logger.info("[SUBMIT DONE]")


@cli.command("init")
@click.option("--start-time", type=click.DateTime(), default=None)
@click.option("--end-time", type=click.DateTime(), default=None)
@click.option("--src-repo", default="rwe/era5-0p25-6h-nonprod-ohio")
@click.option("--dst-repo", default="kafou/aurora-era5-samples")
@click.option("--src-branch", default="main")
@click.option("--dst-branch", default="main")
@click.option("--token", default=None)
@click.option(
    "--force-init",
    is_flag=True,
    help="DANGER: delete and recreate the destination schema if it already exists",
)
def init(
    start_time: datetime.datetime | None,
    end_time: datetime.datetime | None,
    src_repo: str,
    src_branch: str,
    dst_repo: str,
    dst_branch: str,
    token: str,
    force_init: bool = False,
) -> None:
    """Initialize channelized repository."""
    client = arraylake_client(token)
    if token is None and (src_repo.startswith("rwe/") or dst_repo.startswith("rwe/")):
        client.login()

    sfc_ds, pl_ds, inv_ds, _ = open_src_datasets(
        src_repo=src_repo,
        src_branch=src_branch,
        token=token,
    )

    _, session = get_or_create_repo_branch(
        client=client,
        repo_name=dst_repo,
        branch_name=dst_branch,
        base_branch="main",
    )
    root = zarr.open_group(session.store, mode="a")

    if force_init:
        logger.info("  --force-init: deleting existing schema")
        for key in ("samples", "invariant"):
            if key in root:
                del root[key]
    elif "samples" in root or "invariant" in root:
        raise click.ClickException(
            "Destination store already initialized.\n"
            "Refusing to overwrite. Re-run with --force-init to destroy and recreate."
        )

    init_store(
        session.store,
        sfc_ds=sfc_ds,
        pl_ds=pl_ds,
        inv_ds=inv_ds,
        start_time=start_time,
        end_time=end_time,
    )

    set_metadata(session=session, start_time=None, end_time=None)

    commit_id = commit_with_retries(session=session, msg="Initialized sample data store.")
    logger.info(f"  initialized: commit_id={commit_id}")

    ds = xr.open_zarr(
        session.store,
        zarr_format=3,
        consolidated=False,
        group="samples",
        chunks=None,
    )
    logger.info(ds)


if __name__ == "__main__":
    cli()
