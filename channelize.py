#!/usr/bin/env python
"""Copyright (c) RWE Supply & Trading GmbH. Licensed under the MIT license.

conda activate aurora

# Initialize the destination zarr v3 store
python -u channelize.py init \
    --start-time 2025-01-01T00:00:00 \
    --end-time 2025-01-31T18:00:00 \
    --src-repo rwe/era5-0p25-6h-nonprod-ohio \
    --src-branch main \
    --dst-store s3://kafou-data/tutorial-channel.zarr \
    --token "$ARRAYLAKE_TOKEN" \
    --force-init

# Submit distributed channelization jobs. Re-running the same command after
# a partial failure skips batches whose done marker already exists.
python -u channelize.py submit-jobs \
    2025-01-01T00:00:00 \
    2025-01-31T18:00:00 \
    --src-repo rwe/era5-0p25-6h-nonprod-ohio \
    --src-branch main \
    --dst-store s3://kafou-data/tutorial-channel.zarr \
    --token "$ARRAYLAKE_TOKEN" \
    --times-at-once 120 \
    --cpus 4
"""

import datetime
import logging
import re
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
from obstore_utils import create_bucket_if_not_exists, open_s3_zarr_store
from zarr.core.config import config as zarr_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

xr.set_options(keep_attrs=True)

zarr_config.set({"async.concurrency": 8})

# Repo-level config: concurrency tuning for icechunk storage backend.
# Passed as `config=` to client.get_repo().
REPO_CONFIG = icechunk.RepositoryConfig.default()

# Storage-level config: network timeout tuning.
# minimum_throughput_bytes_per_second=0 disables the throughput floor check.
# STORAGE_OPTIONS: dict = {"network_stream_timeout_seconds": 600}

INVARIANT_REPO = "rwe/era5-0p25-6h-nonprod-ohio"
S3_PROFILE = "kafou"

NAME_MAP = {
    "sfc": {"VAR_2T": "2t", "MSL": "msl", "VAR_10U": "10u", "VAR_10V": "10v"},
    "pl": {"Z": "z", "U": "u", "V": "v", "T": "t", "Q": "q"},
    "inv": {"LSM": "lsm", "Z": "z", "SLT": "slt"},
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
    src_repo_obj = client.get_repo(src_repo, config=REPO_CONFIG)
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

    src_repo_obj = client.get_repo(src_repo, config=REPO_CONFIG)
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
    inv_repo_obj = client.get_repo(INVARIANT_REPO, config=REPO_CONFIG)
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
    in_locs = {"sfc": {}, "pl": {}}
    out_locs = {"sfc": {}, "pl": {}}

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
    time_arr = grp["time"][:]
    units = dict(grp["time"].attrs).get("units")
    if units is not None:
        match = re.match(r"(\w+)\s+since\s+(.+)", units)
        if not match:
            raise ValueError(f"Cannot parse time units: {units}")
        unit, ref_str = match.groups()
        multipliers = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}
        if unit not in multipliers:
            raise ValueError(f"Unsupported time unit: {unit}")
        ref = np.datetime64(ref_str.replace(" ", "T"))
        return (
            ref + (time_arr * multipliers[unit] * 1_000_000_000).astype("timedelta64[ns]")
        ).astype("datetime64[ns]")
    if np.issubdtype(time_arr.dtype, np.integer):
        return time_arr.view("datetime64[ns]")
    return time_arr.astype("datetime64[ns]")


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
    store: zarr.abc.store.Store,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
    extra: dict | None = None,
) -> None:
    root = zarr.open_group(store, path="samples", zarr_format=3)
    attrs = dict(root.attrs)

    attrs["last_updated"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S")

    if extra:
        attrs.update(extra)

    root.attrs.put(attrs)


def batch_marker_path(dst_store: str, start: datetime.datetime, end: datetime.datetime) -> str:
    """S3 path of the marker for a single sbatch submission.

    Lives alongside (not inside) the zarr store so marker listing never has
    to walk zarr chunk keys. Filename encodes the batch's start/end range.
    """
    stem = f"{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}"
    return f"{dst_store.rstrip('/')}__markers/{stem}.done"


def write_batch_marker(*, dst_store: str, start: datetime.datetime, end: datetime.datetime) -> str:
    fs = s3_filesystem()
    path = batch_marker_path(dst_store, start, end)
    with fs.open(path, "wb") as fobj:
        fobj.write(b"")
    return path


def batch_marker_exists(
    *, dst_store: str, start: datetime.datetime, end: datetime.datetime
) -> bool:
    return s3_filesystem().exists(batch_marker_path(dst_store, start, end))


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
@click.option("--dst-store", required=True, help="S3 URL of destination zarr v3 store")
@click.option("--token", default=None)
def channelize_worker(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_repo: str,
    src_branch: str,
    dst_store: str,
    token: str | None,
) -> None:
    logger.info(f"\n{'=' * 80}")
    logger.info(f"[BATCH] {start_time:%Y-%m-%d} -> {end_time:%Y-%m-%d}")
    logger.info("=" * 80)

    # -- Arraylake auth (for source only) --
    client = arraylake_client(token)
    if token is None and src_repo.startswith("rwe/"):
        logger.info("[AUTH] logging in to Arraylake")
        client.login()

    # -- Destination store (plain zarr v3 over obstore, no sessions) --
    logger.info(f"[DEST] opening zarr store at {dst_store}")
    dst_zstore = open_s3_zarr_store(dst_store, profile=S3_PROFILE)

    logger.info("[DEST] reading destination timestamps")
    ds_store = xr.open_zarr(
        dst_zstore,
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

    # Force manifest fetch for all variables we'll use by reading a single
    # scalar value from each. This warms the manifest cache and lets us time
    # the cold-start cost before the real load begins.
    logger.info("[MANIFEST] warming manifest cache...")
    t0 = time.perf_counter()
    for v in NAME_MAP["sfc"]:
        _ = sfc_grp[v][0, 0, 0]
    for v in NAME_MAP["pl"]:
        _ = pl_grp[v][0, 0, 0, 0]
    logger.info(f"[MANIFEST] cache warm in {time.perf_counter() - t0:.1f}s")

    time_ns = decode_zarr_time(sfc_grp)
    start_ns = np.datetime64(start_time).astype("datetime64[ns]")
    end_ns = np.datetime64(end_time).astype("datetime64[ns]")
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

    def _load(grp: zarr.Group, var: str) -> np.ndarray:
        return grp[var][t_slice_aligned][trim_start:trim_end]

    sfc_vars = list(NAME_MAP["sfc"])
    pl_vars = list(NAME_MAP["pl"])
    with ThreadPoolExecutor(max_workers=len(sfc_vars) + len(pl_vars)) as executor:
        sfc_futures = {v: executor.submit(_load, sfc_grp, v) for v in sfc_vars}
        pl_futures = {v: executor.submit(_load, pl_grp, v) for v in pl_vars}
        sfc_data = {v: f.result() for v, f in sfc_futures.items()}
        pl_data = {v: f.result() for v, f in pl_futures.items()}

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
        dst_zstore,
        group="samples",
        region="auto",
        mode="r+",
        consolidated=False,
    )

    logger.info("  write complete")

    # Written last so its presence certifies all chunks for this batch landed.
    marker = write_batch_marker(dst_store=dst_store, start=start_time, end=end_time)
    logger.info(f"[MARKER] batch marker written: {marker}")

    logger.info(f"[DONE] {start_time.isoformat()} -> {end_time.isoformat()}")


@cli.command("submit-jobs")
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--src-repo", required=True)
@click.option("--src-branch", default="main")
@click.option("--dst-store", required=True, help="S3 URL of destination zarr v3 store")
@click.option("--token", default=None)
@click.option("--times-at-once", type=int, default=56)
@click.option("--cpus", type=int, default=16)
def submit_jobs(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_repo: str,
    src_branch: str,
    dst_store: str,
    token: str | None,
    times_at_once: int,
    cpus: int,
) -> None:
    """Distributed channelization against a plain zarr v3 store.

    Each SLURM worker writes a disjoint, chunk-aligned time slice via
    region="auto", then writes a per-batch done marker. The coordinator
    extends the time axis once up front (single-writer), submits sbatch
    jobs, waits for the queue to drain, and updates store metadata.
    """

    logger.info(f"\n{'=' * 80}")
    logger.info(f"[SUBMIT] {start_time:%Y-%m-%d %H:%M} -> {end_time:%Y-%m-%d %H:%M}")
    logger.info("=" * 80)

    # -- Setup --
    logger.info("[SETUP] initializing")

    client = arraylake_client(token)
    if token is None and src_repo.startswith("rwe/"):
        client.login()

    # -- Extend time axis (single-writer, before any worker starts) --
    logger.info(f"[DEST] opening zarr store at {dst_store}")
    dst_zstore = open_s3_zarr_store(dst_store, profile=S3_PROFILE)

    logger.info(f"[DEST] ensuring time axis through {end_time}")
    msg = ensure_time_in_arrays(
        store=dst_zstore,
        timestamp=end_time,
        time_frequency="6h",
        group="samples",
    )
    if msg is not None:
        logger.info(f"  extended: {msg}")
    else:
        logger.info("  time axis already sufficient")

    # -- Batch planning --
    batches = get_batches(start_time, end_time, times_at_once)

    logger.info(f"[BATCH] {len(batches)} batches, {times_at_once} timesteps each")
    if batches:
        logger.info(f"  first: {batches[0][0]} -> {batches[0][-1]}")
        logger.info(f"  last:  {batches[-1][0]} -> {batches[-1][-1]}")

    # SLURM job-name prefix used only to group + wait on this run's workers.
    # Random so concurrent submit-jobs calls do not wait on each other.
    job_prefix = f"chan_{random_job_string(10)}"

    # -- Submit SLURM jobs --
    # Existing batch markers are skipped so re-running the same submit-jobs
    # command after a partial failure only resubmits the batches that did
    # not complete last time.
    logger.info(f"[SLURM] submitting up to {len(batches)} jobs (cpus={cpus})")

    submitted = 0
    skipped = 0
    for i, batch in enumerate(batches, start=1):
        batch_start, batch_end = batch[0], batch[-1]
        start_s, end_s = batch_start.isoformat(), batch_end.isoformat()

        if batch_marker_exists(dst_store=dst_store, start=batch_start, end=batch_end):
            logger.info(f"  [{i}/{len(batches)}] {start_s} -> {end_s} — SKIP (marker exists)")
            skipped += 1
            continue

        logger.info(f"  [{i}/{len(batches)}] {start_s} -> {end_s}")

        cmd = [
            "sbatch",
            "--ntasks=1",
            f"--cpus-per-task={cpus}",
            "--mem=220G",
            f"--job-name={job_prefix}_{start_s}_{end_s}",
            sys.argv[0],
            "channelize-worker",
            start_s,
            end_s,
            f"--src-repo={src_repo}",
            f"--src-branch={src_branch}",
            f"--dst-store={dst_store}",
            *([] if token is None else [f"--token={token}"]),
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"sbatch failed:\n{res.stderr}")
        submitted += 1

        # Stagger submissions to spread manifest fetches across time and
        # avoid thundering-herd on S3 from all workers starting simultaneously.
        time.sleep(10)

    logger.info(f"  submitted={submitted} skipped={skipped}")

    # -- Wait --
    logger.info("[SLURM] waiting for workers")

    time.sleep(60)
    while (remaining := get_job_count(job_prefix)) > 0:
        logger.info(f"  {remaining} jobs remaining...")
        time.sleep(60)

    logger.info("  all jobs complete")

    # -- Metadata --
    # Workers write a per-batch marker as the last step of a successful run.
    # Missing markers after SLURM is empty indicate crashed/failed batches —
    # inspect the markers prefix alongside dst-store to determine what to re-run.
    logger.info("[META] updating store metadata")
    set_metadata(store=dst_zstore, start_time=start_time, end_time=end_time)

    logger.info("[SUBMIT DONE]")


@cli.command("init")
@click.option("--start-time", type=click.DateTime(), default=None)
@click.option("--end-time", type=click.DateTime(), default=None)
@click.option("--src-repo", default="rwe/era5-0p25-6h-nonprod-ohio")
@click.option("--src-branch", default="main")
@click.option("--dst-store", required=True, help="S3 URL of destination zarr v3 store")
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
    dst_store: str,
    token: str | None,
    force_init: bool = False,
) -> None:
    """Initialize a plain zarr v3 destination store with the channelized schema."""
    client = arraylake_client(token)
    if token is None and src_repo.startswith("rwe/"):
        client.login()

    sfc_ds, pl_ds, inv_ds, _ = open_src_datasets(
        src_repo=src_repo,
        src_branch=src_branch,
        token=token,
    )

    bucket = dst_store.removeprefix("s3://").split("/", 1)[0]
    create_bucket_if_not_exists(bucket, profile=S3_PROFILE)

    logger.info(f"[DEST] opening zarr store at {dst_store}")
    dst_zstore = open_s3_zarr_store(dst_store, profile=S3_PROFILE)
    root = zarr.open_group(dst_zstore, mode="a")

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
        dst_zstore,
        sfc_ds=sfc_ds,
        pl_ds=pl_ds,
        inv_ds=inv_ds,
        start_time=start_time,
        end_time=end_time,
    )

    set_metadata(store=dst_zstore, start_time=None, end_time=None)
    logger.info("[INIT] done")

    ds = xr.open_zarr(
        dst_zstore,
        zarr_format=3,
        consolidated=False,
        group="samples",
        chunks=None,
    )
    logger.info(ds)


if __name__ == "__main__":
    cli()
