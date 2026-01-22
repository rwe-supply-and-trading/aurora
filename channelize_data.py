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
import xarray as xr
import zarr
from dataset_io import ensure_time_in_arrays
from icechunk.distributed import merge_sessions

# -----------------#
# GLOBAL VARIABLES #
# -----------------#
# Ensure unbuffered output for real-time logging
# I could use python -u or use logging instead, but this works for now
os.environ["PYTHONUNBUFFERED"] = "1"

READ_RETRIES = 6
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
def get_clamped_time_range(
    *,
    ds: xr.Dataset,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> tuple[datetime.datetime, datetime.datetime]:
    """
    Clamp (start_time, end_time) to the available time range in ds.
    Raises if start_time is entirely outside the dataset.
    """
    return start_time, end_time


def random_job_string(length: int = 8) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(chars) for _ in range(length))


def open_src_datasets(
    *,
    src_repo: str,
    src_branch: str = "main",
    token: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Grabs the base datasets from the source repo. Assumes either ERA5 or ECMWF ENFO format.
    """
    client = arraylake.Client(token=token)
    if token is None:
        client.login()

    # detect what dataset we're working with
    if "era5" in src_repo:
        dataset_type = "era5"
    elif "ecmwf" in src_repo:
        dataset_type = "ecmwf"
    else:
        raise ValueError("Source repo must be ERA5 or ECMWF ecmwf dataset.")

    src_repo = client.get_repo(src_repo)
    src_session = src_repo.readonly_session(src_branch)

    if dataset_type == "ecmwf":
        # get static ds
        sfc_ds = xr.open_zarr(
            src_session.store,
            group="ENFO-T0",
            zarr_format=3,
            consolidated=False,
            chunks=None,
        )

        # drop ensemble dimension
        sfc_ds = sfc_ds.isel(ensemble_member=0, drop=True)

        # convert lat / lon to latitude / longitude
        if "lat" in sfc_ds.coords and "lon" in sfc_ds.coords:
            sfc_ds = sfc_ds.rename(
                {"lat": "latitude", "lon": "longitude", "pressure_level": "level"}
            )
        # match ERA5 variable names
        if "t2m" in sfc_ds.data_vars:
            sfc_ds = sfc_ds.rename(
                {"t2m": "VAR_2T", "slp": "MSL", "u10m": "VAR_10U", "v10m": "VAR_10V"}
            )

        # get pressure_level ds
        pl_ds = sfc_ds.copy(deep=True)

        # match ERA5 variable names
        if "z" in pl_ds.data_vars:
            pl_ds = pl_ds.rename({"z": "Z", "u": "U", "v": "V", "t": "T", "q": "Q"})

        # sort by level
        pl_ds = pl_ds.sortby("level")

    elif dataset_type == "era5":
        # get static ds
        sfc_ds = xr.open_zarr(
            src_session.store,
            group="surface",
            zarr_format=3,
            consolidated=False,
            chunks=None,
        )

        # get pressure_level ds
        pl_ds = xr.open_zarr(
            src_session.store,
            group="pressure_level",
            zarr_format=3,
            consolidated=False,
            chunks=None,
        )

        # convert lat / lon to latitude / longitude
        pl_ds = pl_ds.sortby("level")
    else:
        raise ValueError("dataset_type must be 'era5' or 'ecmwf'")

    # get invariant ds from ERA5 dataset
    inv_repo = "rwe/era5-0p25-6h-nonprod-ohio"
    inv_repo = client.get_repo(inv_repo)
    inv_session = inv_repo.readonly_session("main")
    inv_ds = xr.open_zarr(
        inv_session.store,
        group="invariant",
        zarr_format=3,
        consolidated=False,
        chunks=None,
    )

    return sfc_ds, pl_ds, inv_ds


def get_variable_locations(n_pressure_levels: int = 13) -> tuple[dict, dict]:
    in_locs = {"sfc": {}, "pl": {}}
    out_locs = {"sfc": {}, "pl": {}}

    i = 0
    sfc_map = NAME_MAP["sfc"]
    for var in sfc_map:
        loc = (i, 1)
        in_locs["sfc"][var] = loc
        out_locs["sfc"][sfc_map[var]] = loc
        i += 1

    pl_map = NAME_MAP["pl"]
    for var in pl_map:
        loc = (i, n_pressure_levels)
        in_locs["pl"][var] = loc
        out_locs["pl"][pl_map[var]] = loc
        i += n_pressure_levels

    return in_locs, out_locs


def init_store(
    store: zarr.abc.store.Store,
    *,
    sfc_ds: xr.Dataset,
    pl_ds: xr.Dataset,
    inv_ds: xr.Dataset,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
) -> None:
    """
    Initialize the destination Zarr layout.

    Creates:
      - invariant/        : static invariant fields
      - samples/          : coordinates + empty sample_data array
        * coords          : time, channel, latitude, longitude, atmos_levels
        * attrs           : var_locs (channel layout)
        * sample_data     : (time, channel, latitude, longitude)

    Assumes:
      - time is 6-hourly
      - channel layout is fixed for the lifetime of the store
    """
    print("Initializing destination store...")

    # Invariant fields
    print("  • Writing 'invariant' group...")
    inv_ds = inv_ds[["Z", "LSM", "SLT"]]
    inv_ds = inv_ds.rename(NAME_MAP["inv"])
    inv_ds.to_zarr(store, group="invariant", zarr_format=3, consolidated=False, mode="a")

    # Schema derivation
    time_coord, lat_coord, lon_coord = sfc_ds.time, sfc_ds.latitude, sfc_ds.longitude
    time_coord = time_coord.sel(
        time=slice(
            np.datetime64(start_time) if start_time is not None else time_coord.min().values,
            np.datetime64(end_time) if end_time is not None else time_coord.max().values,
        )
    )

    n_levels = int(pl_ds.level.size)
    levels = pl_ds.level.values.astype("int32")

    _, out_var_locs = get_variable_locations(n_levels)
    n_channels = len(NAME_MAP["sfc"]) + len(NAME_MAP["pl"]) * n_levels

    time_len, lat_len, lon_len = len(time_coord), len(lat_coord), len(lon_coord)

    # Samples coordinates
    print("  • Writing 'samples' coordinates...")
    time_chunk, lat_chunk, lon_chunk = 1, 103, 72
    coords_ds = xr.Dataset(
        data_vars={
            "atmos_levels": ("atmos_levels", levels),
        },
        coords={
            "time": time_coord,
            "latitude": lat_coord,
            "longitude": lon_coord,
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
            "time": {"chunks": (len(time_coord),)},
            "latitude": {"chunks": (len(lat_coord),)},
            "longitude": {"chunks": (len(lon_coord),)},
            "atmos_levels": {"chunks": (len(levels),)},
        },
    )

    # Empty sample_data array
    group = zarr.open_group(store, path="samples", mode="a")

    print("  • Creating sample_data array:")
    print(f"      shape  = ({time_len}, {n_channels}, {lat_len}, {lon_len})")
    print(f"      chunks = ({time_chunk}, {n_channels}, {lat_chunk}, {lon_chunk})")

    compressors = [zarr.codecs.BloscCodec(clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)]

    # TODO: tune chunking strategy ??
    # we want to store in 1 MB chunks to optimize for read performance
    group.create_array(
        "sample_data",
        shape=(time_len, n_channels, lat_len, lon_len),
        chunks=(time_chunk, n_channels, lat_chunk, lon_chunk),
        dtype="float32",
        dimension_names=("time", "channel", "latitude", "longitude"),
        fill_value=np.nan,
        compressors=compressors,
        overwrite=True,
    )

    print("Initialization complete.")


def get_batches(ds: xr.Dataset, start: datetime.datetime, end: datetime.datetime, n: int = 1):
    """
    Generate batches of timestamps from ds between start and end, n at a time.
    """
    step = datetime.timedelta(hours=6)
    cur = start
    while cur <= end:
        nxt = min(cur + step * (n - 1), end)
        yield [cur + i * step for i in range(int((nxt - cur) / step) + 1)]
        cur = nxt + step


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


def commit_with_retries(session, msg: str, max_attempts: int = 6):
    for attempt in range(1, max_attempts + 1):
        try:
            return session.commit(msg)
        except (RuntimeError, icechunk.IcechunkError):
            if attempt == max_attempts:
                raise
            time.sleep(2**attempt)


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


def set_metadata(
    *,
    session,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
    extra: dict | None = None,
) -> None:
    root = zarr.open_group(session.store, path="samples", zarr_format=3)

    attrs = dict(root.attrs)

    old = attrs.get("valid_times")
    if old:
        attrs["valid_times"] = [
            min(old[0], start_time.isoformat()),
            max(old[1], end_time.isoformat()),
        ]
    elif start_time is None and end_time is None:
        attrs["valid_times"] = [
            None,
            None,
        ]
    else:
        attrs["valid_times"] = [
            start_time.isoformat(),
            end_time.isoformat(),
        ]
    attrs["last_updated"] = datetime.datetime.utcnow().isoformat()

    if extra:
        attrs.update(extra)

    root.attrs.clear()
    root.attrs.update(attrs)


# -------------------------------#
# PARALLEL SUBMISSION VIA SLURM #
# -------------------------------#
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
@click.option("--write-session-location", required=True)
@click.option("--token", default=None)
def channelize_worker(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_repo: str,
    src_branch: str,
    dst_repo: str,
    dst_branch: str,
    write_session_location: str,
    token: str | None,
) -> None:
    print(f"\n=== PROCESSING {start_time:%Y-%m-%d}-{end_time:%Y-%m-%d} ===")
    # --------------------------------------------------
    # Session handling
    # --------------------------------------------------
    client = arraylake.Client(token=token)

    if token is None and src_repo.startswith("rwe/"):
        client.login()

    if write_session_location is not None:
        fs = fsspec.filesystem("s3", profile="kafou")
        with fs.open(os.path.join(write_session_location, "session.pickle"), "rb") as fobj:
            dst_session = pickle.load(fobj)
    else:
        repo = client.get_repo(dst_repo)
        dst_session = repo.writable_session(dst_branch)

    # --------------------------------------------------
    # Get Destination Session Store
    # --------------------------------------------------
    ds_store = xr.open_zarr(
        dst_session.store,
        zarr_format=3,
        consolidated=False,
        group="samples",
        chunks=None,
    )

    # Get timestamps
    timestamps = (
        ds_store["time"].sel(time=slice(start_time, end_time)).values.astype("datetime64[ns]")
    )
    print(timestamps)

    # --------------------------------------------------
    # The meat of it all
    # --------------------------------------------------
    sfc_ds, pl_ds, _ = open_src_datasets(
        src_repo=src_repo,
        src_branch=src_branch,
        token=token,
    )
    sfc_vars, pl_vars = list(NAME_MAP["sfc"]), list(NAME_MAP["pl"])

    # TODO: avoiding Icechunk Bytes Streaming Error
    # Read batch with retries catch Icechunk streaming, S3 stalls, network issues, etc.
    for attempt in range(1, READ_RETRIES + 1):
        try:
            print("    → loading sfc")
            sfc = sfc_ds[sfc_vars].sel(time=timestamps).load()
            print("    → loading pl")
            pl = pl_ds[pl_vars].sel(time=timestamps).load()
            print("    → constructing block")
            break
        except (OSError, RuntimeError, icechunk.IcechunkError) as e:
            print(f"   → exception type: {type(e).__name__}")
            print(f"   → exception: {e}")
            if attempt == READ_RETRIES:
                print(
                    f"Read failed for {start_time:%Y%m%dT%H%M%S} -> {end_time:%Y%m%dT%H%M%S} "
                    f"after {READ_RETRIES} attempts — aborting batch."
                )
                raise

    if not np.array_equal(sfc.time.values, pl.time.values):
        raise RuntimeError("sfc/pl time mismatch")

    n_levels = pl.level.size
    in_locs, _ = get_variable_locations(n_levels)

    T, Y, X = sfc.time.size, sfc.latitude.size, sfc.longitude.size
    n_channels = len(NAME_MAP["sfc"]) + len(NAME_MAP["pl"]) * n_levels

    block = np.empty((T, n_channels, Y, X), dtype=np.float32)

    for v, (cidx, _) in in_locs["sfc"].items():
        block[:, cidx, :, :] = sfc[v].values

    for v, (cidx, size) in in_locs["pl"].items():
        block[:, cidx : cidx + size, :, :] = pl[v].values

    write_ds = xr.Dataset(
        {
            "sample_data": (
                ("time", "channel", "latitude", "longitude"),
                block,
            )
        },
        coords={
            "time": sfc.time.values,
            "latitude": sfc.latitude.values,
            "longitude": sfc.longitude.values,
        },
    )

    # Region writes
    print("→ writing to zarr")
    write_ds.to_zarr(
        dst_session.store,
        group="samples",
        region="auto",  # region="auto" → xarray resolves correct indices from time coords
        mode="a",  # mode="a" → append / overwrite existing regions
        consolidated=False,
    )
    print("→ write complete")

    print(f"   ✔ Finished writing batch {start_time:%Y%m%dT%H%M%S}-{end_time:%Y%m%dT%H%M%S}.")

    # --------------------------------------------------
    # Write out session pickle
    # --------------------------------------------------
    if write_session_location is None:
        commit_id = dst_session.commit(f"Added {timestamps.min()} to {timestamps.max()}")
        print(f"[channelize_worker] Committed session: {commit_id}")
        pass
    else:
        outpath = os.path.join(
            write_session_location,
            f"chan_{start_time:%Y%m%dT%H%M%S}_{end_time:%Y%m%dT%H%M%S}_.pickle",
        )
        fs = fsspec.filesystem("s3", profile="kafou")
        print(f"Writing {outpath}")
        with fs.open(outpath, "wb") as fobj:
            pickle.dump(dst_session, fobj)
        print(f"[channelize_worker] wrote session {outpath}")


@cli.command("submit-jobs")
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--src-repo", required=True)
@click.option("--src-branch", default="main")
@click.option("--dst-repo", required=True)
@click.option("--dst-branch", default="main")
@click.option("--token", default=None)
@click.option("--times-at-once", type=int, default=14)
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
):
    """
    Distributed channelization using SLURM + Icechunk sessions.

    Contract:
      - init_store MUST already have been run
      - time axis is extended ONCE here before submitting jobs
      - workers only write data
      - final merge + commit is atomic, so a driver script is recomended
    """

    # Get repo
    client = arraylake.Client(token=token)
    if token is None and (src_repo.startswith("rwe/") or dst_repo.startswith("rwe/")):
        client.login()
    repo = client.get_repo(dst_repo)

    # Open source once (for batch planning + time extension)
    sfc_ds, _, _ = open_src_datasets(
        src_repo=src_repo,
        src_branch=src_branch,
        token=token,
    )

    # Clamp requested time range to available source data
    start_time, end_time = get_clamped_time_range(
        ds=sfc_ds, start_time=start_time, end_time=end_time
    )

    # Extend time axis for new data
    print(f"[SUBMIT] Ensuring time axis through {end_time}")

    base_session = repo.writable_session(dst_branch)
    msg = ensure_time_in_arrays(
        store=base_session.store,
        timestamp=end_time,
        time_frequency="6h",
        group="samples",
    )
    if msg is not None:
        commit_with_retries(session=base_session, msg=str(msg))

    # reopen extended repo
    client = arraylake.Client()
    repo = client.get_repo(dst_repo)
    base_session = repo.writable_session(dst_branch)

    batches = list(get_batches(sfc_ds, start_time, end_time, times_at_once))

    # Create base Icechunk session
    job_id = random_job_string(10)
    session_location = os.path.join(coordination_location, job_id)
    session_pickle = os.path.join(session_location, "session.pickle")

    print(f"[SUBMIT] Saving base session → {session_pickle}")
    fs = fsspec.filesystem("s3", profile="kafou")
    with fs.open(session_pickle, "wb") as fobj:
        with base_session.allow_pickling():
            pickle.dump(base_session, fobj)

    print(f"[SUBMIT] Submitting {len(batches)} SLURM jobs ({times_at_once} timesteps per job)")
    job_prefix = f"chan_{job_id}"
    expected = set()

    for batch in batches:
        start, end = batch[0], batch[-1]  # datetime.datetime, datetime.datetime
        start_s, end_s = start.isoformat(), end.isoformat()  # str, str
        expected.add((start_s, end_s))

        cmd = [
            "sbatch",
            "--ntasks=1",
            f"--cpus-per-task={cpus}",
            f"--job-name={job_prefix}_{start_s}_{end_s}",
            sys.argv[0],
            "channelize-worker",
            start_s,
            end_s,
            f"--src-repo={src_repo}",
            f"--src-branch={src_branch}",
            f"--dst-repo={dst_repo}",
            f"--dst-branch={dst_branch}",
            f"--write-session-location={session_location}",
        ]

        if token is not None:
            cmd.append(f"--token={token}")

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"sbatch failed:\n{res.stderr}")

    time.sleep(60)

    # Wait for jobs to finish
    print("[SUBMIT] Waiting for workers...")
    job_count = get_job_count(job_prefix)
    while job_count != 0:
        job_count = get_job_count(job_prefix)
        print(f"[SUBMIT] {job_count} jobs remaining ...")
        time.sleep(60)

    # Merge sessions
    print("[SUBMIT] All Jobs complete. Merging worker sessions...")
    time.sleep(10)

    print("All jobs completed, gathering results...")
    sessions = []
    for fspath in fs.ls(session_location):
        filename = fspath.split("/")[-1]
        if filename.startswith("chan_") and filename.endswith(".pickle"):
            with fs.open(fspath, "rb") as fobj:
                sessions.append(pickle.load(fobj))
        fs.rm(fspath)
    merged_session = merge_sessions(base_session, *sessions)

    # Add Metadata
    init_time = sfc_ds.time.values.min().item().astype("datetime64[us]").astype(object)
    set_metadata(session=merged_session, start_time=init_time, end_time=end_time)

    # Commit
    msg = f"channelize {start_time.isoformat()} → {end_time.isoformat()}"
    commit_id = commit_with_retries(session=merged_session, msg=msg)
    print(f"[SUBMIT] Commit complete: {commit_id}")


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
    """
    Initializes channelized repository.

    - Opens and cleans source datasets
    - Gets and/or created new repo and branch
    - You may elect to force overwriting of an old repo using force_init flag
    - Adds metadata and commits
    """
    client = arraylake.Client(token=token)
    if token is None and (src_repo.startswith("rwe/") or dst_repo.startswith("rwe/")):
        client.login()

    # Open source datasets
    sfc_ds, pl_ds, inv_ds = open_src_datasets(
        src_repo=src_repo,
        src_branch=src_branch,
        token=token,
    )

    # Get or create destination repo + branch
    repo, session = get_or_create_repo_branch(
        client=client,
        repo_name=dst_repo,
        branch_name=dst_branch,
        base_branch="main",
    )
    root = zarr.open_group(session.store, mode="a")

    # Check for existing schema, Delete if forced
    if force_init:
        print("⚠️  --force-init enabled: deleting existing schema")
        for key in ("samples", "invariant"):
            if key in root:
                del root[key]
    else:
        if "samples" in root or "invariant" in root:
            raise click.ClickException(
                "Destination store already initialized.\n"
                "Refusing to overwrite existing schema.\n\n"
                "If you REALLY want to destroy and recreate it, re-run with --force-init."
            )

    # Initialize layout
    init_store(
        session.store,
        sfc_ds=sfc_ds,
        pl_ds=pl_ds,
        inv_ds=inv_ds,
        start_time=start_time,
        end_time=end_time,
    )

    # Add Metadata
    set_metadata(session=session, start_time=None, end_time=None)

    # Commit
    commit_id = commit_with_retries(session=session, msg="Initialized sample data store.")
    print(f"✓ Initialized; commit_id={commit_id}")

    ds = xr.open_zarr(
        session.store,
        zarr_format=3,
        consolidated=False,
        group="samples",
        chunks=None,
    )

    print(ds)


if __name__ == "__main__":
    cli()
