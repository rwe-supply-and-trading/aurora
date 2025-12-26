#!/usr/bin/env python

"""
Aurora Latent Vector Writer (ERA5 → Zarr / Icechunk)

This script generates Aurora latent vectors from ERA5 inputs and stores them in
a Zarr v3 dataset backed by Icechunk. It supports both single-process execution
and distributed GPU execution via SLURM with safe concurrent writes and atomic
commits.

For each 6-hourly ERA5 timestamp `t`, the model writes a latent vector at
`t + 6h`, reflecting the next Aurora prediction timestep. This offset is
intentional and centralized in `LatentVectorExtractor`.

Main commands
-------------
init
  Initialize a new latent-vector repository and preallocate storage.

save-lvs
  Run Aurora inference for a time range and write results directly.

submit-jobs
  Split a large time range into SLURM jobs, merge results, update metadata,
  and commit once.

The dataset tracks a monotonic `valid_time_range` attribute indicating the
temporal coverage of written latent vectors.

Intended for large-scale historical backfills and ongoing incremental updates
in HPC environments.


To Run:
-------

tmux

conda activate aurora

sbatch   --ntasks=1   --cpus-per-task=8   --mem=50G  --job-name=lv-submit   --wrap="
    python latent_vector_writer.py submit-jobs \
      2025-01-01T00:00:00 \
      2025-04-01T00:00:00 \
      --src-repo kafou/aurora-era5-samples \
      --src-branch extend-2025 \
      --dest-repo kafou/aurora-era5-t1-latent-vectors \
      --dest-branch main \
      --aws-profile kafou \
      --timesteps-per-job 8 \
      --coordination-location s3://icechunk-write-coordination"

"""

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
from dataset_io import ensure_time_in_arrays
from icechunk.distributed import merge_sessions

from aurora import AuroraPretrained
from aurora.data import ERA5DataLoaderFOAM


def init_zarr_store(*, store, n_init, n_lead, latent_dim, init_times, lead_times, valid_times):
    """
    Initialize an EMPTY latent-forecast Zarr v3 store.

    - No compression
    - No chunking
    - No in-memory allocation of huge arrays
    - Only metadata + empty array definition
    """

    print("\n====================== INIT_ZARR_STORE ======================")
    print("[DEBUG] n_init:", n_init)
    print("[DEBUG] n_lead:", n_lead)
    print("[DEBUG] latent_dim:", latent_dim)
    print("[DEBUG] init_times (len):", len(init_times))
    print("[DEBUG] lead_times (len):", len(lead_times))
    print("[DEBUG] valid_times.shape:", valid_times.shape)

    # ---- 1. Coordinate Dataset ----
    print("\n[DEBUG] Building coordinate dataset...")
    coord_ds = xr.Dataset(
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
            "valid_time": (("init_time", "lead_time"), valid_times),
        },
        attrs={
            "description": "Latent-space forecast dataset (initialized)",
        },
    )
    print("[DEBUG] coord_ds.dims:", coord_ds.dims)
    print("[DEBUG] coord_ds.coords:", list(coord_ds.coords))

    # ---- Write only coordinate metadata ----
    print("\n[DEBUG] Writing coordinate dataset to Zarr...")
    coord_ds.to_zarr(
        store,
        zarr_format=3,
        mode="w",
        consolidated=False,
        write_empty_chunks=False,
    )
    print("[DEBUG] Coordinate dataset written.")

    # ---- 2. Create empty array for latent_forecast ----
    print("\n[DEBUG] Creating empty zarr array for latent_forecast...")
    import zarr

    print("[DEBUG] zarr.create_array parameters:")
    print("        name        = latent_forecast")
    print("        shape       =", (n_init, n_lead, latent_dim))
    print("        dtype       = float32")
    print("        fill_value  = NaN")
    print("        dimension_names =", ("init_time", "lead_time", "lv"))

    zarr.create_array(
        store,
        name="latent_forecast",
        shape=(n_init, n_lead, latent_dim),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=("init_time", "lead_time", "lv"),
    )

    print("[DEBUG] latent_forecast array created (metadata only).")
    print("==============================================================\n")

    return coord_ds


def write_metadata(
    store: zarr.abc.store.Store,
    *,
    written_times: list[datetime.datetime],
) -> None:
    """
    Update derived metadata for the latent-vector dataset.

    Specifically maintains the `valid_time_range` root attribute with
    the following invariants:

      - valid_time_range[0] (start) is set once and never changes
      - valid_time_range[1] (end) monotonically advances as new data
        is written

    This function is intentionally conservative: it requires at least
    one successfully written timestamp and will refuse to run otherwise.

    Parameters
    ----------
    store
        The Zarr store backing the latent-vector dataset.
    written_times
        One or more timestamps that were successfully written during
        the current job (typically t+6h).
    """
    if not written_times:
        raise RuntimeError(
            "write_metadata() called with no written_times; refusing to update valid_time_range"
        )

    root = zarr.open_group(store)

    existing = root.attrs.get("valid_time_range", [None, None])

    def _parse(t):
        return datetime.datetime.fromisoformat(t) if isinstance(t, str) else t

    existing_start = _parse(existing[0])
    existing_end = _parse(existing[1])

    job_end = max(written_times)

    # Start is set once, then frozen
    new_start = existing_start if existing_start is not None else min(written_times)

    # End always advances (monotonic)
    new_end = max(existing_end, job_end) if existing_end is not None else job_end

    root.attrs["valid_time_range"] = [
        new_start.isoformat(),
        new_end.isoformat(),
    ]


def random_job_string(length: int) -> str:
    """Generate a short random identifier suitable for SLURM job grouping."""
    choices = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(choices) for _ in range(length))


def get_job_count(lv_job_id: str, retries: int = 2, delay: int = 5) -> int:
    """
    Count active SLURM jobs matching a latent-vector job identifier.

    This function queries `squeue` and counts jobs whose names contain
    `lv_job_id`. Retries are performed to tolerate transient scheduler
    failures.

    Returns
    -------
    int
        The number of currently running or queued jobs matching the ID.

    """
    last_err = None
    for _ in range(retries):
        result = subprocess.run(
            ["squeue", "-ho", "%j"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return sum(lv_job_id in line for line in result.stdout.splitlines())
        last_err = result.stderr
        time.sleep(delay)

    raise RuntimeError(f"squeue failed after {retries} retries: {last_err}")


def reshape_rollout_to_cube(lv_ds: xr.Dataset, init_time: datetime.datetime):
    print("\n[DEBUG][reshape_rollout_to_cube] lv_ds:")
    print("  lv_ds.dims:", lv_ds.dims)
    print("  lv_ds['lv'].shape:", lv_ds["lv"].shape)

    times = lv_ds["time"].values
    print("  raw times:", times)

    lead_hours = ((times - np.datetime64(init_time)) / np.timedelta64(1, "h")).astype("int64")
    print("  computed lead_hours:", lead_hours)

    lv = lv_ds["lv"].values
    print("  lv raw np.shape:", lv.shape)

    lv_flat = lv.reshape(lv.shape[0], -1)
    print("  lv_flat shape (steps × flattened):", lv_flat.shape)

    ds = xr.Dataset(
        coords={
            "init_time": [np.datetime64(init_time)],
            "lead_time": lead_hours,
        },
        data_vars={
            "latent_forecast": (("init_time", "lead_time", "lv"), lv_flat[None, :, :]),
        },
    )

    print("  cube dataset dims:", ds.dims)

    valid_time = np.datetime64(init_time) + lead_hours * np.timedelta64(1, "h")
    ds = ds.assign_coords(valid_time=(("init_time", "lead_time"), valid_time[None, :]))

    print("[DEBUG][reshape_rollout_to_cube] final cube dims:", ds.dims)
    return ds


class LatentVectorExtractor:
    """
    Aurora inference wrapper for ERA5 latent-vector extraction.

    Responsibilities:
      - Load sample and invariant datasets from a source repository
      - Run the Aurora model in GPU inference mode
      - Return latent vectors as an xarray.Dataset

    Temporal semantics:
      For an input timestamp `t`, the returned dataset is labeled at
      `t + 6h`, corresponding to the next Aurora prediction timestep.
      This offset is intentional, fixed, and centralized here.
    """

    def __init__(
        self,
        source_branch: str = "main",
        *,
        source_repo: str,
        client: arraylake.Client | None = None,
    ):
        print("\n[DEBUG][LVE.__init__] Loading repo:", source_repo)

        if client is None:
            client = arraylake.Client()

        repo = client.get_repo(source_repo)
        session = repo.readonly_session(source_branch)

        sample_ds = xr.open_zarr(
            session.store, group="samples", zarr_format=3, consolidated=False, chunks=None
        )
        inv_ds = xr.open_zarr(
            session.store, group="invariant", zarr_format=3, consolidated=False, chunks=None
        )

        self.data_loader = ERA5DataLoaderFOAM(sample_ds=sample_ds, invariant_ds=inv_ds)

        self.model = AuroraPretrained()
        self.model.load_checkpoint()
        self.model.eval()
        self.model.to("cuda")

    def __getitem__(self, item: datetime.datetime) -> xr.Dataset:
        if not isinstance(item, datetime.datetime):
            raise KeyError("Invalid key; must be datetime object")

        batch = self.data_loader[item]
        with torch.inference_mode():
            lv = self.model.forward(batch, lv_only=True).to("cpu").numpy()

        # Return a latent vector dataset with the timestamp moved forward 6 hours to match
        # the next Aurora prediction timestep corresponding to the latent vector extracted.
        out_time = item + datetime.timedelta(hours=6)  # ← single, explicit semantic step

        return xr.Dataset(
            coords={
                "time": xr.DataArray([out_time], dims=("time",)),
            },
            data_vars={
                "lv": xr.DataArray(lv, dims=("time", "spatial_location", "feature")),
            },
        )


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("start-time", type=click.DateTime())
@click.argument("end-time", type=click.DateTime())
@click.option("--src-repo", type=str, required=True, help="Source repository", show_default=True)
@click.option(
    "--dest-repo",
    type=str,
    required=True,
    help="Repository to be created",
    show_default=True,
)
@click.option(
    "--dest-branch", type=str, default="main", help="Destination branch", show_default=True
)
@click.option(
    "--src-branch", type=str, default="main", help="Destination branch", show_default=True
)
@click.option(
    "--rollout-steps", type=int, default=10, help="Number of lead times", show_default=True
)
def init(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_repo: str,
    dest_repo: str,
    dest_branch: str,
    src_branch: str,
    rollout_steps: int,
) -> None:
    """
    Initialize a new latent-vector repository.

    This command creates a destination repository and writes the
    initial Zarr layout, including the time coordinate and preallocated
    latent-vector array.
    """
    client = arraylake.Client()

    try:
        dest_repo_obj = client.create_repo(dest_repo)
    except Exception:
        dest_repo_obj = client.get_repo(dest_repo)

    dest_session = dest_repo_obj.writable_session(dest_branch)

    # ----------------------------------------
    # Time coordinate construction
    # ----------------------------------------
    init_times = pd.date_range(start_time, end_time, freq="6H")
    n_init = len(init_times)

    lead_times = pd.to_timedelta(np.arange(rollout_steps) * 6, unit="h")
    n_lead = len(lead_times)

    valid_times = np.array(init_times, dtype="datetime64[ns]")[:, None] + lead_times.values[None, :]

    # ----------------------------------------
    # Known Aurora latent dimension
    # ----------------------------------------
    latent_dim = 259200 * 1024

    print("\n[DEBUG][init] init_times:", init_times)
    print("[DEBUG] n_init:", n_init)
    print("[DEBUG] lead_times:", lead_times)
    print("[DEBUG] n_lead:", n_lead)
    print("[DEBUG] valid_times.shape:", valid_times.shape)
    print("[DEBUG] latent_dim:", latent_dim)

    # ----------------------------------------
    # Call lazy Zarr initializer
    # ----------------------------------------
    ds = init_zarr_store(
        store=dest_session.store,
        n_init=n_init,
        n_lead=n_lead,
        latent_dim=latent_dim,
        init_times=init_times,
        lead_times=lead_times,
        valid_times=valid_times,
    )

    commit_id = dest_session.commit("Initialized latent forecast Zarr schema.")
    print(f"[DEBUG] Zarr initialized and committed: {commit_id}")

    return ds, dest_session


@cli.command()
@click.argument("start-time", type=click.DateTime())
@click.argument("end-time", type=click.DateTime())
@click.option("--src-repo", required=True, type=str, help="Source repository", show_default=True)
@click.option("--dest-repo", type=str, required=True, help="Destination repo", show_default=True)
@click.option(
    "--write-session-location",
    type=str,
    default=None,
    help="An S3 bucket where coordinated write sessions live (optional)",
)
@click.option(
    "--aws-profile", type=str, default="kafou", help="AWS profile name", show_default=True
)
@click.option(
    "--dest-branch", type=str, default="main", help="Destination branch", show_default=True
)
@click.option(
    "--src-branch", type=str, default="main", help="Destination branch", show_default=True
)
def save_lvs(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_repo: str,
    dest_repo: str,
    write_session_location: str | None,
    aws_profile: str,
    src_branch: str,
    dest_branch: str,
) -> None:
    """
    Generate and write Aurora latent vectors for a contiguous time range.

    This command:
      - Validates 6-hourly time alignment
      - Runs Aurora inference for each timestep
      - Writes latent vectors at (t + 6h)
      - Ensures the destination time axis is extended safely
      - Updates dataset metadata to reflect newly written coverage

    When `write_session_location` is provided, writes occur into a
    coordinated session for later merging rather than committing
    immediately.
    """

    if (
        start_time.hour not in (0, 6, 12, 18)
        or start_time.minute != 0
        or start_time.second != 0
        or start_time.microsecond != 0
    ):
        raise click.ClickException("Invalid start time")
    if (
        end_time.hour not in (0, 6, 12, 18)
        or end_time.minute != 0
        or end_time.second != 0
        or end_time.microsecond != 0
    ):
        raise click.ClickException("Invalid end time")

    times = []
    this_time = start_time
    while this_time <= end_time:
        times.append(this_time)
        this_time = this_time + datetime.timedelta(hours=6)

    client = arraylake.Client()

    if write_session_location is not None:
        fs = fsspec.filesystem("s3", profile=aws_profile)
        with fs.open(os.path.join(write_session_location, "session.pickle"), "rb") as fobj:
            dest_session = pickle.load(fobj)
    else:
        repo = client.get_repo(dest_repo)
        dest_session = repo.writable_session(dest_branch)

    lve = LatentVectorExtractor(source_branch=src_branch, source_repo=src_repo, client=client)

    # compute max time this job will write
    final_write_time = end_time + datetime.timedelta(hours=6)

    # 🔒 ensure time axis ONCE
    ensure_time_in_arrays(
        dest_session.store,
        final_write_time,
        time_dim="time",
        time_frequency="auto",
    )

    last_written = None

    for timestamp in times:
        print(f"{timestamp:%Y-%m-%d %H:%M:%S}")

        lv = lve[timestamp]

        lv.to_zarr(
            dest_session.store,
            zarr_format=3,
            consolidated=False,
            region="auto",
            mode="a",
        )

        last_written = timestamp + datetime.timedelta(hours=6)

    if write_session_location is None and last_written is not None:
        write_metadata(dest_session.store, written_times=[last_written])

    if write_session_location is None:
        commit_id = dest_session.commit(
            f"Added {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S}"
        )
        print(f"Commited data: {commit_id}")
    else:
        outpath = os.path.join(
            write_session_location,
            f"lv_{start_time:%Y%m%dT%H%M%S}_{end_time:%Y%m%dT%H%M%S}_.pickle",
        )
        fs = fsspec.filesystem("s3", profile=aws_profile)
        print(f"Writing {outpath}")
        with fs.open(outpath, "wb") as fobj:
            pickle.dump(dest_session, fobj)


@cli.command()
@click.argument("start-time", type=click.DateTime())
@click.argument("end-time", type=click.DateTime())
@click.option("--src-repo", type=str, required=True, help="Source repository", show_default=True)
@click.option("--dest-repo", type=str, required=True, help="Destination repo", show_default=True)
@click.option(
    "--aws-profile", type=str, default="kafou", help="AWS profile name", show_default=True
)
@click.option(
    "--coordination-location",
    type=str,
    default="s3://icechunk-write-coordination",
    show_default=True,
)
@click.option("--timesteps-per-job", type=click.INT, default=4 * 2)
@click.option(
    "--dest-branch", type=str, default="main", help="Destination branch", show_default=True
)
@click.option(
    "--src-branch", type=str, default="main", help="Destination branch", show_default=True
)
def submit_jobs(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_repo: str,
    dest_repo: str,
    aws_profile: str,
    coordination_location: str,
    timesteps_per_job: int,
    src_branch: str,
    dest_branch: str,
) -> None:
    """
    Run distributed latent-vector generation via SLURM.

    This command:
      - Splits a large time range into smaller chunks
      - Submits one GPU job per chunk
      - Pre-extends the destination time axis once
      - Merges completed write sessions
      - Updates metadata and commits atomically

    Intended for large historical backfills and high-throughput
    production pipelines.
    """
    if (
        start_time.hour not in (0, 6, 12, 18)
        or start_time.minute != 0
        or start_time.second != 0
        or start_time.microsecond != 0
    ):
        raise click.ClickException("Invalid start time")
    if (
        end_time.hour not in (0, 6, 12, 18)
        or end_time.minute != 0
        or end_time.second != 0
        or end_time.microsecond != 0
    ):
        raise click.ClickException("Invalid end time")

    time_delta = datetime.timedelta(hours=6 * (timesteps_per_job - 1))

    next_start = start_time
    next_end = min(start_time + time_delta, end_time)

    time_spans = []

    while next_start < end_time:
        time_spans.append((next_start, next_end))
        next_start = next_end + datetime.timedelta(hours=6)
        next_end = min(next_start + time_delta, end_time)

    lv_job_id = random_job_string(10)
    session_location = os.path.join(coordination_location, lv_job_id)
    session_pickle = os.path.join(session_location, "session.pickle")

    client = arraylake.Client()
    repo = client.get_repo(dest_repo)
    session = repo.writable_session(dest_branch)

    # 🔒 PRE-EXTEND TIME AXIS (optimization only)
    final_write_time = end_time + datetime.timedelta(hours=6)
    ensure_time_in_arrays(
        session.store,
        final_write_time,
        time_dim="time",
        time_frequency="auto",
    )

    # Optional but recommended: commit the pre-extension
    session.commit(f"Pre-extend time axis through {final_write_time}")

    print(f"Saving the session pickle to {session_pickle}")
    fs = fsspec.filesystem("s3", profile=aws_profile)
    with fs.open(session_pickle, "wb") as fobj:
        with session.allow_pickling():
            pickle.dump(session, fobj)

    # Use this later to tell the user who ran this if any time spans were missing.
    ts_tracking = set()

    for start, end in time_spans:
        start_string = start.strftime("%Y-%m-%dT%H:%M:%S")
        end_string = end.strftime("%Y-%m-%dT%H:%M:%S")
        ts_tracking.add((start.strftime("%Y%m%dT%H%M%S"), end.strftime("%Y%m%dT%H%M%S")))
        command = [
            "sbatch",
            "--ntasks=1",
            "--cpus-per-task=32",
            "--gpus=1",
            f"--job-name=lv-{lv_job_id}_{start_string}_{end_string}",
            sys.argv[0],
            "save-lvs",
            f"--src-repo={src_repo}",
            f"--dest-repo={dest_repo}",
            f"--aws-profile={aws_profile}",
            f"--write-session-location={session_location}",
            f"--src-branch={src_branch}",
            f"--dest-branch={dest_branch}",
            start_string,
            end_string,
        ]
        subprocess.run(command)

        res = subprocess.run(command, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"sbatch failed: {res.stderr}\ncmd={' '.join(command)}")

    time.sleep(10)

    job_count = get_job_count(lv_job_id)
    while job_count != 0:
        time.sleep(60)
        job_count = get_job_count(lv_job_id)
        print(f"{job_count} jobs remaining")

    print("All jobs completed, gathering results...")

    sessions = []
    for fspath in fs.ls(session_location):
        filename = fspath.split("/")[-1]
        if filename.startswith("lv_") and filename.endswith(".pickle"):
            start_string, end_string = filename.split("_")[1:3]
            ts_tracking.remove((start_string, end_string))
            with fs.open(fspath, "rb") as fobj:
                sessions.append(pickle.load(fobj))
        fs.rm(fspath)

    base_session = repo.writable_session(dest_branch)

    merged = merge_sessions(base_session, *sessions)

    write_metadata(merged.store, written_times=[end_time + datetime.timedelta(hours=6)])

    if ts_tracking:
        commit_message = (
            f"Partial add of {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S}"
        )
    else:
        commit_message = f"Add {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S}"
    commit_id = merged.commit(commit_message)
    print(f"Committed data: {commit_id}")

    if ts_tracking:
        print("Missing time spans:")
        for start, end in sorted(ts_tracking):
            start = datetime.datetime.strptime(start, "%Y%m%dT%H%M%S")
            end = datetime.datetime.strptime(end, "%Y%m%dT%H%M%S")
            print(f"    {start:%Y-%m-%dT%H:%M:%S} {end:%Y-%m-%dT%H:%M:%S}")


if __name__ == "__main__":
    cli()
