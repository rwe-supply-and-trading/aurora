#!/usr/bin/env python

import datetime
import io
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
from icechunk.distributed import merge_sessions

from aurora import AuroraPretrained
from aurora.data import ERA5DataLoaderFOAM

SOURCE_REPO = "kafou/aurora-era5-samples"
SOURCE_BRANCH = "extend-2025"

DESTINATION_REPO = "kafou/aurora-era5-t1-latent-vectors"
DESTINATION_BRANCH = "main"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def random_job_string(length: int) -> str:
    return "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(length))


def get_job_count(lv_job_id):
    result = subprocess.run(["squeue", "-ho", "%j"], capture_output=True)
    if result.returncode != 0:
        return -1

    count = 0
    for line in io.BytesIO(result.stdout):
        if lv_job_id in line.decode():
            count += 1
    return count


# ---------------------------------------------------------------------
# Latent vector extraction
# ---------------------------------------------------------------------


class LatentVectorExtractor:
    def __init__(self, *, source_repo, client, source_branch):
        print("[LVE] Initializing LatentVectorExtractor")

        repo = client.get_repo(source_repo)
        session = repo.readonly_session(source_branch)

        print("[LVE] Opening sample + invariant datasets")
        sample_ds = xr.open_zarr(
            session.store, group="samples", zarr_format=3, consolidated=False, chunks=None
        )
        inv_ds = xr.open_zarr(
            session.store, group="invariant", zarr_format=3, consolidated=False, chunks=None
        )

        self.data_loader = ERA5DataLoaderFOAM(sample_ds=sample_ds, invariant_ds=inv_ds)

        print("[LVE] Loading Aurora model")
        self.model = AuroraPretrained()
        self.model.load_checkpoint()
        self.model.eval()
        self.model.to("cuda")

    def __getitem__(self, ts: datetime.datetime):
        print(f"[LVE] Extracting LV for source time {ts}")
        batch = self.data_loader[ts]

        with torch.inference_mode():
            lv = self.model.forward(batch, lv_only=True).to("cpu").numpy()

        out_time = ts + datetime.timedelta(hours=6)
        print(f"[LVE] Produced LV for target time {out_time}")

        return xr.Dataset(
            coords={"time": xr.DataArray([out_time], dims=("time",))},
            data_vars={"lv": xr.DataArray(lv, dims=("time", "spatial_location", "feature"))},
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


@click.group()
def cli():
    pass


# ---------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------


def init_zarr_store(*, store, sample_ds):
    ds = xr.Dataset(coords={"time": sample_ds.time[1:]})
    ds.to_zarr(
        store, zarr_format=3, consolidated=False, encoding={"time": {"chunks": (len(ds.time),)}}
    )

    zarr.create_array(
        store,
        name="lv",
        shape=(len(ds.time), 259200, 1024),
        chunks=(1, 2025, 256),
        dimension_names=("time", "spatial_location", "feature"),
        compressors=[],
        dtype=np.float32,
        fill_value=np.nan,
    )


@cli.command()
@click.option(
    "--src-repo", type=str, default=SOURCE_REPO, help="Source repository", show_default=True
)
@click.option(
    "--dest-repo",
    type=str,
    default=DESTINATION_REPO,
    help="Repository to be created",
    show_default=True,
)
@click.option(
    "--src-branch", type=str, default=SOURCE_BRANCH, help="Source repository", show_default=True
)
@click.option(
    "--dest-branch",
    type=str,
    default=DESTINATION_BRANCH,
    help="Repository to be created",
    show_default=True,
)
def init(src_repo, dest_repo, src_branch, dest_branch):
    """Initialize a new latent vector repository."""
    dest_repo_name = dest_repo

    client = arraylake.Client()

    src_repo = client.get_repo(src_repo)
    src_session = src_repo.readonly_session(src_branch)

    dest_repo = client.create_repo(dest_repo)
    dest_session = dest_repo.writable_session(dest_branch)

    sample_ds = xr.open_zarr(
        src_session.store, group="samples", zarr_format=3, consolidated=False, chunks=None
    )

    init_zarr_store(store=dest_session.store, sample_ds=sample_ds)
    commit_id = dest_session.commit("Initialized latent vector store.")
    print(f"Initialized repo {dest_repo_name}: {commit_id}")


# ---------------------------------------------------------------------
# SAVE LVS (explicit overwrite semantics)
# ---------------------------------------------------------------------


@cli.command()
@click.argument("start-time", type=click.DateTime())
@click.argument("end-time", type=click.DateTime())
@click.option("--src-repo", default=SOURCE_REPO)
@click.option("--dest-repo", default=DESTINATION_REPO)
@click.option("--src-branch", default=SOURCE_BRANCH)
@click.option("--dest-branch", default=DESTINATION_BRANCH)
@click.option("--aws-profile", default="kafou")
@click.option("--write-session-location", default=None)
@click.option("--overwrite", is_flag=True, help="Explicitly allow overwriting existing timestamps")
def save_lvs(
    start_time,
    end_time,
    src_repo,
    dest_repo,
    src_branch,
    dest_branch,
    aws_profile,
    write_session_location,
    overwrite,
):
    if not overwrite:
        raise click.ClickException("Refusing to run without --overwrite")

    print(f"[SAVE] Overwrite job from {start_time} → {end_time}")

    # Build source times (Python datetimes)
    times = []
    t = start_time
    while t <= end_time:
        times.append(t)
        t += datetime.timedelta(hours=6)

    client = arraylake.Client()

    if write_session_location:
        fs = fsspec.filesystem("s3", profile=aws_profile)
        with fs.open(os.path.join(write_session_location, "session.pickle"), "rb") as f:
            dest_session = pickle.load(f)
        print("[SAVE] Loaded coordinated session from S3")
    else:
        repo = client.get_repo(dest_repo)
        dest_session = repo.writable_session(dest_branch)
        print("[SAVE] Opened direct writable session")

    root = zarr.open_group(dest_session.store, mode="r+", zarr_format=3)

    # Load time index ONCE and canonicalize
    time_index = root["time"][:].astype("datetime64[ns]")

    lve = LatentVectorExtractor(
        source_repo=src_repo,
        client=client,
        source_branch=src_branch,
    )

    for ts in times:
        lv_ds = lve[ts]

        # Canonical target time
        target_ns = np.datetime64(lv_ds.time.values[0], "ns")

        print(f"[SAVE] Verifying target timestamp {target_ns}")

        matches = np.where(time_index == target_ns)[0]
        if len(matches) != 1:
            raise RuntimeError(
                f"[SAVE][ERROR] Expected exactly one match for time {target_ns}, "
                f"found {len(matches)}"
            )

        idx = int(matches[0])
        print(f"[SAVE] Overwriting time index {idx}")

        # Explicit region write (NO auto inference)
        root["lv"][
            idx : idx + 1,
            :,
            :,
        ] = lv_ds["lv"].values

        # Cheap sanity check (sample, not full slab)
        nan_after = np.isnan(root["lv"][idx, :1000, :100]).sum()
        print(f"[SAVE] Sample NaNs after write: {nan_after}")

    if not write_session_location:
        commit_id = dest_session.commit(
            f"Overwrite LVs {start_time:%Y-%m-%dT%H:%M:%S} → {end_time:%Y-%m-%dT%H:%M:%S}"
        )
        print(f"[SAVE] Commit complete: {commit_id}")
    else:
        out = os.path.join(
            write_session_location,
            f"lv_{start_time:%Y%m%dT%H%M%S}_{end_time:%Y%m%dT%H%M%S}.pickle",
        )
        fs = fsspec.filesystem("s3", profile=aws_profile)
        print(f"[SAVE] Writing session fragment → {out}")
        with fs.open(out, "wb") as f:
            pickle.dump(dest_session, f)


@cli.command()
@click.argument("start-time", type=click.DateTime())
@click.argument("end-time", type=click.DateTime())
@click.option(
    "--src-repo", type=str, default=SOURCE_REPO, help="Source repository", show_default=True
)
@click.option(
    "--dest-repo", type=str, default=DESTINATION_REPO, help="Destination repo", show_default=True
)
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
    "--src-branch", type=str, default=SOURCE_BRANCH, help="Source repository", show_default=True
)
@click.option(
    "--dest-branch",
    type=str,
    default=DESTINATION_BRANCH,
    help="Repository to be created",
    show_default=True,
)
def submit_jobs(
    start_time,
    end_time,
    src_repo,
    dest_repo,
    aws_profile,
    coordination_location,
    timesteps_per_job,
    src_branch,
    dest_branch,
):
    # -----------------------#
    # Get time spans to add #
    # -----------------------#
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

    # ---------------------------------------#
    # Fill repo with NaNs for new timestamps #
    # ---------------------------------------#

    # get dest session to write to
    client = arraylake.Client()
    repo = client.get_repo(dest_repo)
    session = repo.writable_session(dest_branch)

    # ensure all time coords exist
    ensure_time_in_arrays(store=session.store, timestamp=end_time, time_dim="time")

    try:
        # commit so that the new time coords are saved
        session.commit("Added NaN time coords to lv destination.")
    except icechunk.IcechunkError as e:
        error_msg = str(e)
        if "no changes" in error_msg:
            print(f"Timestamps already exist, skipping commit: {e}")
        else:
            # Re-raise if it's a different error
            raise

    # reopen session after commit
    session = repo.writable_session(dest_branch)

    # ---------------------------------------------------#
    # Save the session pickle to S3 for the jobs to use #
    # ---------------------------------------------------#

    # Generate JobIDs for each process
    lv_job_id = random_job_string(10)
    session_location = os.path.join(coordination_location, lv_job_id)
    session_pickle = os.path.join(session_location, "session.pickle")

    print(f"Saving the session pickle to {session_pickle}")
    fs = fsspec.filesystem("s3", profile=aws_profile)
    with fs.open(session_pickle, "wb") as fobj:
        with session.allow_pickling():
            pickle.dump(session, fobj)

    # Use this later to tell the user who ran this if any time spans were missing.
    ts_tracking = set()

    # ---------------------------------------#
    # Create jobs to process each time span #
    # ---------------------------------------#
    for start, end in time_spans:
        start_string = start.strftime("%Y-%m-%dT%H:%M:%S")
        end_string = end.strftime("%Y-%m-%dT%H:%M:%S")
        ts_tracking.add((start.strftime("%Y%m%dT%H%M%S"), end.strftime("%Y%m%dT%H%M%S")))
        command = [
            "sbatch",
            "--ntasks=1",
            "--cpus-per-task=32",
            "--gpus=1",
            f"--job-name=lv-{lv_job_id} {start_string} {end_string}",
            sys.argv[0],
            "save-lvs",
            f"--src-repo={src_repo}",
            f"--dest-repo={dest_repo}",
            f"--aws-profile={aws_profile}",
            f"--src-branch={src_branch}",
            f"--dest-branch={dest_branch}",
            f"--write-session-location={session_location}",
            "--overwrite",
            start_string,
            end_string,
        ]
        subprocess.run(command)

    time.sleep(10)

    job_count = get_job_count(lv_job_id)
    while job_count != 0:
        time.sleep(60)
        job_count = get_job_count(lv_job_id)
        print(f"{job_count} jobs remaining")

    print("All jobs completed, gathering results...")

    # -------------------------------------------------------- #
    # Gather results from each job run, merge into one session #
    # -------------------------------------------------------- #
    sessions = []
    for fspath in fs.ls(session_location):
        filename = fspath.split("/")[-1]
        if filename.startswith("lv_") and filename.endswith(".pickle"):
            start_string, end_string = filename.split("_")[1:3]
            ts_tracking.remove((start_string, end_string))  # store failed timestamps
            with fs.open(fspath, "rb") as fobj:
                sessions.append(pickle.load(fobj))
        fs.rm(fspath)

    session = merge_sessions(session, *sessions)

    if ts_tracking:
        commit_message = (
            f"Partial add of {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S}"
        )
    else:
        commit_message = f"Add {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S}"

    # zarr_group = zarr.open_group(session.store, mode="r+", zarr_format=3)
    # zarr_group.attrs["valid_time_range"] = [start_string, end_string]
    # zarr_group.attrs["FAILED_TIMESTAMPS"] = (
    #     [list(item) for item in ts_tracking] if ts_tracking else []
    # )

    commit_id = session.commit(commit_message)
    print(f"Committed data: {commit_id}")

    if ts_tracking:
        print("Missing time spans:")
        for start, end in sorted(ts_tracking):
            start = datetime.datetime.strptime(start, "%Y%m%dT%H%M%S")
            end = datetime.datetime.strptime(end, "%Y%m%dT%H%M%S")
            print(f"    {start:%Y-%m-%dT%H:%M:%S} {end:%Y-%m-%dT%H:%M:%S}")


if __name__ == "__main__":
    cli()
