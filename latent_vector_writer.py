#!/usr/bin/env python

import datetime
import io
import os
import pickle
import random
import subprocess
import sys
import time
from pprint import pprint

import fsspec
import kafou_arraylake as arraylake
import click
import numpy as np
import torch
import xarray as xr
import zarr

from icechunk.distributed import merge_sessions
from aurora import AuroraPretrained
from aurora.data import ERA5DataLoaderFOAM

SOURCE_REPO = "kafou/aurora-era5-samples"
DESTINATION_REPO = "kafou/aurora-era5-t1-latent-vectors"


def init_zarr_store(*, store, sample_ds):
    ds = xr.Dataset(coords={"time": sample_ds.time[1:]})
    ds.to_zarr(store, zarr_format=3, consolidated=False,
               encoding={"time": {"chunks": (len(ds.time),)}})

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


def random_job_string(length: int) -> str:
    choices = "abcdefghijklmnopqrstuvwxyz0123456789"

    return "".join(random.choice(choices) for _ in range(length))


def get_job_count(lv_job_id):
    result = subprocess.run(
        ["squeue", "-ho", "%j"],
        capture_output=True,
    )
    if result.returncode != 0 or result.stderr:
        return -1

    job_count = 0

    for line in io.BytesIO(result.stdout):
        line = line.decode("utf-8")
        if lv_job_id in line:
            job_count += 1

    return job_count


class LatentVectorExtractor:
    def __init__(self, *, source_repo: str = SOURCE_REPO, client: arraylake.Client | None = None):
        if client is None:
            client = arraylake.Client()

        repo = client.get_repo(source_repo)
        session = repo.readonly_session("main")
        sample_ds = xr.open_zarr(session.store, group="samples", zarr_format=3, consolidated=False, chunks=None)
        inv_ds = xr.open_zarr(session.store, group="invariant", zarr_format=3, consolidated=False, chunks=None)

        self.data_loader = ERA5DataLoaderFOAM(sample_ds=sample_ds, invariant_ds=inv_ds)

        self.model = AuroraPretrained()
        self.model.load_checkpoint()
        self.model.eval()
        self.model.to("cuda")

    def __getitem__(self, item: datetime.datetime):
        if not isinstance(item, datetime.datetime):
            raise KeyError("Invalid key; must be datetime object")

        batch = self.data_loader[item]
        with torch.inference_mode():
            lv = self.model.forward(batch, lv_only=True).to("cpu").numpy()

        # Return a latent vector dataset with the timestamp moved forward 6 hours to match
        # the next Aurora prediction timestep corresponding to the latent vector extracted.
        return xr.Dataset(
            coords={
                "time": xr.DataArray([item + datetime.timedelta(hours=6)], dims=("time",)),
            },
            data_vars={
                "lv": xr.DataArray(lv, dims=("time", "spatial_location", "feature"))
            }
        )




@click.group()
def cli():
    pass


@cli.command()
@click.option("--src-repo", type=str, default=SOURCE_REPO, help="Source repository", show_default=True)
@click.option("--dest-repo", type=str, default=DESTINATION_REPO, help="Repository to be created", show_default=True)
def init(src_repo, dest_repo):
    """Initialize a new latent vector repository."""
    dest_repo_name = dest_repo

    client = arraylake.Client()

    src_repo = client.get_repo(src_repo)
    src_session = src_repo.readonly_session("main")

    dest_repo = client.create_repo(dest_repo)
    dest_session = dest_repo.writable_session("main")

    sample_ds = xr.open_zarr(src_session.store, group="samples", zarr_format=3, consolidated=False, chunks=None)

    init_zarr_store(store=dest_session.store, sample_ds=sample_ds)
    commit_id = dest_session.commit("Initialized latent vector store.")
    print(f"Initialized repo {dest_repo_name}: {commit_id}")


@cli.command()
@click.argument("start-time", type=click.DateTime())
@click.argument("end-time", type=click.DateTime())
@click.option("--src-repo", type=str, default=SOURCE_REPO, help="Source repository", show_default=True)
@click.option("--dest-repo", type=str, default=DESTINATION_REPO, help="Destination repo", show_default=True)
@click.option("--write-session-location", type=str, default=None, help="An S3 bucket where coordinated write sessions live (optional)")
@click.option("--aws-profile", type=str, default="kafou", help="AWS profile name", show_default=True)
def save_lvs(start_time, end_time, src_repo, dest_repo, write_session_location, aws_profile):
    """Generate and save latent vectors."""

    if start_time.hour not in (0, 6, 12, 18) or start_time.minute != 0 or start_time.second != 0 or start_time.microsecond != 0:
        raise click.ClickException("Invalid start time")
    if end_time.hour not in (0, 6, 12, 18) or end_time.minute != 0 or end_time.second != 0 or end_time.microsecond != 0:
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
        dest_session = repo.writable_session("main")

    lve = LatentVectorExtractor(source_repo=src_repo, client=client)

    for timestamp in times:
        print(f"{timestamp:%Y-%m-%d %H:%M:%S}")
        lv = lve[timestamp]
        lv.to_zarr(dest_session.store, zarr_format=3, consolidated=False, region="auto")

    if write_session_location is None:
        commit_id = dest_session.commit(f"Added {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S}")
        print(f"Commited data: {commit_id}")
    else:
        outpath = os.path.join(
            write_session_location,
            f"lv_{start_time:%Y%m%dT%H%M%S}_{end_time:%Y%m%dT%H%M%S}_.pickle"
        )
        fs = fsspec.filesystem("s3", profile=aws_profile)
        print(f"Writing {outpath}")
        with fs.open(outpath, "wb") as fobj:
            pickle.dump(dest_session, fobj)


@cli.command()
@click.argument("start-time", type=click.DateTime())
@click.argument("end-time", type=click.DateTime())
@click.option("--src-repo", type=str, default=SOURCE_REPO, help="Source repository", show_default=True)
@click.option("--dest-repo", type=str, default=DESTINATION_REPO, help="Destination repo", show_default=True)
@click.option("--aws-profile", type=str, default="kafou", help="AWS profile name", show_default=True)
@click.option("--coordination-location", type=str, default="s3://icechunk-write-coordination", show_default=True)
@click.option("--timesteps-per-job", type=click.INT, default=4 * 2)
def submit_jobs(start_time, end_time, src_repo, dest_repo, aws_profile, coordination_location, timesteps_per_job):
    if start_time.hour not in (0, 6, 12, 18) or start_time.minute != 0 or start_time.second != 0 or start_time.microsecond != 0:
        raise click.ClickException("Invalid start time")
    if end_time.hour not in (0, 6, 12, 18) or end_time.minute != 0 or end_time.second != 0 or end_time.microsecond != 0:
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
    session = repo.writable_session("main")

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
            f"--job-name=lv-{lv_job_id} {start_string} {end_string}",
            sys.argv[0],
            "save-lvs",
            f"--src-repo={src_repo}",
            f"--dest-repo={dest_repo}",
            f"--aws-profile={aws_profile}",
            f"--write-session-location={session_location}",
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

    sessions = []
    for fspath in fs.ls(session_location):
        filename = fspath.split("/")[-1]
        if filename.startswith("lv_") and filename.endswith(".pickle"):
            start_string, end_string = filename.split("_")[1:3]
            ts_tracking.remove((start_string, end_string))
            with fs.open(fspath, "rb") as fobj:
                sessions.append(pickle.load(fobj))
        fs.rm(fspath)

    session = merge_sessions(session, *sessions)

    if ts_tracking:
        commit_message = f"Partial add of {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S}"
    else:
        commit_message = f"Add {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S}"
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