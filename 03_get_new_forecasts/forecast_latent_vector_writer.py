#!/usr/bin/env python

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

sbatch   --ntasks=1   --cpus-per-task=8   --mem=50G   --job-name=lv-submit   --wrap="
    python latent_forecast_writer.py submit-jobs \
      2025-07-01T00:00:00 \
      2025-07-31T18:00:00 \
      --src-repo kafou/aurora-era5-samples \
      --src-branch extend-2025 \
      --dest-repo kafou/aurora-era5-forecast-latent-vectors \
      --dest-branch main \
      --aws-profile kafou \
      --timesteps-per-job 8 \
      --coordination-location s3://icechunk-write-coordination
"

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


# ============================================================
# Extractor
# ============================================================


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
        *,
        source_repo: str = SOURCE_REPO,
        client: arraylake.Client | None = None,
        source_branch: str = SOURCE_BRANCH,
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

        print(f"[LVE] sample_ds.sizes={dict(sample_ds.sizes)}")
        print(f"[LVE] invariant_ds.sizes={dict(inv_ds.sizes)}")

        self.data_loader = ERA5DataLoaderFOAM(sample_ds=sample_ds, invariant_ds=inv_ds)

        print("[LVE] Loading Aurora model checkpoint...")
        self.model = AuroraPretrained()
        self.model.load_checkpoint()
        self.model.eval()
        self.model.to(device)
        self.device = device
        print("[LVE] Model ready.")

    def rollout_lvs(self, item: datetime.datetime, steps: int) -> xr.Dataset:
        if not isinstance(item, datetime.datetime):
            raise KeyError("Invalid key; must be datetime object")

        print(f"\n[LVE.rollout_lvs] item={item} steps={steps}")
        batch = self.data_loader[item]

        times = []
        lvs = []

        with torch.inference_mode():
            for step, (_pred, latent) in enumerate(rollout_with_latents(self.model, batch, steps)):
                t = item + datetime.timedelta(hours=6 * (step + 1))
                times.append(t)

                # latent: (1, S, F) -> (S, F)
                latent_np = latent.detach().to("cpu").numpy().squeeze(0)
                print(f"[LVE.rollout_lvs] step={step} latent_np.shape={latent_np.shape}")
                lvs.append(latent_np)

        lv_arr = np.stack(lvs, axis=0)  # (steps, S, F)
        print(f"[LVE.rollout_lvs] lv_arr.shape={lv_arr.shape}")

        return xr.Dataset(
            coords={"time": ("time", times)},
            data_vars={"lv": (("time", "spatial_location", "feature"), lv_arr)},
        )


# ============================================================
# CLI
# ============================================================


@click.group()
def cli():
    pass


def init_forecast_zarr_store(
    *,
    store,
    src_ds: xr.Dataset,
    init_times: np.ndarray,
    rollout_steps: int,
):
    """
    Create an EMPTY latent forecast Zarr store.

    Resulting schema:

      latent_forecast(
        init_time,
        lead_time,
        spatial_location,
        feature
      )

    All latent dimensions are inferred from src_ds.
    """

    n_spatial = int(src_ds.sizes["spatial_location"])
    n_feature = int(src_ds.sizes["feature"])

    lead_times = (np.arange(1, rollout_steps + 1) * 6).astype("int64")
    valid_times = init_times[:, None] + lead_times[None, :] * np.timedelta64(1, "h")

    # Write coordinate-only dataset
    coord_ds = xr.Dataset(
        coords={
            "init_time": ("init_time", init_times),
            "lead_time": ("lead_time", lead_times),
            "spatial_location": ("spatial_location", np.arange(n_spatial, dtype="int64")),
            "feature": ("feature", np.arange(n_feature, dtype="int64")),
            "valid_time": (("init_time", "lead_time"), valid_times),
        },
        attrs={
            "description": "Aurora latent forecast dataset",
            "source_schema": "lv(time, spatial_location, feature)",
            "rollout_steps": int(rollout_steps),
            "schema_version": "v1",
        },
    )

    coord_ds.to_zarr(
        store,
        zarr_format=3,
        mode="w",
        consolidated=False,
        write_empty_chunks=False,
    )

    # Create empty latent_forecast array (metadata-only)
    zarr.create_array(
        store,
        name="latent_forecast",
        shape=(
            len(init_times),
            len(lead_times),
            n_spatial,
            n_feature,
        ),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=(
            "init_time",
            "lead_time",
            "spatial_location",
            "feature",
        ),
    )


@cli.command("init")
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--dest-repo", required=True, show_default=True)
@click.option("--dest-branch", default="main", show_default=True)
@click.option("--src-repo", required=True, show_default=True)
@click.option("--src-branch", default="main", show_default=True)
@click.option("--rollout-steps", type=int, default=10, show_default=True)
def init(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    dest_repo: str,
    dest_branch: str,
    src_repo: str,
    src_branch: str,
    rollout_steps: int,
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

    # Validate 6-hour alignment
    for t, name in [(start_time, "start_time"), (end_time, "end_time")]:
        if t.hour not in (0, 6, 12, 18) or t.minute or t.second or t.microsecond:
            raise click.ClickException(f"{name} must be 6-hour aligned (00/06/12/18)")

    if start_time > end_time:
        raise click.ClickException("start_time must be <= end_time")

    client = arraylake.Client()

    # Open source dataset (for dimension inference)
    src_repo_obj = client.get_repo(src_repo)
    src_session = src_repo_obj.readonly_session(src_branch)

    src_ds = xr.open_zarr(
        src_session.store,
        group="samples",
        zarr_format=3,
        consolidated=False,
    )

    # Build init_time axis (pure datetime)
    init_times = []
    this_time = start_time
    while this_time <= end_time:
        init_times.append(this_time)
        this_time += datetime.timedelta(hours=6)

    # Create destination repo + session
    dest_repo_obj = client.create_repo(dest_repo)
    session = dest_repo_obj.writable_session(dest_branch)

    # Delegate all real work to helper
    init_forecast_zarr_store(
        store=session.store,
        src_ds=src_ds,
        init_times=init_times,
        rollout_steps=rollout_steps,
    )

    commit_id = session.commit(
        f"init forecast schema: {start_time.isoformat()} .. {end_time.isoformat()} "
        f"steps={rollout_steps}"
    )
    print(f"[INIT] Committed: {commit_id}")


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
def save_lvs(
    start_time,
    end_time,
    src_repo,
    src_branch,
    dest_repo,
    dest_branch,
    write_session_location,
    aws_profile,
    rollout_steps,
):
    """
    Fill latent_forecast with rollouts.

    Writes one (init_time, :) cube at a time using region writes.
    """

    # Validate start/end time
    for t in (start_time, end_time):
        if t.hour not in (0, 6, 12, 18) or t.minute or t.second or t.microsecond:
            raise click.ClickException(
                "Invalid start/end time: must be 6-hour aligned (00/06/12/18)"
            )

    # Build init_times (pure datetime, pandas OK)
    init_times = []
    this_time = start_time
    while this_time <= end_time:
        init_times.append(this_time)
        this_time += datetime.timedelta(hours=6)

    client = arraylake.Client()

    # Destination session
    if write_session_location is not None:
        fs = fsspec.filesystem("s3", profile=aws_profile)
        with fs.open(os.path.join(write_session_location, "session.pickle"), "rb") as f:
            dest_session = pickle.load(f)
        print(f"[SAVE_LVS] Loaded session from {write_session_location}")
    else:
        repo = client.get_repo(dest_repo)
        dest_session = repo.writable_session(dest_branch)
        print(f"[SAVE_LVS] Opened writable session {dest_repo}:{dest_branch}")

    # Open store metadata for index mapping
    root = zarr.open_group(dest_session.store, mode="r", zarr_format=3)

    store_init = root["init_time"][:]  # datetime64
    store_lead = root["lead_time"][:]  # int64
    store_spatial = root["spatial_location"].size
    store_feature = root["feature"].size

    print(f"[SAVE_LVS] store init_time n={len(store_init)}")
    print(f"[SAVE_LVS] store lead_time={store_lead}")
    print(f"[SAVE_LVS] spatial_location={store_spatial}, feature={store_feature}")

    # Extractor
    lve = LatentVectorExtractor(
        source_repo=src_repo,
        source_branch=src_branch,
        client=client,
    )

    # compute max time this job will write
    # final_write_time = end_time + datetime.timedelta(hours=6)

    # ensure_time_in_arrays(
    #     dest_session.store,
    #     final_write_time,
    #     time_dim="time",
    #     time_frequency="auto",
    # )

    # Write cubes
    for init_time in init_times:
        print(f"\n[SAVE_LVS] Rolling out init={init_time}")

        lv_ds = lve.rollout_lvs(init_time, steps=rollout_steps)
        lv = lv_ds["lv"].values  # (steps, spatial_location, feature)

        # ------------------ build cube ------------------
        cube = xr.Dataset(
            data_vars={
                "latent_forecast": (
                    ("init_time", "lead_time", "spatial_location", "feature"),
                    lv[None, ...].astype("float32"),
                )
            },
            coords={
                "init_time": ("init_time", [init_time]),
                "lead_time": ("lead_time", store_lead),
                "spatial_location": ("spatial_location", np.arange(store_spatial)),
                "feature": ("feature", np.arange(store_feature)),
            },
        )

        # ------------------ find index ------------------
        matches = np.where(store_init == np.datetime64(init_time))[0]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly 1 match for init_time={init_time}, got {len(matches)}"
            )
        i = int(matches[0])

        print(f"[SAVE_LVS] Writing cube at init_time index {i}")

        cube.to_zarr(
            dest_session.store,
            zarr_format=3,
            consolidated=False,
            region={
                "init_time": slice(i, i + 1),
                "lead_time": slice(0, len(store_lead)),
                "spatial_location": slice(0, store_spatial),
                "feature": slice(0, store_feature),
            },
        )

    # Commit or pickle
    if write_session_location is None:
        commit_id = dest_session.commit(
            f"Added {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S}"
        )
        print(f"[SAVE_LVS] Committed: {commit_id}")
    else:
        outpath = os.path.join(
            write_session_location,
            f"lv_{start_time:%Y%m%dT%H%M%S}_{end_time:%Y%m%dT%H%M%S}.pickle",
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
@click.option("--rollout-steps", type=int, default=10, show_default=True)
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
    if expected_spans:
        commit_msg = (
            f"PARTIAL save-lvs {start_time.isoformat()}..{end_time.isoformat()} "
            f"(missing {len(expected_spans)} spans)"
        )
    else:
        commit_msg = f"save-lvs {start_time.isoformat()}..{end_time.isoformat()}"

    commit_id = merged.commit(commit_msg)
    print(f"[SUBMIT] Commit complete: {commit_id}")

    if expected_spans:
        print("[SUBMIT] Missing spans:")
        for s, e in sorted(expected_spans):
            print(f"  {s} → {e}")


if __name__ == "__main__":
    cli()
