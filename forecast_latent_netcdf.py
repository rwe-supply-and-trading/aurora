#!/usr/bin/env python

import os
import subprocess
import sys

import click
import kafou_arraylake as arraylake
import numpy as np
import xarray as xr
import zarr
from dataset_io import ensure_time_in_arrays
from forecast_utils import get_spatial_indices_from_bounds
from LatentVectorExtractor import LatentVectorExtractor
from obstore_utils import open_s3_zarr_store

# Prints from workers in real-time instead of buffering until the end of the job.
os.environ["PYTHONUNBUFFERED"] = "1"
xr.set_options(keep_attrs=True)

# @Zora has weird enviornment issues with certs. This prevents SSL errors when accessing S3.
for var in [
    "CURL_CA_BUNDLE",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
]:
    os.environ.pop(var, None)


# ------------------------------------------------------
# Initialize Zarr dataset
# ------------------------------------------------------
def initialize_dataset(
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
        Dimensions:         (init_time=N lead_time=rollout_steps, spatial_location=M, feature=1024)
        Coordinates:
            * feature       (feature) int64
            * init_time     (init_time) datetime64[ns]
            * lead_time     (lead_time) int64
            * spatial_location  (spatial_location) int64
        Data variables:
            lv              (init_time, lead_time, spatial_location, feature) float32


    Aurora Latent Vector Shape:
        259200 = 360 (lon) × 180 (lat) × 4 (levels)


    Flattened layout in memory:
        - The raw flat vector is stored C-style as [level, lat, lon]
          which we can reshape to (4, 180, 360, -1):
            dim 0: level  (4)
            dim 1: lat    (180)
            dim 2: lon    (360)
            dim 3: -1     (time dim)

        ┌──────────┬──────────┬──────────┬──────────┐
        │ Level 0  │ Level 1  │ Level 2  │ Level 3  │
        │  64,800  │  64,800  │  64,800  │  64,800  │
        └──────────┴──────────┴──────────┴──────────┘
        0        64799    129599    194399    259199


        ┌────────┬───────┬─────┬─────┐
        | index  │ level │ lat │ lon │
        ├────────┼───────┼─────┼─────┤
        |      0 │   0   │  0  │   0 │
        |      1 │   0   │  0  │   1 │
        |      2 │   0   │  0  │   2 │
        |   ...  │       │     │     │
        |   359  │   0   │  0  │ 359 │  ← end of lat=0
        |   360  │   0   │  1  │   0 │  ← next latitude row
        |   361  │   0   │  1  │   1 │
        |   ...  │       │     │     │
        |   719  │   0   │  1  │ 359 │  ← end of lat=1
        |   720  │   0   │  2  │   0 │
        |   ...  │       │     │     │
        | 64439  │   0   │ 178 │ 359 │  ← end of lat=178
        | 64440  │   0   │ 179 │   0 │  ← last latitude row
        | 64441  │   0   │ 179 │   1 │
        |   ...  │       │     │     │
        | 64799  │   0   │ 179 │ 359 │  ← end of level 0
        ├───────-┼───────┼─────┼─────┤
        |  64800 │   1   │  0  │   0 │  ← start of level 1
        |  64801 │   1   │  0  │   1 │
        |    ... │       │     │     │
        | 129599 │   1   │ 179 │ 359 │  ← end of level 1
        ├────────┼───────┼─────┼─────┤
        | 129600 │   2   │  0  │   0 │  ← start of level 2
        |    ... │       │     │     │
        | 194399 │   2   │ 179 │ 359 │  ← end of level 2
        ├────────┼───────┼─────┼─────┤
        | 194400 │   3   │  0  │   0 │  ← start of level 3
        | 194401 │   3   │  0  │   1 │
        |    ... │       │     │     │
        | 259199 │   3   │ 179 │ 359 │  ← end of level 3
        └────────┴───────┴─────┴─────┘

    """
    print("[INIT] Initializing destination store...")

    # This is always the same
    n_feature = 1024

    # Handle subsetting the latent grid
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

    # Else just return the whole latent grid
    else:
        print(
            "[INIT] Not subsetting coordinates, using full grid with lat=[-89.75, 90], lon=[-180, 179.75]"
        )
        spatial_coord = np.arange(259200, dtype="int64")
        n_spatial = 259200

    # Get lead / valid time
    lead_times = (np.arange(1, rollout_steps + 1) * 6).astype("int64")

    # Generate Coordinate xr.Dataset
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
            "lead_times": lead_times.tolist(),
            "spatial_subset_type": ("bounding_box" if lat_range and lon_range else "global"),
            "lat_range": (list(lat_range) if lat_range and lon_range else None),
            "lon_range": (list(lon_range) if lat_range and lon_range else None),
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

    # Open existing group (preserves attrs) and add array
    root = zarr.open_group(store, mode="a", zarr_format=3)
    root.create_array(
        name="lv",
        shape=(len(init_times), len(lead_times), n_spatial, n_feature),
        chunks=(1, len(lead_times), 1024, 128),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=("init_time", "lead_time", "spatial_location", "feature"),
    )
    root.attrs.update(coord_ds.attrs)  # restore attrs clobbered by open_group

    # Print dataset for verification
    print("[INIT] Createing lv DataArray...")
    ds = xr.open_zarr(store, consolidated=False)
    print(ds)

    print("[INIT] Initialization complete.")


# ------------------------------------------------------
# LATENT Worker
# ------------------------------------------------------
def run_worker(start_time: str, end_time: str, store_path: str, src: str) -> None:
    store = open_s3_zarr_store(location=store_path, profile="kafou")

    # Get coordinate information from Store
    ds_store = xr.open_zarr(store, consolidated=False)

    print(ds_store)
    spatial_indices = ds_store.spatial_location.values.tolist()
    rollout_steps = len(ds_store.lead_time.data)
    init_times = ds_store.init_time.sel(init_time=slice(start_time, end_time)).values

    print(
        f"[WORKER] Store spatial_indicies: [{spatial_indices[0:3]} ... {spatial_indices[-3:-1]}], len={len(spatial_indices)}"
    )

    # Instantiate LV Extractors
    if src == "ecmwf":
        lve = LatentVectorExtractor(
            source_repo="kafou/aurora-ecmwf-samples",
            source_branch="main",
            client=arraylake.Client(),
        )
    elif src == "era5":
        lve = LatentVectorExtractor(
            source_repo="kafou/aurora-era5-samples",
            source_branch="extend-2025",
            client=arraylake.Client(),
        )
    else:
        raise ValueError(f"Unknown source: {src}. Must be 'ecmwf' or 'era5'")

    for init_time in init_times:
        print("Processing:", init_time, flush=True)

        lv_ds = lve.rollout_lvs(
            item=init_time.astype("datetime64[s]").item(),
            steps=rollout_steps,
        )

        print(lv_ds)
        print(lv_ds.spatial_location)

        lv_arr = lv_ds["lv"].values  # (lead_time, spatial_location, feature)
        if len(spatial_indices) != (4 * 180 * 360):
            lv_arr = lv_arr[:, spatial_indices, :]

        lv_da = xr.DataArray(
            lv_arr[None, ...],
            dims=("init_time", "lead_time", "spatial_location", "feature"),
            coords={
                "init_time": [init_time],
                "lead_time": ds_store["lead_time"],
                "spatial_location": ds_store["spatial_location"],
                "feature": ds_store["feature"],
            },
            name="lv",
        )

        lv_da.to_zarr(
            store,
            zarr_format=3,
            consolidated=False,
            mode="a",
            region="auto",
        )

    print("[WORKER] Worker complete.")


# ------------------------------------------------------
# SLURM job submission
# ------------------------------------------------------
def submit_jobs(start_time, end_time, store_path, src):
    init_times = np.arange(
        np.datetime64(start_time),
        np.datetime64(end_time) + np.timedelta64(6, "h"),
        np.timedelta64(6, "h"),
    )

    batch_size = 180
    batches = [init_times[i : i + batch_size] for i in range(0, len(init_times), batch_size)]

    store = open_s3_zarr_store(
        location=store_path,
        profile="kafou",
    )

    root = zarr.open_group(store, mode="a")
    if "source" not in root.attrs:
        root.attrs["source"] = src
    else:
        # validate subsequent workers are consistent
        assert root.attrs["source"] == src, (
            f"Store source={root.attrs['source']!r} != worker src={src!r}"
        )
        print("[WORKER] Metadata already present, skipping write.")

    ensure_time_in_arrays(
        store=store,
        timestamp=end_time,
        time_dim="init_time",
        time_frequency="6h",
    )

    for batch in batches:
        start = str(batch[0])
        end = str(batch[-1])

        cmd = [
            "sbatch",
            "--ntasks=1",
            "--cpus-per-task=32",
            "--gpus=1",
            f"--job-name={start}_{end}",
            "--wrap",
            f"python {sys.argv[0]} worker {start} {end} {store_path} {src}",
        ]

        print("Submitting:", " ".join(cmd))
        subprocess.run(cmd, check=True)


# ------------------------------------------------------
# CLI
# ------------------------------------------------------
@click.group()
def cli():
    pass


@cli.command()
@click.argument("location", type=str)
@click.argument("start")
@click.argument("end")
@click.argument("rollout_steps", type=int)
@click.option("--lat-range", nargs=2, type=float, default=None, help="Latitude range (min max)")
@click.option("--lon-range", nargs=2, type=float, default=None, help="Longitude range (min max)")
def init(
    location,
    start,
    end,
    rollout_steps=8,
    lat_range=None,
    lon_range=None,
):
    store = open_s3_zarr_store(
        location=location,
        profile="kafou",
    )

    init_times = np.arange(
        np.datetime64(start),
        np.datetime64(end) + np.timedelta64(6, "h"),
        np.timedelta64(6, "h"),
    )

    initialize_dataset(
        store=store,
        init_times=init_times,
        rollout_steps=rollout_steps,
        lat_range=lat_range,
        lon_range=lon_range,
    )


@cli.command()
@click.argument("start")
@click.argument("end")
@click.argument("store")
@click.argument("src")
def worker(start, end, store, src):
    run_worker(start_time=start, end_time=end, store_path=store, src=src)


@cli.command()
@click.argument("store")
@click.argument("start")
@click.argument("end")
@click.argument("src")
def submit(store, start, end, src):
    submit_jobs(start_time=start, end_time=end, store_path=store, src=src)


if __name__ == "__main__":
    cli()
