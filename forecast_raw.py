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
from forecast_utils import convert_aurora_to_greenwich
from LatentVectorExtractor import LatentVectorExtractor
from obstore_utils import open_s3_zarr_store

# Prints from workers in real-time instead of buffering until the end of the job.
os.environ["PYTHONUNBUFFERED"] = "1"

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
    Initialize forecast store with coords + empty `normalized_sample`.

    Creates:
        Dimensions:         (feature=72, init_time=N, lead_time=rollout, lat=720, lon=1440)
        Coordinates:
            * feature       (feature) int64
            * init_time     (init_time) datetime64[ns]
            * lead_time     (lead_time) int64
            * lat           (lat) float32
            * lon           (lon) float32
        Data variables:
            normalized_sample (init_time, lead_time, lat, lon, feature) float32


    Greenwich-centered coordinates matching Aurora's 720x1440 grid

    (North Pole) lat=90         +────────────+────────────+    ← index 719
                                |  US        |EU    ASIA  |
    (Equator)    lat=0          +────────────+────────────+
                                |  SA        |AF    AUS   |
    (South Pole) lat=-89.75     +────────────+────────────+    ← index 0
                             lon=-180      lon=0       lon=179.75
                            (Date Line)  (Greenwich)  (Date Line)

                   ←── Western Hemisphere ──→←── Eastern Hemisphere ──→
                        lon: -180 ... 0            lon: 0 ... 179.75

    """
    print("[INIT] Initializing destination store...")

    n_feature = 72

    lat = np.linspace(-89.75, 90, 720, dtype=np.float32)  # ascending (south → north)
    lon = np.linspace(-180.0, 179.75, 1440, dtype=np.float32)  # Greenwich-centered

    if lat_range and lon_range:
        print(f"[INIT] User-specified ranges: lat={lat_range}, lon={lon_range}")

        lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
        lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])

        lat = lat[lat_mask]
        lon = lon[lon_mask]

        print(f"[INIT] Subsetting to: {len(lat)} lat points, {len(lon)} lon points")
        print(
            f"[INIT] Actual ranges: lat=[{lat.min():.2f}, {lat.max():.2f}], lon=[{lon.min():.2f}, {lon.max():.2f}]"
        )

        if lat.size == 0 or lon.size == 0:
            raise ValueError("[INIT] No latitude or longitude values remain after filtering.")

    # Get lead / valid time
    lead_times = (np.arange(1, rollout_steps + 1) * 6).astype("int64")

    # Generate Coordinate xr.Dataset
    print("[INIT] Writing coordinates...")
    coord_ds = xr.Dataset(
        coords={
            "init_time": ("init_time", init_times),
            "lead_time": ("lead_time", lead_times),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "feature": ("feature", np.arange(n_feature, dtype="int64")),
        },
        attrs={
            "description": "Aurora raw predictions forecast dataset.",
            "rollout_steps": int(rollout_steps),
            "lead_times": lead_times.tolist(),
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
            "lat": {"chunks": (len(lat),)},
            "lon": {"chunks": (len(lon),)},
            "feature": {"chunks": (n_feature,)},
        },
    )

    zarr.create_array(
        store,
        name="normalized_sample",
        shape=(len(init_times), len(lead_times), len(lat), len(lon), n_feature),
        chunks=(1, len(lead_times), 128, 128, 128),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=("init_time", "lead_time", "lat", "lon", "feature"),
    )

    # Print dataset for verification
    print("[INIT] Creating normalized_sample DataArray...")
    ds = xr.open_zarr(store, consolidated=False)
    print(ds)

    print("[INIT] Initialization complete.")


# ------------------------------------------------------
# RAW Worker
# ------------------------------------------------------
def run_worker(start_time: str, end_time: str, store_path: str, src: str) -> None:
    store = open_s3_zarr_store(location=store_path, profile="kafou")

    # Get coordinate information from Store
    ds_store = xr.open_zarr(store, consolidated=False)

    print(ds_store)
    target_lat = ds_store.lat.values
    target_lon = ds_store.lon.values
    rollout_steps = len(ds_store.lead_time.data)
    init_times = ds_store.init_time.sel(init_time=slice(start_time, end_time)).values

    print(f"[WORKER] Store lat: [{target_lat[0]}, {target_lat[-1]}], len={len(target_lat)}")
    print(f"[WORKER] Store lon: [{target_lon[0]}, {target_lon[-1]}], len={len(target_lon)}")

    # Instantiate Extractors
    if src == "ecmwf":
        src_repo = "kafou/aurora-ecmwf-samples"
        src_branch = "main"
        lve = LatentVectorExtractor(
            source_repo=src_repo,
            source_branch=src_branch,
            client=arraylake.Client(),
        )

    elif src == "era5":
        src_repo = "kafou/aurora-era5-samples"
        src_branch = "extend-2025"
        lve = LatentVectorExtractor(
            source_repo=src_repo,
            source_branch=src_branch,
            client=arraylake.Client(),
        )
    else:
        raise ValueError(f"Unknown source: {src}. Must be 'ecmwf' or 'era5'")

    for init_time in init_times:
        print("Processing:", init_time, flush=True)

        lv_ds = lve.rollout(
            item=init_time.astype("datetime64[s]").item(),
            steps=rollout_steps,
        )

        # print(lv_ds)
        # print(lv_ds.lat.values.min(), lv_ds.lat.values.max(), len(lv_ds.lat))
        # print(lv_ds.lon.values.min(), lv_ds.lon.values.max(), len(lv_ds.lon))

        # Convert from Aurora's coordinates to Greenwich-centered
        lv_ds = convert_aurora_to_greenwich(lv_ds)

        if len(target_lat) < len(lv_ds.lat) or len(target_lon) < len(lv_ds.lon):
            lv_ds = lv_ds.sel(
                lat=target_lat,
                lon=target_lon,
                method="nearest",
                tolerance=1e-4,
            )

        lv_arr = lv_ds["normalized_sample"].values  # (lead_time, lat, lon, feature)

        lv_da = xr.DataArray(
            lv_arr[None, ...],
            dims=("init_time", "lead_time", "lat", "lon", "feature"),
            coords={
                "init_time": [init_time],
                "lead_time": ds_store["lead_time"],
                "lat": ds_store["lat"],
                "lon": ds_store["lon"],
                "feature": ds_store["feature"],
            },
            name="normalized_sample",
        )

        lv_da.to_zarr(
            store,
            zarr_format=3,
            consolidated=False,
            mode="a",
            region="auto",
        )


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
