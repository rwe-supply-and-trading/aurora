"""Copyright (c) RWE Supply & Trading GmbH. Licensed under the MIT license."""

import logging

import kafou_arraylake as arraylake
import numpy as np
import xarray as xr
import zarr
from forecast_utils import convert_aurora_to_greenwich
from LatentVectorExtractor import LatentVectorExtractor
from obstore_utils import open_s3_zarr_store

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# Initialize Zarr dataset
# ------------------------------------------------------
def initialize_dataset(
    *,
    store,
    init_times: np.ndarray,
    rollout_steps: int,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
) -> None:
    """
    Initialize forecast store with coords + empty `normalized_sample`.

    Args:
        - store: Zarr store to initialize (e.g. S3 path)
        - init_times: Array of initialization times (datetime64)
        - rollout_steps: Number of forecast steps (e.g. 16 for 6-hourly steps up to 4 days)
        - lat_range: Optional tuple specifying (min_lat, max_lat) for spatial subsetting
        - lon_range: Optional tuple specifying (min_lon, max_lon) for spatial subsetting

    Returns:
        - None

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
    logger.info("[INIT] Initializing destination store...")

    n_feature = 72

    lat = np.linspace(-89.75, 90, 720, dtype=np.float32)  # ascending (south → north)
    lon = np.linspace(-180.0, 179.75, 1440, dtype=np.float32)  # Greenwich-centered

    if lat_range and lon_range:
        logger.info(f"[INIT] User-specified ranges: lat={lat_range}, lon={lon_range}")

        lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
        lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])

        lat = lat[lat_mask]
        lon = lon[lon_mask]

        logger.info(f"[INIT] Subsetting to: {len(lat)} lat points, {len(lon)} lon points")
        logger.info(
            f"[INIT] Actual ranges: lat=[{lat.min():.2f}, {lat.max():.2f}], lon=[{lon.min():.2f}, {lon.max():.2f}]"
        )

        if lat.size == 0 or lon.size == 0:
            raise ValueError("[INIT] No latitude or longitude values remain after filtering.")

    # Get lead / valid time
    lead_times = (np.arange(1, rollout_steps + 1) * 6).astype("int64")

    # Generate Coordinate xr.Dataset
    logger.info("[INIT] Writing coordinates...")
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
            "lat": {"chunks": (len(lat),)},
            "lon": {"chunks": (len(lon),)},
            "feature": {"chunks": (n_feature,)},
        },
    )

    # Open existing group (preserves attrs) and add array
    # zarr.create_array(fill_value=np.nan) does not allocate or write any chunk data at all.
    # It only writes a small metadata file (zarr.json) with  the array's shape, chunks, dtype,
    # fill_value, and dimension names. This is an O(1) operation regardless of array size.
    # When a Zarr reader later encounters an unwritten chunk, it returns the fill_value (NaN
    # automatically — this is a core Zarr design principle called "virtual fill."
    # No physical storage is consumed for chunks that have never been written to.
    root = zarr.open_group(store, mode="a", zarr_format=3)
    root.create_array(
        name="normalized_sample",
        shape=(len(init_times), len(lead_times), len(lat), len(lon), n_feature),
        chunks=(1, len(lead_times), 128, 128, 128),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=("init_time", "lead_time", "lat", "lon", "feature"),
    )

    # Print dataset for verification
    logger.info("[INIT] Creating normalized_sample DataArray...")
    ds = xr.open_zarr(store, consolidated=False)
    logger.info(ds)

    logger.info("[INIT] Initialization complete.")


# ------------------------------------------------------
# RAW Worker
# ------------------------------------------------------
def run_worker(*, start_time: str, end_time: str, store_path: str, src: str) -> None:
    """Worker function to generate forecasts for a given time range and store in Zarr.

    Args:
        - start_time (str): Start of the time range (inclusive) in ISO format, e.g. "2024-01-01T00:00:00Z"
        - end_time (str): End of the time range (inclusive) in ISO format, e.g. "2024-01-31T18:00:00Z"
        - store_path (str): S3 path to the Zarr store, e.g. "s3://my-bucket/forecast.zarr"
        - src (str): Source dataset to use for forecasts, either "ecmwf" or "era5"
    Returns:
        - None
    """
    store = open_s3_zarr_store(location=store_path, profile="kafou")

    # Get coordinate information from Store
    ds_store = xr.open_zarr(store, consolidated=False)

    logger.info(ds_store)
    target_lat = ds_store.lat.values
    target_lon = ds_store.lon.values
    rollout_steps = len(ds_store.lead_time.data)
    init_times = ds_store.init_time.sel(init_time=slice(start_time, end_time)).values

    logger.info(f"[WORKER] Store lat: [{target_lat[0]}, {target_lat[-1]}], len={len(target_lat)}")
    logger.info(f"[WORKER] Store lon: [{target_lon[0]}, {target_lon[-1]}], len={len(target_lon)}")

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
        logger.info(f"Processing: {init_time}")

        lv_ds = lve.rollout(
            item=init_time.astype("datetime64[s]").item(),
            steps=rollout_steps,
        )

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
            mode="r+",
            region="auto",
        )

    logger.info("[WORKER] Worker complete.")
