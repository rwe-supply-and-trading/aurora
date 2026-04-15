#!/usr/bin/env python

import logging
import os
import subprocess
import sys

import click
import numpy as np
import xarray as xr
import zarr
from dataset_io import ensure_time_in_arrays
from obstore_utils import open_s3_zarr_store

xr.set_options(keep_attrs=True)


logger = logging.getLogger(__name__)
xr.set_options(keep_attrs=True)

os.environ["PYTHONUNBUFFERED"] = "1"

for var in [
    "CURL_CA_BUNDLE",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
]:
    os.environ.pop(var, None)


def _get_module(mode: str) -> object:
    """Helper to import the correct module based on mode.
    Args:
        - mode: Forecast mode ("latent" or "raw")

    Returns:
        - The imported module (forecast_latent or forecast_raw)
    """
    if mode == "latent":
        import forecast_latent

        return forecast_latent
    elif mode == "raw":
        import forecast_raw

        return forecast_raw
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Must be 'latent' or 'raw'")


# ------------------------------------------------------
# SLURM job submission
# ------------------------------------------------------
def submit_jobs(*, start_time: str, end_time: str, store_path: str, src: str, mode: str) -> None:
    """Submit SLURM jobs for each batch of initialization times.

    Args:
        - start_time: Start of initialization time range (e.g. "2024-01-01T00:00")
        - end_time: End of initialization time range (e.g. "2024-01-31T18:00")
        - store_path: S3 path to Zarr store (e.g. "s3://my-bucket/forecast.zarr")
        - src: Source dataset ("ecmwf" or "era5")
        - mode: Forecast mode ("latent" or "raw")

    Returns:
        - None
    """

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
        assert root.attrs["source"] == src, (
            f"Store source={root.attrs['source']!r} != worker src={src!r}"
        )

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
            f"--job-name={mode}_{start}_{end}",
            "--wrap",
            f"python {sys.argv[0]} worker {mode} {start} {end} {store_path} {src}",
        ]

        logger.info(f"Submitting: {cmd}")
        subprocess.run(cmd, check=True)


# ------------------------------------------------------
# CLI
# ------------------------------------------------------
@click.group()
def cli():
    pass


@cli.command()
@click.argument("mode", type=click.Choice(["latent", "raw"]))
@click.argument("location", type=str)
@click.argument("start")
@click.argument("end")
@click.argument("rollout_steps", type=int)
@click.option("--lat-range", nargs=2, type=float, default=None, help="Latitude range (min max)")
@click.option("--lon-range", nargs=2, type=float, default=None, help="Longitude range (min max)")
def init(
    *,
    mode: str,
    location: str,
    start: str,
    end: str,
    rollout_steps: int,
    lat_range: tuple[float, float] | None,
    lon_range: tuple[float, float] | None,
) -> None:
    """Initialize the Zarr store with coordinates and metadata.

    Args:
        - mode: Forecast mode ("latent" or "raw")
        - location: S3 path to Zarr store (e.g. "s3://my-bucket/forecast.zarr")
        - start: Start of initialization time range (e.g. "2024-01-01T00:00")
        - end: End of initialization time range (e.g. "2024-01-31T18:00")
        - rollout_steps: Number of forecast steps to initialize (e.g. 72 for 3 days at 1 step per hour)
        - lat_range: Optional latitude range for spatial subset (e.g. --lat-range -30 30)
        - lon_range: Optional longitude range for spatial subset (e.g. --lon-range -60 60)

    Returns:
        - None
    """
    mod = _get_module(mode)

    store = open_s3_zarr_store(
        location=location,
        profile="kafou",
    )

    init_times = np.arange(
        np.datetime64(start),
        np.datetime64(end) + np.timedelta64(6, "h"),
        np.timedelta64(6, "h"),
    )

    mod.initialize_dataset(
        store=store,
        init_times=init_times,
        rollout_steps=rollout_steps,
        lat_range=lat_range,
        lon_range=lon_range,
    )


@cli.command()
@click.argument("mode", type=click.Choice(["latent", "raw"]))
@click.argument("start")
@click.argument("end")
@click.argument("store")
@click.argument("src")
def worker(*, mode: str, start: str, end: str, store: str, src: str) -> None:
    """Checks for an runs the correct run_worker function based on the mode argument."""
    mod = _get_module(mode)
    mod.run_worker(start_time=start, end_time=end, store_path=store, src=src)


@cli.command()
@click.argument("mode", type=click.Choice(["latent", "raw"]))
@click.argument("start")
@click.argument("end")
@click.argument("store")
@click.argument("src")
def submit(*, mode: str, start: str, end: str, store: str, src: str) -> None:
    """Submit SLURM jobs for the specified time range and mode."""
    submit_jobs(start_time=start, end_time=end, store_path=store, src=src, mode=mode)


if __name__ == "__main__":
    cli()
