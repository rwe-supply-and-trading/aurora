#!/usr/bin/env python
"""
conda activate aurora
srun --exclusive --ntasks=1 --mem=0 python -u export_zarr_to_netcdf.py s3://icechunk-write-coordination/dedk-ecmwf /shared/rwx/data/sailboat/silly.nc
"""

import logging

import click
import netCDF4
import xarray as xr
from obstore_utils import open_s3_zarr_store

logger = logging.getLogger(__name__)


@click.command()
@click.argument("store_path")
@click.argument("output_file")
@click.option("--profile", default="kafou", help="AWS / object store profile")
@click.option("--step", default=3000, show_default=True, help="Timesteps per chunk")
def export(store_path: str, output_file: str, profile: str, step: int) -> None:
    """Export a Zarr dataset to NetCDF, one slice at a time. No dask."""
    logger.info("Opening Zarr store...")
    store = open_s3_zarr_store(location=store_path, profile=profile, read_only=True)

    logger.info("Opening dataset (lazy, no dask)...")
    ds = xr.open_zarr(store, consolidated=False, chunks=None)
    logger.info(ds)

    size_gb = ds.nbytes / 1e9
    logger.info(f"Est. Dataset size: {size_gb:.1f} GB")

    if size_gb < 450:
        logger.info("Dataset is small enough to load eagerly, skipping chunked write...")
        ds.load().to_netcdf(output_file, engine="netcdf4", mode="w")

    else:
        n_times = ds.sizes["init_time"]
        logger.info(f"Total timesteps: {n_times}, writing {step} at a time")

        # Write the first chunk to initialize the file
        # You could use netCDF4 for everything, but you'd need to manually
        # create dimensions, variables, set attributes, handle encoding, etc.
        # The hybrid approach leverages xarray's convenience for setup while
        # using netCDF4's flexibility for appending.
        logger.info(f"  Slice 1: init_time=0:{min(step, n_times)}  ({min(step, n_times)} steps)...")
        first_chunk = ds.isel(init_time=slice(0, step)).load()

        encoding = {
            var: {
                "chunksizes": tuple(
                    1 if dim == "init_time" else size
                    for dim, size in zip(ds[var].dims, ds[var].shape)
                )
            }
            for var in ds.data_vars
        }
        first_chunk.to_netcdf(
            output_file,
            engine="netcdf4",
            mode="w",
            unlimited_dims=["init_time"],
            encoding=encoding,
        )
        del first_chunk

        # Append remaining chunks using netCDF4 directly
        for i, start in enumerate(range(step, n_times, step), start=2):
            stop = min(start + step, n_times)
            logger.info(f"  Slice {i}: init_time={start}:{stop}  ({stop - start} steps)...")

            chunk = ds.isel(init_time=slice(start, stop)).load()

            with netCDF4.Dataset(output_file, "a") as nc:
                idx = nc.dimensions["init_time"].size
                times = chunk["init_time"].values.astype("datetime64[ns]")
                nc.variables["init_time"][idx:] = netCDF4.date2num(
                    times.astype("M8[ms]").astype(object),
                    units=nc.variables["init_time"].units,
                    calendar=nc.variables["init_time"].calendar,
                )
                for var in ds.data_vars:
                    nc.variables[var][idx:] = chunk[var].values

            del chunk

    logger.info("Finished:", output_file)
    return


if __name__ == "__main__":
    export()
