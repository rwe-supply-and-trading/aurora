#!/usr/bin/env python
"""
conda activate aurora
srun --exclusive --ntasks=1 --mem=0 python -u export_zarr_to_netcdf.py s3://icechunk-write-coordination/dedk-ecmwf /shared/rwx/data/sailboat/silly.nc
"""

import click
import netCDF4
import numpy as np
import xarray as xr
from obstore_utils import open_s3_zarr_store


@click.command()
@click.argument("store_path")
@click.argument("output_file")
@click.option("--profile", default="kafou", help="AWS / object store profile")
@click.option("--step", default=3000, show_default=True, help="Timesteps per chunk")
def export(store_path: str, output_file: str, profile: str, step: int) -> None:
    """Export a Zarr dataset to NetCDF, one slice at a time. No dask."""
    print("Opening Zarr store...")
    store = open_s3_zarr_store(location=store_path, profile=profile, read_only=True)

    print("Opening dataset (lazy, no dask)...")
    ds = xr.open_zarr(store, consolidated=False, chunks=None)
    print(ds)

    size_gb = ds.nbytes / 1e9
    print(f"Est. Dataset size: {size_gb:.1f} GB")

    if size_gb < 450:
        print("Dataset is small enough to load eagerly, skipping chunked write...")
        ds.load().to_netcdf(output_file, engine="netcdf4", mode="w")
        print("Finished:", output_file)
        return

    n_times = ds.sizes["init_time"]
    print(f"Total timesteps: {n_times}, writing {step} at a time")

    for i, start in enumerate(range(0, n_times, step)):
        stop = min(start + step, n_times)
        print(f"  Slice {i + 1}: init_time={start}:{stop}  ({stop - start} steps)...")

        chunk = ds.isel(init_time=slice(start, stop)).load()

        if start == 0:
            encoding = {
                var: {
                    "chunksizes": tuple(
                        1 if dim == "init_time" else size
                        for dim, size in zip(ds[var].dims, ds[var].shape)
                    )
                }
                for var in ds.data_vars
            }
            chunk.to_netcdf(
                output_file,
                engine="netcdf4",
                mode="w",
                unlimited_dims=["init_time"],
                encoding=encoding,
            )
        else:
            # Use netCDF4 directly — xarray validates chunk size on append
            # and rejects the final slice when it's smaller than step.
            with netCDF4.Dataset(output_file, "a") as nc:
                idx = nc.dimensions["init_time"].size
                nc.variables["init_time"][idx:] = (
                    chunk["init_time"].values.astype("datetime64[ns]").astype(np.int64)
                )
                for var in ds.data_vars:
                    nc.variables[var][idx:] = chunk[var].values

    print("Finished:", output_file)


if __name__ == "__main__":
    export()
