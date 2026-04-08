import argparse

import numpy as np
import xarray as xr


def merge_netcdfs(path_a: str, path_b: str, path_out: str, var: str) -> None:
    ds_a = xr.open_dataset(path_a, chunks=None)
    ds_b = xr.open_dataset(path_b, chunks=None)

    print("A:", ds_a)
    print("B:", ds_b)

    a_times = ds_a.init_time.values
    b_times = ds_b.init_time.values

    valid_mask = np.isin(b_times, a_times)
    if not valid_mask.all():
        print(f"[WARN] Dropping {(~valid_mask).sum()} init_times from B not in A")
    else:
        print("All init_times in B are present in A")

    ds_b = ds_b.isel(init_time=valid_mask)
    b_times = ds_b.init_time.values

    a_indices = np.searchsorted(a_times, b_times)

    a_arr = ds_a[var].values
    b_arr = ds_b[var].values
    print("Loaded both arrays:", a_arr.shape, b_arr.shape)

    a_arr[a_indices, ...] = b_arr

    ds_out = ds_a.copy(data={var: a_arr})
    ds_out.to_netcdf(path_out)
    print(f"Done combining {path_a} and {path_b} into {path_out}.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge two NetCDF files by overwriting matching init_times from B into A."
    )
    parser.add_argument("a", help="Path to the base NetCDF file (A)")
    parser.add_argument("b", help="Path to the overlay NetCDF file (B)")
    parser.add_argument("out", help="Path to write the merged output")
    parser.add_argument(
        "--var",
        default="normalized_sample",
        help="Variable name to merge (default: normalized_sample)",
    )
    args = parser.parse_args()
    merge_netcdfs(args.a, args.b, args.out, args.var)


if __name__ == "__main__":
    main()
