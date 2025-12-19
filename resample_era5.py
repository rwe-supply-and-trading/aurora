import datetime
import time

import click
import kafou_arraylake as arraylake
import numpy as np
import xarray as xr
from dataset_io import ensure_time_in_arrays

# ---------------------------------------------------------------------
# VARIABLE MAP
# ---------------------------------------------------------------------

NAME_MAP = {
    "sfc": {
        "VAR_2T": "2t",
        "MSL": "msl",
        "VAR_10U": "10u",
        "VAR_10V": "10v",
    },
    "pl": {
        "Z": "z",
        "U": "u",
        "V": "v",
        "T": "t",
        "Q": "q",
    },
    "inv": {
        "LSM": "lsm",
        "Z": "z",
        "SLT": "slt",
    },
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def day_bounds(ts: datetime.datetime):
    """Return numpy datetime64 bounds for exactly D00Z → D18Z."""
    d0 = np.datetime64(ts.date())                # YYYY-MM-DDT00:00:00
    d1 = d0 + np.timedelta64(18, "h")            # D18Z
    return d0, d1


def get_day_batches(start, end, days_at_once):
    cursor = start
    one = datetime.timedelta(days=1)

    while cursor <= end:
        batch_end = min(cursor + one * (days_at_once - 1), end)
        batch = []

        d = cursor
        while d <= batch_end:
            batch.append(d)
            d += one

        yield batch
        cursor = batch_end + one


def get_variable_locations(n_pressure_levels: int):
    in_locs = {"sfc": {}, "pl": {}}
    out_locs = {"sfc": {}, "pl": {}}

    idx = 0

    for var in NAME_MAP["sfc"]:
        in_locs["sfc"][var] = (idx, 1)
        out_locs["sfc"][NAME_MAP["sfc"][var]] = (idx, 1)
        idx += 1

    for var in NAME_MAP["pl"]:
        in_locs["pl"][var] = (idx, n_pressure_levels)
        out_locs["pl"][NAME_MAP["pl"][var]] = (idx, n_pressure_levels)
        idx += n_pressure_levels

    return in_locs, out_locs


# ---------------------------------------------------------------------
# IDEMPOTENCY CHECK
# ---------------------------------------------------------------------

def day_already_written(store, day):
    """
    Return True if all 4 timestamps for the given day already exist in Zarr.
    """
    ds = xr.open_zarr(store, group="samples", zarr_format=3, consolidated=False)

    d0, d1 = day_bounds(day)
    existing = ds.time.sel(time=slice(d0, d1))

    # ERA5 should have D00Z, D06Z, D12Z, D18Z → 4 records
    return existing.size == 4


# ---------------------------------------------------------------------
# MAIN PER-DAY INGEST FUNCTION
# ---------------------------------------------------------------------

def process_day_from_datasets(
    *,
    store,
    timestamp: datetime.datetime,
    sfc_ds: xr.Dataset,
    pl_ds: xr.Dataset,
):
    # IDEMPOTENCY — skip day if already written
    if day_already_written(store, timestamp):
        print(f"   ✔ Day {timestamp:%Y-%m-%d} already written — skipping.")
        return 0

    print(f"\n=== PROCESSING {timestamp:%Y-%m-%d} ===")

    # 1. Slice ERA5 correctly: D00Z → D18Z
    d0, d1 = day_bounds(timestamp)
    sfc_vars = list(NAME_MAP["sfc"])
    pl_vars = list(NAME_MAP["pl"])

    sfc_day = sfc_ds[sfc_vars].sel(time=slice(d0, d1)).compute()
    pl_day = pl_ds[pl_vars].sel(time=slice(d0, d1)).compute()

    if sfc_day.time.size != 4:
        raise RuntimeError(
            f"Expected 4 timesteps for {timestamp:%Y-%m-%d}, got {sfc_day.time.size}"
        )

    # 2. Determine channels
    n_levels = pl_day.level.size
    in_locs, _ = get_variable_locations(n_levels)

    T = 4
    Y = sfc_day.latitude.size
    X = sfc_day.longitude.size
    n_channels = len(NAME_MAP["sfc"]) + len(NAME_MAP["pl"]) * n_levels

    block = np.zeros((T, n_channels, Y, X), dtype=np.float32)

    # 3. Fill block
    for ti in range(T):
        for varname, (cidx, _) in in_locs["sfc"].items():
            block[ti, cidx] = sfc_day[varname].isel(time=ti).values

        for varname, (cidx, size) in in_locs["pl"].items():
            block[ti, cidx:cidx+size] = pl_day[varname].isel(time=ti).values

    new_times = sfc_day.time.values

    # 4. Ensure timestamps exist in Zarr arrays
    ensure_time_in_arrays(
        store,
        timestamp=new_times[-1],
        time_frequency="6h",
        group="samples",
    )

    # 5. Build dataset for region write
    new_ds = xr.Dataset(
        {"sample_data": (("time", "channel", "latitude", "longitude"), block)},
        coords={
            "time": new_times,
            "channel": np.arange(n_channels),
            "latitude": sfc_day.latitude.values,
            "longitude": sfc_day.longitude.values,
            "atmos_levels": pl_day.level.values,
        },
    )

    new_ds.to_zarr(
        store,
        group="samples",
        region="auto",
        mode="a",
        consolidated=False,
    )

    print(f"   ✔ Finished writing day {timestamp:%Y-%m-%d}")
    return T


# ---------------------------------------------------------------------
# BATCHED INGEST + IDEMPOTENT COMMIT
# ---------------------------------------------------------------------

def parallel_reorg(
    *,
    src_repo_name,
    dst_repo_name,
    start_time,
    end_time,
    src_branch="main",
    dst_branch="main",
    token=None,
    days_at_once=14,
):
    client = arraylake.Client(token=token)
    if token is None:
        client.login()

    src_repo = client.get_repo(src_repo_name)
    src_session = src_repo.readonly_session(src_branch)

    sfc_ds = xr.open_zarr(src_session.store, "surface", zarr_format=3, consolidated=False)
    pl_ds  = xr.open_zarr(src_session.store, "pressure_level", zarr_format=3, consolidated=False)

    dst_repo = client.get_repo(dst_repo_name)

    batches = get_day_batches(start_time, end_time, days_at_once)

    for batch in batches:
        print(f"\n▶️  BATCH {batch[0]:%Y-%m-%d} → {batch[-1]:%Y-%m-%d}")
        t0 = time.monotonic()

        dst_session = dst_repo.writable_session(dst_branch)

        try:
            # ingest each day
            for day in batch:
                process_day_from_datasets(
                    store=dst_session.store,
                    timestamp=day,
                    sfc_ds=sfc_ds,
                    pl_ds=pl_ds,
                )

            # commit with retries
            commit_msg = f"Added batch {batch[0]:%Y-%m-%d} → {batch[-1]:%Y-%m-%d}"

            for attempt in range(1, 7):
                try:
                    cid = dst_session.commit(commit_msg)
                    print(f"   ✅ COMMIT SUCCESS — {cid}")
                    break
                except Exception as e:
                    print(f"   ❌ Commit attempt #{attempt} failed: {e}")
                    if attempt == 6:
                        raise RuntimeError("Max commit retries reached.")
                    time.sleep(2 ** attempt)

            print(f"   ⏱ Batch completed in {time.monotonic() - t0:.2f}s")

        except Exception as e:
            print("\n   ❌ Batch failure:", e)
            print(f"   ❌ Stopping at day {batch[0]:%Y-%m-%d}")
            raise SystemExit(1)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

@click.command()
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--src-repo", default="rwe/era5-0p25-6h-nonprod-ohio")
@click.option("--dest-repo", default="kafou/aurora-era5-samples")
@click.option("--src-branch", default="main")
@click.option("--dest-branch", default="main")
@click.option("--token", default=None)
@click.option("--days-at-once", type=int, default=14)
def main(start_time, end_time, src_repo, dest_repo, src_branch, dest_branch, token, days_at_once):
    parallel_reorg(
        src_repo_name=src_repo,
        dst_repo_name=dest_repo,
        start_time=start_time,
        end_time=end_time,
        src_branch=src_branch,
        dst_branch=dest_branch,
        token=token,
        days_at_once=days_at_once,
    )


if __name__ == "__main__":
    main()
