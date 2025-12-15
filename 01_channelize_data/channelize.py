"""
tmux
codna activate aurora
srun --ntasks=1 --cpus-per-task=16 --gpus=0 --mem=300G --pty /bin/bash

python channelize.py \
    2009-01-01 \
    2025-12-05 \
    --init \
    --src-repo rwe/era5-0p25-6h-nonprod-ohio \
    --dst-repo kafou/aurora-era5-samples \
    --src-branch main \
    --dst-branch testing \
    --token $ARRAYLAKE_TOKEN

"""

import datetime
import time
import click
import kafou_arraylake as arraylake
import numpy as np
import xarray as xr
import zarr
from dataset_io import ensure_time_in_arrays

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


# ------------------#
# HELPER FUNCTIONS #
# ------------------#
def day_bounds(ts: datetime.datetime) -> tuple[np.datetime64, np.datetime64]:
    """
    Return numpy datetime64 bounds for exactly:
        start = YYYY-MM-DDT00:00
        end   = YYYY-MM-DDT18:00
    """
    d0 = np.datetime64(ts.replace(hour=0, minute=0, second=0, microsecond=0))
    d1 = d0 + np.timedelta64(18, "h")
    return d0, d1


def get_day_batches(start: datetime.datetime, end: datetime.datetime, days_at_once: int):
    """
    Yield batches of consecutive days, each up to `days_at_once` in length.
    Always inclusive of endpoints.
    """
    one_day = datetime.timedelta(days=1)
    cursor = start

    while cursor <= end:
        batch_end = min(cursor + one_day * (days_at_once - 1), end)
        batch = [cursor + i * one_day for i in range((batch_end - cursor).days + 1)]
        yield batch
        cursor = batch_end + one_day


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


def process_day_from_datasets(
    *,
    store,
    timestamp: datetime.datetime,
    sfc_ds: xr.Dataset,
    pl_ds: xr.Dataset,
):
    print(f"\n=== PROCESSING {timestamp:%Y-%m-%d} ===")

    # 1) Slice day: 00Z → 18Z
    d0, d1 = day_bounds(timestamp)

    sfc_vars = list(NAME_MAP["sfc"])
    pl_vars = list(NAME_MAP["pl"])

    sfc_day = sfc_ds[sfc_vars].sel(time=slice(d0, d1)).load()
    pl_day = pl_ds[pl_vars].sel(time=slice(d0, d1)).load()

    if sfc_day.time.size != 4:
        raise RuntimeError(
            f"{timestamp:%Y-%m-%d}: expected 4 timesteps (00/06/12/18Z), got {sfc_day.time.size}"
        )

    # sanity: pl must match the same 4 times
    if pl_day.time.size != 4 or not np.array_equal(pl_day.time.values, sfc_day.time.values):
        raise RuntimeError(f"{timestamp:%Y-%m-%d}: pl times don’t match sfc times")

    new_times = sfc_day.time.values

    ensure_time_in_arrays(
        store,
        timestamp=new_times[-1],  # last timestamp we will write
        time_frequency="6h",
        group="samples",
    )

    # 2) Channel layout
    n_levels = int(pl_day.level.size)
    in_locs, _ = get_variable_locations(n_levels)

    T = 4
    Y = int(sfc_day.latitude.size)
    X = int(sfc_day.longitude.size)
    n_channels = len(NAME_MAP["sfc"]) + (len(NAME_MAP["pl"]) * n_levels)

    block = np.zeros((T, n_channels, Y, X), dtype=np.float32)

    for varname, (cidx, _) in in_locs["sfc"].items():
        block[:, cidx, :, :] = sfc_day[varname].values

    for varname, (cidx, size) in in_locs["pl"].items():
        v = pl_day[varname].values  # (time, level, lat, lon)
        block[:, cidx : cidx + size, :, :] = v

    # 3) Build dataset to write (ONLY sample_data)
    write_ds = xr.Dataset(
        {
            "sample_data": (
                ("time", "channel", "latitude", "longitude"),
                block,
            )
        },
        coords={
            "time": new_times,
            "channel": np.arange(n_channels, dtype=np.int32),
            "latitude": sfc_day.latitude.values,
            "longitude": sfc_day.longitude.values,
        },
    )

    # 4) Region write by coordinate labels
    write_ds.to_zarr(
        store,
        group="samples",
        region="auto",
        mode="a",
        consolidated=False,
    )

    print(f"   ✔ Finished writing day {timestamp:%Y-%m-%d}")
    return write_ds


# BATCHED INGEST


def open_src_datasets(
    *,
    src_repo: str,
    src_branch: str = "main",
    token: str | None = None,
):
    """
    Grabs the base datasets from the source repo.
    """
    client = arraylake.Client(token=token)
    if token is None:
        client.login()

    # detect what dataset we're working with
    print(src_repo)
    if "era5" in src_repo:
        dataset_type = "era5"
    elif "ecmwf" in src_repo:
        dataset_type = "ecmwf"
    else:
        raise ValueError("Source repo must be ERA5 or ECMWF ecmwf dataset.")

    src_repo = client.get_repo(src_repo)
    src_session = src_repo.readonly_session(src_branch)

    if dataset_type == "ecmwf":
        # get static ds
        sfc_ds = xr.open_zarr(
            src_session.store, group="ENFO-T0", zarr_format=3, consolidated=False, chunks=None
        )

        # drop ensemble dimension
        sfc_ds = sfc_ds.isel(ensemble_member=0, drop=True)

        # convert lat / lon to latitude / longitude
        if "lat" in sfc_ds.coords and "lon" in sfc_ds.coords:
            sfc_ds = sfc_ds.rename(
                {"lat": "latitude", "lon": "longitude", "pressure_level": "level"}
            )
        # match ERA5 variable names
        if "t2m" in sfc_ds.data_vars:
            sfc_ds = sfc_ds.rename(
                {"t2m": "VAR_2T", "slp": "MSL", "u10m": "VAR_10U", "v10m": "VAR_10V"}
            )

        # get pressure_level ds
        pl_ds = xr.open_zarr(
            src_session.store, group="ENFO-T0", zarr_format=3, consolidated=False, chunks=None
        )
        # drop ensemble dimension
        pl_ds = pl_ds.isel(ensemble_member=0, drop=True)

        # convert lat / lon to latitude / longitude
        if "lat" in pl_ds.coords and "lon" in pl_ds.coords:
            pl_ds = pl_ds.rename({"lat": "latitude", "lon": "longitude", "pressure_level": "level"})

        # match ERA5 variable names
        if "z" in pl_ds.data_vars:
            pl_ds = pl_ds.rename({"z": "Z", "u": "U", "v": "V", "t": "T", "q": "Q"})

        # sort by level
        pl_ds = pl_ds.sortby("level")

    elif dataset_type == "era5":
        # get static ds
        sfc_ds = xr.open_zarr(
            src_session.store, group="surface", zarr_format=3, consolidated=False, chunks=None
        )

        # get pressure_level ds
        pl_ds = xr.open_zarr(
            src_session.store,
            group="pressure_level",
            zarr_format=3,
            consolidated=False,
            chunks=None,
        )

        # convert lat / lon to latitude / longitude
        pl_ds = pl_ds.sortby("level")
    else:
        raise ValueError("dataset_type must be 'era5' or 'ecmwf'")

    # get invariant ds from ERA5 dataset
    inv_repo = "rwe/era5-0p25-6h-nonprod-ohio"
    inv_repo = client.get_repo(inv_repo)
    inv_session = inv_repo.readonly_session("main")
    inv_ds = xr.open_zarr(
        inv_session.store, group="invariant", zarr_format=3, consolidated=False, chunks=None
    )

    return sfc_ds, pl_ds, inv_ds


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

    sfc_ds, pl_ds, inv_ds = open_src_datasets(
        src_repo=src_repo_name,
        src_branch=src_branch,
        token=token,
    )

    dst_repo = client.get_repo(dst_repo_name)

    last_committed_day = None  # TRACK THIS

    batches = get_day_batches(start_time, end_time, days_at_once)
    for batch in batches:
        print(f"\n▶️  BATCH {batch[0]:%Y-%m-%d} → {batch[-1]:%Y-%m-%d}")
        t0 = time.monotonic()

        dst_session = dst_repo.writable_session(dst_branch)

        try:
            for day in batch:
                process_day_from_datasets(
                    store=dst_session.store,
                    timestamp=day,
                    sfc_ds=sfc_ds,
                    pl_ds=pl_ds,
                )

            commit_msg = f"Added batch {batch[0]:%Y-%m-%d} → {batch[-1]:%Y-%m-%d}"

            for attempt in range(1, 7):
                try:
                    cid = dst_session.commit(commit_msg)
                    print(f"   COMMIT SUCCESS — {cid}")
                    last_committed_day = batch[-1].date()  # UPDATE HERE
                    break
                except Exception as e:
                    print(f"   Commit attempt #{attempt} failed: {e}")
                    if attempt == 6:
                        raise RuntimeError("Max commit retries reached.")
                    time.sleep(2**attempt)

            print(f"   ⏱ Batch completed in {time.monotonic() - t0:.2f}s")

        except Exception as e:
            print("\n   Batch failure:", e)
            if last_committed_day is not None:
                print(f"   Last successfully written day: {last_committed_day:%Y-%m-%d}")
            else:
                print("   No days were successfully written.")

            raise SystemExit(1)


# -------------------------
# Initialization
# -------------------------


def init_store(
    store: zarr.abc.store.Store,
    *,
    sfc_ds: xr.Dataset,
    pl_ds: xr.Dataset,
    inv_ds: xr.Dataset,
):
    """
    Initialize the destination Zarr layout.

    Creates:
      - invariant/        : static invariant fields
      - samples/          : coordinates + empty sample_data array
        * coords          : time, channel, latitude, longitude, atmos_levels
        * attrs           : var_locs (channel layout)
        * sample_data     : (time, channel, latitude, longitude)

    Assumes:
      - time is 6-hourly
      - channel layout is fixed for the lifetime of the store
    """

    print("Initializing destination store...")

    # ------------------------------------------------------------------
    # Invariant fields
    # ------------------------------------------------------------------
    print("  • Writing 'invariant' group...")
    (
        inv_ds[["Z", "LSM", "SLT"]]
        .rename(NAME_MAP["inv"])
        .to_zarr(
            store,
            group="invariant",
            zarr_format=3,
            consolidated=False,
            mode="w",
        )
    )

    # ------------------------------------------------------------------
    # Schema derivation
    # ------------------------------------------------------------------
    time_coord = sfc_ds.time
    lat_coord = sfc_ds.latitude
    lon_coord = sfc_ds.longitude

    n_levels = int(pl_ds.level.size)
    levels = pl_ds.level.values.astype("int32")

    _, out_var_locs = get_variable_locations(n_levels)
    n_channels = len(NAME_MAP["sfc"]) + len(NAME_MAP["pl"]) * n_levels

    # ------------------------------------------------------------------
    # Samples coordinates
    # ------------------------------------------------------------------
    print("  • Writing 'samples' coordinates...")
    coords_ds = xr.Dataset(
        data_vars={
            "atmos_levels": ("atmos_levels", levels),
        },
        coords={
            "time": time_coord,
            "channel": np.arange(n_channels, dtype="int32"),
            "latitude": lat_coord,
            "longitude": lon_coord,
        },
        attrs={
            "var_locs": out_var_locs,
        },
    )

    coords_ds.to_zarr(
        store,
        group="samples",
        zarr_format=3,
        consolidated=False,
        mode="w",
        encoding={
            "time": {"chunks": (len(time_coord),)},
            "latitude": {"chunks": (len(lat_coord),)},
            "longitude": {"chunks": (len(lon_coord),)},
            "atmos_levels": {"chunks": (len(levels),)},
        },
    )

    # ------------------------------------------------------------------
    # Empty sample_data array
    # ------------------------------------------------------------------
    group = zarr.open_group(store, path="samples", mode="a")

    time_len = len(time_coord)
    lat_len = len(lat_coord)
    lon_len = len(lon_coord)

    print("  • Creating sample_data array:")
    print(f"      shape  = ({time_len}, {n_channels}, {lat_len}, {lon_len})")
    print(f"      chunks = (1, {n_channels}, 103, 72)")

    compressors = [
        zarr.codecs.BloscCodec(
            clevel=3,
            shuffle=zarr.codecs.BloscShuffle.bitshuffle,
        )
    ]

    group.create_array(
        name="sample_data",
        shape=(time_len, n_channels, lat_len, lon_len),
        chunks=(1, n_channels, 103, 72),
        dtype="float32",
        dimension_names=("time", "channel", "latitude", "longitude"),
        fill_value=np.nan,
        compressors=compressors,
        overwrite=True,
    )

    print("Initialization complete.")


@click.command()
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--src-repo", default="rwe/era5-0p25-6h-nonprod-ohio")
@click.option("--dst-repo", default="kafou/aurora-era5-samples")
@click.option("--src-branch", default="main")
@click.option("--dst-branch", default="main")
@click.option("--token", default=None)
@click.option("--days-at-once", type=int, default=14)
@click.option("--init/--no-init", default=False)
def main(
    start_time,
    end_time,
    src_repo,
    dst_repo,
    src_branch,
    dst_branch,
    token,
    days_at_once,
    init,
):
    """
    Reorganize RAW IC data.

    Use:
      --init     once, to create and initialize the dst repo
      --no-init  for subsequent appends between start_time and end_time
    """

    # To initialize the dst repo
    if init:
        client = arraylake.Client(token=token)

        # Initialization path: create dst repo and layout
        if token is None and (src_repo.startswith("rwe/") or dst_repo.startswith("rwe/")):
            print("Logging in to Arraylake for init...")
            client.login()

        sfc_ds, pl_ds, inv_ds = open_src_datasets(
            src_repo=src_repo,
            src_branch=src_branch,
            token=token,
        )

        print(f"Creating destination repo: {dst_repo}")

        # Create or open existing repo
        try:
            dst_repo_obj = client.create_repo(dst_repo)
            print(f"✓ Created new repo: {dst_repo}")
        except Exception:
            print(f"Repo already exists; opening existing repo: {dst_repo}")
            dst_repo_obj = client.get_repo(dst_repo)

        # ---------------------------------------------------------
        # Ensure the destination branch exists
        # ---------------------------------------------------------
        branches = dst_repo_obj.list_branches()

        if dst_branch not in branches:
            print(f"Branch '{dst_branch}' does not exist — creating from 'main'...")
            main_id = dst_repo_obj.lookup_branch("main")
            dst_repo_obj.create_branch(dst_branch, main_id)
        else:
            print(f"Branch '{dst_branch}' already exists.")

        # Open writable session on the correct branch
        dst_session = dst_repo_obj.writable_session(dst_branch)

        # Initialize layout
        init_store(dst_session.store, sfc_ds=sfc_ds, pl_ds=pl_ds, inv_ds=inv_ds)

        commit_id = dst_session.commit("Initialized sample data store.")
        print(f"Initialized; commit_id={commit_id}")
        return

    # otherwise, we want to fill it with data
    parallel_reorg(
        src_repo_name=src_repo,
        dst_repo_name=dst_repo,
        start_time=start_time,
        end_time=end_time,
        src_branch=src_branch,
        dst_branch=dst_branch,
        token=token,
        days_at_once=days_at_once,
    )


if __name__ == "__main__":
    main()
