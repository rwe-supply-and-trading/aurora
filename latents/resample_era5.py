import datetime
import time
from concurrent.futures import ProcessPoolExecutor

import kafou_arraylake as arraylake
import click
import icechunk
import numpy as np
import xarray as xr
import zarr
import zarr.abc.store

from icechunk.distributed import merge_sessions

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


def open_src_datasets(session: icechunk.Session) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    sfc_ds = xr.open_zarr(
        session.store, group="surface", zarr_format=3, consolidated=False, chunks=None
    )
    pl_ds = xr.open_zarr(
        session.store, group="pressure_level", zarr_format=3, consolidated=False, chunks=None
    )
    inv_ds = xr.open_zarr(
        session.store, group="invariant", zarr_format=3, consolidated=False, chunks=None
    )

    return sfc_ds, pl_ds, inv_ds


def get_variable_locations(n_pressure_levels: int = 13) -> tuple[dict, dict]:
    in_locs = {"sfc": {}, "pl": {}}
    out_locs = {"sfc": {}, "pl": {}}

    i = 0
    sfc_map = NAME_MAP["sfc"]
    for var in sfc_map:
        loc = (i, 1)
        in_locs["sfc"][var] = loc
        out_locs["sfc"][sfc_map[var]] = loc
        i += 1

    pl_map = NAME_MAP["pl"]
    for var in pl_map:
        loc = (i, n_pressure_levels)
        in_locs["pl"][var] = loc
        out_locs["pl"][pl_map[var]] = loc
        i += n_pressure_levels

    return in_locs, out_locs


def init_store(
    store: zarr.abc.store.Store, *, sfc_ds: xr.Dataset, pl_ds: xr.Dataset, inv_ds: xr.Dataset
):
    """
    Initialize a zarr storage based upon our input datasets.
    """

    # Select just the invariant variables we want and rename them appropriately.
    inv_ds = inv_ds[["Z", "LSM", "SLT"]]
    inv_ds = inv_ds.rename(NAME_MAP["inv"])
    inv_ds.to_zarr(store, group="invariant", zarr_format=3, consolidated=False)

    # Use xarray to create coordinates.
    _, out_var_locs = get_variable_locations(len(pl_ds.level))
    coords_ds = xr.Dataset(
        data_vars={
            "atmos_levels": [int(x) for x in pl_ds.level]
        },
        coords={
            "time": sfc_ds.time,
            "latitude": sfc_ds.latitude,
            "longitude": sfc_ds.longitude,
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
        encoding={
            "time": {"chunks": (len(sfc_ds.time),)},
            "latitude": {"chunks": (len(sfc_ds.latitude),)},
            "longitude": {"chunks": (len(sfc_ds.longitude),)},
            "atmos_levels": {"chunks": (len(coords_ds.atmos_levels),)},
        },
    )

    # Finally, use zarr to add an additional sparse data array which will be filled later.
    group = zarr.open_group(store, path="samples", mode="a")
    compressors = [zarr.codecs.BloscCodec(clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)]
    channels = len(NAME_MAP["sfc"]) + (len(NAME_MAP["pl"]) * len(coords_ds.atmos_levels))
    group.create_array(
        "sample_data",
        shape=(len(sfc_ds.time), channels, len(sfc_ds.latitude), len(sfc_ds.longitude)),
        chunks=(1, channels, 103, 72),
        dtype="float32",
        dimension_names=("time", "channel", "latitude", "longitude"),
        fill_value=np.nan,
        compressors=compressors,
    )


def process_day_from_datasets(
    *,
    store: zarr.abc.store.Store,
    timestamp: datetime.datetime,
    sfc_ds: xr.Dataset,
    pl_ds: xr.Dataset,
):
    time_selector = timestamp.strftime("%Y-%m-%d")

    in_locs, _ = get_variable_locations(len(pl_ds.level))
    sfc_vars = list(in_locs["sfc"])
    pl_vars = list(in_locs["pl"])

    # Load all data for the vars we care about all at once as a performance improvement.
    sfc_ds = sfc_ds[sfc_vars].sel(time=time_selector).compute()
    pl_ds = pl_ds[pl_vars].sel(time=time_selector).compute()

    n_channels = len(NAME_MAP["sfc"]) + (len(NAME_MAP["pl"]) * len(pl_ds.level))

    sample_data = np.zeros(
        (len(sfc_ds.time), n_channels, len(sfc_ds.latitude), len(sfc_ds.longitude)),
        dtype=np.float32,
    )

    for i in range(len(sfc_ds.time)):
        for varname, (d_idx, size) in in_locs["sfc"].items():
            sample_data[i, d_idx] = sfc_ds[varname][i]

        for varname, (d_idx, size) in in_locs["pl"].items():
            sample_data[i, d_idx : d_idx + size] = pl_ds[varname][i]

    data_array = xr.DataArray(sample_data, dims=("time", "channel", "latitude", "longitude"))

    ds = xr.Dataset(
        {"sample_data": data_array},
        coords={"time": sfc_ds.time, "latitude": sfc_ds.latitude, "longitude": sfc_ds.longitude},
    )
    ds.to_zarr(store, group="samples", zarr_format=3, consolidated=False, region="auto")

    return ds


def process_day_from_repo(
    *,
    timestamp: datetime.datetime,
    src_repo_name: str,
    session: icechunk.Session,
    src_branch: str = "main",
    token: str | None = None,
):
    client = arraylake.Client(token=token)
    if token is None:
        client.login()

    src_repo = client.get_repo(src_repo_name)
    src_session = src_repo.readonly_session(src_branch)
    sfc_ds = xr.open_zarr(
        src_session.store, group="surface", zarr_format=3, consolidated=False, chunks=None
    )
    pl_ds = xr.open_zarr(
        src_session.store, group="pressure_level", zarr_format=3, consolidated=False, chunks=None
    )
    process_day_from_datasets(store=session.store, timestamp=timestamp, sfc_ds=sfc_ds, pl_ds=pl_ds)

    return session


def parallel_reorg(
    *,
    src_repo_name: str,
    dst_repo_name: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    src_branch: str = "main",
    dst_branch: str = "main",
    token: str | None = None,
    days_at_once: int = 30,
    n_workers: int = 1,
):
    client = arraylake.Client(token=token)
    if token is None and dst_repo_name.startswith("rwe/"):
        client.login()

    dst_repo = client.get_repo(dst_repo_name)

    day_groups = []
    i = 0
    group = []
    while start_time <= end_time:
        if i == days_at_once:
            day_groups.append(group)
            group = []
            i = 0
        i += 1
        group.append(start_time)
        start_time += datetime.timedelta(days=1)
    if group:
        day_groups.append(group)

    for day_group in day_groups:
        try:
            p_start = time.monotonic()
            print(f"Processing {day_group[0]:%Y-%m-%d} to {day_group[-1]:%Y-%m-%d}")
            session = dst_repo.writable_session(dst_branch)

            with ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=1) as executor:
                with session.allow_pickling():
                    futures = [
                        executor.submit(
                            process_day_from_repo,
                            timestamp=x,
                            src_repo_name=src_repo_name,
                            session=session,
                            src_branch=src_branch,
                            token=token,
                        )
                        for x in day_group
                    ]
                    sessions = [f.result() for f in futures]
            session = merge_sessions(session, *sessions)
            session.commit(f"Added {day_group[0]:%Y-%m-%d} to {day_group[-1]:%Y-%m-%d}")
            p_end = time.monotonic()
            p_total = p_end - p_start
            print(f"Took {p_total:0.2f} seconds")
        except Exception:
            print(f"FAILED: {day_group[0]:%Y-%m-%d} {day_group[-1]:%Y-%m-%d}")


@click.command()
@click.argument("start_time", type=click.DateTime())
@click.argument("end_time", type=click.DateTime())
@click.option("--src-repo", default="rwe/era5-0p25-6h-nonprod-ohio")
@click.option("--dest-repo", default="kafou/aurora-era5-samples")
@click.option("--src-branch", default="main")
@click.option("--dest-branch", default="main")
@click.option("--token", default=None)
@click.option("--days-at-once", type=int, default=14)
@click.option("--n-workers", type=int, default=14)
@click.option("--init/--no-init", default=False)
def main(
    start_time,
    end_time,
    src_repo,
    dest_repo,
    src_branch,
    dest_branch,
    token,
    days_at_once,
    n_workers,
    init,
):
    if init:
        client = arraylake.Client(token=token)
        if token is None and (src_repo.startswith("rwe/") or dest_repo.startswith("rwe/")):
            client.login()

        src_repo = client.get_repo(src_repo)
        src_session = src_repo.readonly_session(src_branch)

        dest_repo = client.create_repo(dest_repo)
        dest_session = dest_repo.writable_session(dest_branch)

        inv_ds = xr.open_zarr(
            src_session.store, group="invariant", zarr_format=3, consolidated=False, chunks=None
        )
        sfc_ds = xr.open_zarr(
            src_session.store, group="surface", zarr_format=3, consolidated=False, chunks=None
        )
        pl_ds = xr.open_zarr(
            src_session.store,
            group="pressure_level",
            zarr_format=3,
            consolidated=False,
            chunks=None,
        )

        init_store(dest_session.store, sfc_ds=sfc_ds, pl_ds=pl_ds, inv_ds=inv_ds)

        commit_id = dest_session.commit("Initialized sample data store.")
        print(f"Initialized; {commit_id=}")
        return

    parallel_reorg(
        src_repo_name=src_repo,
        dst_repo_name=dest_repo,
        start_time=start_time,
        end_time=end_time,
        src_branch=src_branch,
        dst_branch=dest_branch,
        token=token,
        days_at_once=days_at_once,
        n_workers=n_workers,
    )


if __name__ == "__main__":
    main()
