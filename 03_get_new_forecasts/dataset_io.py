"""Utilities to manage read/write of data using arraylake/icechunk/zarr."""

import logging
import os
from datetime import datetime, timedelta
from typing import Literal, Optional, Union

try:
    import kafou_arraylake as arraylake
except ImportError:
    import arraylake
import cftime
import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr.abc.store
import zarr.codecs
from icechunk import IcechunkError

# from rwx_pmf.ecmwf_grib import (
#     DEFAULT_PRESSURE_LEVELS_TO_EXTRACT,
#     interpolate_missing_pressure_levels,
# )

ARRAYLAKE_IC_REPO_NAME: str = "rwe/model-ecmwf-t0-nonprod-frankfurt"
ARRAYLAKE_IC_ZARR_GROUP_NAME: str = "ENFO-T0"

logger = logging.getLogger(__name__)


def ensure_time_in_arrays(
    store: zarr.abc.store.Store,
    timestamp: datetime,
    *,
    time_dim: str = "time",
    time_frequency: Union[str, timedelta, Literal["auto"]] = "auto",
    group: str | None = None,
) -> str | None:
    """Update the arrays in a Zarr group to ensure the given timestamp exists in a coordinate.

    This is important so that when a data set is updated, we can use xarray's `region="auto"`
    functionality to incrementally update data arrays.

    The time coordinate named by `time_dim` is assumed to be an xarray-compatible timestamp
    variable with all the time values in ascending order. If the given timestamp is not already
    in the time coordinate variable and is later than the last timestamp in that variable,
    the time coordinate will be extended.

    Additional timestamps up to and including the given timestamp at a resolution of the given
    time frequency will be appended to the array. All other arrays in the group that refer to
    the named time dimension will also be resized appropriately.

    If the given time frequency is left at the default of "auto", then the frequency is
    determined by the time delta between the last two timestamps in the time coordinate array.

    This function:

        1. Checks for the time_dim coordinate. Raises a KeyError if not present.
        2. If the time value is already in the time coordinate, it does nothing.
        3. If the time value is before the first time in the time coordinate,
           it raises a ValueError
        4. If the time value is between the first and last value in the time
           coordinate but is NOT in the time coordinate values it raises a ValueError.
        5. If the time value is not in the time coordinates and after the last value
           in the time coordinates, it appends either all time values between the
           last time in the time_coordinate and the requested time, following the
           time_frequency.
        6. All data arrays are then resized to add empty data corresponding to the
           number of time values added to the time.

    Args:
        store: A zarr store where the dataset resides.
        timestamp: The time value to ensure.
        time_dim: The name of the time dimension coordinate in which we ensure a timestamp.
                  (Default: "time")
        time_frequency: A time frequency for the time dimension. (Default: "auto")
        group: Optional group within the ZarrV3 store where the dataset is located. If None, the
               root group is used.

    Returns:
        None if no changes needed to be made, or a commit message suitable for Icechunk sessions
        if changes were made.
    """
    # Do this part in xarray, though we'll ultimately do the appending with
    # zarr.  Xarray decodes time times for us for comparisons.
    ds = xr.open_zarr(store, group=group, zarr_format=3, consolidated=False)

    # Check for the time_dim coordinate
    if time_dim not in ds.dims:
        raise KeyError(
            f"Requested time coordinate {time_dim!r} not found in dataset "
            f"at store: {store} and group {group}."
        )

    # Use pandas to convert input time to datetime64 for consistency
    time_dt64 = pd.to_datetime(timestamp)

    # If the time is in the range already, return
    if time_dt64 in ds[time_dim].values:
        logger.info(
            f"Requested time {timestamp} already exists in output dataset.Not adding new times."
        )
        return

    # If the time is before the first time or less than the last time, raise
    # an error.  Use pd.to_datetime to ensure we're comparing the same time types.
    if time_dt64 < ds[time_dim][0].values:
        raise ValueError(
            f"Requested time to write ({timestamp}) is before the first time "
            f"({ds[time_dim][0]}) in the dataset.  Cannot prepend to the beginning of "
            "the array."
        )
    elif time_dt64 < ds[time_dim][-1].values:
        raise ValueError(
            f"Requested time to write ({timestamp}) is before the last time "
            f"({ds[time_dim][-1]}) in the dataset, but not already in the dataset. "
            "Cannot insert into the middle of the array."
        )

    # If the time frequency requested is "auto", we calculate a timedelta from the
    # last two items of the time dimension.
    if time_frequency == "auto":
        if len(ds[time_dim]) < 2:
            raise ValueError(
                f"time_frequency must be specified as dimension {time_dim!r} is too small"
            )
        time_frequency = (
            (ds[time_dim][-1] - ds[time_dim][-2]).to_numpy().astype("timedelta64[s]").item()
        )

    # If we're here, then we can append this value.  Build the range of times
    # between the last time and the requested time at the time_frequency.  Exclude
    # the first time in this range (since it will match the last time already in the
    # dataset)

    datetimes_to_append = pd.date_range(
        start=ds[time_dim].values[-1], end=timestamp, freq=time_frequency
    )[1:]

    # Also figure out which variables will have to be extended
    variables_to_adjust = []
    for var in ds.data_vars:
        if time_dim in ds[var].dims:
            # Figure out dimension index
            dim_index = ds[var].dims.index(time_dim)
            variables_to_adjust.append((var, dim_index))

    # We need to encode these to CF integers following the conventions
    # in the Zarr dataset.  Xarray can obscure the needed attributes
    # when decoding the times, so switch to using zarr here
    group = zarr.open_group(store=store, path=group)
    time_coord_array = group[time_dim]
    time_coord_attrs = dict(time_coord_array.attrs)

    # Build the actual encoded time values here
    encoded_times_to_append = encode_datetimes_following_cf_conventions(
        datetimes_to_append,
        units=time_coord_attrs["units"],
        calendar=time_coord_attrs["calendar"],
    )

    # How many times are we appending?
    num_new_times = encoded_times_to_append.shape[0]

    # Now we do the actual appending of the time coordinates.
    # This will adjust the Zarr data.
    time_coord_array.append(encoded_times_to_append)

    # We now need to adjust all data variables that had the time dimension
    # as a dimension.  We use this by doing a "resize" on the zarr arrays,
    # since we don't have actual data to append, just want to fill with NaN.
    for var, dim_index in variables_to_adjust:
        # Get this array and its shape
        zarr_array = group[var]
        current_shape = list(zarr_array.shape)
        # Add the number of new times to the appropriate dimension
        current_shape[dim_index] += num_new_times
        # Resize
        zarr_array.resize(tuple(current_shape))

    # Commit the changes if we have a session
    commit_message = (
        f"Added {num_new_times} new time(s) to {time_dim} dimension "
        f"({datetimes_to_append[0]} - {datetimes_to_append[-1]})."
    )
    logger.info(commit_message)
    return commit_message


def init_ifs_dataset(
    store: zarr.abc.store.Store,
    start_time: Union[datetime, np.datetime64],
    end_time: Union[datetime, np.datetime64],
    *,
    time_resolution: Optional[Union[timedelta, np.timedelta64]] = None,
    group: Optional[str] = None,
) -> None:
    """Initializes an IFS dataset and stores it in the specified Zarr store.

    The dataset includes metadata and multidimensional arrays for meteorological variables.
    This function sets up the coordinates, such as time, pressure levels, latitude, longitude,
    and ensemble members, with their respective resolutions, and configures the
    arrays for storing the data.

    This function is mostly useful during development and testing and is included to document
    the general way in which the production data store was initialized.

    Args:
        store (zarr.abc.store.Store): The Zarr store where the dataset will be saved.
        start_time (Union[datetime, np.datetime64]): The starting time of the dataset.
        end_time (Union[datetime, np.datetime64]): The ending time of the dataset.
        time_resolution (Optional[Union[timedelta, np.timedelta64]]): Temporal
            resolution for the dataset. If not provided, it defaults to the entire
            time span.
        group (Optional[str]): The Zarr group where the dataset will be saved within
            the store.
    """
    if isinstance(start_time, datetime):
        start_time = np.datetime64(start_time)
    if isinstance(end_time, datetime):
        end_time = np.datetime64(end_time)

    start_time = start_time.astype("datetime64[ns]")
    end_time = end_time.astype("datetime64[ns]")

    if time_resolution is None:
        time_resolution = end_time - start_time
        if time_resolution == np.timedelta64(0, "ns"):
            time_resolution = np.timedelta64(1, "ns")
    elif isinstance(time_resolution, timedelta):
        time_resolution = np.timedelta64(time_resolution)

    time_data = np.arange(start_time, end_time + time_resolution, time_resolution)
    pressure_level_data = np.array(
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=np.int64
    )
    lat_data = np.arange(90.0, -90.25, -0.25, dtype=np.float64)
    lon_data = np.arange(0.0, 360.0, 0.25, dtype=np.float64)
    ensemble_member_data = np.arange(0, 52, 1, dtype=np.int64)

    ds = xr.Dataset(
        coords={
            "ensemble_member": xr.DataArray(ensemble_member_data, dims=("ensemble_member",)),
            "lat": xr.DataArray(lat_data, dims=("lat",)),
            "lon": xr.DataArray(lon_data, dims=("lon",)),
            "pressure_level": xr.DataArray(pressure_level_data, dims=("pressure_level",)),
            "time": xr.DataArray(time_data, dims=("time",)),
        }
    )

    ds.to_zarr(
        store,
        group=group,
        zarr_format=3,
        consolidated=False,
        encoding={
            "ensemble_member": {"chunks": (len(ds.ensemble_member),)},
            "lat": {"chunks": (len(ds.lat),)},
            "lon": {"chunks": (len(ds.lon),)},
            "pressure_level": {"chunks": (len(ds.pressure_level),)},
            "time": {"chunks": (10000,)},
        },
    )

    zgroup = zarr.open_group(store, path=group)
    compressors = [zarr.codecs.BloscCodec(clevel=3, shuffle="bitshuffle")]

    for varname in (
        "d2m",
        "ps",
        "slp",
        "t2m",
        "u100m",
        "u10m",
        "v100m",
        "v10m",
    ):
        zgroup.create_array(
            name=varname,
            shape=(len(ds.time), len(ds.ensemble_member), len(ds.lat), len(ds.lon)),
            chunks=(1, 1, len(ds.lat), len(ds.lon)),
            dtype=np.float32,
            fill_value=np.nan,
            compressors=compressors,
            dimension_names=("time", "ensemble_member", "lat", "lon"),
        )

    for varname in ("q", "t", "u", "v", "z"):
        zgroup.create_array(
            name=varname,
            shape=(
                len(ds.time),
                len(ds.ensemble_member),
                len(ds.pressure_level),
                len(ds.lat),
                len(ds.lon),
            ),
            chunks=(1, 1, 1, len(ds.lat), len(ds.lon)),
            dtype=np.float32,
            fill_value=np.nan,
            compressors=compressors,
            dimension_names=("time", "ensemble_member", "pressure_level", "lat", "lon"),
        )


def get_icechunk_session(
    repo: str,
    readonly_session: bool = False,
    branch: str = "main",
) -> icechunk.session.Session:
    """Get an icechunk session based on the parameters provided.

    If a requested repo does not exist, it will raise a ValueError.

    Args:
        repo: The path to the icechunk repository.
        readonly_session: Set to True to load return a readonly session instead
           of a writable session (default).
        branch: Optional branch instead of "main".

    Returns:
        An icechunk session object of the requested mode (writable or readonly).
    """
    if repo.startswith("kafou/") or repo.startswith("rwe/"):
        # This is an icechunk repo on object store
        client = arraylake.Client(token=os.environ.get("ARRAYLAKE_TOKEN", None))
        try:
            repository = client.get_repo(repo)
        except IcechunkError as ierr:
            raise KeyError(
                f"Tried to access this repo on S3 arraylake but it did "
                f"not exist: {repo}.  You must have manually created the repo on S3 "
                f"arraylake (including selecting the appropriate bucket, etc.) "
                f"to write to the S3 arraylake repo."
            ) from ierr
    else:
        # This is an icechunk store on local disk (not managed by arraylake)
        storage = icechunk.local_filesystem_storage(repo)
        repository = icechunk.Repository.open(storage)

    # Get session of appropriate type
    session = (
        repository.readonly_session(branch)
        if readonly_session
        else repository.writable_session(branch)
    )

    return session


def encode_datetimes_following_cf_conventions(
    times: pd.DatetimeIndex,
    units: str,
    calendar: str,
) -> np.ndarray[np.int64]:
    """Encode datetime64 values to CF-compliant integers using date2num.

    Args:
        times: Pandas-generated DatetimeIndex
        units: CF-compliant string descrbing time units.  Should be of the
            form "{time unit} since {starting date}"
        calendar: A CF-compliant calendar name (e.g., 'gregorian')

    Returns:
        A numpy array of integers corresponding to the datetimes provided
        but encoded with the CF conventions
    """
    return np.array([cftime.date2num(d, units=units, calendar=calendar) for d in times])


def ensure_time_in_arrays(
    store: zarr.abc.store.Store,
    timestamp: datetime,
    *,
    time_dim: str = "time",
    time_frequency: Union[str, timedelta, Literal["auto"]] = "auto",
    group: str | None = None,
) -> str | None:
    """Update the arrays in a Zarr group to ensure the given timestamp exists in a coordinate.

    This is important so that when a data set is updated, we can use xarray's `region="auto"`
    functionality to incrementally update data arrays.

    The time coordinate named by `time_dim` is assumed to be an xarray-compatible timestamp
    variable with all the time values in ascending order. If the given timestamp is not already
    in the time coordinate variable and is later than the last timestamp in that variable,
    the time coordinate will be extended.

    Additional timestamps up to and including the given timestamp at a resolution of the given
    time frequency will be appended to the array. All other arrays in the group that refer to
    the named time dimension will also be resized appropriately.

    If the given time frequency is left at the default of "auto", then the frequency is
    determined by the time delta between the last two timestamps in the time coordinate array.

    This function:

        1. Checks for the time_dim coordinate. Raises a KeyError if not present.
        2. If the time value is already in the time coordinate, it does nothing.
        3. If the time value is before the first time in the time coordinate,
           it raises a ValueError
        4. If the time value is between the first and last value in the time
           coordinate but is NOT in the time coordinate values it raises a ValueError.
        5. If the time value is not in the time coordinates and after the last value
           in the time coordinates, it appends either all time values between the
           last time in the time_coordinate and the requested time, following the
           time_frequency.
        6. All data arrays are then resized to add empty data corresponding to the
           number of time values added to the time.

    Args:
        store: A zarr store where the dataset resides.
        timestamp: The time value to ensure.
        time_dim: The name of the time dimension coordinate in which we ensure a timestamp.
                  (Default: "time")
        time_frequency: A time frequency for the time dimension. (Default: "auto")
        group: Optional group within the ZarrV3 store where the dataset is located. If None, the
               root group is used.

    Returns:
        None if no changes needed to be made, or a commit message suitable for Icechunk sessions
        if changes were made.
    """
    # Do this part in xarray, though we'll ultimately do the appending with
    # zarr.  Xarray decodes time times for us for comparisons.
    ds = xr.open_zarr(store, group=group, zarr_format=3, consolidated=False)

    # Check for the time_dim coordinate
    if time_dim not in ds.dims:
        raise KeyError(
            f"Requested time coordinate {time_dim!r} not found in dataset "
            f"at store: {store} and group {group}."
        )

    # Use pandas to convert input time to datetime64 for consistency
    time_dt64 = pd.to_datetime(timestamp)

    # If the time is in the range already, return
    if time_dt64 in ds[time_dim].values:
        logger.info(
            f"Requested time {timestamp} already exists in output dataset.Not adding new times."
        )
        return

    # If the time is before the first time or less than the last time, raise
    # an error.  Use pd.to_datetime to ensure we're comparing the same time types.
    if time_dt64 < ds[time_dim][0].values:
        raise ValueError(
            f"Requested time to write ({timestamp}) is before the first time "
            f"({ds[time_dim][0]}) in the dataset.  Cannot prepend to the beginning of "
            "the array."
        )
    elif time_dt64 < ds[time_dim][-1].values:
        raise ValueError(
            f"Requested time to write ({timestamp}) is before the last time "
            f"({ds[time_dim][-1]}) in the dataset, but not already in the dataset. "
            "Cannot insert into the middle of the array."
        )

    # If the time frequency requested is "auto", we calculate a timedelta from the
    # last two items of the time dimension.
    if time_frequency == "auto":
        if len(ds[time_dim]) < 2:
            raise ValueError(
                f"time_frequency must be specified as dimension {time_dim!r} is too small"
            )
        time_frequency = (
            (ds[time_dim][-1] - ds[time_dim][-2]).to_numpy().astype("timedelta64[s]").item()
        )

    # If we're here, then we can append this value.  Build the range of times
    # between the last time and the requested time at the time_frequency.  Exclude
    # the first time in this range (since it will match the last time already in the
    # dataset)

    datetimes_to_append = pd.date_range(
        start=ds[time_dim].values[-1], end=timestamp, freq=time_frequency
    )[1:]

    # Also figure out which variables will have to be extended
    variables_to_adjust = []
    for var in ds.data_vars:
        if time_dim in ds[var].dims:
            # Figure out dimension index
            dim_index = ds[var].dims.index(time_dim)
            variables_to_adjust.append((var, dim_index))

    # We need to encode these to CF integers following the conventions
    # in the Zarr dataset.  Xarray can obscure the needed attributes
    # when decoding the times, so switch to using zarr here
    group = zarr.open_group(store=store, path=group)
    time_coord_array = group[time_dim]
    time_coord_attrs = dict(time_coord_array.attrs)

    # Build the actual encoded time values here
    encoded_times_to_append = encode_datetimes_following_cf_conventions(
        datetimes_to_append,
        units=time_coord_attrs["units"],
        calendar=time_coord_attrs["calendar"],
    )

    # How many times are we appending?
    num_new_times = encoded_times_to_append.shape[0]

    # Now we do the actual appending of the time coordinates.
    # This will adjust the Zarr data.
    time_coord_array.append(encoded_times_to_append)

    # We now need to adjust all data variables that had the time dimension
    # as a dimension.  We use this by doing a "resize" on the zarr arrays,
    # since we don't have actual data to append, just want to fill with NaN.
    for var, dim_index in variables_to_adjust:
        # Get this array and its shape
        zarr_array = group[var]
        current_shape = list(zarr_array.shape)
        # Add the number of new times to the appropriate dimension
        current_shape[dim_index] += num_new_times
        # Resize
        zarr_array.resize(tuple(current_shape))

    # Commit the changes if we have a session
    commit_message = (
        f"Added {num_new_times} new time(s) to {time_dim} dimension "
        f"({datetimes_to_append[0]} - {datetimes_to_append[-1]})."
    )
    logger.info(commit_message)
    return commit_message


# def get_arraylake_ic_pair(
#     init_time: datetime,
#     ensemble_number: int,
#     hours_back: int = 6,
# ) -> xr.Dataset:
#     """Loads and normalises initial conditions from arraylake.

#     This function provides a handy interface for getting the initial conditions pair required to
#     initialise the aurora simulation. The expected use of this is to produce an xarray dataset
#     that is ready to create an aurora batch object from.

#     **This function should ONLY be used when expecting a initial condition pair
#     from init_time and init_time - 6 hours.**

#     This function handles the the pressure levels by removing any that are NaN in the input
#     and removing them. Once they are removed, they are interpolated by the
#     `interpolate_missing_pressure_levels` function.

#     Args:
#         init_time: The initialisation time for the dataset.
#         ensemble_number: The ensemble member number.
#         hours_back: how many hours back the pair should be fetched.
#                     Defaults to 6 as that is the aurora requirement

#     Returns:
#         xarray dataset containing the initial condition data for both init time & init time - 6.
#     """
#     session = get_icechunk_session(ARRAYLAKE_IC_REPO_NAME, readonly_session=True)

#     dset = xr.open_zarr(
#         session.store,
#         group=ARRAYLAKE_IC_ZARR_GROUP_NAME,
#         consolidated=False,
#         decode_timedelta=True,
#     )

#     prior_dt = init_time - timedelta(hours=hours_back)

#     try:
#         # always need a time & ensemble member to produce a 1x721x1440x<pressure_levels>
#         levels = list(DEFAULT_PRESSURE_LEVELS_TO_EXTRACT)
#         current_ics = dset.sel(
#             time=init_time,
#             ensemble_member=ensemble_number,
#             pressure_level=levels,
#         ).reindex(pressure_level=levels)
#         prior_ics = dset.sel(
#             time=prior_dt,
#             ensemble_member=ensemble_number,
#             pressure_level=levels,
#         ).reindex(pressure_level=levels)

#     except KeyError as e:
#         raise KeyError(
#             f"Could not find data for init_time {init_time},"
#             f"ensemble_member {ensemble_number}"
#             f"in arraylake repo {ARRAYLAKE_IC_REPO_NAME} "
#             f"and group {ARRAYLAKE_IC_ZARR_GROUP_NAME}. "
#             f"Please check that the requested data exists."
#         ) from e

#     # rename the co-ordinates
#     # interpolate function demands pressure levels be called "level"
#     prior_ics = prior_ics.rename(
#         {"lat": "latitude", "lon": "longitude", "pressure_level": "level"}
#     )
#     current_ics = current_ics.rename(
#         {"lat": "latitude", "lon": "longitude", "pressure_level": "level"}
#     )

#     # Remove any pressure levels that are all NaN by only selecting
#     # pressure levels where no NaN data is present.
#     # Assumes temperature variable is included and missing
#     # data in temperature is representative of all other pressure
#     # level data
#     prior_levels_initial = len(prior_ics["level"])
#     current_levels_initial = len(current_ics["level"])
#     print(
#         f"Initial pressure levels - Prior IC: {prior_levels_initial}, "
#         f"Current IC: {current_levels_initial}"
#     )

#     prior_ics = prior_ics.sel(
#         level=(prior_ics["t"].isnull().sum(dim=("latitude", "longitude")) == 0)
#     )
#     current_ics = current_ics.sel(
#         level=(current_ics["t"].isnull().sum(dim=("latitude", "longitude")) == 0)
#     )

#     # Check if we have any pressure levels left.  If not, it means
#     # that the initial condition data was NaN for all pressure levels,
#     # so abort
#     if len(prior_ics.level) == 0 or len(current_ics.level) == 0:
#         raise ValueError(
#             f"Initial condition data for init_time {init_time} or "
#             f"prior time {prior_dt} "
#             f"ensemble_member {ensemble_number} "
#             f"in arraylake repo {ARRAYLAKE_IC_REPO_NAME} "
#             f"and group {ARRAYLAKE_IC_ZARR_GROUP_NAME} "
#             f"was NaN for all pressure levels. Cannot proceed."
#         )

#     prior_levels_filtered = len(prior_ics["level"])
#     current_levels_filtered = len(current_ics["level"])
#     print(
#         f"NaN pressure levels - Prior IC: {prior_levels_filtered}, "
#         f"Current IC: {current_levels_filtered}"
#     )

#     # Now, backfill missing levels
#     prior_ics = interpolate_missing_pressure_levels(prior_ics)
#     current_ics = interpolate_missing_pressure_levels(current_ics)

#     ic_dset = xr.concat([prior_ics, current_ics], dim="time")

#     return ic_dset
