"""Utilities to manage read/write of data using arraylake/icechunk/zarr."""

import logging
from datetime import datetime, timedelta
from typing import Literal, Union

try:
    import kafou_arraylake as arraylake
except ImportError:
    pass
import cftime
import numpy as np
import pandas as pd
import xarray as xr
import zarr.abc.store
import zarr.codecs

logger = logging.getLogger(__name__)


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
    init_hour: int | None = None,
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

    # ------------------------------
    # init_hour filtering
    # ------------------------------
    if init_hour is not None:
        datetimes_to_append = datetimes_to_append[datetimes_to_append.hour == init_hour]

    if len(datetimes_to_append) == 0:
        logger.info(f"No datetimes to append after init_hour={init_hour} filtering.")
        return None

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
