import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def convert_aurora_to_greenwich(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert Aurora's output coordinates to Greenwich-centered grid.

    Aurora outputs:
        lat: [90, -89.75] (descending, 720 points)
        lon: [0, 359.75] (0-360 range, 1440 points)

    Converts to:
        lat: [-89.75, 90] (ascending, 720 points)
        lon: [-180, 179.75] (Greenwich-centered, 1440 points)
    """
    logger.info(f"[CONVERT] Input: lat[0]={ds.lat.values[0]}, lat[-1]={ds.lat.values[-1]}")
    logger.info(f"[CONVERT] Input: lon[0]={ds.lon.values[0]}, lon[-1]={ds.lon.values[-1]}")

    # Flip latitude to ascending order: [90, -89.75] → [-89.75, 90]
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))
    else:
        logger.info("[CONVERT] Latitude already ascending, skipping flip.")

    # Convert longitude from [0, 360) to Greenwich-centered [-180, 180)
    if ds.lon.values[0] >= 0 and ds.lon.values[-1] >= 180:
        old_lon = ds.lon.values
        new_lon = np.where(old_lon >= 180, old_lon - 360, old_lon)

        # Find where to split: index of first lon >= 180 (which becomes negative)
        split_idx = int(np.argmax(old_lon >= 180))
        logger.info(f"[CONVERT] split_idx={split_idx} (lon[split_idx]={old_lon[split_idx]})")
        # xarray's roll_coords=True has known bugs with this pattern
        # Roll data so negative longitudes come first, then positive
        ds = ds.roll(lon=-split_idx, roll_coords=False)

        # Create properly ordered longitude: [-180, -179.75, ..., 179.75]
        new_lon_sorted = np.concatenate([new_lon[split_idx:], new_lon[:split_idx]])
        ds = ds.assign_coords(lon=new_lon_sorted)
    else:
        logger.info("[CONVERT] Longitude already Greenwich-centered, skipping conversion.")

    logger.info(f"[CONVERT] Output: lat[0]={ds.lat.values[0]}, lat[-1]={ds.lat.values[-1]}")
    logger.info(f"[CONVERT] Output: lon[0]={ds.lon.values[0]}, lon[-1]={ds.lon.values[-1]}")

    return ds


def get_spatial_indices_from_bounds(
    *,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    n_lev: int = 4,
    n_lat: int = 180,
    n_lon: int = 360,
    expected_spatial: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return flat spatial_location indices for a lat/lon bounding box.

    Latitude is assumed to be ERA-style (north → south).
    Longitude bounds must be Greenwich-centered in [-180, 180].

    The returned indices always refer to the ORIGINAL flattened
    (lev, lat, lon) ordering:
        spatial_location = lev * (n_lat * n_lon) + lat * n_lon + lon

    THEY WILL NOT BE MONOTONICALLY INCREASING.
    """

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    # Coordinate grids (centers)
    lat_vals = np.linspace(89.5, -89.5, n_lat)  # north → south
    lon_vals_0360 = np.linspace(0.5, 359.5, n_lon)  # 0–360
    lon_vals = ((lon_vals_0360 + 180) % 360) - 180  # −180…180

    # Latitude mask
    lat_mask = (lat_vals >= lat_min) & (lat_vals <= lat_max)
    lat_idx = np.where(lat_mask)[0]

    if lat_idx.size == 0:
        raise ValueError("Latitude range selects no grid points.")

    # Longitude mask (wrap-aware)
    if lon_min <= lon_max:
        lon_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)
    else:
        # wrapped interval across prime meridian
        lon_mask = (lon_vals >= lon_min) | (lon_vals <= lon_max)

    lon_idx = np.where(lon_mask)[0]

    if lon_idx.size == 0:
        raise ValueError("Longitude range selects no grid points.")

    # Sort longitude indices by actual longitude value
    # (fixes visual seam issues when crossing 0°)
    lon_idx = lon_idx[np.argsort(lon_vals[lon_idx])]

    # Build flat spatial_location indices
    # lev-major, then lat, then lon
    spatial_indices = np.concatenate(
        [lev * n_lat * n_lon + lat * n_lon + lon_idx for lev in range(n_lev) for lat in lat_idx]
    ).astype(np.int64)

    spatial_indices = spatial_indices.astype(np.int64)

    if expected_spatial is not None:
        expected_spatial = expected_spatial.astype(np.int64)
        if not np.array_equal(expected_spatial, spatial_indices):
            raise RuntimeError(
                "Spatial indices derived from lat/lon bounds do not match "
                "the destination store.\n"
                "Did you run `init` with the same spatial bounds?"
            )

    return spatial_indices
