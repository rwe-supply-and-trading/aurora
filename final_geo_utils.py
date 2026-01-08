import cartopy.crs as ccrs
import cartopy.feature as cfeature
import kafou_arraylake as arraylake
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
import zarr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER


def recenter_to_greenwich(ds, *, n_lat=180, n_lon=360):
    """Recenter a flattened (lev, lat, lon) grid from dateline-centered (0–360)
    to Greenwich-centered (-180–180).

    Assumes spatial_location ordering:
        lev-major, then lat-major, then lon-major.
    """
    # normalize longitude to [-180, 180)
    lon_new = ((ds.lon.values + 180) % 360) - 180

    flat = np.arange(ds.sizes["spatial_location"])
    lev = flat // (n_lat * n_lon)
    rem = flat % (n_lat * n_lon)
    lat_i = rem // n_lon
    lon_i = rem % n_lon

    # correct stable ordering: lev → lat → lon_value
    order = np.lexsort((lon_new, lat_i, lev))

    return ds.isel(spatial_location=order).assign_coords(lon=("spatial_location", lon_new[order]))


def subset_lv_dataset(
    ds: xr.Dataset,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
) -> xr.Dataset:
    """Spatial subset ONLY.

    Keeps:
      - ALL times
      - ALL levels
      - ALL features
      - spatial_location dimension intact

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset. Assumes a single dimension 'spatial_location' that is a flattened
        (lev, lat, lon) grid, with lev-major, then lat-major, then lon-major ordering.
        (i.e., for each level, all latitudes, for each latitude, all longitudes).
    lat_range : tuple of float
        (min_lat, max_lat) in degrees, specifying the latitude bounds for subsetting.
        Must be in Greenwich-centered coordinates (i.e., -90 to 90, north positive).
    lon_range : tuple of float
        (min_lon, max_lon) in degrees, specifying the longitude bounds for subsetting.
        Must be in Greenwich-centered coordinates (i.e., -180 to 180, east positive).

    Notes:
    -----
    - The function expects the input dataset to use dateline-centered (0–360) longitudes.
    - It automatically recenters the longitude coordinates to Greenwich-centered (-180–180).
    - The function is memory-safe.
    """
    # GRID SHAPE (MUST MATCH DAeTA)
    n_lev = 4
    n_lat = 180
    n_lon = 360

    # latitude centers (north → south)
    lats = np.linspace(89.5, -89.5, n_lat)

    # longitude centers (0.5 → 359.5, dateline-centered)
    lons = np.linspace(0.5, 359.5, n_lon)

    # RECONSTRUCT FLATTENED INDEXING
    # spatial_location = lev-major, lat-major, lon-major
    flat = np.arange(ds.sizes["spatial_location"])

    lev = flat // (n_lat * n_lon)
    rem = flat % (n_lat * n_lon)
    lat_idx = rem // n_lon
    lon_idx = rem % n_lon

    lat_vals = lats[lat_idx]
    lon_vals = lons[lon_idx]

    # ATTACH COORDINATES
    ds = ds.assign_coords(
        lat=("spatial_location", lat_vals),
        lon=("spatial_location", lon_vals),
        lev=("spatial_location", lev),
    ).set_coords(["lat", "lon", "lev"])

    print(ds)

    ds_greenwich = recenter_to_greenwich(ds)

    print("SUCCESS: lat / lon / lev coordinates attached")
    print("lat range:", float(ds_greenwich.lat.min()), float(ds_greenwich.lat.max()))
    print("lon range:", float(ds_greenwich.lon.min()), float(ds_greenwich.lon.max()))
    print("levels:", np.unique(ds_greenwich.lev.values))

    if lat_range and lon_range:
        mask = (
            (ds_greenwich.lat >= lat_range[0])
            & (ds_greenwich.lat <= lat_range[1])
            & (ds_greenwich.lon >= lon_range[0])
            & (ds_greenwich.lon <= lon_range[1])
        )

        ds_sub = ds_greenwich.isel(spatial_location=mask)
    else:
        ds_sub = ds_greenwich

    ds_sub.attrs.update(
        dict(
            lat_range=lat_range,
            lon_range=lon_range,
            created_by="Zora Zorkic",
        )
    )

    return ds_sub


def plot_lv_dataset(
    ds: xr.Dataset,
    *,
    time: str,
    lev: int,
    feature: int,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    cmap: str = "viridis",
    figsize: tuple = (8, 6),
):
    """Plot a single (time, lev, feature) slice from a spatially-subsetted,
    Greenwich-centered latent-vector dataset.

    If lat_range or lon_range is None, it is inferred from the data.
    """
    # --------------------------------------------------
    # 1) SELECT ONE SLICE (cheap)
    # --------------------------------------------------
    if "init_time" in ds.coords:
        da = (
            ds["lv"]
            .sel(init_time=np.datetime64(time), lead_time=12, feature=feature)
            .where(ds.lev == lev, drop=True)
        )

    else:
        da = ds.lv.sel(time=time, feature=feature).where(ds.lev == lev, drop=True)

    lat = ds.lat.where(ds.lev == lev, drop=True).values
    lon = ds.lon.where(ds.lev == lev, drop=True).values

    # --------------------------------------------------
    # 2) UNSTACK TO (lat, lon) GRID (safe)
    # --------------------------------------------------
    da2 = (
        da.assign_coords(
            lat=("spatial_location", lat),
            lon=("spatial_location", lon),
        )
        .set_index(spatial_location=("lat", "lon"))
        .unstack("spatial_location")
        .sortby("lat")
        .sortby("lon")
    )

    grid = da2.values
    lat_u = da2.lat.values
    lon_u = da2.lon.values

    # --------------------------------------------------
    # 3) INFER BOUNDS IF NEEDED
    # --------------------------------------------------
    if lat_range is None:
        lat_range = (float(lat_u.min()) - 0.5, float(lat_u.max()) + 0.5)

    if lon_range is None:
        lon_range = (float(lon_u.min()) - 0.5, float(lon_u.max()) + 0.5)

    # --------------------------------------------------
    # 4) PLOT
    # --------------------------------------------------
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)

    pcm = ax.pcolormesh(
        lon_u,
        lat_u,
        grid,
        shading="nearest",
        cmap=cmap,
        transform=ccrs.PlateCarree(),
    )

    # --------------------------------------------------
    # 5) GRIDLINES + TICKS (1°)
    # --------------------------------------------------
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle=":",
    )

    gl.top_labels = False
    gl.right_labels = False

    gl.xlocator = mticker.MultipleLocator(1)
    gl.ylocator = mticker.MultipleLocator(1)

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlabel_style = {
        "size": 9,
        "rotation": 45,
        "ha": "right",
    }

    gl.ylabel_style = {
        "size": 9,
        "va": "center",
    }

    # --------------------------------------------------
    # 6) FINAL TOUCHES
    # --------------------------------------------------
    plt.colorbar(pcm, ax=ax, label="lv value")

    ax.set_title(
        f"Greenwich-centered LV data\n"
        f"Feature={feature} | Level={lev} | Time={time}\n"
        f"Lat={lat_range} | Lon={lon_range}",
        fontsize=10,
    )

    plt.show()


import xarray as xr


def _infer_stream_dim(lv_da: xr.DataArray) -> str:
    """Choose the dimension we will stream/region-write over.
    Priority:
      1) 'time'
      2) 'init_time'
      3) any dim whose coordinate is datetime64
    """
    for preferred in ("time", "init_time"):
        if preferred in lv_da.dims:
            return preferred

    # fallback: any datetime-like dim
    for d in lv_da.dims:
        coord = lv_da.coords.get(d, None)
        if coord is not None and np.issubdtype(coord.dtype, np.datetime64):
            return d

    raise ValueError(
        f"Could not infer a streaming dimension from lv dims={lv_da.dims}. "
        "Expected one of ('time','init_time') or a datetime64 coordinate dim."
    )


def _infer_drop_coords_for_region_write(lv_da: xr.DataArray, *, stream_dim: str) -> list[str]:
    """For region writes along stream_dim, drop coords that don't include stream_dim
    (e.g., lat/lon/lev on spatial_location).
    Keep coords that depend on stream_dim (e.g., valid_time(init_time,lead_time)).
    """
    drop: list[str] = []
    for c in lv_da.coords:
        # don't drop dimension coords; those are fine
        if c in lv_da.dims:
            continue
        if stream_dim not in lv_da[c].dims:
            drop.append(c)
    return drop


def _summarize_layout(
    ds: xr.Dataset,
    *,
    lv_name: str,
    stream_dim: str,
    time_batch: int,
    lv_chunks: tuple[int, ...],
    zarr_format: int,
    consolidated: bool,
    allow_overwrite: bool,
) -> str:
    lv_da = ds[lv_name]
    lines: list[str] = []

    # Dataset basics
    lines.append("=== DATASET LAYOUT SUMMARY ===")
    lines.append(f"lv_name: {lv_name!r}")
    lines.append(f"ds.dims: {dict(ds.sizes)}")
    lines.append(f"ds.data_vars: {list(ds.data_vars)}")
    lines.append(f"ds.coords: {list(ds.coords)}")
    lines.append(f"ds.attrs keys: {list(ds.attrs.keys())}")

    # LV basics
    lines.append("")
    lines.append("--- LV VARIABLE ---")
    lines.append(f"lv.dims: {lv_da.dims}")
    lines.append(f"lv.shape: {lv_da.shape}")
    lines.append(f"lv.dtype: {str(lv_da.dtype)}")
    lines.append(f"stream_dim: {stream_dim!r} (batch={time_batch})")

    # Coordinate classification
    dim_coords = [d for d in lv_da.dims if d in lv_da.coords]
    aux_coords = [c for c in lv_da.coords if c not in lv_da.dims]
    lines.append(f"lv dim-coords: {dim_coords}")
    lines.append(f"lv aux-coords: {aux_coords}")

    # Drop plan
    drop_coords = _infer_drop_coords_for_region_write(lv_da, stream_dim=stream_dim)
    lines.append(f"drop coords during region write: {drop_coords}")

    # Chunking plan
    lines.append("")
    lines.append("--- STORAGE PLAN ---")
    lines.append(f"zarr_format: {zarr_format}")
    lines.append(f"consolidated: {consolidated}")
    lines.append(f"allow_overwrite: {allow_overwrite}")
    lines.append(f"lv_chunks: {lv_chunks}")
    lines.append("static write: ds.drop_vars([lv]) -> mode='w'")
    lines.append(f"stream write: region={{'{stream_dim}': slice(i0,i1)}}")

    return "\n".join(lines)


def write_full_dataset_to_arraylake_streaming(
    *,
    ds: xr.Dataset,
    dest_repo_name: str,
    dest_branch: str,
    base_branch: str = "main",
    time_batch: int = 7,
    zarr_format: int = 3,
    consolidated: bool = False,
    allow_overwrite: bool = False,
    commit: bool = False,
    commit_message: str | None = None,
    lv_name: str = "lv",
    lv_chunks: tuple[int, ...] | None = None,
    log_layout: bool = True,
):
    if lv_name not in ds.data_vars:
        raise ValueError(f"Dataset missing data var {lv_name!r}")

    lv_da = ds[lv_name]
    stream_dim = _infer_stream_dim(lv_da)

    if lv_chunks is None:
        inferred = []
        for d, s in zip(lv_da.dims, lv_da.shape, strict=False):
            inferred.append(1 if d == stream_dim else s)
        lv_chunks = tuple(inferred)

    if len(lv_chunks) != lv_da.ndim:
        raise ValueError(
            f"lv_chunks rank mismatch: got {lv_chunks} (len={len(lv_chunks)}) "
            f"but lv.ndim={lv_da.ndim} for dims={lv_da.dims}"
        )

    if log_layout:
        print(
            _summarize_layout(
                ds,
                lv_name=lv_name,
                stream_dim=stream_dim,
                time_batch=time_batch,
                lv_chunks=lv_chunks,
                zarr_format=zarr_format,
                consolidated=consolidated,
                allow_overwrite=allow_overwrite,
            )
        )

    client = arraylake.Client()
    try:
        repo = client.get_repo(dest_repo_name)
    except Exception:
        if not allow_overwrite:
            raise RuntimeError(
                f"Repository {dest_repo_name!r} does not exist and allow_overwrite=False"
            )
        print(f"Creating new repository {dest_repo_name!r}")
        repo = client.create_repo(dest_repo_name)

    try:
        session = repo.writable_session(dest_branch)
    except arraylake.exceptions.BranchNotFoundError:
        base_info = repo.lookup_branch(base_branch)
        print(
            f"Branch {dest_branch!r} not found. Creating it from {base_branch!r} (base={base_info})"
        )
        repo.create_branch(dest_branch, base_info)
        session = repo.writable_session(dest_branch)

    # (1) Write static dataset
    ds_static = ds.drop_vars([lv_name])

    if not allow_overwrite:
        try:
            root_existing = zarr.open_group(session.store, mode="r", zarr_format=zarr_format)
            if (len(root_existing.group_keys()) > 0) or (len(root_existing.array_keys()) > 0):
                raise FileExistsError(
                    "Destination branch/store is not empty and allow_overwrite=False. "
                    "Refusing to overwrite."
                )
        except zarr.errors.GroupNotFoundError:
            pass

    ds_static.to_zarr(
        session.store,
        mode="w",
        zarr_format=zarr_format,
        consolidated=consolidated,
    )

    root = zarr.open_group(session.store, mode="a", zarr_format=zarr_format)
    root.attrs.update(dict(ds.attrs))

    # (2) Create full LV array
    if lv_name in root.array_keys():
        print(f"⚠️ Found existing {lv_name!r} array — deleting + recreating with full shape")
        del root[lv_name]

    zarr.create_array(
        session.store,
        name=lv_name,
        shape=lv_da.shape,
        chunks=lv_chunks,
        dtype=lv_da.dtype,
        fill_value=np.nan,
        dimension_names=lv_da.dims,
        compressors=[],
    )

    # (3) Region-write LV streaming
    drop_coords = _infer_drop_coords_for_region_write(lv_da, stream_dim=stream_dim)
    n_stream = ds.sizes[stream_dim]

    for i0 in range(0, n_stream, time_batch):
        i1 = min(i0 + time_batch, n_stream)

        lv_chunk = lv_da.isel(**{stream_dim: slice(i0, i1)})

        if drop_coords:
            lv_chunk = lv_chunk.reset_coords(drop_coords, drop=True)

        # explicitly build dataset w/ only lv, then drop offending coord-vars
        lv_ds = lv_chunk.to_dataset(name=lv_name).drop_vars(
            ["feature", "lead_time", "spatial_location"], errors="ignore"
        )

        lv_ds.to_zarr(
            session.store,
            mode="a",
            zarr_format=zarr_format,
            consolidated=consolidated,
            region={stream_dim: slice(i0, i1)},
        )

        if log_layout:
            coord_note = f" (dropped coords: {drop_coords})" if drop_coords else ""
            print(f"[OK] wrote {lv_name} {stream_dim}[{i0}:{i1}]{coord_note}")

    # -----------------------------
    # (4) Write root attrs LAST
    # -----------------------------
    root = zarr.open_group(session.store, mode="a", zarr_format=zarr_format)

    # overwrite intentionally — attrs are authoritative from ds
    root.attrs.clear()
    root.attrs.update(dict(ds.attrs))

    if commit:
        cid = session.commit(commit_message or f"stream-write dataset + {lv_name}")
        print("Commit:", cid)

    return session
