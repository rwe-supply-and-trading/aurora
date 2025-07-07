try:
    import kafou_arraylake as arraylake
except ImportError:
    import arraylake

import datetime

import xarray as xr
import numpy as np
import torch

from aurora import Batch, Metadata


def batch_to_ds(batch: Batch) -> xr.Dataset:
    """
    Convert an Aurora Batch object to an xarray dataset for convenience.

    Args:
        batch: aurora.Batch object

    Returns:
        An xarray.Dataset object

    """
    coord_data = {
        "latitude": xr.DataArray(batch.metadata.lat, dims=("latitude",)),
        "longitude": xr.DataArray(batch.metadata.lon, dims=("longitude",)),
        "level": xr.DataArray(np.array(batch.metadata.atmos_levels), dims=("level",)),
        "time": xr.DataArray(np.array(batch.metadata.time), dims=("time",)),
    }

    data_vars = {}

    data_vars.update(
        {
            dv: xr.DataArray(
                batch.surf_vars[dv],
                dims=("batch", "time", "latitude", "longitude"),
                coords={c: coord_data[c] for c in ("latitude", "longitude", "time")},
            )
            for dv in batch.surf_vars
        }
    )

    data_vars.update(
        {
            dv: xr.DataArray(
                batch.atmos_vars[dv],
                dims=("batch", "time", "level", "latitude", "longitude"),
                coords={c: coord_data[c] for c in ("latitude", "longitude", "level", "time")},
            )
            for dv in batch.atmos_vars
        }
    )

    return xr.Dataset(data_vars)


class ERA5DataLoaderFOAM:
    """
    An ERA5 data loader.

    Goes with resample_era5.py which reorganizes ERA5 data into a scheme which is fast to load from an object
    store for the FOAM use-case.

    Example use:

        from datetime import datetime

        import kafou_arraylake as arraylake

        from aurora import AuroraPretrained

        # Get an icechunk session.
        client = arraylake.Client()
        repo = client.get_repo("kafou/aurora-era5-samples")
        session = repo.readonly_session("main")

        # Load invariant and sample data for 2009-2024.
        invariant_data = xr.open_zarr(session.store, group="invariant", zarr_format=3, consolidated=False, chunks=None)
        sample_data = xr.open_zarr(session.store, group="samples", zarr_format=3, consolidated=False, chunks=None)
        sample_data = sample_data.sel(time=slice(datetime(2009, 1, 1, 0), datetime(2024, 12, 31, 18)))

        # Create a data loader.
        loader = ERA5DataLoaderFOAM(sample_data, invariant_data)

        # Load Aurora.
        model = AuroraPretrained()
        model.load_checkpoint()
        model.eval()
        model.to("cuda")

        # Iterate over batches, getting latent vectors.
        for batch in loader:
            pred = model.forward(batch, lv_only=True)
            ...
    """

    def __init__(
        self,
        sample_ds: xr.Dataset,
        invariant_ds: xr.Dataset,
    ):
        self.sample_ds = sample_ds

        self.static_vars = {
            k: torch.from_numpy(invariant_ds[k].to_numpy()) for k in ("lsm", "z", "slt")
        }

        self.meta_lat = torch.from_numpy(sample_ds.latitude.to_numpy())
        self.meta_lon = torch.from_numpy(sample_ds.longitude.to_numpy())
        self.meta_atmos_levels = tuple(x.item() for x in sample_ds.atmos_levels)

    def __getitem__(self, timestamp: datetime.datetime | int) -> Batch | tuple[Batch, Batch]:
        if isinstance(timestamp, datetime.datetime):
            if (
                timestamp.hour not in (0, 6, 12, 18)
                or timestamp.minute != 0
                or timestamp.second != 0
                or timestamp.microsecond != 0
            ):
                raise KeyError(f"Invalid sample time: {timestamp!r}")

            delta = datetime.timedelta(hours=6)
            ds = self.sample_ds.sel(time=slice(timestamp - delta, timestamp))
            if len(ds.time) != 2:
                raise KeyError(f"Unavailable sample time: {timestamp!r}")
        elif isinstance(timestamp, int):
            if timestamp > (len(self.sample_ds) - 1):
                raise KeyError(f"Invalid index: {timestamp}")
            ds = self.sample_ds.isel(time=slice(timestamp, timestamp + 2))
            timestamp = datetime.datetime.fromtimestamp(ds.time[1].item() / 1_000_000_000, tz=datetime.UTC).replace(tzinfo=None)
        else:
            raise KeyError(f"Invalid key: {timestamp!r}")

        ds = ds.compute()
        ds = ds.expand_dims(dim="batch", axis=0)
        sfc_locs = ds.attrs["var_locs"]["sfc"]
        pl_locs = ds.attrs["var_locs"]["pl"]

        surf_vars = {}
        for varname, (idx, n_channels) in sfc_locs.items():
            ary = ds["sample_data"][:, :, idx].to_numpy()
            surf_vars[varname] = torch.from_numpy(ary)

        atmos_vars = {}
        for varname, (idx, n_channels) in pl_locs.items():
            ary = ds["sample_data"][:, :, idx : idx + n_channels].to_numpy()
            atmos_vars[varname] = torch.from_numpy(ary)

        return Batch(
            surf_vars=surf_vars,
            static_vars=self.static_vars,
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=self.meta_lat,
                lon=self.meta_lon,
                atmos_levels=self.meta_atmos_levels,
                time=(timestamp,),
            ),
        )

    def __iter__(self):
        for i in range(len(self.sample_ds.time) - 1):
            yield self[i]