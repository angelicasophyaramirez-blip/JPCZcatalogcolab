"""NOAA OLR access helpers for JPCZ event review."""

from __future__ import annotations

import pandas as pd
import xarray as xr

from .config import BoundingBox, WORKING_DOMAIN

NOAA_OLR_DAILY_OPENDAP_TEMPLATE = (
    "https://www.ncei.noaa.gov/thredds/dodsC/cdr/olr-daily/"
    "olr-daily_v01r02_{year}0101_{year}1231.nc"
)


def daily_olr_dataset_url(target_date: pd.Timestamp | str) -> str:
    """Return the official NOAA NCEI OPeNDAP URL for the event year."""
    year = pd.Timestamp(target_date).year
    return NOAA_OLR_DAILY_OPENDAP_TEMPLATE.format(year=year)


def load_daily_olr_field(
    target_date: pd.Timestamp | str,
    *,
    domain: BoundingBox = WORKING_DOMAIN,
) -> xr.DataArray:
    """Load one daily OLR field on the working domain from NOAA NCEI."""
    target_date = pd.Timestamp(target_date).normalize()
    ds = xr.open_dataset(daily_olr_dataset_url(target_date))

    olr_field = ds["olr"].sel(
        time=target_date,
        lon=slice(domain.lon_min, domain.lon_max),
        lat=slice(domain.lat_min, domain.lat_max),
    )

    if "time" in olr_field.dims:
        olr_field = olr_field.squeeze("time", drop=True)

    olr_field = olr_field.rename("olr")
    olr_field = olr_field.rename({"lon": "longitude", "lat": "latitude"})
    olr_field.attrs["units"] = "W m^-2"
    olr_field.attrs["display_units"] = "W m^-2"
    return olr_field.load()
