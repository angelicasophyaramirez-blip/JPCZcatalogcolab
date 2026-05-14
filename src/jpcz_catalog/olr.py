"""NOAA OLR access helpers for JPCZ event review."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import xarray as xr

from .config import BoundingBox, WORKING_DOMAIN

NOAA_OLR_DAILY_FILESERVER_TEMPLATE = (
    "https://www.ncei.noaa.gov/thredds/fileServer/cdr/olr-daily/"
    "olr-daily_v01r02_{year}0101_{year}1231.nc"
)


def daily_olr_dataset_url(target_date: pd.Timestamp | str) -> str:
    """Return the official NOAA NCEI direct-download URL for the event year."""
    year = pd.Timestamp(target_date).year
    return NOAA_OLR_DAILY_FILESERVER_TEMPLATE.format(year=year)


def cached_daily_olr_dataset_path(
    target_date: pd.Timestamp | str,
    *,
    cache_dir: str | Path = ".cache/noaa_olr",
) -> Path:
    """Return the local cache path for one yearly NOAA OLR file."""
    target_date = pd.Timestamp(target_date).normalize()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"olr-daily_v01r02_{target_date.year}0101_{target_date.year}1231.nc"


def ensure_daily_olr_dataset(
    target_date: pd.Timestamp | str,
    *,
    cache_dir: str | Path = ".cache/noaa_olr",
) -> Path:
    """Download and cache the yearly NOAA OLR file if needed."""
    target_date = pd.Timestamp(target_date).normalize()
    cache_path = cached_daily_olr_dataset_path(target_date, cache_dir=cache_dir)

    if not cache_path.exists():
        urlretrieve(daily_olr_dataset_url(target_date), cache_path)

    return cache_path


def load_daily_olr_field(
    target_date: pd.Timestamp | str,
    *,
    domain: BoundingBox = WORKING_DOMAIN,
    cache_dir: str | Path = ".cache/noaa_olr",
) -> xr.DataArray:
    """Load one daily OLR field on the working domain from NOAA NCEI."""
    target_date = pd.Timestamp(target_date).normalize()
    dataset_path = ensure_daily_olr_dataset(target_date, cache_dir=cache_dir)
    ds = xr.open_dataset(dataset_path)

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
