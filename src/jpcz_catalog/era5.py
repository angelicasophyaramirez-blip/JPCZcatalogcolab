"""Cloud-hosted ERA5 access helpers."""

from __future__ import annotations

import calendar
from collections.abc import Iterable, Sequence

import xarray as xr

from .config import (
    ARCO_ERA5_ZARR_STORE,
    BASELINE_END_UTC,
    BASELINE_START_UTC,
    BoundingBox,
    DECEMBER_BENCHMARK_YEARS,
    WORKING_DOMAIN,
)

ANALYSIS_VARIABLES: tuple[str, ...] = (
    "u_component_of_wind",
    "v_component_of_wind",
    "mean_sea_level_pressure",
)


def open_arco_era5(
    store: str = ARCO_ERA5_ZARR_STORE,
    *,
    chunks: dict[str, int] | None = None,
    storage_options: dict[str, str] | None = None,
) -> xr.Dataset:
    """Open the public ARCO ERA5 Zarr store with anonymous access."""
    if storage_options is None:
        storage_options = {"token": "anon"}
    return xr.open_zarr(store, chunks=chunks, storage_options=storage_options)


def subset_era5_window(
    ds: xr.Dataset,
    start: str,
    end: str,
    *,
    domain: BoundingBox = WORKING_DOMAIN,
    variables: Sequence[str] = ANALYSIS_VARIABLES,
    level: int = 925,
) -> xr.Dataset:
    """Subset ERA5 by time, domain, and optional level."""
    subset = ds[list(variables)].sel(
        time=slice(start, end),
        longitude=slice(domain.lon_min, domain.lon_max),
        latitude=slice(domain.lat_max, domain.lat_min),
    )

    if "level" in subset.dims or "level" in subset.coords:
        subset = subset.sel(level=level)

    return subset


def december_window(year: int) -> tuple[str, str]:
    """Return the start/end slice for one December."""
    return f"{year}-12-01", f"{year}-12-31T23:00:00"


def month_window(year: int, month: int) -> tuple[str, str]:
    """Return the start/end slice for one calendar month."""
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}T23:00:00"


def iter_december_windows(
    years: Iterable[int] = DECEMBER_BENCHMARK_YEARS,
) -> Iterable[tuple[int, str, str]]:
    """Yield year/start/end tuples for December benchmark windows."""
    for year in years:
        start, end = december_window(year)
        yield year, start, end


def iter_month_windows(
    years: Iterable[int],
    months: Iterable[int],
) -> Iterable[tuple[int, int, str, str]]:
    """Yield calendar-year/month start/end tuples for month-based catalog runs."""
    for year in years:
        for month in months:
            start, end = month_window(year, month)
            yield year, month, start, end


def open_december_month(
    ds: xr.Dataset,
    year: int,
    *,
    domain: BoundingBox = WORKING_DOMAIN,
    variables: Sequence[str] = ANALYSIS_VARIABLES,
    level: int = 925,
) -> xr.Dataset:
    """Open one December analysis window."""
    start, end = december_window(year)
    return subset_era5_window(
        ds,
        start,
        end,
        domain=domain,
        variables=variables,
        level=level,
    )


def open_month(
    ds: xr.Dataset,
    year: int,
    month: int,
    *,
    domain: BoundingBox = WORKING_DOMAIN,
    variables: Sequence[str] = ANALYSIS_VARIABLES,
    level: int = 925,
) -> xr.Dataset:
    """Open one calendar-month analysis window."""
    start, end = month_window(year, month)
    return subset_era5_window(
        ds,
        start,
        end,
        domain=domain,
        variables=variables,
        level=level,
    )


def open_baseline_window(
    ds: xr.Dataset,
    *,
    domain: BoundingBox = WORKING_DOMAIN,
    variables: Sequence[str] = ANALYSIS_VARIABLES,
    level: int = 925,
) -> xr.Dataset:
    """Open the known February 2018 baseline event window."""
    return subset_era5_window(
        ds,
        BASELINE_START_UTC,
        BASELINE_END_UTC,
        domain=domain,
        variables=variables,
        level=level,
    )
