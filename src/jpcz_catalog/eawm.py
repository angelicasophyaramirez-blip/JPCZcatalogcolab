"""Helpers for event-scale EAWM-oriented review diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import BoundingBox, PACIFIC_EAST_OF_JAPAN_BOX
from .era5 import open_arco_era5

EAWM_DIAGNOSTIC_DOMAIN = BoundingBox(
    lon_min=80.0,
    lon_max=180.0,
    lat_min=-5.0,
    lat_max=80.0,
)

EAWM_LOW_LEVEL_DOMAIN = BoundingBox(
    lon_min=115.0,
    lon_max=155.0,
    lat_min=20.0,
    lat_max=55.0,
)

SIBERIAN_MONGOLIAN_HIGH_BOX = BoundingBox(
    lon_min=80.0,
    lon_max=120.0,
    lat_min=40.0,
    lat_max=60.0,
)

NORTH_PACIFIC_PRESSURE_BOX = BoundingBox(
    lon_min=150.0,
    lon_max=180.0,
    lat_min=30.0,
    lat_max=55.0,
)

MARITIME_CONTINENT_BOX = BoundingBox(
    lon_min=100.0,
    lon_max=130.0,
    lat_min=-5.0,
    lat_max=15.0,
)

JET_ENTRANCE_REGION_BOX = BoundingBox(
    lon_min=120.0,
    lon_max=150.0,
    lat_min=25.0,
    lat_max=40.0,
)

LOW_LEVEL_NORTH_FLOW_BOX = BoundingBox(
    lon_min=125.0,
    lon_max=145.0,
    lat_min=30.0,
    lat_max=45.0,
)

EAST_OF_JAPAN_Z500_BOX = PACIFIC_EAST_OF_JAPAN_BOX


def subset_box(field: xr.DataArray, box: BoundingBox) -> xr.DataArray:
    """Subset one field to a latitude/longitude box."""
    latitude = field["latitude"]
    latitude_slice = (
        slice(float(box.lat_max), float(box.lat_min))
        if float(latitude[0]) >= float(latitude[-1])
        else slice(float(box.lat_min), float(box.lat_max))
    )
    return field.sel(
        longitude=slice(float(box.lon_min), float(box.lon_max)),
        latitude=latitude_slice,
    )


def area_weighted_mean(field: xr.DataArray, box: BoundingBox | None = None) -> float:
    """Return a cosine-latitude-weighted mean value."""
    subset = subset_box(field, box) if box is not None else field
    if subset.size == 0:
        return float("nan")
    weights = xr.DataArray(
        np.cos(np.deg2rad(subset["latitude"])),
        coords={"latitude": subset["latitude"]},
        dims=("latitude",),
    )
    weighted = subset.weighted(weights)
    return float(weighted.mean(dim=("latitude", "longitude"), skipna=True).values)


def compute_monthly_field_climatology(
    ds: xr.Dataset,
    *,
    years: Sequence[int],
    months: Sequence[int],
    domain: BoundingBox,
    variables: Sequence[str],
    level: int | None,
    field_getter: Callable[[xr.Dataset], xr.DataArray],
    output_name: str,
) -> xr.DataArray:
    """Compute one monthly-mean climatology from a derived field getter."""
    climatology_slices: list[xr.DataArray] = []

    for month in sorted({int(value) for value in months}):
        month_sum = None
        month_count = 0
        for year in years:
            start = pd.Timestamp(year=int(year), month=int(month), day=1)
            end = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
            subset = ds[list(variables)].sel(
                time=slice(start, end),
                longitude=slice(float(domain.lon_min), float(domain.lon_max)),
                latitude=slice(float(domain.lat_max), float(domain.lat_min)),
            )
            if level is not None and ("level" in subset.dims or "level" in subset.coords):
                subset = subset.sel(level=int(level))

            field = field_getter(subset)
            if "time" not in field.dims:
                raise RuntimeError(
                    f"Field getter for {output_name!r} must return a DataArray with a time dimension."
                )
            field = field.astype(float).load()
            if field.sizes.get("time", 0) == 0:
                continue

            window_sum = field.sum("time", skipna=True)
            window_count = int(field.sizes["time"])
            month_sum = window_sum if month_sum is None else (month_sum + window_sum)
            month_count += window_count

        if month_sum is None or month_count == 0:
            continue

        month_mean = (month_sum / month_count).expand_dims(month=[int(month)])
        climatology_slices.append(month_mean)

    if not climatology_slices:
        raise RuntimeError(f"Unable to compute any climatology slices for {output_name!r}.")

    climatology = xr.concat(climatology_slices, dim="month").rename(output_name).sortby("month")
    return climatology


def load_or_update_monthly_field_climatology(
    path: str | Path,
    *,
    years: Sequence[int],
    months: Sequence[int],
    domain: BoundingBox,
    variables: Sequence[str],
    level: int | None,
    field_getter: Callable[[xr.Dataset], xr.DataArray],
    output_name: str,
    current_ds: xr.Dataset | None = None,
    chunks: dict[str, int] | None = None,
    storage_options: dict[str, str] | None = None,
) -> tuple[xr.DataArray, xr.Dataset]:
    """Load one cached monthly climatology or compute any missing months."""
    path = Path(path)
    climatology = None
    cached_months: set[int] = set()

    if path.exists():
        climatology = xr.open_dataarray(path).load()
        cached_months = {int(month_value) for month_value in climatology["month"].values.tolist()}

    missing_months = sorted({int(value) for value in months} - cached_months)
    runtime_ds = current_ds
    if missing_months:
        runtime_ds = open_arco_era5(chunks=chunks, storage_options=storage_options) if runtime_ds is None else runtime_ds
        for month in missing_months:
            month_climatology = compute_monthly_field_climatology(
                runtime_ds,
                years=years,
                months=[month],
                domain=domain,
                variables=variables,
                level=level,
                field_getter=field_getter,
                output_name=output_name,
            )
            climatology = (
                month_climatology
                if climatology is None
                else xr.concat([climatology, month_climatology], dim="month").sortby("month")
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            climatology.to_netcdf(path)

    if climatology is None:
        raise RuntimeError(f"Unable to load or compute climatology at {path}.")
    return climatology, runtime_ds if runtime_ds is not None else current_ds
