"""ERA5 radiation helpers for OLR-like cloud-proxy panels."""

from __future__ import annotations

import xarray as xr

AUTO_ERA5_OLR_TOKEN = "AUTO_ERA5_OLR"

ERA5_OLR_CANDIDATE_VARIABLES = (
    "mean_top_net_long_wave_radiation_flux",
    "avg_tnlwrf",
    "top_net_thermal_radiation",
    "ttr",
)


def available_era5_olr_variables(ds: xr.Dataset) -> list[str]:
    """Return OLR-like ERA5 radiation variables present in the dataset."""
    return [name for name in ERA5_OLR_CANDIDATE_VARIABLES if name in ds.data_vars]


def resolve_era5_olr_variable(ds: xr.Dataset, requested: str | None) -> str | None:
    """Resolve an ERA5 OLR-like variable name from the request."""
    if requested is None:
        return None
    if requested != AUTO_ERA5_OLR_TOKEN:
        return requested if requested in ds.data_vars else None

    available = available_era5_olr_variables(ds)
    if available:
        return available[0]
    return None


def era5_olr_like_field(snapshot: xr.Dataset, variable_name: str) -> tuple[xr.DataArray, str]:
    """Convert an ERA5 radiation variable into a positive-up OLR-like field."""
    field = snapshot[variable_name]
    units = str(field.attrs.get("units", "")).lower()

    if variable_name in {"mean_top_net_long_wave_radiation_flux", "avg_tnlwrf"}:
        # ERA5 net long-wave flux is positive downward; outgoing long-wave radiation is the opposite sign.
        olr_like = (-field).rename("era5_olr_like")
        olr_like.attrs["units"] = "W m^-2"
        olr_like.attrs["display_units"] = "W m^-2"
        label = f"ERA5 OLR proxy [W m^-2] ({variable_name})"
        return olr_like.load(), label

    if variable_name in {"top_net_thermal_radiation", "ttr"}:
        # ERA5 top net thermal radiation is an accumulated energy term (J m^-2) over the forecast step.
        # For hourly data, divide by 3600 s to approximate a mean flux, then flip sign for positive-up OLR.
        scale = 3600.0 if "j" in units else 1.0
        olr_like = (-(field / scale)).rename("era5_olr_like")
        olr_like.attrs["units"] = "W m^-2" if scale != 1.0 else field.attrs.get("units", "")
        olr_like.attrs["display_units"] = "W m^-2"
        label = f"ERA5 OLR proxy [W m^-2] ({variable_name})"
        return olr_like.load(), label

    return field.load(), variable_name
