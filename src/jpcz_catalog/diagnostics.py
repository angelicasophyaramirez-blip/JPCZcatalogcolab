"""Synoptic diagnostic helpers for the JPCZ event-atlas workflow."""

from __future__ import annotations

import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units

from .config import BoundingBox, EXTENDED_DOMAIN
from .detect import compute_grid_deltas

STANDARD_GRAVITY = 9.80665


def load_snapshot(
    ds: xr.Dataset,
    analysis_time: pd.Timestamp | str,
    *,
    variables: tuple[str, ...] | list[str],
    domain: BoundingBox = EXTENDED_DOMAIN,
    level: int | None = None,
) -> xr.Dataset:
    """Load one analysis-time snapshot on the requested domain and level."""
    snapshot = ds[list(variables)].sel(
        time=pd.Timestamp(analysis_time),
        longitude=slice(domain.lon_min, domain.lon_max),
        latitude=slice(domain.lat_max, domain.lat_min),
    )

    if level is not None and ("level" in snapshot.dims or "level" in snapshot.coords):
        snapshot = snapshot.sel(level=level)

    if "time" in snapshot.dims:
        snapshot = snapshot.squeeze("time", drop=True)

    return snapshot.load()


def load_offset_snapshot(
    ds: xr.Dataset,
    event_peak: pd.Timestamp | str,
    *,
    offset_hours: int,
    variables: tuple[str, ...] | list[str],
    domain: BoundingBox = EXTENDED_DOMAIN,
    level: int | None = None,
) -> xr.Dataset:
    """Load one snapshot offset from the event peak time."""
    target_time = pd.Timestamp(event_peak) + pd.Timedelta(hours=offset_hours)
    return load_snapshot(
        ds,
        target_time,
        variables=variables,
        domain=domain,
        level=level,
    )


def compute_geopotential_height_field(
    snapshot: xr.Dataset,
    *,
    z_name: str = "geopotential",
) -> xr.DataArray:
    """Return geopotential height in gpm from ERA5 geopotential."""
    geopotential_height = snapshot[z_name] / STANDARD_GRAVITY
    geopotential_height = geopotential_height.rename("geopotential_height")
    geopotential_height.attrs["units"] = "gpm"
    geopotential_height.attrs["display_units"] = "gpm"
    return geopotential_height


def compute_relative_vorticity_field(
    snapshot: xr.Dataset,
    *,
    u_name: str = "u_component_of_wind",
    v_name: str = "v_component_of_wind",
    dx=None,
    dy=None,
) -> xr.DataArray:
    """Compute relative vorticity for one analysis snapshot."""
    if dx is None or dy is None:
        dx, dy = compute_grid_deltas(snapshot.longitude, snapshot.latitude)

    u = snapshot[u_name].values * units("m/s")
    v = snapshot[v_name].values * units("m/s")
    zeta = mpcalc.vorticity(u, v, dx=dx, dy=dy).m

    return xr.DataArray(
        zeta,
        coords={"latitude": snapshot.latitude, "longitude": snapshot.longitude},
        dims=("latitude", "longitude"),
        name="relative_vorticity",
        attrs={"units": "s^-1", "display_units": "1e-5 s^-1"},
    )


def compute_temperature_gradient_magnitude(
    snapshot: xr.Dataset,
    *,
    temperature_name: str = "temperature",
    dx=None,
    dy=None,
) -> xr.DataArray:
    """Compute the horizontal temperature-gradient magnitude."""
    if dx is None or dy is None:
        dx, dy = compute_grid_deltas(snapshot.longitude, snapshot.latitude)

    temperature = snapshot[temperature_name].values * units.kelvin
    dtdy, dtdx = mpcalc.gradient(temperature, deltas=(dy, dx))
    grad_mag = ((dtdx**2 + dtdy**2) ** 0.5).m

    return xr.DataArray(
        grad_mag,
        coords={"latitude": snapshot.latitude, "longitude": snapshot.longitude},
        dims=("latitude", "longitude"),
        name="temperature_gradient_magnitude",
        attrs={
            "units": "K m^-1",
            "display_units": "K (100 km)^-1",
            "display_scale_factor": 1e5,
        },
    )


def compute_domain_relative_anomaly(field: xr.DataArray) -> xr.DataArray:
    """Subtract the domain mean to emphasize departures within one snapshot."""
    anomaly = field - field.mean(dim=("latitude", "longitude"))
    anomaly = anomaly.rename(f"{field.name}_domain_relative_anomaly")
    anomaly.attrs.update(field.attrs)
    return anomaly
