"""Synoptic diagnostic helpers for the JPCZ event-atlas workflow."""

from __future__ import annotations

import numpy as np
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


def compute_wind_speed_field(
    snapshot: xr.Dataset,
    *,
    u_name: str = "u_component_of_wind",
    v_name: str = "v_component_of_wind",
) -> xr.DataArray:
    """Compute horizontal wind-speed magnitude for one analysis snapshot."""
    wind_speed = ((snapshot[u_name] ** 2 + snapshot[v_name] ** 2) ** 0.5).rename("wind_speed")
    wind_speed.attrs["units"] = "m s^-1"
    wind_speed.attrs["display_units"] = "m s^-1"
    return wind_speed


def compute_frontogenesis_field(
    snapshot: xr.Dataset,
    *,
    temperature_name: str = "temperature",
    u_name: str = "u_component_of_wind",
    v_name: str = "v_component_of_wind",
    dx=None,
    dy=None,
) -> xr.DataArray:
    """Compute Petterssen frontogenesis on one pressure-level snapshot."""
    if dx is None or dy is None:
        dx, dy = compute_grid_deltas(snapshot.longitude, snapshot.latitude)

    temperature = snapshot[temperature_name].values * units.kelvin
    u = snapshot[u_name].values * units("m/s")
    v = snapshot[v_name].values * units("m/s")
    frontogenesis = mpcalc.frontogenesis(temperature, u, v, dx=dx, dy=dy).m

    return xr.DataArray(
        frontogenesis,
        coords={"latitude": snapshot.latitude, "longitude": snapshot.longitude},
        dims=("latitude", "longitude"),
        name="frontogenesis",
        attrs={
            "units": "K m^-1 s^-1",
            "display_units": "K (100 km)^-1 (3 h)^-1",
            "display_scale_factor": 1e5 * 10800.0,
        },
    )


def compute_geostrophic_wind_fields(
    snapshot: xr.Dataset,
    *,
    geopotential_height: xr.DataArray | None = None,
    z_name: str = "geopotential",
    dx=None,
    dy=None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute geostrophic wind components from a geopotential-height field."""
    if dx is None or dy is None:
        dx, dy = compute_grid_deltas(snapshot.longitude, snapshot.latitude)

    if geopotential_height is None:
        geopotential_height = compute_geopotential_height_field(snapshot, z_name=z_name)

    height = geopotential_height.values * units.meter
    dzdy, dzdx = mpcalc.gradient(height, deltas=(dy, dx))
    coriolis_1d = mpcalc.coriolis_parameter(snapshot.latitude.values * units.degrees).to("s^-1").m
    coriolis_2d = np.broadcast_to(coriolis_1d[:, np.newaxis], geopotential_height.shape)

    dzdy_values = dzdy.to_base_units().m
    dzdx_values = dzdx.to_base_units().m
    with np.errstate(divide="ignore", invalid="ignore"):
        ug = np.where(np.abs(coriolis_2d) > 0.0, -(STANDARD_GRAVITY * dzdy_values) / coriolis_2d, np.nan)
        vg = np.where(np.abs(coriolis_2d) > 0.0, (STANDARD_GRAVITY * dzdx_values) / coriolis_2d, np.nan)

    ug_field = xr.DataArray(
        ug,
        coords={"latitude": snapshot.latitude, "longitude": snapshot.longitude},
        dims=("latitude", "longitude"),
        name="geostrophic_u_wind",
        attrs={"units": "m s^-1", "display_units": "m s^-1"},
    )
    vg_field = xr.DataArray(
        vg,
        coords={"latitude": snapshot.latitude, "longitude": snapshot.longitude},
        dims=("latitude", "longitude"),
        name="geostrophic_v_wind",
        attrs={"units": "m s^-1", "display_units": "m s^-1"},
    )
    return ug_field, vg_field


def compute_ageostrophic_wind_fields(
    snapshot: xr.Dataset,
    *,
    geopotential_height: xr.DataArray | None = None,
    z_name: str = "geopotential",
    u_name: str = "u_component_of_wind",
    v_name: str = "v_component_of_wind",
    dx=None,
    dy=None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute ageostrophic wind components by subtracting geostrophic flow."""
    ug_field, vg_field = compute_geostrophic_wind_fields(
        snapshot,
        geopotential_height=geopotential_height,
        z_name=z_name,
        dx=dx,
        dy=dy,
    )
    uag_field = (snapshot[u_name] - ug_field).rename("ageostrophic_u_wind")
    vag_field = (snapshot[v_name] - vg_field).rename("ageostrophic_v_wind")
    uag_field.attrs["units"] = "m s^-1"
    uag_field.attrs["display_units"] = "m s^-1"
    vag_field.attrs["units"] = "m s^-1"
    vag_field.attrs["display_units"] = "m s^-1"
    return uag_field, vag_field


def compute_ageostrophic_divergence_field(
    snapshot: xr.Dataset,
    *,
    geopotential_height: xr.DataArray | None = None,
    z_name: str = "geopotential",
    u_name: str = "u_component_of_wind",
    v_name: str = "v_component_of_wind",
    dx=None,
    dy=None,
) -> xr.DataArray:
    """Compute divergence of the ageostrophic wind field."""
    if dx is None or dy is None:
        dx, dy = compute_grid_deltas(snapshot.longitude, snapshot.latitude)

    uag_field, vag_field = compute_ageostrophic_wind_fields(
        snapshot,
        geopotential_height=geopotential_height,
        z_name=z_name,
        u_name=u_name,
        v_name=v_name,
        dx=dx,
        dy=dy,
    )
    uag = uag_field.values * units("m/s")
    vag = vag_field.values * units("m/s")
    div_ag = mpcalc.divergence(uag, vag, dx=dx, dy=dy).m

    return xr.DataArray(
        div_ag,
        coords={"latitude": snapshot.latitude, "longitude": snapshot.longitude},
        dims=("latitude", "longitude"),
        name="ageostrophic_divergence",
        attrs={"units": "s^-1", "display_units": "1e-5 s^-1"},
    )


def compute_domain_relative_anomaly(field: xr.DataArray) -> xr.DataArray:
    """Subtract the domain mean to emphasize departures within one snapshot."""
    anomaly = field - field.mean(dim=("latitude", "longitude"))
    anomaly = anomaly.rename(f"{field.name}_domain_relative_anomaly")
    anomaly.attrs.update(field.attrs)
    return anomaly
