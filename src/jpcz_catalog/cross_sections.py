"""Cross-section diagnostics for Notebook 26 timing-group manual verification."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .config import BoundingBox

EARTH_RADIUS_M = 6_371_000.0
STANDARD_GRAVITY = 9.80665
OMEGA_VECTOR_SCALE = 140.0
PRESSURE_BOTTOM_HPA = 1000.0
PRESSURE_TOP_HPA = 150.0
DEFAULT_PRESSURE_LEVELS_HPA: tuple[int, ...] = (
    1000,
    975,
    950,
    925,
    900,
    875,
    850,
    825,
    800,
    775,
    750,
    700,
    650,
    600,
    550,
    500,
    450,
    400,
    350,
    300,
    250,
    225,
    200,
)
DEFAULT_THETA_LEVELS_K = np.arange(270.0, 366.0, 2.0)
DEFAULT_PV_LEVELS_PVU = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
DEFAULT_OMEGA_LEVELS_PA_S = np.array(
    [-0.80, -0.60, -0.45, -0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80]
)
DEFAULT_MOISTURE_PROXY_LEVELS = np.array(
    [-2.5, -2.0, -1.5, -1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5]
)


@dataclass(frozen=True)
class Transect:
    """Simple geographic transect with projected distance."""

    start_lon: float
    start_lat: float
    end_lon: float
    end_lat: float
    lon: xr.DataArray
    lat: xr.DataArray
    distance_km: xr.DataArray
    along_unit_x: float
    along_unit_y: float


def build_transect(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    *,
    num_points: int = 141,
) -> Transect:
    """Construct evenly spaced lat/lon sample points and along-track distance."""
    lon_values = np.linspace(float(start_lon), float(end_lon), int(num_points))
    lat_values = np.linspace(float(start_lat), float(end_lat), int(num_points))
    point_index = np.arange(int(num_points), dtype=int)
    lon_da = xr.DataArray(lon_values, dims=("point",), coords={"point": point_index}, name="longitude")
    lat_da = xr.DataArray(lat_values, dims=("point",), coords={"point": point_index}, name="latitude")

    lat_rad = np.deg2rad(lat_values)
    lon_rad = np.deg2rad(lon_values)
    segment_dist_m = []
    for idx in range(1, num_points):
        mean_lat = 0.5 * (lat_rad[idx] + lat_rad[idx - 1])
        dlat = lat_rad[idx] - lat_rad[idx - 1]
        dlon = lon_rad[idx] - lon_rad[idx - 1]
        dx = EARTH_RADIUS_M * np.cos(mean_lat) * dlon
        dy = EARTH_RADIUS_M * dlat
        segment_dist_m.append((dx**2 + dy**2) ** 0.5)
    cumulative_km = np.concatenate([[0.0], np.cumsum(segment_dist_m) / 1000.0])
    distance_da = xr.DataArray(
        cumulative_km,
        dims=("point",),
        coords={"point": point_index},
        name="distance_km",
        attrs={"units": "km"},
    )

    mean_lat_for_unit = np.deg2rad(0.5 * (start_lat + end_lat))
    dx_total = EARTH_RADIUS_M * np.cos(mean_lat_for_unit) * np.deg2rad(end_lon - start_lon)
    dy_total = EARTH_RADIUS_M * np.deg2rad(end_lat - start_lat)
    length_total = max((dx_total**2 + dy_total**2) ** 0.5, 1.0)
    along_unit_x = dx_total / length_total
    along_unit_y = dy_total / length_total

    return Transect(
        start_lon=float(start_lon),
        start_lat=float(start_lat),
        end_lon=float(end_lon),
        end_lat=float(end_lat),
        lon=lon_da,
        lat=lat_da,
        distance_km=distance_da,
        along_unit_x=float(along_unit_x),
        along_unit_y=float(along_unit_y),
    )


def bounding_box_from_transect(
    transect: Transect,
    *,
    lon_pad: float = 4.0,
    lat_pad: float = 3.5,
) -> BoundingBox:
    """Build a padded domain around the transect endpoints."""
    lon_min = max(0.0, min(transect.start_lon, transect.end_lon) - float(lon_pad))
    lon_max = min(360.0, max(transect.start_lon, transect.end_lon) + float(lon_pad))
    lat_min = max(-90.0, min(transect.start_lat, transect.end_lat) - float(lat_pad))
    lat_max = min(90.0, max(transect.start_lat, transect.end_lat) + float(lat_pad))
    return BoundingBox(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)


def select_available_levels(
    ds: xr.Dataset,
    requested_levels_hpa: tuple[int, ...] = DEFAULT_PRESSURE_LEVELS_HPA,
) -> list[int]:
    """Return only those requested pressure levels that exist in the dataset."""
    if "level" not in ds.coords:
        raise RuntimeError("ERA5 runtime dataset does not expose pressure levels.")
    available_levels = {int(level) for level in np.asarray(ds["level"].values).astype(int)}
    selected = [int(level) for level in requested_levels_hpa if int(level) in available_levels]
    if len(selected) < 4:
        raise RuntimeError("Fewer than four requested pressure levels are available for cross-section plotting.")
    return selected


def load_pressure_volume(
    ds: xr.Dataset,
    analysis_time: pd.Timestamp | str,
    *,
    domain: BoundingBox,
    levels_hpa: list[int],
    variables: tuple[str, ...] | list[str],
) -> xr.Dataset:
    """Load one multi-level pressure-coordinate ERA5 subset on the requested domain."""
    subset = ds[list(variables)].sel(
        time=pd.Timestamp(analysis_time),
        longitude=slice(domain.lon_min, domain.lon_max),
        latitude=slice(domain.lat_max, domain.lat_min),
        level=levels_hpa,
    )
    if "time" in subset.dims:
        subset = subset.squeeze("time", drop=True)
    subset = subset.sortby("level", ascending=False)
    subset = subset.sortby("longitude")
    subset = subset.sortby("latitude")
    return subset.load()


def compute_potential_temperature_3d(
    ds: xr.Dataset,
    *,
    temperature_name: str = "temperature",
) -> xr.DataArray:
    """Compute potential temperature from a pressure-level temperature cube."""
    pressure_hpa = xr.DataArray(ds["level"].astype(float), dims=("level",), coords={"level": ds["level"]})
    kappa = 287.05 / 1004.0
    theta = ds[temperature_name] * (1000.0 / pressure_hpa) ** kappa
    theta = theta.rename("potential_temperature")
    theta.attrs["units"] = "K"
    theta.attrs["display_units"] = "K"
    return theta


def compute_vertical_moisture_flux_proxy_3d(
    ds: xr.Dataset,
    *,
    specific_humidity_name: str = "specific_humidity",
    vertical_velocity_name: str = "vertical_velocity",
) -> xr.DataArray:
    """Compute the -1000*q*omega moist-ascent proxy on all available pressure levels."""
    proxy = (-1000.0 * ds[specific_humidity_name] * ds[vertical_velocity_name]).rename("moist_ascent_proxy")
    proxy.attrs["units"] = "1e-3 Pa s^-1"
    proxy.attrs["display_units"] = "1e-3 Pa s^-1"
    return proxy


def _coordinate_arrays_m(field: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    lat_values = np.asarray(field["latitude"].values, dtype=float)
    lon_values = np.asarray(field["longitude"].values, dtype=float)
    mean_lat_rad = np.deg2rad(np.nanmean(lat_values))
    x_coords_m = EARTH_RADIUS_M * np.cos(mean_lat_rad) * np.deg2rad(lon_values)
    y_coords_m = EARTH_RADIUS_M * np.deg2rad(lat_values)
    return x_coords_m, y_coords_m


def compute_baroclinic_potential_vorticity_3d(
    ds: xr.Dataset,
    *,
    temperature_name: str = "temperature",
    u_name: str = "u_component_of_wind",
    v_name: str = "v_component_of_wind",
) -> xr.DataArray:
    """Approximate baroclinic Ertel PV in PVU on pressure levels."""
    theta = compute_potential_temperature_3d(ds, temperature_name=temperature_name)
    u = ds[u_name].astype(float)
    v = ds[v_name].astype(float)

    pressures_pa = np.asarray(theta["level"].values, dtype=float) * 100.0
    x_coords_m, y_coords_m = _coordinate_arrays_m(theta.isel(level=0, drop=True))

    theta_values = np.asarray(theta.values, dtype=float)
    u_values = np.asarray(u.values, dtype=float)
    v_values = np.asarray(v.values, dtype=float)

    dtheta_dp = np.gradient(theta_values, pressures_pa, axis=0, edge_order=2)
    du_dp = np.gradient(u_values, pressures_pa, axis=0, edge_order=2)
    dv_dp = np.gradient(v_values, pressures_pa, axis=0, edge_order=2)

    dtheta_dy, dtheta_dx = np.gradient(theta_values, y_coords_m, x_coords_m, axis=(-2, -1), edge_order=2)
    du_dy, _du_dx = np.gradient(u_values, y_coords_m, x_coords_m, axis=(-2, -1), edge_order=2)
    _dv_dy, dv_dx = np.gradient(v_values, y_coords_m, x_coords_m, axis=(-2, -1), edge_order=2)
    relative_vorticity = dv_dx - du_dy

    coriolis_1d = 2.0 * 7.2921159e-5 * np.sin(np.deg2rad(np.asarray(theta["latitude"].values, dtype=float)))
    coriolis_2d = np.broadcast_to(coriolis_1d[np.newaxis, :, np.newaxis], theta_values.shape)

    pv_values = -STANDARD_GRAVITY * (
        du_dp * dtheta_dy
        - dv_dp * dtheta_dx
        + (relative_vorticity + coriolis_2d) * dtheta_dp
    )
    pv_pvu = pv_values * 1e6

    pv = xr.DataArray(
        pv_pvu,
        coords=theta.coords,
        dims=theta.dims,
        name="baroclinic_potential_vorticity",
        attrs={"units": "PVU", "display_units": "PVU"},
    )
    return pv


def section_from_field(field: xr.DataArray, transect: Transect) -> xr.DataArray:
    """Interpolate a 2-D or 3-D field along the transect."""
    return field.interp(longitude=transect.lon, latitude=transect.lat)


def compute_along_transect_wind_section(
    ds: xr.Dataset,
    transect: Transect,
    *,
    u_name: str = "u_component_of_wind",
    v_name: str = "v_component_of_wind",
) -> xr.DataArray:
    """Project the horizontal wind onto the cross-section plane."""
    along_wind = (
        ds[u_name] * float(transect.along_unit_x)
        + ds[v_name] * float(transect.along_unit_y)
    ).rename("along_section_wind")
    along_wind.attrs["units"] = "m s^-1"
    along_wind.attrs["display_units"] = "m s^-1"
    return section_from_field(along_wind, transect)


def compute_terrain_section_pressure_hpa(
    terrain_field: xr.DataArray | None,
    transect: Transect,
) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    """Return terrain height and approximate surface pressure along the transect."""
    if terrain_field is None:
        return None, None
    terrain_section = terrain_field.interp(longitude=transect.lon, latitude=transect.lat, method="nearest")
    terrain_section = terrain_section.where(np.isfinite(terrain_section), 0.0)
    terrain_pressure_hpa = 1013.25 * np.exp(-terrain_section / 8400.0)
    terrain_pressure_hpa = terrain_pressure_hpa.rename("terrain_pressure_hpa")
    terrain_pressure_hpa.attrs["units"] = "hPa"
    return terrain_section, terrain_pressure_hpa


def interpolate_section_to_theta_coordinates(
    theta_section: xr.DataArray,
    value_section: xr.DataArray,
    *,
    theta_levels_k: np.ndarray = DEFAULT_THETA_LEVELS_K,
) -> xr.DataArray:
    """Interpolate one section field from pressure levels to theta levels point by point."""
    theta_levels_k = np.asarray(theta_levels_k, dtype=float)
    point_values = []
    theta_values = np.asarray(theta_section.values, dtype=float)
    value_values = np.asarray(value_section.values, dtype=float)

    for point_idx in range(theta_section.sizes["point"]):
        theta_profile = theta_values[:, point_idx]
        value_profile = value_values[:, point_idx]
        mask = np.isfinite(theta_profile) & np.isfinite(value_profile)
        if mask.sum() < 2:
            point_values.append(np.full(theta_levels_k.shape, np.nan))
            continue
        theta_profile = theta_profile[mask]
        value_profile = value_profile[mask]
        order = np.argsort(theta_profile)
        theta_sorted = theta_profile[order]
        value_sorted = value_profile[order]
        theta_unique, unique_idx = np.unique(theta_sorted, return_index=True)
        value_unique = value_sorted[unique_idx]
        if theta_unique.size < 2:
            point_values.append(np.full(theta_levels_k.shape, np.nan))
            continue
        point_values.append(
            np.interp(theta_levels_k, theta_unique, value_unique, left=np.nan, right=np.nan)
        )

    stacked = np.stack(point_values, axis=1)
    return xr.DataArray(
        stacked,
        dims=("theta_level", "point"),
        coords={
            "theta_level": theta_levels_k,
            "point": theta_section["point"],
            "distance_km": ("point", np.asarray(theta_section["distance_km"].values, dtype=float)),
        },
        name=value_section.name or "theta_section_value",
    )


def _section_mesh_coords(section_field: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(section_field["distance_km"].values, dtype=float)
    y_name = "level" if "level" in section_field.coords else "theta_level"
    y_values = np.asarray(section_field[y_name].values, dtype=float)
    return np.meshgrid(x_values, y_values)


def _rounded_levels_from_data(values: np.ndarray, *, step: float) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([0.0, step], dtype=float)
    lower = np.floor(np.nanmin(finite) / step) * step
    upper = np.ceil(np.nanmax(finite) / step) * step
    if lower == upper:
        upper = lower + step
    return np.arange(lower, upper + 0.5 * step, step)


def _draw_pressure_panel(
    ax,
    *,
    section_field: xr.DataArray,
    cmap: str,
    levels: np.ndarray,
    colorbar_label: str,
    title: str,
    theta_section: xr.DataArray,
    along_wind_section: xr.DataArray,
    omega_section: xr.DataArray,
    terrain_pressure_hpa: xr.DataArray | None,
    terrain_height_m: xr.DataArray | None,
    theta_color: str = "#8b0000",
    panel_extend: str = "both",
    extra_contours: tuple[xr.DataArray, np.ndarray, str, float, str] | None = None,
) -> tuple[object, object]:
    """Render one pressure-coordinate section panel."""
    x_grid, y_grid = _section_mesh_coords(section_field)
    fill = ax.contourf(
        x_grid,
        y_grid,
        np.asarray(section_field.values, dtype=float),
        levels=np.asarray(levels, dtype=float),
        cmap=cmap,
        extend=panel_extend,
    )

    theta_levels = _rounded_levels_from_data(theta_section.values, step=2.0)
    theta_contours = ax.contour(
        x_grid,
        y_grid,
        np.asarray(theta_section.values, dtype=float),
        levels=theta_levels,
        colors=theta_color,
        linewidths=0.8,
    )
    ax.clabel(theta_contours, fmt="%d", fontsize=7, inline=True)

    if extra_contours is not None:
        contour_field, contour_levels, contour_color, contour_width, contour_label = extra_contours
        special = ax.contour(
            x_grid,
            y_grid,
            np.asarray(contour_field.values, dtype=float),
            levels=np.asarray(contour_levels, dtype=float),
            colors=contour_color,
            linewidths=contour_width,
        )
        ax.clabel(special, fmt="%g", fontsize=7, inline=True)
        if contour_label:
            ax.text(
                0.99,
                0.93,
                contour_label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=contour_color,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
            )

    point_stride = max(1, section_field.sizes["point"] // 22)
    level_stride = max(1, section_field.sizes["level"] // 8)
    q = ax.quiver(
        np.asarray(section_field["distance_km"].values, dtype=float)[::point_stride],
        np.asarray(section_field["level"].values, dtype=float)[::level_stride],
        np.asarray(along_wind_section.values, dtype=float)[::level_stride, ::point_stride],
        np.asarray(omega_section.values, dtype=float)[::level_stride, ::point_stride] * OMEGA_VECTOR_SCALE,
        color="black",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0018,
        headwidth=3.4,
        headlength=4.5,
        headaxislength=3.8,
        alpha=0.75,
        zorder=5,
    )
    ax.quiverkey(
        q,
        0.99,
        1.02,
        20.0,
        "20 m s$^{-1}$ along-section wind",
        coordinates="axes",
        labelpos="E",
        fontproperties={"size": 8},
    )

    if terrain_pressure_hpa is not None:
        x_distance = np.asarray(section_field["distance_km"].values, dtype=float)
        terrain_pressure = np.asarray(terrain_pressure_hpa.values, dtype=float)
        terrain_pressure = np.clip(terrain_pressure, PRESSURE_TOP_HPA, 1050.0)
        ax.fill_between(
            x_distance,
            terrain_pressure,
            np.full_like(terrain_pressure, 1050.0),
            facecolor="#8f8f8f",
            edgecolor="#3f3f3f",
            linewidth=1.0,
            alpha=0.92,
            zorder=6,
        )
        ax.plot(x_distance, terrain_pressure, color="#2b2b2b", linewidth=1.2, zorder=7)
        if terrain_height_m is not None and np.isfinite(terrain_height_m.values).any():
            ax2 = ax.twinx()
            ax2.set_ylim(0.0, max(1000.0, float(np.nanmax(np.asarray(terrain_height_m.values, dtype=float))) * 1.05))
            ax2.set_ylabel("Terrain [m]", fontsize=8, color="#4a4a4a")
            ax2.tick_params(axis="y", labelsize=7, colors="#4a4a4a")
            ax2.spines["right"].set_alpha(0.35)
            ax2.set_zorder(0)

    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel("Pressure [hPa]")
    ax.set_ylim(1050.0, PRESSURE_TOP_HPA)
    ax.grid(True, color="0.78", linestyle="--", linewidth=0.45, alpha=0.6)
    return fill, theta_contours


def plot_pressure_cross_section_figure(
    *,
    transect: Transect,
    theta_section: xr.DataArray,
    omega_section: xr.DataArray,
    moisture_proxy_section: xr.DataArray,
    pv_section: xr.DataArray,
    along_wind_section: xr.DataArray,
    terrain_pressure_hpa: xr.DataArray | None,
    terrain_height_m: xr.DataArray | None,
    analysis_time: pd.Timestamp,
    group_id: str,
    time_role: str,
) -> plt.Figure:
    """Create the stacked pressure-coordinate section figure."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.2, 12.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]},
    )
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.09, right=0.92, hspace=0.24)

    omega_fill, _ = _draw_pressure_panel(
        axes[0],
        section_field=omega_section,
        cmap="RdBu_r",
        levels=DEFAULT_OMEGA_LEVELS_PA_S,
        colorbar_label="Omega [Pa s$^{-1}$]",
        title="Jet-normal pressure section: omega shading, theta contours, and cross-section wind vectors",
        theta_section=theta_section,
        along_wind_section=along_wind_section,
        omega_section=omega_section,
        terrain_pressure_hpa=terrain_pressure_hpa,
        terrain_height_m=terrain_height_m,
    )

    moisture_fill, _ = _draw_pressure_panel(
        axes[1],
        section_field=moisture_proxy_section,
        cmap="BrBG",
        levels=DEFAULT_MOISTURE_PROXY_LEVELS,
        colorbar_label="q × (-omega) [1e-3 Pa s$^{-1}$]",
        title="Moist-ascent surrogate section: q × (-omega) shading with theta and vectors",
        theta_section=theta_section,
        along_wind_section=along_wind_section,
        omega_section=omega_section,
        terrain_pressure_hpa=terrain_pressure_hpa,
        terrain_height_m=terrain_height_m,
    )

    pv_fill, _ = _draw_pressure_panel(
        axes[2],
        section_field=pv_section,
        cmap="viridis",
        levels=DEFAULT_PV_LEVELS_PVU,
        colorbar_label="Potential vorticity [PVU]",
        title="PV-focused section: PV shading, theta contours, vectors, and 2-PVU line",
        theta_section=theta_section,
        along_wind_section=along_wind_section,
        omega_section=omega_section,
        terrain_pressure_hpa=terrain_pressure_hpa,
        terrain_height_m=terrain_height_m,
        extra_contours=(pv_section, np.array([2.0]), "#f59e0b", 2.4, "2 PVU"),
        panel_extend="max",
    )
    axes[2].set_xlabel("Distance along section [km]")

    for axis, fill, label in [
        (axes[0], omega_fill, "Omega [Pa s$^{-1}$]"),
        (axes[1], moisture_fill, "q × (-omega) [1e-3 Pa s$^{-1}$]"),
        (axes[2], pv_fill, "Potential vorticity [PVU]"),
    ]:
        colorbar = fig.colorbar(fill, ax=axis, orientation="horizontal", pad=0.08, aspect=45)
        colorbar.set_label(label)

    fig.suptitle(
        f"Cross sections | {group_id} | {time_role} | {analysis_time:%Y-%m-%d %H:%M UTC}",
        fontsize=13,
        y=0.98,
    )
    return fig


def plot_isentropic_cross_section_figure(
    *,
    transect: Transect,
    theta_section: xr.DataArray,
    pressure_on_theta: xr.DataArray,
    moist_proxy_on_theta: xr.DataArray,
    pv_on_theta: xr.DataArray,
    analysis_time: pd.Timestamp,
    group_id: str,
    time_role: str,
) -> plt.Figure:
    """Render a simple theta-coordinate section with pressure contours."""
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 8.8), sharex=True)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.09, right=0.92, hspace=0.26)

    x_values = np.asarray(transect.distance_km.values, dtype=float)
    theta_levels = np.asarray(pressure_on_theta["theta_level"].values, dtype=float)
    x_grid, y_grid = np.meshgrid(x_values, theta_levels)

    for ax, field, cmap, levels, title, cbar_label in [
        (
            axes[0],
            moist_proxy_on_theta,
            "BrBG",
            DEFAULT_MOISTURE_PROXY_LEVELS,
            "Isentropic diagnostic: q × (-omega) interpolated onto theta surfaces",
            "q × (-omega) [1e-3 Pa s$^{-1}$]",
        ),
        (
            axes[1],
            pv_on_theta,
            "viridis",
            DEFAULT_PV_LEVELS_PVU,
            "Isentropic diagnostic: PV interpolated onto theta surfaces",
            "Potential vorticity [PVU]",
        ),
    ]:
        fill = ax.contourf(
            x_grid,
            y_grid,
            np.asarray(field.values, dtype=float),
            levels=np.asarray(levels, dtype=float),
            cmap=cmap,
            extend="both" if cmap == "BrBG" else "max",
        )
        pressure_contours = ax.contour(
            x_grid,
            y_grid,
            np.asarray(pressure_on_theta.values, dtype=float),
            levels=np.arange(200.0, 1001.0, 100.0),
            colors="black",
            linewidths=0.8,
        )
        ax.clabel(pressure_contours, fmt="%d", fontsize=7, inline=True)
        ax.set_ylabel(r"$\theta$ [K]")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, color="0.78", linestyle="--", linewidth=0.45, alpha=0.6)
        colorbar = fig.colorbar(fill, ax=ax, orientation="horizontal", pad=0.10, aspect=45)
        colorbar.set_label(cbar_label)

    axes[1].set_xlabel("Distance along section [km]")
    fig.suptitle(
        f"Isentropic section | {group_id} | {time_role} | {analysis_time:%Y-%m-%d %H:%M UTC}",
        fontsize=13,
        y=0.98,
    )
    return fig


def build_cross_section_diagnostics(
    ds: xr.Dataset,
    analysis_time: pd.Timestamp | str,
    *,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    terrain_field: xr.DataArray | None = None,
    levels_hpa: tuple[int, ...] = DEFAULT_PRESSURE_LEVELS_HPA,
    domain_pad_lon: float = 4.0,
    domain_pad_lat: float = 3.5,
    num_points: int = 141,
) -> dict[str, object]:
    """Load, interpolate, and derive all fields needed for Notebook 26 cross sections."""
    transect = build_transect(
        start_lon=float(start_lon),
        start_lat=float(start_lat),
        end_lon=float(end_lon),
        end_lat=float(end_lat),
        num_points=int(num_points),
    )
    cross_section_domain = bounding_box_from_transect(
        transect,
        lon_pad=float(domain_pad_lon),
        lat_pad=float(domain_pad_lat),
    )
    selected_levels = select_available_levels(ds, requested_levels_hpa=levels_hpa)
    pressure_volume = load_pressure_volume(
        ds,
        analysis_time,
        domain=cross_section_domain,
        levels_hpa=selected_levels,
        variables=(
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "specific_humidity",
        ),
    )

    theta_3d = compute_potential_temperature_3d(pressure_volume)
    omega_3d = pressure_volume["vertical_velocity"].astype(float).rename("omega")
    omega_3d.attrs["units"] = "Pa s^-1"
    moisture_proxy_3d = compute_vertical_moisture_flux_proxy_3d(pressure_volume)
    pv_3d = compute_baroclinic_potential_vorticity_3d(pressure_volume)
    along_wind_section = compute_along_transect_wind_section(pressure_volume, transect)

    theta_section = section_from_field(theta_3d, transect)
    theta_section = theta_section.assign_coords(distance_km=transect.distance_km)
    omega_section = section_from_field(omega_3d, transect).assign_coords(distance_km=transect.distance_km)
    moisture_proxy_section = section_from_field(moisture_proxy_3d, transect).assign_coords(distance_km=transect.distance_km)
    pv_section = section_from_field(pv_3d, transect).assign_coords(distance_km=transect.distance_km)
    along_wind_section = along_wind_section.assign_coords(distance_km=transect.distance_km)

    terrain_height_section, terrain_pressure_section = compute_terrain_section_pressure_hpa(terrain_field, transect)

    pressure_template = xr.DataArray(
        np.broadcast_to(
            np.asarray(theta_section["level"].values, dtype=float)[:, np.newaxis],
            theta_section.shape,
        ),
        dims=theta_section.dims,
        coords=theta_section.coords,
        name="pressure_hpa",
    )
    pressure_on_theta = interpolate_section_to_theta_coordinates(theta_section, pressure_template)
    moist_proxy_on_theta = interpolate_section_to_theta_coordinates(theta_section, moisture_proxy_section)
    pv_on_theta = interpolate_section_to_theta_coordinates(theta_section, pv_section)

    return {
        "analysis_time": pd.Timestamp(analysis_time),
        "transect": transect,
        "domain": cross_section_domain,
        "pressure_volume": pressure_volume,
        "theta_section": theta_section,
        "omega_section": omega_section,
        "moisture_proxy_section": moisture_proxy_section,
        "pv_section": pv_section,
        "along_wind_section": along_wind_section,
        "terrain_height_section": terrain_height_section,
        "terrain_pressure_section": terrain_pressure_section,
        "pressure_on_theta": pressure_on_theta,
        "moist_proxy_on_theta": moist_proxy_on_theta,
        "pv_on_theta": pv_on_theta,
    }
