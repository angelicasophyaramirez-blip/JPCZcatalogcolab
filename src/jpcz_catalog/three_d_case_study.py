"""3-D single-event case-study helpers for Notebook 27."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .config import BoundingBox
from .cross_sections import build_transect, section_from_field
from .detect import compute_divergence_field
from .diagnostics import compute_wind_speed_field

EARTH_RADIUS_M = 6_371_000.0
STANDARD_GRAVITY = 9.80665

DEFAULT_3D_DOMAIN = BoundingBox(
    lon_min=124.0,
    lon_max=146.0,
    lat_min=33.0,
    lat_max=48.0,
)

DEFAULT_3D_PRESSURE_LEVELS_HPA: tuple[int, ...] = (
    925,
    850,
    700,
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

DEFAULT_SLICE_START = (130.5, 36.0)
DEFAULT_SLICE_END = (141.5, 42.0)


@dataclass(frozen=True)
class ThreeDCaseStudyData:
    analysis_time: pd.Timestamp
    domain: BoundingBox
    center_lon: float
    center_lat: float
    pressure_volume: xr.Dataset
    geopotential_height_km: xr.DataArray
    wind_speed: xr.DataArray
    relative_humidity_volume: xr.DataArray
    moisture_proxy_volume: xr.DataArray
    moisture_proxy_700: xr.DataArray
    divergence_925_display: xr.DataArray
    terrain_m: xr.DataArray | None
    terrain_x_km: np.ndarray | None
    terrain_y_km: np.ndarray | None
    x_km_3d: np.ndarray
    y_km_3d: np.ndarray
    x_km_regular_2d: np.ndarray | None
    y_km_regular_2d: np.ndarray | None
    z_levels_regular_km: np.ndarray | None
    wind_speed_regular_volume: np.ndarray | None
    relative_humidity_regular_volume: np.ndarray | None
    moisture_proxy_regular_volume: np.ndarray | None
    omega_regular_volume: np.ndarray | None
    slice_start: tuple[float, float] | None
    slice_end: tuple[float, float] | None
    slice_x_km: np.ndarray | None
    slice_y_km: np.ndarray | None
    slice_z_km: xr.DataArray | None
    slice_omega: xr.DataArray | None
    slice_lon: xr.DataArray | None
    slice_lat: xr.DataArray | None
    slice_terrain_km: np.ndarray | None


def subset_to_bounds(field: xr.DataArray, domain: BoundingBox) -> xr.DataArray:
    """Subset a 2-D field to one latitude/longitude box."""
    return field.where(
        (field.longitude >= float(domain.lon_min))
        & (field.longitude <= float(domain.lon_max))
        & (field.latitude >= float(domain.lat_min))
        & (field.latitude <= float(domain.lat_max)),
        drop=True,
    )


def load_surface_elevation_field(
    cache_path: str | Path,
    *,
    domain: BoundingBox | None = None,
) -> xr.DataArray | None:
    """Load the cached ETOPO1-derived terrain field used in earlier notebooks."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    with xr.open_dataset(cache_path) as terrain_ds:
        if "surface_elevation_m" not in terrain_ds.data_vars:
            return None
        terrain_field = terrain_ds["surface_elevation_m"].astype(float).load()

    if "longitude" in terrain_field.coords and float(terrain_field.longitude.min().values) < 0.0:
        terrain_field = terrain_field.assign_coords(
            longitude=((terrain_field.longitude + 360.0) % 360.0)
        )
    if "longitude" in terrain_field.coords:
        terrain_field = terrain_field.sortby("longitude")
    if "latitude" in terrain_field.coords:
        terrain_field = terrain_field.sortby("latitude")

    if domain is not None:
        terrain_field = subset_to_bounds(terrain_field, domain)
    return terrain_field


def load_case_event_catalog(catalog_path: str | Path) -> pd.DataFrame:
    """Load the manual-verification catalog and parse event timestamps."""
    catalog_df = pd.read_csv(catalog_path)
    for column_name in ["event_start", "event_end", "event_peak"]:
        if column_name in catalog_df.columns:
            catalog_df[column_name] = pd.to_datetime(catalog_df[column_name])
    return catalog_df


def select_case_event(
    catalog_df: pd.DataFrame,
    *,
    peak_time_utc: str | pd.Timestamp,
) -> pd.Series:
    """Return the requested event row from the catalog."""
    peak_time_utc = pd.Timestamp(peak_time_utc)
    match = catalog_df.loc[catalog_df["event_peak"] == peak_time_utc]
    if match.empty:
        raise RuntimeError(f"No event was found at peak time {peak_time_utc}.")
    return match.iloc[0].copy()


def _first_existing_value(event_row: pd.Series, names: tuple[str, ...], default=None):
    for name in names:
        if name in event_row.index:
            value = event_row.get(name)
            if pd.notna(value) and value != "":
                return value
    return default


def _load_pressure_volume(
    ds: xr.Dataset,
    analysis_time: pd.Timestamp | str,
    *,
    domain: BoundingBox,
    levels_hpa: tuple[int, ...],
    variables: tuple[str, ...],
) -> xr.Dataset:
    available_levels = {int(level) for level in np.asarray(ds["level"].values).astype(int)}
    selected_levels = [int(level) for level in levels_hpa if int(level) in available_levels]
    if len(selected_levels) < 4:
        raise RuntimeError("Fewer than four requested pressure levels are available for the 3-D case-study volume.")

    volume = ds[list(variables)].sel(
        time=pd.Timestamp(analysis_time),
        longitude=slice(float(domain.lon_min), float(domain.lon_max)),
        latitude=slice(float(domain.lat_max), float(domain.lat_min)),
        level=selected_levels,
    )
    if "time" in volume.dims:
        volume = volume.squeeze("time", drop=True)
    volume = volume.sortby("level", ascending=False)
    volume = volume.sortby("longitude")
    volume = volume.sortby("latitude")
    return volume.load()


def _local_xy_km(
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    center_lon: float,
    center_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    lon_values = np.asarray(longitude, dtype=float)
    lat_values = np.asarray(latitude, dtype=float)
    x_values = (
        EARTH_RADIUS_M
        * np.cos(np.deg2rad(float(center_lat)))
        * np.deg2rad(lon_values - float(center_lon))
        / 1000.0
    )
    y_values = EARTH_RADIUS_M * np.deg2rad(lat_values - float(center_lat)) / 1000.0
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return x_grid, y_grid


def _local_xy_km_from_lonlat(
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    *,
    center_lon: float,
    center_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    lon_values = np.asarray(lon_values, dtype=float)
    lat_values = np.asarray(lat_values, dtype=float)
    x_values = (
        EARTH_RADIUS_M
        * np.cos(np.deg2rad(float(center_lat)))
        * np.deg2rad(lon_values - float(center_lon))
        / 1000.0
    )
    y_values = EARTH_RADIUS_M * np.deg2rad(lat_values - float(center_lat)) / 1000.0
    return x_values, y_values


def _surface_height_km(snapshot: xr.Dataset) -> xr.DataArray:
    height_km = (snapshot["geopotential"].astype(float) / STANDARD_GRAVITY / 1000.0).rename("geopotential_height_km")
    height_km.attrs["units"] = "km"
    return height_km


def _compute_relative_humidity_volume(snapshot: xr.Dataset) -> xr.DataArray:
    """Approximate relative humidity from pressure, temperature, and specific humidity."""
    temperature_k = snapshot["temperature"].astype(float)
    specific_humidity = snapshot["specific_humidity"].astype(float)
    pressure_pa = (
        xr.DataArray(
            np.asarray(snapshot["level"].values, dtype=float) * 100.0,
            coords={"level": snapshot["level"]},
            dims=("level",),
        )
        .broadcast_like(temperature_k)
        .astype(float)
    )
    temperature_c = temperature_k - 273.15
    saturation_vapor_pressure_pa = 611.2 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))
    vapor_pressure_pa = (specific_humidity * pressure_pa) / (0.622 + 0.378 * specific_humidity)
    relative_humidity = (vapor_pressure_pa / saturation_vapor_pressure_pa).clip(min=0.0, max=1.2).rename("relative_humidity")
    relative_humidity.attrs["units"] = "fraction"
    relative_humidity.attrs["display_units"] = "%"
    return relative_humidity


def _smooth_section(
    section: xr.DataArray,
    *,
    vertical_window: int = 3,
    horizontal_window: int = 7,
) -> xr.DataArray:
    """Lightly smooth a 2-D section field for a more readable curtain."""
    if section.ndim != 2:
        return section
    vertical_dim, horizontal_dim = section.dims[0], section.dims[1]
    smoothed = section
    if int(vertical_window) > 1:
        smoothed = smoothed.rolling({vertical_dim: int(vertical_window)}, center=True, min_periods=1).mean()
    if int(horizontal_window) > 1:
        smoothed = smoothed.rolling({horizontal_dim: int(horizontal_window)}, center=True, min_periods=1).mean()
    return smoothed


def _build_regular_height_volume(
    field: xr.DataArray,
    height_km: xr.DataArray,
    x_grid_2d: np.ndarray,
    y_grid_2d: np.ndarray,
    *,
    horizontal_stride: int = 2,
    z_step_km: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate one pressure-level field onto a regular height grid."""
    stride = max(1, int(horizontal_stride))
    field_values = np.asarray(field.values, dtype=float)[:, ::stride, ::stride]
    height_values = np.asarray(height_km.values, dtype=float)[:, ::stride, ::stride]
    x_regular = np.asarray(x_grid_2d, dtype=float)[::stride, ::stride]
    y_regular = np.asarray(y_grid_2d, dtype=float)[::stride, ::stride]

    max_height_km = float(np.nanmax(height_values))
    z_levels_km = np.arange(0.5, max(12.0, np.ceil(max_height_km / z_step_km) * z_step_km) + 0.5 * z_step_km, z_step_km)
    regular_volume = np.full((len(z_levels_km), field_values.shape[1], field_values.shape[2]), np.nan, dtype=float)

    for j in range(field_values.shape[1]):
        for i in range(field_values.shape[2]):
            z_column = height_values[:, j, i]
            field_column = field_values[:, j, i]
            valid = np.isfinite(z_column) & np.isfinite(field_column)
            if valid.sum() < 2:
                continue
            z_valid = z_column[valid]
            field_valid = field_column[valid]
            sort_order = np.argsort(z_valid)
            z_sorted = z_valid[sort_order]
            field_sorted = field_valid[sort_order]
            z_unique, unique_idx = np.unique(z_sorted, return_index=True)
            field_unique = field_sorted[unique_idx]
            if z_unique.size < 2:
                continue
            inside = (z_levels_km >= float(z_unique.min())) & (z_levels_km <= float(z_unique.max()))
            if not np.any(inside):
                continue
            regular_volume[inside, j, i] = np.interp(z_levels_km[inside], z_unique, field_unique)

    return x_regular, y_regular, z_levels_km, regular_volume


def build_3d_case_study_data(
    ds: xr.Dataset,
    analysis_time: pd.Timestamp | str,
    *,
    domain: BoundingBox = DEFAULT_3D_DOMAIN,
    levels_hpa: tuple[int, ...] = DEFAULT_3D_PRESSURE_LEVELS_HPA,
    terrain_field: xr.DataArray | None = None,
    slice_start: tuple[float, float] | None = DEFAULT_SLICE_START,
    slice_end: tuple[float, float] | None = DEFAULT_SLICE_END,
) -> ThreeDCaseStudyData:
    """Load one event-centered 3-D volume and derive the first-pass visualization fields."""
    pressure_volume = _load_pressure_volume(
        ds,
        analysis_time,
        domain=domain,
        levels_hpa=levels_hpa,
        variables=(
            "u_component_of_wind",
            "v_component_of_wind",
            "geopotential",
            "temperature",
            "vertical_velocity",
            "specific_humidity",
        ),
    )

    center_lon = 0.5 * (float(domain.lon_min) + float(domain.lon_max))
    center_lat = 0.5 * (float(domain.lat_min) + float(domain.lat_max))
    x_grid_2d, y_grid_2d = _local_xy_km(
        pressure_volume.longitude.values,
        pressure_volume.latitude.values,
        center_lon=center_lon,
        center_lat=center_lat,
    )
    x_km_3d = np.broadcast_to(x_grid_2d[np.newaxis, :, :], pressure_volume["geopotential"].shape)
    y_km_3d = np.broadcast_to(y_grid_2d[np.newaxis, :, :], pressure_volume["geopotential"].shape)

    geopotential_height_km = _surface_height_km(pressure_volume)
    wind_speed = compute_wind_speed_field(pressure_volume)
    x_km_regular_2d, y_km_regular_2d, z_levels_regular_km, wind_speed_regular_volume = _build_regular_height_volume(
        wind_speed,
        geopotential_height_km,
        x_grid_2d,
        y_grid_2d,
    )
    relative_humidity_volume = _compute_relative_humidity_volume(pressure_volume)
    _, _, _, relative_humidity_regular_volume = _build_regular_height_volume(
        relative_humidity_volume,
        geopotential_height_km,
        x_grid_2d,
        y_grid_2d,
    )
    moisture_proxy_volume = (
        -1000.0
        * pressure_volume["specific_humidity"].astype(float)
        * pressure_volume["vertical_velocity"].astype(float)
    ).rename("moisture_proxy_volume")
    moisture_proxy_volume.attrs["units"] = "1e-3 Pa s^-1"
    moisture_proxy_700 = moisture_proxy_volume.sel(level=700).rename("moisture_proxy_700")
    moisture_proxy_700.attrs["units"] = "1e-3 Pa s^-1"
    _, _, _, moisture_proxy_regular_volume = _build_regular_height_volume(
        moisture_proxy_volume,
        geopotential_height_km,
        x_grid_2d,
        y_grid_2d,
    )
    _, _, _, omega_regular_volume = _build_regular_height_volume(
        pressure_volume["vertical_velocity"].astype(float).rename("omega"),
        geopotential_height_km,
        x_grid_2d,
        y_grid_2d,
    )

    snapshot_925 = pressure_volume.sel(level=925)
    divergence_925_display = (compute_divergence_field(snapshot_925) * 1e5).rename("divergence_925_display")
    divergence_925_display.attrs["units"] = "1e-5 s^-1"

    terrain_subset = None
    terrain_x_km = None
    terrain_y_km = None
    if terrain_field is not None:
        terrain_subset = subset_to_bounds(terrain_field, domain)
        terrain_x_km, terrain_y_km = _local_xy_km(
            terrain_subset.longitude.values,
            terrain_subset.latitude.values,
            center_lon=center_lon,
            center_lat=center_lat,
        )

    slice_x_km = None
    slice_y_km = None
    slice_z_km = None
    slice_omega = None
    slice_lon = None
    slice_lat = None
    slice_terrain_km = None
    if slice_start is not None and slice_end is not None:
        transect = build_transect(
            float(slice_start[0]),
            float(slice_start[1]),
            float(slice_end[0]),
            float(slice_end[1]),
            num_points=121,
        )
        slice_lon = transect.lon
        slice_lat = transect.lat
        x_line_km, y_line_km = _local_xy_km_from_lonlat(
            transect.lon.values,
            transect.lat.values,
            center_lon=center_lon,
            center_lat=center_lat,
        )
        slice_z_km = section_from_field(geopotential_height_km, transect)
        slice_omega = section_from_field(
            pressure_volume["vertical_velocity"].astype(float).rename("omega"),
            transect,
        )
        slice_x_km = np.broadcast_to(x_line_km[np.newaxis, :], slice_z_km.shape)
        slice_y_km = np.broadcast_to(y_line_km[np.newaxis, :], slice_z_km.shape)
        if terrain_subset is not None:
            terrain_along_slice = terrain_subset.interp(
                longitude=transect.lon,
                latitude=transect.lat,
                method="nearest",
            )
            slice_terrain_km = (
                terrain_along_slice.where(np.isfinite(terrain_along_slice), 0.0)
                .astype(float)
                .values
                / 1000.0
            )

    return ThreeDCaseStudyData(
        analysis_time=pd.Timestamp(analysis_time),
        domain=domain,
        center_lon=center_lon,
        center_lat=center_lat,
        pressure_volume=pressure_volume,
        geopotential_height_km=geopotential_height_km,
        wind_speed=wind_speed,
        relative_humidity_volume=relative_humidity_volume,
        moisture_proxy_volume=moisture_proxy_volume,
        moisture_proxy_700=moisture_proxy_700,
        divergence_925_display=divergence_925_display,
        terrain_m=terrain_subset,
        terrain_x_km=terrain_x_km,
        terrain_y_km=terrain_y_km,
        x_km_3d=x_km_3d,
        y_km_3d=y_km_3d,
        x_km_regular_2d=x_km_regular_2d,
        y_km_regular_2d=y_km_regular_2d,
        z_levels_regular_km=z_levels_regular_km,
        wind_speed_regular_volume=wind_speed_regular_volume,
        relative_humidity_regular_volume=relative_humidity_regular_volume,
        moisture_proxy_regular_volume=moisture_proxy_regular_volume,
        omega_regular_volume=omega_regular_volume,
        slice_start=slice_start,
        slice_end=slice_end,
        slice_x_km=slice_x_km,
        slice_y_km=slice_y_km,
        slice_z_km=slice_z_km,
        slice_omega=slice_omega,
        slice_lon=slice_lon,
        slice_lat=slice_lat,
        slice_terrain_km=slice_terrain_km,
    )


def build_case_metadata_table(event_row: pd.Series) -> pd.DataFrame:
    """Return a compact event-summary table for notebook display."""
    summary = {
        "event_peak_utc": pd.Timestamp(event_row["event_peak"]),
        "duration_hours": event_row.get("duration_hours"),
        "primary_label": _first_existing_value(
            event_row,
            ("manual_objective_label", "cleaned_k3_label", "monsoon_type"),
        ),
        "secondary_label": _first_existing_value(
            event_row,
            ("cleaned_k3_label", "shinoda_class"),
        ),
        "candidate_peak_convergence_1e5_s-1": event_row.get("candidate_peak_convergence_1e5_s-1"),
        "peak_max_convergence_lat": event_row.get("peak_max_convergence_lat"),
        "peak_max_convergence_lon": event_row.get("peak_max_convergence_lon"),
        "verification_flag": _first_existing_value(
            event_row,
            ("manual_verification", "verified_event"),
        ),
        "verification_notes": _first_existing_value(
            event_row,
            ("manual_notes", "verification_notes"),
        ),
        "upper_level_forcing_note": event_row.get("upper_level_forcing_note"),
    }
    return pd.DataFrame({"field": list(summary), "value": list(summary.values())})


def build_case_runtime_diagnostics(case_data: ThreeDCaseStudyData) -> pd.DataFrame:
    """Summarize what the 3-D runtime actually loaded."""
    level_values = np.asarray(case_data.pressure_volume["level"].values, dtype=float)
    wind_max_by_level = [
        f"{int(level)} hPa: {float(case_data.wind_speed.sel(level=level).max().values):.1f} m s^-1"
        for level in level_values
    ]
    relative_humidity_max_by_level = [
        f"{int(level)} hPa: {100.0 * float(case_data.relative_humidity_volume.sel(level=level).max().values):.0f}%"
        for level in level_values
    ]
    moisture_max_by_level = [
        f"{int(level)} hPa: {float(case_data.moisture_proxy_volume.sel(level=level).max().values):.2f}"
        for level in level_values
    ]
    omega_extrema_by_level = [
        f"{int(level)} hPa: {float(case_data.pressure_volume['vertical_velocity'].sel(level=level).min().values):.2f} to "
        f"{float(case_data.pressure_volume['vertical_velocity'].sel(level=level).max().values):.2f} Pa s^-1"
        for level in level_values
    ]
    regular_volume = case_data.wind_speed_regular_volume
    regular_volume_max = float(np.nanmax(regular_volume)) if regular_volume is not None else np.nan
    terrain_loaded = case_data.terrain_m is not None
    terrain_max_m = (
        float(np.nanmax(np.asarray(case_data.terrain_m.values, dtype=float)))
        if terrain_loaded
        else np.nan
    )
    z_min_km = float(np.nanmin(np.asarray(case_data.geopotential_height_km.values, dtype=float)))
    z_max_km = float(np.nanmax(np.asarray(case_data.geopotential_height_km.values, dtype=float)))

    summary = {
        "analysis_time_utc": case_data.analysis_time,
        "cube_lon_range": f"{case_data.domain.lon_min:.1f} to {case_data.domain.lon_max:.1f}",
        "cube_lat_range": f"{case_data.domain.lat_min:.1f} to {case_data.domain.lat_max:.1f}",
        "levels_hpa": ", ".join(str(int(level)) for level in level_values),
        "terrain_loaded": terrain_loaded,
        "terrain_max_m": terrain_max_m,
        "z_min_km": z_min_km,
        "z_max_km": z_max_km,
        "wind_max_by_level": " | ".join(wind_max_by_level),
        "relative_humidity_max_by_level": " | ".join(relative_humidity_max_by_level),
        "moisture_proxy_max_by_level": " | ".join(moisture_max_by_level),
        "omega_extrema_by_level": " | ".join(omega_extrema_by_level),
        "regular_volume_max_wind": regular_volume_max,
    }
    return pd.DataFrame({"field": list(summary), "value": list(summary.values())})


def create_3d_case_figure(
    case_data: ThreeDCaseStudyData,
    *,
    title: str | None = None,
    show_cube_frame: bool = True,
    show_jet_volume: bool = True,
    show_moisture_sheet: bool = True,
    show_divergence_sheet: bool = True,
    show_slice_curtain: bool = True,
    show_moisture_volume: bool = False,
    show_ascent_descent_points: bool = False,
    show_convergence_floor: bool = False,
    moisture_volume_mode: str = "relative_humidity",
    jet_isomin: float = 25.0,
    jet_isomax: float | None = None,
    jet_surface_count: int = 6,
    jet_opacity: float = 0.50,
    jet_top_pressure_hpa: int = 400,
    show_jet_points: bool = False,
    jet_point_threshold: float = 20.0,
    jet_point_size: float = 3.0,
    max_jet_points: int = 5000,
    relative_humidity_min: float = 0.82,
    relative_humidity_max: float = 1.00,
    require_ascent_for_rh_volume: bool = True,
    moisture_isomin: float | None = None,
    moisture_isomax: float | None = None,
    moisture_surface_count: int = 4,
    moisture_opacity: float = 0.25,
    moisture_quantile: float = 0.97,
    omega_point_quantile: float = 0.98,
    omega_point_size: float = 3.4,
    max_omega_points: int = 1800,
    convergence_quantile: float = 0.82,
    convergence_opacity: float = 0.60,
    slice_omega_smoothing_levels: int = 3,
    slice_omega_smoothing_points: int = 7,
    slice_omega_max_abs: float | None = None,
    slice_omega_opacity: float = 0.52,
    vertical_exaggeration: float = 28.0,
) -> Any:
    """Build the rotatable Plotly figure for one event-centered case-study cube."""
    import plotly.graph_objects as go

    z_values = np.asarray(case_data.geopotential_height_km.values, dtype=float)
    full_jet_values = np.asarray(case_data.wind_speed.values, dtype=float)

    if (
        case_data.wind_speed_regular_volume is not None
        and case_data.z_levels_regular_km is not None
        and case_data.x_km_regular_2d is not None
        and case_data.y_km_regular_2d is not None
    ):
        regular_z = np.asarray(case_data.z_levels_regular_km, dtype=float)
        jet_height_floor = float(np.nanmin(np.asarray(case_data.geopotential_height_km.sel(level=int(jet_top_pressure_hpa)).values, dtype=float)))
        jet_height_mask = regular_z >= jet_height_floor
        if np.any(jet_height_mask):
            jet_values = np.asarray(case_data.wind_speed_regular_volume, dtype=float)[jet_height_mask, :, :]
            z_jet = np.broadcast_to(regular_z[jet_height_mask][:, np.newaxis, np.newaxis], jet_values.shape)
        else:
            jet_values = np.asarray(case_data.wind_speed_regular_volume, dtype=float)
            z_jet = np.broadcast_to(regular_z[:, np.newaxis, np.newaxis], jet_values.shape)
        x_jet = np.broadcast_to(np.asarray(case_data.x_km_regular_2d, dtype=float), jet_values.shape)
        y_jet = np.broadcast_to(np.asarray(case_data.y_km_regular_2d, dtype=float), jet_values.shape)
    else:
        level_values = np.asarray(case_data.wind_speed["level"].values, dtype=float)
        jet_level_mask = level_values <= float(jet_top_pressure_hpa)
        if np.any(jet_level_mask):
            jet_values = full_jet_values[jet_level_mask, :, :]
            z_jet = z_values[jet_level_mask, :, :]
            x_jet = case_data.x_km_3d[jet_level_mask, :, :]
            y_jet = case_data.y_km_3d[jet_level_mask, :, :]
        else:
            jet_values = full_jet_values
            z_jet = z_values
            x_jet = case_data.x_km_3d
            y_jet = case_data.y_km_3d
    if jet_isomax is None:
        jet_isomax = float(np.nanmax(jet_values))
    jet_valid = np.isfinite(jet_values) & np.isfinite(z_jet) & np.isfinite(x_jet) & np.isfinite(y_jet)

    figure = go.Figure()

    x_min = float(np.nanmin(case_data.x_km_3d))
    x_max = float(np.nanmax(case_data.x_km_3d))
    y_min = float(np.nanmin(case_data.y_km_3d))
    y_max = float(np.nanmax(case_data.y_km_3d))
    z_min = 0.0
    z_max = float(np.nanmax(z_values))

    if show_cube_frame:
        cube_edges = [
            ((x_min, y_min, z_min), (x_max, y_min, z_min)),
            ((x_min, y_max, z_min), (x_max, y_max, z_min)),
            ((x_min, y_min, z_max), (x_max, y_min, z_max)),
            ((x_min, y_max, z_max), (x_max, y_max, z_max)),
            ((x_min, y_min, z_min), (x_min, y_max, z_min)),
            ((x_max, y_min, z_min), (x_max, y_max, z_min)),
            ((x_min, y_min, z_max), (x_min, y_max, z_max)),
            ((x_max, y_min, z_max), (x_max, y_max, z_max)),
            ((x_min, y_min, z_min), (x_min, y_min, z_max)),
            ((x_max, y_min, z_min), (x_max, y_min, z_max)),
            ((x_min, y_max, z_min), (x_min, y_max, z_max)),
            ((x_max, y_max, z_min), (x_max, y_max, z_max)),
        ]
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edge_z: list[float | None] = []
        for start_point, end_point in cube_edges:
            edge_x.extend([float(start_point[0]), float(end_point[0]), None])
            edge_y.extend([float(start_point[1]), float(end_point[1]), None])
            edge_z.extend([float(start_point[2]), float(end_point[2]), None])
        figure.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line={"color": "#475569", "width": 4},
                name="Cube frame",
                hoverinfo="skip",
            )
        )

    if case_data.terrain_m is not None and case_data.terrain_x_km is not None and case_data.terrain_y_km is not None:
        terrain_values_km = np.asarray(case_data.terrain_m.values, dtype=float) / 1000.0
        figure.add_trace(
            go.Surface(
                x=case_data.terrain_x_km,
                y=case_data.terrain_y_km,
                z=terrain_values_km,
                surfacecolor=np.asarray(case_data.terrain_m.values, dtype=float),
                colorscale="Earth",
                cmin=0.0,
                cmax=max(3000.0, float(np.nanmax(np.asarray(case_data.terrain_m.values, dtype=float)))),
                showscale=False,
                opacity=1.0,
                name="Terrain",
                hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>terrain=%{surfacecolor:.0f} m<extra></extra>",
            )
        )

    if show_jet_volume:
        figure.add_trace(
            go.Isosurface(
                x=x_jet[jet_valid],
                y=y_jet[jet_valid],
                z=z_jet[jet_valid],
                value=jet_values[jet_valid],
                isomin=float(jet_isomin),
                isomax=float(jet_isomax),
                surface_count=max(8, int(jet_surface_count)),
                opacity=float(jet_opacity),
                colorscale="Blues",
                caps={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
                colorbar={"title": "Wind [m s^-1]", "x": 1.02, "y": 0.82, "len": 0.30},
                name="Upper-level jet volume",
                hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.2f} km<br>wind=%{value:.1f} m s^-1<extra></extra>",
            )
        )

    if show_jet_points:
        point_mask = jet_valid & (jet_values >= float(jet_point_threshold))
        if np.any(point_mask):
            point_x = x_jet[point_mask]
            point_y = y_jet[point_mask]
            point_z = z_jet[point_mask]
            point_wind = jet_values[point_mask]
            if point_wind.size > int(max_jet_points):
                selection = np.linspace(0, point_wind.size - 1, int(max_jet_points), dtype=int)
                point_x = point_x[selection]
                point_y = point_y[selection]
                point_z = point_z[selection]
                point_wind = point_wind[selection]
            figure.add_trace(
                go.Scatter3d(
                    x=point_x,
                    y=point_y,
                    z=point_z,
                    mode="markers",
                    marker={
                        "size": float(jet_point_size),
                        "color": point_wind,
                        "colorscale": "Turbo",
                        "opacity": 0.78,
                        "colorbar": {"title": "Jet points [m s^-1]", "x": 1.02, "y": 0.82, "len": 0.18},
                    },
                    name="Jet core points",
                    hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.2f} km<br>wind=%{marker.color:.1f} m s^-1<extra></extra>",
                )
            )

    if (
        show_moisture_volume
        and case_data.z_levels_regular_km is not None
        and case_data.x_km_regular_2d is not None
        and case_data.y_km_regular_2d is not None
    ):
        if moisture_volume_mode == "relative_humidity" and case_data.relative_humidity_regular_volume is not None:
            moisture_values = np.asarray(case_data.relative_humidity_regular_volume, dtype=float)
            moisture_valid = np.isfinite(moisture_values)
            if require_ascent_for_rh_volume and case_data.omega_regular_volume is not None:
                moisture_valid &= np.asarray(case_data.omega_regular_volume, dtype=float) < 0.0
            moisture_valid &= moisture_values >= float(relative_humidity_min)
            if np.any(moisture_valid) and float(relative_humidity_max) > float(relative_humidity_min):
                x_moisture = np.broadcast_to(np.asarray(case_data.x_km_regular_2d, dtype=float), moisture_values.shape)
                y_moisture = np.broadcast_to(np.asarray(case_data.y_km_regular_2d, dtype=float), moisture_values.shape)
                z_moisture = np.broadcast_to(
                    np.asarray(case_data.z_levels_regular_km, dtype=float)[:, np.newaxis, np.newaxis],
                    moisture_values.shape,
                )
                figure.add_trace(
                    go.Isosurface(
                        x=x_moisture[moisture_valid],
                        y=y_moisture[moisture_valid],
                        z=z_moisture[moisture_valid],
                        value=moisture_values[moisture_valid],
                        isomin=float(relative_humidity_min),
                        isomax=float(relative_humidity_max),
                        surface_count=max(2, int(moisture_surface_count)),
                        opacity=float(moisture_opacity),
                        colorscale=[
                            [0.0, "#ccfbf1"],
                            [0.35, "#99f6e4"],
                            [0.70, "#2dd4bf"],
                            [1.0, "#0f766e"],
                        ],
                        caps={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
                        colorbar={"title": "RH fraction", "x": 1.02, "y": 0.56, "len": 0.16},
                        name="Humid ascent volume",
                        hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.2f} km<br>RH=%{value:.2f}<extra></extra>",
                    )
                )
        elif case_data.moisture_proxy_regular_volume is not None:
            moisture_values = np.asarray(case_data.moisture_proxy_regular_volume, dtype=float)
            moisture_valid = np.isfinite(moisture_values) & (moisture_values > 0.0)
            if np.any(moisture_valid):
                if moisture_isomin is None:
                    moisture_isomin = float(np.nanquantile(moisture_values[moisture_valid], float(moisture_quantile)))
                if moisture_isomax is None:
                    moisture_isomax = float(np.nanmax(moisture_values[moisture_valid]))
                if float(moisture_isomax) > float(moisture_isomin):
                    x_moisture = np.broadcast_to(np.asarray(case_data.x_km_regular_2d, dtype=float), moisture_values.shape)
                    y_moisture = np.broadcast_to(np.asarray(case_data.y_km_regular_2d, dtype=float), moisture_values.shape)
                    z_moisture = np.broadcast_to(
                        np.asarray(case_data.z_levels_regular_km, dtype=float)[:, np.newaxis, np.newaxis],
                        moisture_values.shape,
                    )
                    figure.add_trace(
                        go.Isosurface(
                            x=x_moisture[moisture_valid],
                            y=y_moisture[moisture_valid],
                            z=z_moisture[moisture_valid],
                            value=moisture_values[moisture_valid],
                            isomin=float(moisture_isomin),
                            isomax=float(moisture_isomax),
                            surface_count=max(3, int(moisture_surface_count)),
                            opacity=float(moisture_opacity),
                            colorscale=[
                                [0.0, "#d9f99d"],
                                [0.35, "#86efac"],
                                [0.70, "#22c55e"],
                                [1.0, "#166534"],
                            ],
                            caps={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
                            colorbar={"title": "Moist-ascent proxy", "x": 1.02, "y": 0.56, "len": 0.16},
                            name="Moist-ascent proxy volume",
                            hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.2f} km<br>q x (-omega)=%{value:.2f}<extra></extra>",
                        )
                    )

    if show_ascent_descent_points:
        omega_values = np.asarray(case_data.pressure_volume["vertical_velocity"].values, dtype=float)
        omega_valid = np.isfinite(omega_values) & np.isfinite(z_values)
        ascent_mag = -omega_values[omega_valid & (omega_values < 0.0)]
        descent_mag = omega_values[omega_valid & (omega_values > 0.0)]

        if ascent_mag.size > 0:
            ascent_threshold = float(np.nanquantile(ascent_mag, float(omega_point_quantile)))
            ascent_mask = omega_valid & (omega_values <= -ascent_threshold)
            if np.any(ascent_mask):
                ascent_x = case_data.x_km_3d[ascent_mask]
                ascent_y = case_data.y_km_3d[ascent_mask]
                ascent_z = z_values[ascent_mask]
                ascent_omega = omega_values[ascent_mask]
                if ascent_omega.size > int(max_omega_points):
                    selection = np.linspace(0, ascent_omega.size - 1, int(max_omega_points), dtype=int)
                    ascent_x = ascent_x[selection]
                    ascent_y = ascent_y[selection]
                    ascent_z = ascent_z[selection]
                    ascent_omega = ascent_omega[selection]
                figure.add_trace(
                    go.Scatter3d(
                        x=ascent_x,
                        y=ascent_y,
                        z=ascent_z,
                        mode="markers",
                        marker={"size": float(omega_point_size), "color": "#2563eb", "opacity": 0.72},
                        name="Strong ascent",
                        hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.2f} km<br>omega=%{customdata:.2f} Pa s^-1<extra></extra>",
                        customdata=ascent_omega,
                    )
                )

        if descent_mag.size > 0:
            descent_threshold = float(np.nanquantile(descent_mag, float(omega_point_quantile)))
            descent_mask = omega_valid & (omega_values >= descent_threshold)
            if np.any(descent_mask):
                descent_x = case_data.x_km_3d[descent_mask]
                descent_y = case_data.y_km_3d[descent_mask]
                descent_z = z_values[descent_mask]
                descent_omega = omega_values[descent_mask]
                if descent_omega.size > int(max_omega_points):
                    selection = np.linspace(0, descent_omega.size - 1, int(max_omega_points), dtype=int)
                    descent_x = descent_x[selection]
                    descent_y = descent_y[selection]
                    descent_z = descent_z[selection]
                    descent_omega = descent_omega[selection]
                figure.add_trace(
                    go.Scatter3d(
                        x=descent_x,
                        y=descent_y,
                        z=descent_z,
                        mode="markers",
                        marker={"size": float(omega_point_size), "color": "#f97316", "opacity": 0.72},
                        name="Strong descent",
                        hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.2f} km<br>omega=%{customdata:.2f} Pa s^-1<extra></extra>",
                        customdata=descent_omega,
                    )
                )

    if show_moisture_sheet:
        height_700 = np.asarray(case_data.geopotential_height_km.sel(level=700).values, dtype=float)
        figure.add_trace(
            go.Surface(
                x=case_data.x_km_3d[0],
                y=case_data.y_km_3d[0],
                z=height_700,
                surfacecolor=np.asarray(case_data.moisture_proxy_700.values, dtype=float),
                colorscale="BrBG",
                cmin=-2.5,
                cmax=2.5,
                opacity=0.58,
                showscale=True,
                colorbar={"title": "700 hPa q x (-omega)", "x": 1.02, "y": 0.46, "len": 0.22},
                name="700 hPa moisture proxy",
                hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.2f} km<br>q x (-omega)=%{surfacecolor:.2f}<extra></extra>",
            )
        )

    if show_convergence_floor:
        convergence_magnitude = -np.asarray(case_data.divergence_925_display.values, dtype=float)
        convergence_magnitude = np.where(convergence_magnitude > 0.0, convergence_magnitude, np.nan)
        finite_convergence = np.isfinite(convergence_magnitude)
        if np.any(finite_convergence):
            convergence_threshold = float(np.nanquantile(convergence_magnitude[finite_convergence], float(convergence_quantile)))
            convergence_display = np.where(convergence_magnitude >= convergence_threshold, convergence_magnitude, np.nan)
            if case_data.terrain_m is not None:
                terrain_on_volume = case_data.terrain_m.interp(
                    longitude=case_data.pressure_volume.longitude,
                    latitude=case_data.pressure_volume.latitude,
                    method="nearest",
                )
                convergence_z = np.asarray(terrain_on_volume.values, dtype=float) / 1000.0 + 0.04
            else:
                convergence_z = np.full_like(np.asarray(case_data.x_km_3d[0], dtype=float), 0.04)
            figure.add_trace(
                go.Surface(
                    x=case_data.x_km_3d[0],
                    y=case_data.y_km_3d[0],
                    z=convergence_z,
                    surfacecolor=convergence_display,
                    colorscale=[
                        [0.0, "#bfdbfe"],
                        [0.45, "#60a5fa"],
                        [0.8, "#2563eb"],
                        [1.0, "#1d4ed8"],
                    ],
                    cmin=float(convergence_threshold),
                    cmax=float(np.nanmax(convergence_magnitude)),
                    opacity=float(convergence_opacity),
                    showscale=True,
                    colorbar={"title": "925 hPa convergence", "x": 1.02, "y": 0.14, "len": 0.16},
                    name="Strong low-level convergence",
                    hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.2f} km<br>convergence=%{surfacecolor:.2f} [1e-5 s^-1]<extra></extra>",
                )
            )

    if show_divergence_sheet:
        height_925 = np.asarray(case_data.geopotential_height_km.sel(level=925).values, dtype=float)
        figure.add_trace(
            go.Surface(
                x=case_data.x_km_3d[0],
                y=case_data.y_km_3d[0],
                z=height_925,
                surfacecolor=np.asarray(case_data.divergence_925_display.values, dtype=float),
                colorscale="RdBu_r",
                cmin=-6.0,
                cmax=6.0,
                opacity=0.56,
                showscale=True,
                colorbar={"title": "925 hPa div", "x": 1.02, "y": 0.14, "len": 0.22},
                name="925 hPa divergence",
                hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.2f} km<br>div=%{surfacecolor:.2f} [1e-5 s^-1]<extra></extra>",
            )
        )

    if (
        show_slice_curtain
        and case_data.slice_x_km is not None
        and case_data.slice_y_km is not None
        and case_data.slice_z_km is not None
        and case_data.slice_omega is not None
    ):
        smoothed_slice_omega = _smooth_section(
            case_data.slice_omega.astype(float),
            vertical_window=int(slice_omega_smoothing_levels),
            horizontal_window=int(slice_omega_smoothing_points),
        )
        slice_omega_values = np.asarray(smoothed_slice_omega.values, dtype=float)
        finite_slice = np.isfinite(slice_omega_values)
        curtain_max_abs = float(slice_omega_max_abs) if slice_omega_max_abs is not None else (
            float(np.nanquantile(np.abs(slice_omega_values[finite_slice]), 0.97)) if np.any(finite_slice) else 0.4
        )
        curtain_max_abs = max(0.1, curtain_max_abs)
        figure.add_trace(
            go.Surface(
                x=case_data.slice_x_km,
                y=case_data.slice_y_km,
                z=np.asarray(case_data.slice_z_km.values, dtype=float),
                surfacecolor=slice_omega_values,
                colorscale="RdBu_r",
                cmin=-curtain_max_abs,
                cmax=curtain_max_abs,
                opacity=float(slice_omega_opacity),
                showscale=True,
                colorbar={"title": "Smoothed omega", "x": 1.02, "y": 0.36, "len": 0.16},
                name="Smoothed omega curtain",
                hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.2f} km<br>omega=%{surfacecolor:.2f} Pa s^-1<extra></extra>",
            )
        )
        if case_data.slice_terrain_km is not None and case_data.slice_lon is not None and case_data.slice_lat is not None:
            x_line_km, y_line_km = _local_xy_km_from_lonlat(
                case_data.slice_lon.values,
                case_data.slice_lat.values,
                center_lon=case_data.center_lon,
                center_lat=case_data.center_lat,
            )
            figure.add_trace(
                go.Scatter3d(
                    x=x_line_km,
                    y=y_line_km,
                    z=case_data.slice_terrain_km,
                    mode="lines",
                    line={"color": "#7c3aed", "width": 5},
                    name="Slice line",
                    hoverinfo="skip",
                )
            )
            figure.add_trace(
                go.Scatter3d(
                    x=[float(x_line_km[0])],
                    y=[float(y_line_km[0])],
                    z=[float(case_data.slice_terrain_km[0])],
                    mode="markers+text",
                    marker={"color": "#16a34a", "size": 6},
                    text=["Start"],
                    textposition="top center",
                    name="Slice start",
                    hovertemplate="Start<extra></extra>",
                )
            )
            figure.add_trace(
                go.Scatter3d(
                    x=[float(x_line_km[-1])],
                    y=[float(y_line_km[-1])],
                    z=[float(case_data.slice_terrain_km[-1])],
                    mode="markers+text",
                    marker={"color": "#dc2626", "size": 6},
                    text=["End"],
                    textposition="top center",
                    name="Slice end",
                    hovertemplate="End<extra></extra>",
                )
            )

    x_extent = max(1.0, max(abs(x_min), abs(x_max)))
    y_extent = max(1.0, max(abs(y_min), abs(y_max)))
    z_extent = max(3.0, z_max - z_min)
    if title is None:
        title = f"3-D case-study cube | {case_data.analysis_time:%Y-%m-%d %H:%M UTC}"

    z_aspect = min(2.5, max(0.55, float(vertical_exaggeration) * z_extent / max(y_extent, 1.0)))
    figure.update_layout(
        title=title,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
        scene={
            "xaxis": {
                "title": "x [km] relative to cube center",
                "backgroundcolor": "#f8fafc",
                "range": [x_min, x_max],
                "showspikes": False,
            },
            "yaxis": {
                "title": "y [km] relative to cube center",
                "backgroundcolor": "#f8fafc",
                "range": [y_min, y_max],
                "showspikes": False,
            },
            "zaxis": {
                "title": "z [km ASL]",
                "backgroundcolor": "#f8fafc",
                "range": [z_min, z_max],
                "showspikes": False,
            },
            "aspectmode": "manual",
            "aspectratio": {"x": max(1.0, x_extent / y_extent), "y": 1.0, "z": z_aspect},
            "camera": {
                "eye": {"x": 1.55, "y": -1.85, "z": 0.95},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
        },
    )
    return figure
