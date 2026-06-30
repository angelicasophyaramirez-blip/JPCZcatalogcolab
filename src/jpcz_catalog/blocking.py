"""Helpers for clearer East Siberian blocking / Yamazaki-style diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import BoundingBox, PACIFIC_EAST_OF_JAPAN_BOX
from .diagnostics import (
    compute_geopotential_height_field,
    compute_wind_speed_field,
    load_snapshot,
)
from .eawm import area_weighted_mean, load_or_update_monthly_field_climatology, subset_box

YAMAZAKI_DOMAIN = BoundingBox(
    lon_min=70.0,
    lon_max=200.0,
    lat_min=20.0,
    lat_max=90.0,
)

EAST_SIBERIAN_BLOCKING_BOX = BoundingBox(
    lon_min=120.0,
    lon_max=170.0,
    lat_min=55.0,
    lat_max=75.0,
)

SIBERIAN_HIGH_SEARCH_BOX = BoundingBox(
    lon_min=80.0,
    lon_max=160.0,
    lat_min=45.0,
    lat_max=80.0,
)

ALEUTIAN_LOW_SEARCH_BOX = BoundingBox(
    lon_min=160.0,
    lon_max=200.0,
    lat_min=35.0,
    lat_max=70.0,
)

EAST_OF_JAPAN_TROUGH_BOX = PACIFIC_EAST_OF_JAPAN_BOX

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

BLOCKING_CLIMATOLOGY_FILE_NAMES = {
    "mslp_hpa": "mslp_blocking_story_monthly_climatology.nc",
    "z500_gpm": "z500_blocking_story_monthly_climatology.nc",
    "t850_k": "t850_blocking_story_monthly_climatology.nc",
}


def rounded_contour_levels(
    field: xr.DataArray,
    *,
    step: float,
    symmetric: bool = False,
) -> np.ndarray:
    """Return rounded contour levels for one field."""
    values = np.asarray(field.values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.asarray([], dtype=float)

    if symmetric:
        max_abs = max(float(np.nanmax(np.abs(finite))), float(step))
        bound = step * np.ceil(max_abs / step)
        return np.arange(-bound, bound + 0.5 * step, step, dtype=float)

    lower = step * np.floor(float(np.nanmin(finite)) / step)
    upper = step * np.ceil(float(np.nanmax(finite)) / step)
    if np.isclose(lower, upper):
        lower -= 2.0 * step
        upper += 2.0 * step
    return np.arange(lower, upper + 0.5 * step, step, dtype=float)


def positive_contour_levels(field: xr.DataArray, *, step: float) -> np.ndarray:
    """Return positive-only rounded contour levels."""
    values = np.asarray(field.values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.asarray([], dtype=float)
    upper = max(float(step), step * np.ceil(float(np.nanmax(finite)) / step))
    return np.arange(0.0, upper + 0.5 * step, step, dtype=float)


def find_box_extremum(
    field: xr.DataArray,
    box: BoundingBox,
    *,
    mode: str,
) -> dict[str, float]:
    """Return the max or min location inside one box."""
    subset = subset_box(field, box)
    values = np.asarray(subset.values, dtype=float)
    finite_mask = np.isfinite(values)
    if values.size == 0 or not finite_mask.any():
        return {
            "center_lon_degE": np.nan,
            "center_lat_degN": np.nan,
            "extremum_value": np.nan,
            "box_mean": np.nan,
        }

    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported extremum mode: {mode!r}")

    arg_idx = np.unravel_index(
        np.nanargmax(values) if mode == "max" else np.nanargmin(values),
        values.shape,
    )
    lat_idx, lon_idx = int(arg_idx[0]), int(arg_idx[1])
    return {
        "center_lon_degE": float(subset.longitude.values[lon_idx]),
        "center_lat_degN": float(subset.latitude.values[lat_idx]),
        "extremum_value": float(values[lat_idx, lon_idx]),
        "box_mean": float(np.nanmean(values)),
    }


def estimate_ridge_axis(
    field: xr.DataArray,
    box: BoundingBox = EAST_SIBERIAN_BLOCKING_BOX,
    *,
    longitude_stride: int = 3,
    minimum_value: float = 0.0,
) -> pd.DataFrame:
    """Estimate a simple ridge-axis trace by following the latitude of the local maximum."""
    subset = subset_box(field, box)
    if subset.size == 0:
        return pd.DataFrame(columns=["longitude", "latitude", "value"])

    rows: list[dict[str, float]] = []
    longitudes = subset.longitude.values[:: max(1, int(longitude_stride))]
    for longitude in np.asarray(longitudes, dtype=float):
        column = subset.sel(longitude=float(longitude))
        values = np.asarray(column.values, dtype=float)
        if values.size == 0 or not np.isfinite(values).any():
            continue
        lat_idx = int(np.nanargmax(values))
        value = float(values[lat_idx])
        if not np.isfinite(value) or value < float(minimum_value):
            continue
        rows.append(
            {
                "longitude": float(longitude),
                "latitude": float(column.latitude.values[lat_idx]),
                "value": value,
            }
        )

    return pd.DataFrame(rows)


def _pressure_thickness_hpa(level_values: Sequence[float]) -> np.ndarray:
    """Approximate pressure-layer thickness from full levels."""
    levels = np.asarray(level_values, dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("Pressure levels must be a non-empty 1-D array.")
    if levels.size == 1:
        return np.asarray([0.0], dtype=float)

    edges = np.empty(levels.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (levels[:-1] + levels[1:])
    edges[0] = levels[0] + 0.5 * (levels[0] - levels[1])
    edges[-1] = levels[-1] - 0.5 * (levels[-2] - levels[-1])
    return np.abs(np.diff(edges))


def compute_cold_air_mass_flux(
    snapshot: xr.Dataset,
    *,
    theta_threshold_k: float = 280.0,
    temperature_name: str = "temperature",
    u_name: str = "u_component_of_wind",
    v_name: str = "v_component_of_wind",
) -> dict[str, xr.DataArray]:
    """Compute an Iwasaki-style cold-air transport proxy below one theta threshold."""
    pressure_hpa = xr.DataArray(
        np.asarray(snapshot["level"].values, dtype=float),
        coords={"level": snapshot["level"]},
        dims=("level",),
    )
    theta = snapshot[temperature_name] * (1000.0 / pressure_hpa) ** 0.286
    theta = theta.rename("potential_temperature")

    cold_mask = xr.where(theta <= float(theta_threshold_k), 1.0, 0.0)
    thickness_hpa = xr.DataArray(
        _pressure_thickness_hpa(snapshot["level"].values),
        coords={"level": snapshot["level"]},
        dims=("level",),
    )

    u_flux = (snapshot[u_name].astype(float).where(cold_mask > 0.0) * thickness_hpa).sum(
        "level",
        skipna=True,
    )
    v_flux = (snapshot[v_name].astype(float).where(cold_mask > 0.0) * thickness_hpa).sum(
        "level",
        skipna=True,
    )
    cold_depth = (cold_mask.where(np.isfinite(snapshot[temperature_name])) * thickness_hpa).sum(
        "level",
        skipna=True,
    )
    flux_magnitude = ((u_flux**2 + v_flux**2) ** 0.5).rename("cold_air_flux_magnitude_hpa_ms")

    u_flux = u_flux.rename("cold_air_flux_u_hpa_ms")
    v_flux = v_flux.rename("cold_air_flux_v_hpa_ms")
    cold_depth = cold_depth.rename("cold_air_depth_hpa")

    for field in (u_flux, v_flux, flux_magnitude):
        field.attrs["units"] = "hPa m s^-1"
        field.attrs["theta_threshold_k"] = float(theta_threshold_k)
    cold_depth.attrs["units"] = "hPa"
    cold_depth.attrs["theta_threshold_k"] = float(theta_threshold_k)

    return {
        "u": u_flux,
        "v": v_flux,
        "magnitude": flux_magnitude,
        "depth": cold_depth,
    }


def _load_pressure_level_volume(
    ds: xr.Dataset,
    analysis_time: pd.Timestamp | str,
    *,
    variables: Sequence[str],
    domain: BoundingBox = YAMAZAKI_DOMAIN,
    level_min_hpa: float = 500.0,
    level_max_hpa: float = 1000.0,
) -> xr.Dataset:
    """Load one multi-level pressure-coordinate volume inside a level range."""
    snapshot = ds[list(variables)].sel(
        time=pd.Timestamp(analysis_time),
        longitude=slice(float(domain.lon_min), float(domain.lon_max)),
        latitude=slice(float(domain.lat_max), float(domain.lat_min)),
    )

    if "time" in snapshot.dims:
        snapshot = snapshot.squeeze("time", drop=True)

    level_mask = (
        (snapshot["level"].astype(float) >= float(level_min_hpa))
        & (snapshot["level"].astype(float) <= float(level_max_hpa))
    )
    snapshot = snapshot.sel(level=snapshot["level"].where(level_mask, drop=True))
    return snapshot.load()


def load_or_update_blocking_climatology_bundle(
    cache_dir: str | Path,
    *,
    years: Iterable[int],
    months: Iterable[int] = (2,),
    current_ds: xr.Dataset | None = None,
    chunks: dict[str, int] | None = None,
    storage_options: dict[str, str] | None = None,
) -> tuple[dict[str, xr.DataArray], xr.Dataset | None]:
    """Load or compute the monthly climatologies needed by the blocking notebook."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    runtime_ds = current_ds
    bundle: dict[str, xr.DataArray] = {}

    bundle["mslp_hpa"], runtime_ds = load_or_update_monthly_field_climatology(
        cache_dir / BLOCKING_CLIMATOLOGY_FILE_NAMES["mslp_hpa"],
        years=years,
        months=months,
        domain=YAMAZAKI_DOMAIN,
        variables=["mean_sea_level_pressure"],
        level=None,
        field_getter=lambda ds: (ds["mean_sea_level_pressure"] / 100.0).rename("mslp_hpa"),
        output_name="monthly_mslp_hpa_climatology",
        current_ds=runtime_ds,
        chunks=chunks,
        storage_options=storage_options,
    )
    bundle["z500_gpm"], runtime_ds = load_or_update_monthly_field_climatology(
        cache_dir / BLOCKING_CLIMATOLOGY_FILE_NAMES["z500_gpm"],
        years=years,
        months=months,
        domain=YAMAZAKI_DOMAIN,
        variables=["geopotential"],
        level=500,
        field_getter=lambda ds: compute_geopotential_height_field(ds).rename("z500_gpm"),
        output_name="monthly_z500_gpm_climatology",
        current_ds=runtime_ds,
        chunks=chunks,
        storage_options=storage_options,
    )
    bundle["t850_k"], runtime_ds = load_or_update_monthly_field_climatology(
        cache_dir / BLOCKING_CLIMATOLOGY_FILE_NAMES["t850_k"],
        years=years,
        months=months,
        domain=YAMAZAKI_DOMAIN,
        variables=["temperature"],
        level=850,
        field_getter=lambda ds: ds["temperature"].astype(float).rename("t850_k"),
        output_name="monthly_t850_k_climatology",
        current_ds=runtime_ds,
        chunks=chunks,
        storage_options=storage_options,
    )
    return bundle, runtime_ds


def _finalize_story_bundle(
    *,
    analysis_label: str,
    analysis_time: pd.Timestamp | None,
    theta_threshold_k: float,
    mslp_hpa: xr.DataArray,
    mslp_anomaly_hpa: xr.DataArray,
    z500_gpm: xr.DataArray,
    z500_anomaly_gpm: xr.DataArray,
    u300_ms: xr.DataArray,
    v300_ms: xr.DataArray,
    wind_speed_300_ms: xr.DataArray,
    u850_ms: xr.DataArray,
    v850_ms: xr.DataArray,
    t850_k: xr.DataArray,
    t850_anomaly_k: xr.DataArray,
    cold_air_flux_u_hpa_ms: xr.DataArray,
    cold_air_flux_v_hpa_ms: xr.DataArray,
    cold_air_flux_magnitude_hpa_ms: xr.DataArray,
    cold_air_depth_hpa: xr.DataArray,
) -> dict[str, object]:
    """Attach derived labels and centers to one bundle of blocking diagnostics."""
    high_center = find_box_extremum(mslp_anomaly_hpa, SIBERIAN_HIGH_SEARCH_BOX, mode="max")
    low_center = find_box_extremum(mslp_anomaly_hpa, ALEUTIAN_LOW_SEARCH_BOX, mode="min")
    ridge_axis = estimate_ridge_axis(z500_anomaly_gpm)

    return {
        "analysis_label": analysis_label,
        "analysis_time": analysis_time,
        "theta_threshold_k": float(theta_threshold_k),
        "mslp_hpa": mslp_hpa,
        "mslp_anomaly_hpa": mslp_anomaly_hpa,
        "z500_gpm": z500_gpm,
        "z500_anomaly_gpm": z500_anomaly_gpm,
        "u300_ms": u300_ms,
        "v300_ms": v300_ms,
        "wind_speed_300_ms": wind_speed_300_ms,
        "u850_ms": u850_ms,
        "v850_ms": v850_ms,
        "t850_k": t850_k,
        "t850_anomaly_k": t850_anomaly_k,
        "cold_air_flux_u_hpa_ms": cold_air_flux_u_hpa_ms,
        "cold_air_flux_v_hpa_ms": cold_air_flux_v_hpa_ms,
        "cold_air_flux_magnitude_hpa_ms": cold_air_flux_magnitude_hpa_ms,
        "cold_air_depth_hpa": cold_air_depth_hpa,
        "siberian_high_center": high_center,
        "aleutian_low_center": low_center,
        "ridge_axis_df": ridge_axis,
    }


def build_blocking_snapshot_bundle(
    ds: xr.Dataset,
    analysis_time: pd.Timestamp | str,
    climatology_bundle: dict[str, xr.DataArray],
    *,
    theta_threshold_k: float = 280.0,
) -> dict[str, object]:
    """Build one Yamazaki-style diagnostic bundle for one analysis time."""
    analysis_time = pd.Timestamp(analysis_time)

    surface_snapshot = load_snapshot(
        ds,
        analysis_time,
        variables=["mean_sea_level_pressure"],
        domain=YAMAZAKI_DOMAIN,
        level=None,
    )
    snapshot_500 = load_snapshot(
        ds,
        analysis_time,
        variables=["geopotential"],
        domain=YAMAZAKI_DOMAIN,
        level=500,
    )
    snapshot_300 = load_snapshot(
        ds,
        analysis_time,
        variables=["u_component_of_wind", "v_component_of_wind"],
        domain=YAMAZAKI_DOMAIN,
        level=300,
    )
    snapshot_850 = load_snapshot(
        ds,
        analysis_time,
        variables=["u_component_of_wind", "v_component_of_wind", "temperature"],
        domain=YAMAZAKI_DOMAIN,
        level=850,
    )
    cold_volume = _load_pressure_level_volume(
        ds,
        analysis_time,
        variables=["u_component_of_wind", "v_component_of_wind", "temperature"],
        domain=YAMAZAKI_DOMAIN,
        level_min_hpa=500.0,
        level_max_hpa=1000.0,
    )

    month = int(analysis_time.month)
    mslp_hpa = (surface_snapshot["mean_sea_level_pressure"] / 100.0).rename("mslp_hpa")
    mslp_anomaly_hpa = (
        mslp_hpa - climatology_bundle["mslp_hpa"].sel(month=month)
    ).rename("mslp_anomaly_hpa")
    z500_gpm = compute_geopotential_height_field(snapshot_500).rename("z500_gpm")
    z500_anomaly_gpm = (
        z500_gpm - climatology_bundle["z500_gpm"].sel(month=month)
    ).rename("z500_anomaly_gpm")
    u300_ms = snapshot_300["u_component_of_wind"].astype(float).rename("u300_ms")
    v300_ms = snapshot_300["v_component_of_wind"].astype(float).rename("v300_ms")
    wind_speed_300_ms = compute_wind_speed_field(snapshot_300).rename("wind_speed_300_ms")
    u850_ms = snapshot_850["u_component_of_wind"].astype(float).rename("u850_ms")
    v850_ms = snapshot_850["v_component_of_wind"].astype(float).rename("v850_ms")
    t850_k = snapshot_850["temperature"].astype(float).rename("t850_k")
    t850_anomaly_k = (
        t850_k - climatology_bundle["t850_k"].sel(month=month)
    ).rename("t850_anomaly_k")
    cold_air_flux = compute_cold_air_mass_flux(cold_volume, theta_threshold_k=theta_threshold_k)

    return _finalize_story_bundle(
        analysis_label=f"{analysis_time:%Y-%m-%d %H UTC}",
        analysis_time=analysis_time,
        theta_threshold_k=theta_threshold_k,
        mslp_hpa=mslp_hpa,
        mslp_anomaly_hpa=mslp_anomaly_hpa,
        z500_gpm=z500_gpm,
        z500_anomaly_gpm=z500_anomaly_gpm,
        u300_ms=u300_ms,
        v300_ms=v300_ms,
        wind_speed_300_ms=wind_speed_300_ms,
        u850_ms=u850_ms,
        v850_ms=v850_ms,
        t850_k=t850_k,
        t850_anomaly_k=t850_anomaly_k,
        cold_air_flux_u_hpa_ms=cold_air_flux["u"],
        cold_air_flux_v_hpa_ms=cold_air_flux["v"],
        cold_air_flux_magnitude_hpa_ms=cold_air_flux["magnitude"],
        cold_air_depth_hpa=cold_air_flux["depth"],
    )


def average_story_bundles(
    bundles: Sequence[dict[str, object]],
    *,
    analysis_label: str,
) -> dict[str, object]:
    """Average a sequence of snapshot bundles into one mean bundle."""
    if not bundles:
        raise ValueError("At least one bundle is required for averaging.")

    field_names = [
        "mslp_hpa",
        "mslp_anomaly_hpa",
        "z500_gpm",
        "z500_anomaly_gpm",
        "u300_ms",
        "v300_ms",
        "wind_speed_300_ms",
        "u850_ms",
        "v850_ms",
        "t850_k",
        "t850_anomaly_k",
        "cold_air_flux_u_hpa_ms",
        "cold_air_flux_v_hpa_ms",
        "cold_air_flux_magnitude_hpa_ms",
        "cold_air_depth_hpa",
    ]
    mean_fields: dict[str, xr.DataArray] = {}
    for field_name in field_names:
        stacked = xr.concat(
            [bundle[field_name].expand_dims(composite_member=[member_idx]) for member_idx, bundle in enumerate(bundles)],
            dim="composite_member",
        )
        mean_fields[field_name] = stacked.mean("composite_member", skipna=True)

    theta_threshold_k = float(bundles[0]["theta_threshold_k"])
    return _finalize_story_bundle(
        analysis_label=analysis_label,
        analysis_time=None,
        theta_threshold_k=theta_threshold_k,
        mslp_hpa=mean_fields["mslp_hpa"],
        mslp_anomaly_hpa=mean_fields["mslp_anomaly_hpa"],
        z500_gpm=mean_fields["z500_gpm"],
        z500_anomaly_gpm=mean_fields["z500_anomaly_gpm"],
        u300_ms=mean_fields["u300_ms"],
        v300_ms=mean_fields["v300_ms"],
        wind_speed_300_ms=mean_fields["wind_speed_300_ms"],
        u850_ms=mean_fields["u850_ms"],
        v850_ms=mean_fields["v850_ms"],
        t850_k=mean_fields["t850_k"],
        t850_anomaly_k=mean_fields["t850_anomaly_k"],
        cold_air_flux_u_hpa_ms=mean_fields["cold_air_flux_u_hpa_ms"],
        cold_air_flux_v_hpa_ms=mean_fields["cold_air_flux_v_hpa_ms"],
        cold_air_flux_magnitude_hpa_ms=mean_fields["cold_air_flux_magnitude_hpa_ms"],
        cold_air_depth_hpa=mean_fields["cold_air_depth_hpa"],
    )


def build_blocking_story_lines(bundle: dict[str, object]) -> list[str]:
    """Translate one bundle into a short sequential synoptic interpretation."""
    esb_z500 = area_weighted_mean(bundle["z500_anomaly_gpm"], EAST_SIBERIAN_BLOCKING_BOX)
    east_japan_z500 = area_weighted_mean(bundle["z500_anomaly_gpm"], EAST_OF_JAPAN_TROUGH_BOX)
    low_level_v850 = area_weighted_mean(bundle["v850_ms"], LOW_LEVEL_NORTH_FLOW_BOX)
    low_level_t850 = area_weighted_mean(bundle["t850_anomaly_k"], LOW_LEVEL_NORTH_FLOW_BOX)
    camf_depth = area_weighted_mean(bundle["cold_air_depth_hpa"], EAST_SIBERIAN_BLOCKING_BOX)

    lines: list[str] = []
    if np.isfinite(esb_z500):
        if esb_z500 >= 60.0:
            lines.append("A strong positive 500 hPa height anomaly ridge occupies the East Siberian blocking box.")
        elif esb_z500 > 0.0:
            lines.append("A weaker positive 500 hPa height anomaly ridge still appears inside the East Siberian blocking box.")
        else:
            lines.append("The East Siberian blocking box does not show a positive 500 hPa ridge anomaly.")
    if np.isfinite(east_japan_z500):
        if east_japan_z500 <= -20.0:
            lines.append("Downstream, east of Japan, the 500 hPa anomaly is negative, consistent with a trough response.")
        elif east_japan_z500 >= 20.0:
            lines.append("East of Japan the 500 hPa anomaly is positive rather than trough-like.")
        else:
            lines.append("East of Japan the 500 hPa anomaly is weak or mixed.")
    if np.isfinite(low_level_v850):
        if low_level_v850 < 0.0:
            lines.append("The 850 hPa meridional flow in the Japan-sector box is northerly, supporting cold-air export toward East Asia.")
        else:
            lines.append("The 850 hPa meridional flow in the Japan-sector box is not northerly on average.")
    if np.isfinite(low_level_t850):
        if low_level_t850 <= -2.0:
            lines.append("The lower troposphere is colder than the February climatology in the Japan-sector box.")
        elif low_level_t850 >= 2.0:
            lines.append("The lower troposphere is warmer than the February climatology in the Japan-sector box.")
    if np.isfinite(camf_depth):
        lines.append(
            f"The mean pressure-depth of air colder than {float(bundle['theta_threshold_k']):.0f} K inside the blocking box is about {camf_depth:.0f} hPa."
        )
    return lines


def build_blocking_summary_df(bundle: dict[str, object]) -> pd.DataFrame:
    """Summarize the main blocking diagnostics in a compact table."""
    high_center = bundle["siberian_high_center"]
    low_center = bundle["aleutian_low_center"]
    rows = [
        {
            "metric": "East Siberian blocking-box mean z500 anomaly",
            "value": round(area_weighted_mean(bundle["z500_anomaly_gpm"], EAST_SIBERIAN_BLOCKING_BOX), 2),
            "units": "gpm",
            "interpretation": "Positive values support a blocking ridge over East Siberia / Okhotsk.",
        },
        {
            "metric": "East-of-Japan mean z500 anomaly",
            "value": round(area_weighted_mean(bundle["z500_anomaly_gpm"], EAST_OF_JAPAN_TROUGH_BOX), 2),
            "units": "gpm",
            "interpretation": "Negative values support the downstream trough side of the blocking pattern.",
        },
        {
            "metric": "Jet-entrance mean 300 hPa wind speed",
            "value": round(area_weighted_mean(bundle["wind_speed_300_ms"], JET_ENTRANCE_REGION_BOX), 2),
            "units": "m s^-1",
            "interpretation": "Actual upper-level circulation strength near the western Pacific jet entrance.",
        },
        {
            "metric": "Japan-sector mean 850 hPa meridional wind",
            "value": round(area_weighted_mean(bundle["v850_ms"], LOW_LEVEL_NORTH_FLOW_BOX), 2),
            "units": "m s^-1",
            "interpretation": "Negative values indicate northerly low-level cold-air flow.",
        },
        {
            "metric": "Japan-sector mean 850 hPa temperature anomaly",
            "value": round(area_weighted_mean(bundle["t850_anomaly_k"], LOW_LEVEL_NORTH_FLOW_BOX), 2),
            "units": "K",
            "interpretation": "Negative values indicate colder-than-climatology lower-tropospheric air.",
        },
        {
            "metric": "Siberian High search-box max SLP anomaly",
            "value": round(float(high_center["extremum_value"]), 2),
            "units": "hPa",
            "interpretation": f"H center near {high_center['center_lon_degE']:.1f}E, {high_center['center_lat_degN']:.1f}N.",
        },
        {
            "metric": "Aleutian Low search-box min SLP anomaly",
            "value": round(float(low_center["extremum_value"]), 2),
            "units": "hPa",
            "interpretation": f"L center near {low_center['center_lon_degE']:.1f}E, {low_center['center_lat_degN']:.1f}N.",
        },
        {
            "metric": f"Blocking-box mean cold-air depth below {float(bundle['theta_threshold_k']):.0f} K",
            "value": round(area_weighted_mean(bundle["cold_air_depth_hpa"], EAST_SIBERIAN_BLOCKING_BOX), 2),
            "units": "hPa",
            "interpretation": "Larger values mean a deeper reservoir of sub-threshold cold air.",
        },
        {
            "metric": f"Japan-sector mean cold-air flux magnitude below {float(bundle['theta_threshold_k']):.0f} K",
            "value": round(area_weighted_mean(bundle["cold_air_flux_magnitude_hpa_ms"], LOW_LEVEL_NORTH_FLOW_BOX), 2),
            "units": "hPa m s^-1",
            "interpretation": "Larger values indicate stronger threshold-based cold-air transport toward East Asia / Japan.",
        },
    ]
    return pd.DataFrame(rows)


def build_blocking_compact_row(bundle: dict[str, object]) -> dict[str, object]:
    """Return one compact row for time-evolution tables."""
    high_center = bundle["siberian_high_center"]
    low_center = bundle["aleutian_low_center"]
    return {
        "analysis_label": bundle["analysis_label"],
        "esb_z500_mean_gpm": round(area_weighted_mean(bundle["z500_anomaly_gpm"], EAST_SIBERIAN_BLOCKING_BOX), 2),
        "east_japan_z500_mean_gpm": round(area_weighted_mean(bundle["z500_anomaly_gpm"], EAST_OF_JAPAN_TROUGH_BOX), 2),
        "jet_entrance_wind_speed_300_ms": round(area_weighted_mean(bundle["wind_speed_300_ms"], JET_ENTRANCE_REGION_BOX), 2),
        "v850_japan_box_ms": round(area_weighted_mean(bundle["v850_ms"], LOW_LEVEL_NORTH_FLOW_BOX), 2),
        "t850_anomaly_japan_box_k": round(area_weighted_mean(bundle["t850_anomaly_k"], LOW_LEVEL_NORTH_FLOW_BOX), 2),
        "cold_air_depth_esb_box_hpa": round(area_weighted_mean(bundle["cold_air_depth_hpa"], EAST_SIBERIAN_BLOCKING_BOX), 2),
        "high_center_lon_degE": round(float(high_center["center_lon_degE"]), 2),
        "high_center_lat_degN": round(float(high_center["center_lat_degN"]), 2),
        "high_max_slp_anomaly_hpa": round(float(high_center["extremum_value"]), 2),
        "low_center_lon_degE": round(float(low_center["center_lon_degE"]), 2),
        "low_center_lat_degN": round(float(low_center["center_lat_degN"]), 2),
        "low_min_slp_anomaly_hpa": round(float(low_center["extremum_value"]), 2),
    }


def configure_map_axis(ax, domain: BoundingBox, *, title: str):
    """Apply consistent map formatting."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.set_extent(
        [float(domain.lon_min), float(domain.lon_max), float(domain.lat_min), float(domain.lat_max)],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="50m", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="dimgray")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.45, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7.5}
    gl.ylabel_style = {"size": 7.5}
    ax.set_title(title, fontsize=10.0)


def draw_box(
    ax,
    box: BoundingBox,
    *,
    edgecolor: str,
    label: str | None = None,
    linestyle: str = "-",
    linewidth: float = 1.4,
):
    """Draw one labeled latitude/longitude box."""
    import cartopy.crs as ccrs
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (float(box.lon_min), float(box.lat_min)),
        float(box.lon_max - box.lon_min),
        float(box.lat_max - box.lat_min),
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            float(box.lon_min) + 1.2,
            float(box.lat_max) - 1.5,
            label,
            fontsize=7.5,
            color=edgecolor,
            weight="bold",
            transform=ccrs.PlateCarree(),
            zorder=11,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.3},
        )


def plot_high_low_center(
    ax,
    center: dict[str, float],
    *,
    label: str,
    color: str,
    transform=None,
):
    """Mark one H/L center with a short label and anomaly value."""
    import cartopy.crs as ccrs

    if transform is None:
        transform = ccrs.PlateCarree()
    lon = float(center["center_lon_degE"])
    lat = float(center["center_lat_degN"])
    value = float(center["extremum_value"])
    if not np.isfinite(lon) or not np.isfinite(lat):
        return
    ax.text(
        lon,
        lat,
        label,
        fontsize=14,
        weight="bold",
        color=color,
        ha="center",
        va="center",
        transform=transform,
        zorder=13,
    )
    if np.isfinite(value):
        ax.text(
            lon,
            lat - 3.0,
            f"{value:.0f}",
            fontsize=7.8,
            color=color,
            ha="center",
            va="top",
            transform=transform,
            zorder=13,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 0.6},
        )


def add_ridge_axis_annotation(
    ax,
    ridge_axis_df: pd.DataFrame,
    *,
    color: str = "#7c3aed",
):
    """Draw a simple ridge-axis polyline."""
    import cartopy.crs as ccrs

    if ridge_axis_df.empty:
        return
    ax.plot(
        ridge_axis_df["longitude"].values,
        ridge_axis_df["latitude"].values,
        color=color,
        linewidth=2.1,
        linestyle="--",
        transform=ccrs.PlateCarree(),
        zorder=12,
    )
    lead = ridge_axis_df.iloc[len(ridge_axis_df) // 2]
    ax.text(
        float(lead["longitude"]) + 2.0,
        float(lead["latitude"]) + 1.0,
        "ridge axis",
        fontsize=7.8,
        color=color,
        transform=ccrs.PlateCarree(),
        zorder=13,
        bbox={"facecolor": "white", "alpha": 0.68, "edgecolor": "none", "pad": 0.8},
    )


def plot_upper_blocking_panel(ax, bundle: dict[str, object]):
    """Plot the upper-level blocking ridge panel."""
    import cartopy.crs as ccrs

    fill = ax.contourf(
        bundle["z500_anomaly_gpm"].longitude,
        bundle["z500_anomaly_gpm"].latitude,
        bundle["z500_anomaly_gpm"],
        levels=rounded_contour_levels(bundle["z500_anomaly_gpm"], step=30.0, symmetric=True),
        cmap="RdBu_r",
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    configure_map_axis(ax, YAMAZAKI_DOMAIN, title=f"500 hPa anomaly + height contours\n{bundle['analysis_label']}")
    contours = ax.contour(
        bundle["z500_gpm"].longitude,
        bundle["z500_gpm"].latitude,
        bundle["z500_gpm"],
        levels=rounded_contour_levels(bundle["z500_gpm"], step=60.0),
        colors="#2f2f2f",
        linewidths=0.82,
        transform=ccrs.PlateCarree(),
    )
    if len(contours.levels) > 0:
        ax.clabel(contours, fontsize=6.8, inline=True, fmt="%d")
    draw_box(ax, EAST_SIBERIAN_BLOCKING_BOX, edgecolor="#7c3aed", label="ESB box")
    draw_box(ax, EAST_OF_JAPAN_TROUGH_BOX, edgecolor="#0f766e", label="East of Japan")
    add_ridge_axis_annotation(ax, bundle["ridge_axis_df"])
    return fill


def plot_jet_circulation_panel(ax, bundle: dict[str, object]):
    """Plot the upper-level circulation panel with actual 300 hPa winds."""
    import cartopy.crs as ccrs

    fill = ax.contourf(
        bundle["wind_speed_300_ms"].longitude,
        bundle["wind_speed_300_ms"].latitude,
        bundle["wind_speed_300_ms"],
        levels=rounded_contour_levels(bundle["wind_speed_300_ms"], step=10.0),
        cmap="Blues",
        extend="max",
        alpha=0.88,
        transform=ccrs.PlateCarree(),
    )
    configure_map_axis(ax, YAMAZAKI_DOMAIN, title=f"300 hPa circulation vectors\n{bundle['analysis_label']}")
    quiver = ax.quiver(
        bundle["u300_ms"].longitude.values[::6],
        bundle["u300_ms"].latitude.values[::6],
        bundle["u300_ms"].values[::6, ::6],
        bundle["v300_ms"].values[::6, ::6],
        color="#111827",
        scale=420.0,
        width=0.0020,
        transform=ccrs.PlateCarree(),
        zorder=11,
    )
    ax.quiverkey(quiver, 0.90, -0.07, 20, "20 m s$^{-1}$", labelpos="E")
    draw_box(ax, JET_ENTRANCE_REGION_BOX, edgecolor="#d97706", label="Jet entrance")
    return fill


def plot_cold_air_flux_panel(ax, bundle: dict[str, object]):
    """Plot the threshold-based cold-air transport panel."""
    import cartopy.crs as ccrs

    fill = ax.contourf(
        bundle["cold_air_flux_magnitude_hpa_ms"].longitude,
        bundle["cold_air_flux_magnitude_hpa_ms"].latitude,
        bundle["cold_air_flux_magnitude_hpa_ms"],
        levels=positive_contour_levels(bundle["cold_air_flux_magnitude_hpa_ms"], step=100.0),
        cmap="YlGnBu",
        extend="max",
        transform=ccrs.PlateCarree(),
    )
    configure_map_axis(
        ax,
        YAMAZAKI_DOMAIN,
        title=(
            f"Cold-air mass flux below {float(bundle['theta_threshold_k']):.0f} K\n"
            f"{bundle['analysis_label']}"
        ),
    )
    quiver = ax.quiver(
        bundle["cold_air_flux_u_hpa_ms"].longitude.values[::6],
        bundle["cold_air_flux_u_hpa_ms"].latitude.values[::6],
        bundle["cold_air_flux_u_hpa_ms"].values[::6, ::6],
        bundle["cold_air_flux_v_hpa_ms"].values[::6, ::6],
        color="#111827",
        scale=6000.0,
        width=0.0018,
        transform=ccrs.PlateCarree(),
        zorder=11,
    )
    ax.quiverkey(quiver, 0.90, -0.07, 300, "300 hPa m s$^{-1}$", labelpos="E")
    draw_box(ax, EAST_SIBERIAN_BLOCKING_BOX, edgecolor="#7c3aed", label="ESB box")
    return fill


def plot_surface_pressure_panel(ax, bundle: dict[str, object]):
    """Plot the lower-level pressure-anomaly panel with explicit H/L centers."""
    import cartopy.crs as ccrs

    fill = ax.contourf(
        bundle["mslp_anomaly_hpa"].longitude,
        bundle["mslp_anomaly_hpa"].latitude,
        bundle["mslp_anomaly_hpa"],
        levels=rounded_contour_levels(bundle["mslp_anomaly_hpa"], step=2.0, symmetric=True),
        cmap="RdBu_r",
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    configure_map_axis(ax, YAMAZAKI_DOMAIN, title=f"SLP anomaly + labeled H/L centers\n{bundle['analysis_label']}")
    contours = ax.contour(
        bundle["mslp_hpa"].longitude,
        bundle["mslp_hpa"].latitude,
        bundle["mslp_hpa"],
        levels=rounded_contour_levels(bundle["mslp_hpa"], step=4.0),
        colors="#374151",
        linewidths=0.82,
        transform=ccrs.PlateCarree(),
    )
    if len(contours.levels) > 0:
        ax.clabel(contours, fontsize=6.8, inline=True, fmt="%d")
    draw_box(ax, SIBERIAN_HIGH_SEARCH_BOX, edgecolor="#1d4ed8", label="High search")
    draw_box(ax, ALEUTIAN_LOW_SEARCH_BOX, edgecolor="#b91c1c", label="Low search")
    plot_high_low_center(ax, bundle["siberian_high_center"], label="H", color="#1d4ed8")
    plot_high_low_center(ax, bundle["aleutian_low_center"], label="L", color="#b91c1c")
    return fill


def plot_temperature_panel(ax, bundle: dict[str, object]):
    """Plot 850 hPa temperature anomalies with actual low-level winds."""
    import cartopy.crs as ccrs

    configure_map_axis(ax, YAMAZAKI_DOMAIN, title=f"850 hPa temperature anomaly + winds\n{bundle['analysis_label']}")
    negative_levels = np.arange(-18.0, 0.0, 3.0)
    positive_levels_array = np.arange(3.0, 18.0 + 0.1, 3.0)
    if negative_levels.size:
        cold = ax.contour(
            bundle["t850_anomaly_k"].longitude,
            bundle["t850_anomaly_k"].latitude,
            bundle["t850_anomaly_k"],
            levels=negative_levels,
            colors="#2563eb",
            linewidths=1.0,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(cold, fontsize=6.7, inline=True, fmt="%d")
    if positive_levels_array.size:
        warm = ax.contour(
            bundle["t850_anomaly_k"].longitude,
            bundle["t850_anomaly_k"].latitude,
            bundle["t850_anomaly_k"],
            levels=positive_levels_array,
            colors="#dc2626",
            linewidths=1.0,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(warm, fontsize=6.7, inline=True, fmt="%d")
    quiver = ax.quiver(
        bundle["u850_ms"].longitude.values[::6],
        bundle["u850_ms"].latitude.values[::6],
        bundle["u850_ms"].values[::6, ::6],
        bundle["v850_ms"].values[::6, ::6],
        color="#111827",
        scale=260.0,
        width=0.0020,
        transform=ccrs.PlateCarree(),
        zorder=11,
    )
    ax.quiverkey(quiver, 0.90, -0.07, 10, "10 m s$^{-1}$", labelpos="E")
    draw_box(ax, LOW_LEVEL_NORTH_FLOW_BOX, edgecolor="#0f766e", label="Japan-sector northerly box")


def plot_summary_panel(ax, bundle: dict[str, object]):
    """Plot a text-only summary panel."""
    ax.axis("off")
    summary_df = build_blocking_summary_df(bundle)
    story_lines = build_blocking_story_lines(bundle)
    lines = [f"{bundle['analysis_label']}", ""]
    for row in summary_df.itertuples(index=False):
        lines.append(f"{row.metric}: {row.value} {row.units}")
    lines.append("")
    lines.append("Sequential story")
    for idx, line in enumerate(story_lines, start=1):
        lines.append(f"{idx}. {line}")
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=9.0,
        family="monospace",
    )


def plot_blocking_story_figure(bundle: dict[str, object]):
    """Render the full clearer Yamazaki-style six-panel figure."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(19.5, 11.8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.04, right=0.985, wspace=0.10, hspace=0.20)

    m0 = plot_upper_blocking_panel(axes[0, 0], bundle)
    m1 = plot_jet_circulation_panel(axes[0, 1], bundle)
    m2 = plot_cold_air_flux_panel(axes[0, 2], bundle)
    m3 = plot_surface_pressure_panel(axes[1, 0], bundle)
    plot_temperature_panel(axes[1, 1], bundle)
    plot_summary_panel(axes[1, 2], bundle)

    fig.colorbar(m0, ax=axes[0, 0], orientation="horizontal", pad=0.05, fraction=0.05).set_label("500 hPa height anomaly [gpm]")
    fig.colorbar(m1, ax=axes[0, 1], orientation="horizontal", pad=0.05, fraction=0.05).set_label("300 hPa wind speed [m s$^{-1}$]")
    fig.colorbar(m2, ax=axes[0, 2], orientation="horizontal", pad=0.05, fraction=0.05).set_label(f"Cold-air flux magnitude below {float(bundle['theta_threshold_k']):.0f} K [hPa m s$^{-1}$]")
    fig.colorbar(m3, ax=axes[1, 0], orientation="horizontal", pad=0.05, fraction=0.05).set_label("SLP anomaly [hPa]")
    fig.suptitle("Clearer blocking-process view: upper ridge, jet circulation, cold-air flux, and lower-level response", fontsize=14.0, y=0.98)
    return fig


def plot_upper_surface_evolution_gallery(
    bundles: Sequence[dict[str, object]],
    *,
    suptitle: str = "1-10 February 2018 evolution: upper ridge and lower-level pressure/temperature response",
):
    """Render one compact two-column evolution gallery."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    if not bundles:
        raise ValueError("At least one bundle is required.")

    fig, axes = plt.subplots(
        len(bundles),
        2,
        figsize=(14.5, max(3.2 * len(bundles), 4.6)),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if len(bundles) == 1:
        axes = np.asarray([axes])
    fig.subplots_adjust(top=0.97, bottom=0.02, left=0.05, right=0.985, wspace=0.12, hspace=0.25)

    for row_idx, bundle in enumerate(bundles):
        plot_upper_blocking_panel(axes[row_idx, 0], bundle)
        plot_surface_pressure_panel(axes[row_idx, 1], bundle)

        temp_negative = np.arange(-18.0, 0.0, 3.0)
        temp_positive = np.arange(3.0, 18.0 + 0.1, 3.0)
        if temp_negative.size:
            axes[row_idx, 1].contour(
                bundle["t850_anomaly_k"].longitude,
                bundle["t850_anomaly_k"].latitude,
                bundle["t850_anomaly_k"],
                levels=temp_negative,
                colors="#2563eb",
                linewidths=0.85,
                transform=ccrs.PlateCarree(),
            )
        if temp_positive.size:
            axes[row_idx, 1].contour(
                bundle["t850_anomaly_k"].longitude,
                bundle["t850_anomaly_k"].latitude,
                bundle["t850_anomaly_k"],
                levels=temp_positive,
                colors="#dc2626",
                linewidths=0.85,
                transform=ccrs.PlateCarree(),
            )

    fig.suptitle(suptitle, fontsize=13.0, y=0.995)
    return fig


def plot_jet_flux_evolution_gallery(
    bundles: Sequence[dict[str, object]],
    *,
    suptitle: str = "1-10 February 2018 evolution: jet circulation and cold-air export",
):
    """Render one compact jet / cold-air-flux evolution gallery."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    if not bundles:
        raise ValueError("At least one bundle is required.")

    fig, axes = plt.subplots(
        len(bundles),
        2,
        figsize=(14.5, max(3.2 * len(bundles), 4.6)),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if len(bundles) == 1:
        axes = np.asarray([axes])
    fig.subplots_adjust(top=0.97, bottom=0.02, left=0.05, right=0.985, wspace=0.12, hspace=0.25)

    for row_idx, bundle in enumerate(bundles):
        plot_jet_circulation_panel(axes[row_idx, 0], bundle)
        plot_cold_air_flux_panel(axes[row_idx, 1], bundle)

    fig.suptitle(suptitle, fontsize=13.0, y=0.995)
    return fig
