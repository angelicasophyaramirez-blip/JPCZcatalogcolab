"""Classification helpers for Shinoda-style JPCZ event subclasses."""

from __future__ import annotations

import calendar
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units

from .config import (
    BoundingBox,
    DECEMBER_BENCHMARK_YEARS,
    GeographicPoint,
    SAPPORO,
    SEOUL,
    VORTICITY_BOX,
    WORKING_DOMAIN,
)
from .detect import count_events_from_threshold


def nearest_point(
    da: xr.DataArray,
    point: GeographicPoint | tuple[float, float],
    longitude: float | None = None,
) -> xr.DataArray:
    """Select the nearest grid point by lat/lon."""
    if isinstance(point, GeographicPoint):
        latitude = point.latitude
        longitude = point.longitude
    else:
        latitude = point[0]
        if longitude is None:
            longitude = point[1]

    return da.sel(latitude=latitude, longitude=longitude, method="nearest")


def compute_vorticity_box_mean(
    ds_time: xr.Dataset,
    box: BoundingBox = VORTICITY_BOX,
) -> float:
    """Compute area-weighted box-mean relative vorticity at 925 hPa."""
    u = ds_time["u_component_of_wind"].values * units("m/s")
    v = ds_time["v_component_of_wind"].values * units("m/s")

    dx, dy = mpcalc.lat_lon_grid_deltas(
        ds_time.longitude.values,
        ds_time.latitude.values,
    )
    zeta = mpcalc.vorticity(u, v, dx=dx, dy=dy).m

    zeta_da = xr.DataArray(
        zeta,
        coords={"latitude": ds_time.latitude, "longitude": ds_time.longitude},
        dims=("latitude", "longitude"),
        name="relative_vorticity_925hpa",
        attrs={"units": "s^-1", "display_units": "1e-5 s^-1"},
    )

    zeta_box = zeta_da.sel(
        longitude=slice(box.lon_min, box.lon_max),
        latitude=slice(box.lat_max, box.lat_min),
    )

    lat2d, lon2d = np.meshgrid(
        zeta_box.latitude.values,
        zeta_box.longitude.values,
        indexing="ij",
    )
    weights = np.cos(np.deg2rad(lat2d))
    return float((zeta_box.values * weights).sum() / weights.sum())


def annotate_events_with_environment(
    ds: xr.Dataset,
    events_df: pd.DataFrame,
    *,
    domain: BoundingBox = WORKING_DOMAIN,
    box: BoundingBox = VORTICITY_BOX,
    level: int = 925,
    seoul: GeographicPoint = SEOUL,
    sapporo: GeographicPoint = SAPPORO,
    window_hours: int = 12,
) -> pd.DataFrame:
    """Attach 12-hour SLP-difference and vorticity metrics to each event."""
    classified_events = events_df.copy()
    classified_events["event_peak"] = pd.to_datetime(classified_events["event_peak"])

    slp_diff_values: list[float] = []
    zeta_values: list[float] = []

    for peak_time in classified_events["event_peak"]:
        start_time = peak_time - pd.Timedelta(hours=window_hours - 1)
        end_time = peak_time

        event_window = ds[
            [
                "u_component_of_wind",
                "v_component_of_wind",
                "mean_sea_level_pressure",
            ]
        ].sel(
            time=slice(start_time, end_time),
            longitude=slice(domain.lon_min, domain.lon_max),
            latitude=slice(domain.lat_max, domain.lat_min),
        )

        if "level" in event_window.dims or "level" in event_window.coords:
            event_window = event_window.sel(level=level)

        event_window = event_window.load()

        msl_12h = event_window["mean_sea_level_pressure"].mean("time")
        seoul_msl = float(nearest_point(msl_12h, seoul).values) / 100.0
        sapporo_msl = float(nearest_point(msl_12h, sapporo).values) / 100.0
        slp_diff_values.append(seoul_msl - sapporo_msl)

        uv_12h = event_window[["u_component_of_wind", "v_component_of_wind"]].mean("time")
        zeta_values.append(compute_vorticity_box_mean(uv_12h, box=box))

    classified_events["slp_diff_hpa"] = slp_diff_values
    classified_events["zeta_box_mean_s-1"] = zeta_values
    return classified_events


def compute_december_climatological_slp_index(
    ds: xr.Dataset,
    *,
    years: Iterable[int] = DECEMBER_BENCHMARK_YEARS,
    seoul: GeographicPoint = SEOUL,
    sapporo: GeographicPoint = SAPPORO,
) -> tuple[xr.DataArray, float, float]:
    """Compute the December 12-hour Seoul-minus-Sapporo climatological index."""
    dec_slp_index_series = []

    for year in years:
        msl_dec = ds["mean_sea_level_pressure"].sel(
            time=slice(f"{year}-12-01", f"{year}-12-31T23:00:00")
        )

        seoul_da = nearest_point(msl_dec, seoul) / 100.0
        sapporo_da = nearest_point(msl_dec, sapporo) / 100.0

        slp_diff_hourly = (seoul_da - sapporo_da).rename("slp_diff_hpa")
        slp_diff_12h = slp_diff_hourly.rolling(time=12, min_periods=12).mean().dropna("time").load()
        dec_slp_index_series.append(slp_diff_12h)

    all_dec_slp_index = xr.concat(dec_slp_index_series, dim="time").sortby("time")
    mean_value = float(all_dec_slp_index.mean().values)
    std_value = float(all_dec_slp_index.std().values)
    return all_dec_slp_index, mean_value, std_value


def compute_month_climatological_slp_index(
    ds: xr.Dataset,
    month: int,
    *,
    years: Iterable[int] = DECEMBER_BENCHMARK_YEARS,
    seoul: GeographicPoint = SEOUL,
    sapporo: GeographicPoint = SAPPORO,
) -> tuple[xr.DataArray, float, float]:
    """Compute the 12-hour Seoul-minus-Sapporo climatological index for one month."""
    monthly_slp_index_series = []

    for year in years:
        last_day = calendar.monthrange(year, month)[1]
        msl_month = ds["mean_sea_level_pressure"].sel(
            time=slice(f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}T23:00:00")
        )

        seoul_da = nearest_point(msl_month, seoul) / 100.0
        sapporo_da = nearest_point(msl_month, sapporo) / 100.0

        slp_diff_hourly = (seoul_da - sapporo_da).rename("slp_diff_hpa")
        slp_diff_12h = slp_diff_hourly.rolling(time=12, min_periods=12).mean().dropna("time").load()
        monthly_slp_index_series.append(slp_diff_12h)

    all_month_slp_index = xr.concat(monthly_slp_index_series, dim="time").sortby("time")
    mean_value = float(all_month_slp_index.mean().values)
    std_value = float(all_month_slp_index.std().values)
    return all_month_slp_index, mean_value, std_value


def assign_shinoda_classes(
    events_df: pd.DataFrame,
    *,
    slp_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Assign strong/weak monsoon and Shinoda-style subtype labels."""
    classified_events = events_df.copy()

    classified_events["monsoon_type"] = np.where(
        classified_events["slp_diff_hpa"] > slp_threshold,
        "Type 1 strong-monsoon",
        "Type 2 weak-monsoon",
    )

    type1_mask = classified_events["monsoon_type"] == "Type 1 strong-monsoon"
    if type1_mask.sum() == 0:
        raise ValueError("No Type 1 events available to define the vorticity split.")

    zeta_split = (
        classified_events.loc[type1_mask, "zeta_box_mean_s-1"].mean()
        + classified_events.loc[type1_mask, "zeta_box_mean_s-1"].std()
    )

    classified_events["shinoda_class"] = "Type 2 weak-monsoon"
    classified_events.loc[
        type1_mask & (classified_events["zeta_box_mean_s-1"] > zeta_split),
        "shinoda_class",
    ] = "Type 1B higher-vorticity"
    classified_events.loc[
        type1_mask & (classified_events["zeta_box_mean_s-1"] <= zeta_split),
        "shinoda_class",
    ] = "Type 1A lower-vorticity"

    return classified_events, {"slp_threshold_hpa": slp_threshold, "zeta_split_s-1": zeta_split}


def default_vorticity_box_variants(
    box: BoundingBox = VORTICITY_BOX,
) -> dict[str, dict[str, float]]:
    """Return the simple 1-degree box perturbations tested in the notebook."""
    return {
        "original": {
            "lon_min": box.lon_min,
            "lon_max": box.lon_max,
            "lat_min": box.lat_min,
            "lat_max": box.lat_max,
        },
        "west_1deg": {
            "lon_min": box.lon_min - 1.0,
            "lon_max": box.lon_max - 1.0,
            "lat_min": box.lat_min,
            "lat_max": box.lat_max,
        },
        "east_1deg": {
            "lon_min": box.lon_min + 1.0,
            "lon_max": box.lon_max + 1.0,
            "lat_min": box.lat_min,
            "lat_max": box.lat_max,
        },
        "south_1deg": {
            "lon_min": box.lon_min,
            "lon_max": box.lon_max,
            "lat_min": box.lat_min - 1.0,
            "lat_max": box.lat_max - 1.0,
        },
        "north_1deg": {
            "lon_min": box.lon_min,
            "lon_max": box.lon_max,
            "lat_min": box.lat_min + 1.0,
            "lat_max": box.lat_max + 1.0,
        },
        "expanded_1deg": {
            "lon_min": box.lon_min - 1.0,
            "lon_max": box.lon_max + 1.0,
            "lat_min": box.lat_min - 1.0,
            "lat_max": box.lat_max + 1.0,
        },
    }


def cache_type1_mean_winds(
    ds: xr.Dataset,
    classified_events: pd.DataFrame,
    *,
    domain: BoundingBox = WORKING_DOMAIN,
    level: int = 925,
    window_hours: int = 12,
) -> dict[pd.Timestamp, xr.Dataset]:
    """Cache 12-hour mean winds for Type 1 events to speed box sensitivity tests."""
    type1_events = classified_events[
        classified_events["monsoon_type"] == "Type 1 strong-monsoon"
    ].copy()
    type1_events["event_peak"] = pd.to_datetime(type1_events["event_peak"])

    uv12h_cache: dict[pd.Timestamp, xr.Dataset] = {}
    for peak_time in type1_events["event_peak"]:
        start_time = peak_time - pd.Timedelta(hours=window_hours - 1)
        end_time = peak_time

        uv_12h = ds[
            ["u_component_of_wind", "v_component_of_wind"]
        ].sel(
            time=slice(start_time, end_time),
            longitude=slice(domain.lon_min, domain.lon_max),
            latitude=slice(domain.lat_max, domain.lat_min),
        )

        if "level" in uv_12h.dims or "level" in uv_12h.coords:
            uv_12h = uv_12h.sel(level=level)

        uv12h_cache[pd.Timestamp(peak_time)] = uv_12h.mean("time").load()

    return uv12h_cache


def evaluate_vorticity_box_sensitivity(
    classified_events: pd.DataFrame,
    uv12h_cache: Mapping[pd.Timestamp, xr.Dataset],
    *,
    box_variants: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Type 1B sensitivity to simple box shifts."""
    if box_variants is None:
        box_variants = default_vorticity_box_variants()

    type1_events = classified_events[
        classified_events["monsoon_type"] == "Type 1 strong-monsoon"
    ].copy()
    type1_events["event_peak"] = pd.to_datetime(type1_events["event_peak"])
    type1_box_zeta = type1_events[["event_peak"]].copy()

    sensitivity_rows = []

    for name, box in box_variants.items():
        zeta_vals = []

        for peak_time in type1_events["event_peak"]:
            uv_12h = uv12h_cache[pd.Timestamp(peak_time)]
            zeta_vals.append(
                compute_vorticity_box_mean(
                    uv_12h,
                    box=BoundingBox(
                        lon_min=box["lon_min"],
                        lon_max=box["lon_max"],
                        lat_min=box["lat_min"],
                        lat_max=box["lat_max"],
                    ),
                )
            )

        zeta_series = pd.Series(zeta_vals, index=type1_events.index)
        type1_box_zeta[f"zeta_{name}"] = zeta_vals
        split = zeta_series.mean() + zeta_series.std()

        sensitivity_rows.append(
            {
                "box_name": name,
                "lon_min": box["lon_min"],
                "lon_max": box["lon_max"],
                "lat_min": box["lat_min"],
                "lat_max": box["lat_max"],
                "zeta_split_s-1": split,
                "type1a_count": int((zeta_series <= split).sum()),
                "type1b_count": int((zeta_series > split).sum()),
                "max_zeta_1e5_s-1": zeta_series.max() * 1e5,
            }
        )

    box_sensitivity_df = pd.DataFrame(sensitivity_rows).sort_values(
        ["type1b_count", "max_zeta_1e5_s-1"],
        ascending=[False, False],
    )
    return box_sensitivity_df, type1_box_zeta


def evaluate_threshold_box_combinations(
    classified_events: pd.DataFrame,
    all_dec_D: xr.DataArray,
    *,
    D_mean: float,
    D_std: float,
    slp_threshold: float,
    box_variants: Mapping[str, Mapping[str, float]],
    type1_box_zeta: pd.DataFrame,
    n_std_values: Iterable[float] = (2.0, 2.2, 2.3),
) -> pd.DataFrame:
    """Combine threshold sensitivity and vorticity-box sensitivity."""
    rows = []

    for n_std in n_std_values:
        threshold = D_mean - n_std * D_std
        test_events = count_events_from_threshold(all_dec_D, threshold).copy()
        test_events["event_peak"] = pd.to_datetime(test_events["event_peak"])

        subset = classified_events[["event_peak", "slp_diff_hpa"]].copy()
        subset["event_peak"] = pd.to_datetime(subset["event_peak"])
        subset = subset[subset["event_peak"].isin(test_events["event_peak"])].copy()

        subset["monsoon_type"] = np.where(
            subset["slp_diff_hpa"] > slp_threshold,
            "Type 1 strong-monsoon",
            "Type 2 weak-monsoon",
        )

        type1_subset = subset[subset["monsoon_type"] == "Type 1 strong-monsoon"].copy()
        type2_count = int((subset["monsoon_type"] == "Type 2 weak-monsoon").sum())

        for box_name in box_variants.keys():
            merged = type1_subset.merge(
                type1_box_zeta[["event_peak", f"zeta_{box_name}"]],
                on="event_peak",
                how="left",
            )

            zeta_split = (
                merged[f"zeta_{box_name}"].mean()
                + merged[f"zeta_{box_name}"].std()
            )

            rows.append(
                {
                    "threshold": f"mean - {n_std:.1f} std",
                    "box": box_name,
                    "total_events": len(subset),
                    "type1_total": len(merged),
                    "type1a_count": int((merged[f"zeta_{box_name}"] <= zeta_split).sum()),
                    "type1b_count": int((merged[f"zeta_{box_name}"] > zeta_split).sum()),
                    "type2_count": type2_count,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["type1b_count", "total_events"],
        ascending=[False, True],
    )
