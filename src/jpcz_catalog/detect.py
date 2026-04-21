"""JPCZ detection helpers based on Shinoda-style divergence thresholds."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units

from .masks import build_coslat_weights, build_polygon_mask


@dataclass(frozen=True)
class DetectionGeometry:
    """Reusable grid geometry for repeated divergence calculations."""

    dx: object
    dy: object
    polygon_mask: xr.DataArray
    weights: xr.DataArray


def compute_grid_deltas(
    longitude: xr.DataArray | np.ndarray,
    latitude: xr.DataArray | np.ndarray,
):
    """Compute horizontal grid spacing on a lat-lon grid."""
    lon_values = np.asarray(getattr(longitude, "values", longitude))
    lat_values = np.asarray(getattr(latitude, "values", latitude))
    return mpcalc.lat_lon_grid_deltas(lon_values, lat_values)


def prepare_detection_geometry(
    longitude: xr.DataArray | np.ndarray,
    latitude: xr.DataArray | np.ndarray,
    polygon_vertices: Sequence[tuple[float, float]],
) -> DetectionGeometry:
    """Precompute reusable geometry for one lat-lon grid."""
    polygon_mask = build_polygon_mask(longitude, latitude, polygon_vertices)
    weights = build_coslat_weights(
        latitude,
        longitude,
        mask=polygon_mask,
    )
    dx, dy = compute_grid_deltas(longitude, latitude)
    return DetectionGeometry(
        dx=dx,
        dy=dy,
        polygon_mask=polygon_mask,
        weights=weights,
    )


def compute_divergence_stack(
    window_ds: xr.Dataset,
    *,
    u_name: str = "u_component_of_wind",
    v_name: str = "v_component_of_wind",
    dx=None,
    dy=None,
) -> xr.DataArray:
    """Compute hourly 925 hPa divergence over a time window."""
    if dx is None or dy is None:
        dx, dy = compute_grid_deltas(window_ds.longitude, window_ds.latitude)

    divergence_stack = []
    for i in range(window_ds.sizes["time"]):
        t_ds = window_ds.isel(time=i)
        u = t_ds[u_name].values * units("m/s")
        v = t_ds[v_name].values * units("m/s")
        div = mpcalc.divergence(u, v, dx=dx, dy=dy).m
        divergence_stack.append(div)

    return xr.DataArray(
        np.stack(divergence_stack, axis=0),
        coords={
            "time": window_ds.time,
            "latitude": window_ds.latitude,
            "longitude": window_ds.longitude,
        },
        dims=("time", "latitude", "longitude"),
        name="divergence_925hpa",
        attrs={"units": "s^-1", "display_units": "1e-5 s^-1"},
    )


def compute_rolling_mean(
    series: xr.DataArray,
    *,
    window: int = 12,
    min_periods: int | None = None,
    name: str = "D_12h",
) -> xr.DataArray:
    """Compute a rolling-mean time series."""
    if min_periods is None:
        min_periods = window
    rolled = series.rolling(time=window, min_periods=min_periods).mean().rename(name)
    rolled.attrs["units"] = "s^-1"
    rolled.attrs["display_units"] = "1e-5 s^-1"
    return rolled


def compute_polygon_mean_divergence_series(
    window_ds: xr.Dataset,
    polygon_vertices: Sequence[tuple[float, float]] | None = None,
    *,
    geometry: DetectionGeometry | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute hourly polygon-mean divergence and Shinoda-style 12-hour mean D."""
    loaded = window_ds.load()
    if geometry is None:
        if polygon_vertices is None:
            raise ValueError("Provide either polygon_vertices or a precomputed geometry.")
        geometry = prepare_detection_geometry(
            loaded.longitude,
            loaded.latitude,
            polygon_vertices,
        )

    div_925 = compute_divergence_stack(loaded, dx=geometry.dx, dy=geometry.dy)
    polygon_mask = geometry.polygon_mask
    weights = geometry.weights

    hourly = (
        (div_925 * weights).sum(dim=("latitude", "longitude"))
        / weights.sum(dim=("latitude", "longitude"))
    ).rename("polygon_mean_divergence_925hpa")
    hourly.attrs["units"] = "s^-1"
    hourly.attrs["display_units"] = "1e-5 s^-1"

    D = compute_rolling_mean(hourly, window=12, min_periods=12, name="D_12h")
    return hourly, D, polygon_mask


def threshold_from_std(
    D_series: xr.DataArray,
    *,
    n_std: float = 2.0,
) -> tuple[float, float, float]:
    """Return mean, std, and Shinoda-style anomaly threshold."""
    valid = D_series.dropna("time")
    D_mean = float(valid.mean().values)
    D_std = float(valid.std().values)
    threshold = D_mean - n_std * D_std
    return D_mean, D_std, threshold


def detect_threshold_events(D_series: xr.DataArray, threshold: float) -> pd.DataFrame:
    """Group consecutive threshold crossings into events."""
    valid = D_series.dropna("time")
    threshold_hits = (valid < threshold).to_series()

    events: list[dict[str, object]] = []
    in_event = False
    current_times: list[pd.Timestamp] = []

    for timestamp, hit in threshold_hits.items():
        if hit and not in_event:
            in_event = True
            current_times = [timestamp]
        elif hit and in_event:
            current_times.append(timestamp)
        elif (not hit) and in_event:
            events.append(_build_event_record(valid, current_times))
            in_event = False
            current_times = []

    if in_event:
        events.append(_build_event_record(valid, current_times))

    return pd.DataFrame(events)


def _build_event_record(
    D_series: xr.DataArray,
    current_times: Sequence[pd.Timestamp],
) -> dict[str, object]:
    event_slice = D_series.sel(time=current_times)
    peak_time = pd.Timestamp(event_slice.idxmin("time").values)
    peak_value = float(event_slice.min().values)
    return {
        "event_start": current_times[0],
        "event_end": current_times[-1],
        "event_peak": peak_time,
        "event_peak_D_s-1": peak_value,
        "event_peak_D_1e5_s-1": peak_value * 1e5,
        "duration_hours": len(current_times),
    }


def count_events_from_threshold(D_series: xr.DataArray, threshold: float) -> pd.DataFrame:
    """Small convenience wrapper for threshold event detection."""
    return detect_threshold_events(D_series, threshold)


def threshold_sensitivity(
    D_series: xr.DataArray,
    n_std_values: Iterable[float],
) -> pd.DataFrame:
    """Evaluate event-count sensitivity to the anomaly threshold."""
    D_mean, D_std, _ = threshold_from_std(D_series, n_std=2.0)
    rows = []
    for n_std in n_std_values:
        threshold = D_mean - n_std * D_std
        events_df = count_events_from_threshold(D_series, threshold)
        rows.append(
            {
                "threshold_label": f"mean - {n_std:.1f} std",
                "n_std": n_std,
                "threshold_s-1": threshold,
                "threshold_1e5_s-1": threshold * 1e5,
                "event_count": len(events_df),
            }
        )
    return pd.DataFrame(rows)
