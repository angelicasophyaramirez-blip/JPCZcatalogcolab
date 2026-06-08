"""Continuous spell-window helpers for offshore/coastal regime evolution."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .config import (
    BoundingBox,
    OBJECTIVE_SUBTYPE_DOMAIN,
    RUSSIAN_COASTAL_EXCLUSION_BOXES,
)
from .detect import DetectionGeometry, compute_divergence_stack, prepare_detection_geometry
from .era5 import subset_era5_window
from .masks import build_coslat_weights
from .objective_regimes import (
    COASTAL_LABEL,
    DEFAULT_OBJECTIVE_LABEL,
    MIXED_LABEL,
    OFFSHORE_LABEL,
    classify_episode_regime_path,
    collapse_label_sequence,
)


@dataclass(frozen=True)
class ObjectiveRegionWeights:
    """Reusable polygon/coastal weight fields for continuous spell metrics."""

    polygon_geometry: DetectionGeometry
    coastal_geometry: DetectionGeometry
    polygon_mask: xr.DataArray
    coastal_full_mask: xr.DataArray
    overlap_mask: xr.DataArray
    polygon_only_mask: xr.DataArray
    coastal_only_mask: xr.DataArray
    weights_polygon: xr.DataArray
    weights_polygon_only: xr.DataArray
    weights_coastal_full: xr.DataArray
    weights_coastal_only: xr.DataArray
    weights_overlap: xr.DataArray


def lat_weighted_field_mean(field: xr.DataArray, weight_field: xr.DataArray) -> xr.DataArray:
    """Return the cosine-latitude weighted mean over latitude/longitude."""
    valid = xr.apply_ufunc(np.isfinite, field)
    weighted_field = field.where(valid, 0.0) * weight_field.where(valid, 0.0)
    denominator = weight_field.where(valid, 0.0).sum(dim=("latitude", "longitude"))
    numerator = weighted_field.sum(dim=("latitude", "longitude"))
    return xr.where(denominator > 0.0, numerator / denominator, np.nan)


def build_russian_coastal_keep_mask(
    target_field: xr.DataArray,
    *,
    exclusion_boxes: Sequence[BoundingBox] = RUSSIAN_COASTAL_EXCLUSION_BOXES,
) -> xr.DataArray:
    """Mask out terrain-sensitive Russian-coastal cells used in earlier notebooks."""
    lat_vals = np.asarray(target_field.latitude.values, dtype=float)
    lon_vals = np.asarray(target_field.longitude.values, dtype=float)
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    keep_mask = np.ones((len(lat_vals), len(lon_vals)), dtype=bool)
    for box in exclusion_boxes:
        in_box = (
            (lon2d >= box.lon_min)
            & (lon2d <= box.lon_max)
            & (lat2d >= box.lat_min)
            & (lat2d <= box.lat_max)
        )
        keep_mask &= ~in_box
    return xr.DataArray(
        keep_mask,
        coords={"latitude": target_field.latitude, "longitude": target_field.longitude},
        dims=("latitude", "longitude"),
        name="russian_coastal_keep_mask",
    )


def prepare_objective_region_weights(
    longitude: xr.DataArray | np.ndarray,
    latitude: xr.DataArray | np.ndarray,
    *,
    polygon_vertices: Sequence[tuple[float, float]],
    coastal_vertices: Sequence[tuple[float, float]],
) -> ObjectiveRegionWeights:
    """Build the full polygon/coastal/overlap masks used by Notebook 22."""
    polygon_geometry = prepare_detection_geometry(longitude, latitude, polygon_vertices)
    coastal_geometry = prepare_detection_geometry(longitude, latitude, coastal_vertices)

    polygon_mask = polygon_geometry.polygon_mask.astype(bool)
    coastal_full_mask = coastal_geometry.polygon_mask.astype(bool)
    overlap_mask = xr.DataArray(
        polygon_mask.values & coastal_full_mask.values,
        coords=polygon_mask.coords,
        dims=polygon_mask.dims,
        name="polygon_coastal_overlap_mask",
    )
    polygon_only_mask = xr.DataArray(
        polygon_mask.values & (~coastal_full_mask.values),
        coords=polygon_mask.coords,
        dims=polygon_mask.dims,
        name="polygon_only_mask",
    )
    coastal_only_mask = xr.DataArray(
        coastal_full_mask.values & (~polygon_mask.values),
        coords=polygon_mask.coords,
        dims=polygon_mask.dims,
        name="coastal_only_mask",
    )

    weights_polygon = polygon_geometry.weights
    weights_coastal_full = coastal_geometry.weights
    weights_overlap = build_coslat_weights(latitude, longitude, mask=overlap_mask)
    weights_polygon_only = build_coslat_weights(latitude, longitude, mask=polygon_only_mask)
    weights_coastal_only = build_coslat_weights(latitude, longitude, mask=coastal_only_mask)

    return ObjectiveRegionWeights(
        polygon_geometry=polygon_geometry,
        coastal_geometry=coastal_geometry,
        polygon_mask=polygon_mask,
        coastal_full_mask=coastal_full_mask,
        overlap_mask=overlap_mask,
        polygon_only_mask=polygon_only_mask,
        coastal_only_mask=coastal_only_mask,
        weights_polygon=weights_polygon,
        weights_polygon_only=weights_polygon_only,
        weights_coastal_full=weights_coastal_full,
        weights_coastal_only=weights_coastal_only,
        weights_overlap=weights_overlap,
    )


def compute_vertical_moisture_flux_proxy_stack(
    ds_850: xr.Dataset,
    *,
    specific_humidity_name: str = "specific_humidity",
    vertical_velocity_name: str = "vertical_velocity",
) -> xr.DataArray:
    """Compute the -1000*q*omega proxy used in Notebook 17 and 22."""
    return (
        -1000.0 * ds_850[specific_humidity_name] * ds_850[vertical_velocity_name]
    ).rename("vertical_moisture_flux_proxy_850")


def select_regular_time_steps(window_ds: xr.Dataset, *, interval_hours: int) -> xr.Dataset:
    """Thin hourly ERA5 windows to a regular interval without resampling."""
    if interval_hours <= 1 or "time" not in window_ds.dims:
        return window_ds
    return window_ds.isel(time=slice(None, None, int(interval_hours)))


def compute_window_regional_metric_timeseries(
    ds: xr.Dataset,
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    region_weights: ObjectiveRegionWeights,
    domain: BoundingBox = OBJECTIVE_SUBTYPE_DOMAIN,
    time_step_hours: int = 6,
    exclusion_boxes: Sequence[BoundingBox] = RUSSIAN_COASTAL_EXCLUSION_BOXES,
) -> pd.DataFrame:
    """Compute continuous polygon/coastal means over one padded spell window."""
    ds_850 = subset_era5_window(
        ds,
        str(pd.Timestamp(start)),
        str(pd.Timestamp(end)),
        domain=domain,
        variables=("specific_humidity", "vertical_velocity"),
        level=850,
    )
    ds_925 = subset_era5_window(
        ds,
        str(pd.Timestamp(start)),
        str(pd.Timestamp(end)),
        domain=domain,
        variables=("u_component_of_wind", "v_component_of_wind"),
        level=925,
    )

    common_times = pd.Index(pd.to_datetime(ds_850.time.values)).intersection(
        pd.Index(pd.to_datetime(ds_925.time.values))
    )
    if common_times.empty:
        return pd.DataFrame(
            columns=[
                "time",
                "polygon_qflux_850_mean",
                "coastal_qflux_850_mean",
                "polygon_div_925_mean",
                "coastal_div_925_mean",
            ]
        )

    ds_850 = select_regular_time_steps(ds_850.sel(time=common_times), interval_hours=time_step_hours).load()
    ds_925 = select_regular_time_steps(ds_925.sel(time=common_times), interval_hours=time_step_hours).load()

    qflux_850 = compute_vertical_moisture_flux_proxy_stack(ds_850)
    qflux_keep_mask = build_russian_coastal_keep_mask(
        qflux_850.isel(time=0, drop=True),
        exclusion_boxes=exclusion_boxes,
    )
    qflux_850 = qflux_850.where(qflux_keep_mask)

    div_925 = compute_divergence_stack(
        ds_925,
        dx=region_weights.polygon_geometry.dx,
        dy=region_weights.polygon_geometry.dy,
    ) * 1e5
    div_925 = div_925.rename("divergence_925")
    div_keep_mask = build_russian_coastal_keep_mask(
        div_925.isel(time=0, drop=True),
        exclusion_boxes=exclusion_boxes,
    )
    div_925 = div_925.where(div_keep_mask)

    metrics_df = pd.DataFrame(
        {
            "time": pd.to_datetime(qflux_850.time.values),
            "polygon_qflux_850_mean": lat_weighted_field_mean(
                qflux_850,
                region_weights.weights_polygon,
            ).values.astype(float),
            "coastal_qflux_850_mean": lat_weighted_field_mean(
                qflux_850,
                region_weights.weights_coastal_only,
            ).values.astype(float),
            "polygon_div_925_mean": lat_weighted_field_mean(
                div_925,
                region_weights.weights_polygon,
            ).values.astype(float),
            "coastal_div_925_mean": lat_weighted_field_mean(
                div_925,
                region_weights.weights_coastal_only,
            ).values.astype(float),
        }
    )
    return metrics_df


def build_persistent_mask(values: Sequence[bool], *, min_steps: int) -> np.ndarray:
    """Mark runs that stay true for at least ``min_steps`` consecutive samples."""
    raw = np.asarray(values, dtype=bool)
    persistent = np.zeros(raw.shape, dtype=bool)
    start_idx: int | None = None

    for idx, is_true in enumerate(raw):
        if is_true and start_idx is None:
            start_idx = idx
        elif not is_true and start_idx is not None:
            if idx - start_idx >= min_steps:
                persistent[start_idx:idx] = True
            start_idx = None

    if start_idx is not None and len(raw) - start_idx >= min_steps:
        persistent[start_idx:] = True
    return persistent


def apply_continuous_regime_thresholds(
    metrics_df: pd.DataFrame,
    *,
    polygon_qflux_min: float,
    polygon_div_max: float,
    coastal_qflux_split: float,
    coastal_div_max: float,
    persistence_steps: int = 2,
) -> pd.DataFrame:
    """Apply the Notebook 22 offshore/coastal rules to continuous spell metrics."""
    labeled_df = metrics_df.copy()
    labeled_df["offshore_rule_raw"] = (
        np.isfinite(labeled_df["polygon_qflux_850_mean"])
        & np.isfinite(labeled_df["polygon_div_925_mean"])
        & np.isfinite(labeled_df["coastal_qflux_850_mean"])
        & (labeled_df["polygon_qflux_850_mean"] >= float(polygon_qflux_min))
        & (labeled_df["polygon_div_925_mean"] <= float(polygon_div_max))
        & (labeled_df["coastal_qflux_850_mean"] < float(coastal_qflux_split))
    )
    labeled_df["coastal_rule_raw"] = (
        np.isfinite(labeled_df["coastal_qflux_850_mean"])
        & np.isfinite(labeled_df["coastal_div_925_mean"])
        & (labeled_df["coastal_qflux_850_mean"] >= float(coastal_qflux_split))
        & (labeled_df["coastal_div_925_mean"] <= float(coastal_div_max))
    )

    labeled_df["offshore_rule_persistent"] = build_persistent_mask(
        labeled_df["offshore_rule_raw"].to_numpy(dtype=bool),
        min_steps=int(persistence_steps),
    )
    labeled_df["coastal_rule_persistent"] = build_persistent_mask(
        labeled_df["coastal_rule_raw"].to_numpy(dtype=bool),
        min_steps=int(persistence_steps),
    )

    labels = np.full(len(labeled_df), DEFAULT_OBJECTIVE_LABEL, dtype=object)
    offshore_support = labeled_df["offshore_rule_persistent"].to_numpy(dtype=bool)
    coastal_support = labeled_df["coastal_rule_persistent"].to_numpy(dtype=bool)
    labels[offshore_support] = OFFSHORE_LABEL
    labels[coastal_support] = COASTAL_LABEL
    labels[offshore_support & coastal_support] = MIXED_LABEL
    labeled_df["continuous_regime_label"] = labels
    return labeled_df


def summarize_continuous_spell(
    labeled_timeseries_df: pd.DataFrame,
    *,
    spell_id: str,
    threshold_profile: str,
    time_step_hours: int,
) -> dict[str, object]:
    """Reduce one continuous spell window to a compact path / lag summary."""
    if labeled_timeseries_df.empty:
        return {
            "catalog_spell_id": spell_id,
            "threshold_profile": threshold_profile,
            "time_step_hours": int(time_step_hours),
            "continuous_spell_regime_path": "no_data",
            "continuous_clear_sequence": "",
            "time_step_count": 0,
            "offshore_support_step_count": 0,
            "coastal_support_step_count": 0,
            "mixed_support_step_count": 0,
            "weak_step_count": 0,
            "first_offshore_support_time": pd.NaT,
            "first_coastal_support_time": pd.NaT,
            "offshore_precedes_coastal": False,
            "coastal_precedes_offshore": False,
            "offshore_to_coastal_lag_hours": np.nan,
            "coastal_to_offshore_lag_hours": np.nan,
        }

    state_labels = labeled_timeseries_df["continuous_regime_label"].fillna(DEFAULT_OBJECTIVE_LABEL).astype(str).tolist()
    clear_sequence = collapse_label_sequence(state_labels)
    spell_path = classify_episode_regime_path(state_labels)

    offshore_rows = labeled_timeseries_df.loc[labeled_timeseries_df["offshore_rule_persistent"]]
    coastal_rows = labeled_timeseries_df.loc[labeled_timeseries_df["coastal_rule_persistent"]]
    first_offshore = offshore_rows["time"].min() if not offshore_rows.empty else pd.NaT
    first_coastal = coastal_rows["time"].min() if not coastal_rows.empty else pd.NaT

    offshore_precedes_coastal = bool(
        pd.notna(first_offshore) and pd.notna(first_coastal) and first_offshore < first_coastal
    )
    coastal_precedes_offshore = bool(
        pd.notna(first_offshore) and pd.notna(first_coastal) and first_coastal < first_offshore
    )

    return {
        "catalog_spell_id": spell_id,
        "threshold_profile": threshold_profile,
        "time_step_hours": int(time_step_hours),
        "continuous_spell_regime_path": spell_path,
        "continuous_clear_sequence": " -> ".join(clear_sequence) if clear_sequence else "weak_only",
        "time_step_count": int(len(labeled_timeseries_df)),
        "offshore_support_step_count": int(labeled_timeseries_df["offshore_rule_persistent"].sum()),
        "coastal_support_step_count": int(labeled_timeseries_df["coastal_rule_persistent"].sum()),
        "mixed_support_step_count": int(
            (
                labeled_timeseries_df["offshore_rule_persistent"]
                & labeled_timeseries_df["coastal_rule_persistent"]
            ).sum()
        ),
        "weak_step_count": int((labeled_timeseries_df["continuous_regime_label"] == DEFAULT_OBJECTIVE_LABEL).sum()),
        "first_offshore_support_time": first_offshore,
        "first_coastal_support_time": first_coastal,
        "offshore_precedes_coastal": offshore_precedes_coastal,
        "coastal_precedes_offshore": coastal_precedes_offshore,
        "offshore_to_coastal_lag_hours": (
            float((first_coastal - first_offshore).total_seconds() / 3600.0)
            if offshore_precedes_coastal
            else np.nan
        ),
        "coastal_to_offshore_lag_hours": (
            float((first_offshore - first_coastal).total_seconds() / 3600.0)
            if coastal_precedes_offshore
            else np.nan
        ),
    }
