"""Event-review helpers for peak position, candidate intensity, and manual QA."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .config import JPCZ_POLYGON_VERTICES, WORKING_DOMAIN, BoundingBox
from .detect import compute_divergence_field, prepare_detection_geometry


def load_event_peak_snapshot(
    ds: xr.Dataset,
    event_peak: pd.Timestamp | str,
    *,
    domain: BoundingBox = WORKING_DOMAIN,
    level: int = 925,
    cloud_variable: str | None = None,
) -> xr.Dataset:
    """Load one event-peak snapshot for plotting or peak-position diagnostics."""
    variables = ["u_component_of_wind", "v_component_of_wind"]
    if cloud_variable is not None:
        variables.append(cloud_variable)

    snapshot = ds[variables].sel(
        time=pd.Timestamp(event_peak),
        longitude=slice(domain.lon_min, domain.lon_max),
        latitude=slice(domain.lat_max, domain.lat_min),
    )

    if "level" in snapshot.dims or "level" in snapshot.coords:
        snapshot = snapshot.sel(level=level)

    if "time" in snapshot.dims:
        snapshot = snapshot.squeeze("time", drop=True)

    return snapshot.load()


def summarize_peak_convergence_field(
    divergence_field: xr.DataArray,
    *,
    polygon_mask: xr.DataArray,
    weights: xr.DataArray | None = None,
) -> dict[str, float]:
    """Summarize the peak convergence location and centroid inside the polygon."""
    masked_divergence = divergence_field.where(polygon_mask)
    stacked = masked_divergence.stack(point=("latitude", "longitude")).dropna("point")
    if stacked.sizes.get("point", 0) == 0:
        raise ValueError("No valid polygon cells were found for the peak divergence field.")

    peak_point_index = int(stacked.argmin("point").item())
    peak_point = stacked.isel(point=peak_point_index)
    peak_divergence = float(peak_point.values)

    if weights is None:
        lat2d, _ = xr.broadcast(masked_divergence.latitude, masked_divergence.longitude)
        weights = xr.DataArray(
            np.cos(np.deg2rad(lat2d.values)),
            coords=masked_divergence.coords,
            dims=masked_divergence.dims,
        ).where(polygon_mask, other=0.0)

    convergence_magnitude = (-masked_divergence).clip(min=0.0)
    centroid_weights = convergence_magnitude * weights
    total_weight = float(centroid_weights.fillna(0.0).sum().values)

    if total_weight > 0.0:
        lat2d, lon2d = xr.broadcast(masked_divergence.latitude, masked_divergence.longitude)
        centroid_lat = float(((lat2d * centroid_weights).fillna(0.0).sum() / total_weight).values)
        centroid_lon = float(((lon2d * centroid_weights).fillna(0.0).sum() / total_weight).values)
    else:
        centroid_lat = float("nan")
        centroid_lon = float("nan")

    peak_convergence = max(-peak_divergence, 0.0)
    return {
        "peak_max_convergence_s-1": peak_convergence,
        "peak_max_convergence_1e5_s-1": peak_convergence * 1e5,
        "peak_max_convergence_lat": float(peak_point["latitude"].values),
        "peak_max_convergence_lon": float(peak_point["longitude"].values),
        "peak_convergence_centroid_lat": centroid_lat,
        "peak_convergence_centroid_lon": centroid_lon,
    }


def annotate_events_with_peak_position(
    ds: xr.Dataset,
    catalog_df: pd.DataFrame,
    *,
    domain: BoundingBox = WORKING_DOMAIN,
    polygon_vertices: Sequence[tuple[float, float]] = JPCZ_POLYGON_VERTICES,
    level: int = 925,
) -> pd.DataFrame:
    """Attach peak-time convergence-position diagnostics to each event."""
    annotated = catalog_df.copy()
    annotated["event_peak"] = pd.to_datetime(annotated["event_peak"])

    geometry = None
    peak_records: list[dict[str, float]] = []

    for peak_time in annotated["event_peak"]:
        peak_snapshot = load_event_peak_snapshot(
            ds,
            peak_time,
            domain=domain,
            level=level,
        )

        if geometry is None:
            geometry = prepare_detection_geometry(
                peak_snapshot.longitude,
                peak_snapshot.latitude,
                polygon_vertices,
            )

        divergence_field = compute_divergence_field(
            peak_snapshot,
            dx=geometry.dx,
            dy=geometry.dy,
        )
        peak_records.append(
            summarize_peak_convergence_field(
                divergence_field,
                polygon_mask=geometry.polygon_mask,
                weights=geometry.weights,
            )
        )

    peak_metrics = pd.DataFrame(peak_records, index=annotated.index)
    return pd.concat([annotated, peak_metrics], axis=1)


def add_candidate_intensity_metrics(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Add transparent candidate intensity metrics for proposal testing."""
    catalog = catalog_df.copy()

    peak_convergence = -catalog["event_peak_D_1e5_s-1"]
    positive_vorticity = (catalog["zeta_box_mean_s-1"] * 1e5).clip(lower=0.0)

    catalog["candidate_peak_convergence_1e5_s-1"] = peak_convergence
    catalog["candidate_duration_weighted_convergence"] = peak_convergence * catalog["duration_hours"]
    catalog["candidate_positive_box_vorticity_1e5_s-1"] = positive_vorticity
    catalog["candidate_peak_duration_vorticity_index"] = (
        catalog["candidate_duration_weighted_convergence"] * positive_vorticity
    )

    return catalog


def add_position_group_columns(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Add simple north/central/south and west/central/east auto groups."""
    catalog = catalog_df.copy()
    catalog["lat_position_group_auto"] = _assign_tercile_labels(
        catalog["peak_max_convergence_lat"],
        labels=("south-shifted", "central", "north-shifted"),
    )
    catalog["lon_position_group_auto"] = _assign_tercile_labels(
        catalog["peak_max_convergence_lon"],
        labels=("west-shifted", "central", "east-shifted"),
    )
    return catalog


def build_manual_verification_scaffold(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Add blank manual-review columns while preserving the auto diagnostics."""
    scaffold = catalog_df.copy()
    defaults = {
        "verified_event": "",
        "cloud_band_present": "",
        "position_group_manual": "",
        "manual_peak_convergence_lat": pd.NA,
        "manual_peak_convergence_lon": pd.NA,
        "upper_level_forcing_note": "",
        "verification_notes": "",
    }
    for column_name, default_value in defaults.items():
        if column_name not in scaffold.columns:
            scaffold[column_name] = default_value
    return scaffold


def _assign_tercile_labels(
    series: pd.Series,
    *,
    labels: tuple[str, str, str],
) -> pd.Series:
    """Assign stable tercile labels using ranked values."""
    valid = series.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=series.index, dtype="object")

    if valid.nunique() < 3:
        out = pd.Series(labels[1], index=series.index, dtype="object")
        out[series.isna()] = pd.NA
        return out

    ranked = valid.rank(method="first")
    buckets = pd.qcut(ranked, q=3, labels=labels)
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    out.loc[valid.index] = buckets.astype("object")
    return out
