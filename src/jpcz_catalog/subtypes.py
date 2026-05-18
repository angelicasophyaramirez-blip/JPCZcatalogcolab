"""Objective subtype feature helpers for NDJF JPCZ events."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .analysis import load_event_peak_snapshot
from .config import (
    COASTAL_JAPAN_BOX,
    EXTENDED_DOMAIN,
    HOKKAIDO_BOX,
    HOKKAIDO_FRONT_BOX,
    JPCZ_POLYGON_VERTICES,
    PACIFIC_EAST_OF_JAPAN_BOX,
    PACIFIC_FRONT_BOX,
    SEA_OF_JAPAN_BOX,
    BoundingBox,
)
from .detect import compute_divergence_field, prepare_detection_geometry
from .diagnostics import (
    compute_geopotential_height_field,
    compute_relative_vorticity_field,
    compute_temperature_gradient_magnitude,
    load_offset_snapshot,
)
from .era5 import month_window
from .masks import build_coslat_weights


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata for one objective subtype feature."""

    column_name: str
    units: str
    meaning: str
    calculation: str
    region: str
    time_window: str
    purpose: str


FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        column_name="jpcz_polygon_mean_convergence_peak_925",
        units="1e-5 s^-1",
        meaning="Area-weighted mean convergence magnitude in the original JPCZ polygon at peak time.",
        calculation="Compute 925 hPa divergence from ERA5 u and v, multiply by -1, clip negative values to 0, then area-average inside the JPCZ polygon.",
        region="Original JPCZ polygon",
        time_window="event peak only",
        purpose="Canonical convergence strength in the Shinoda detection region.",
    ),
    FeatureDefinition(
        column_name="jpcz_polygon_max_convergence_peak_925",
        units="1e-5 s^-1",
        meaning="Maximum convergence magnitude in the original JPCZ polygon at peak time.",
        calculation="Maximum of the positive-only convergence field inside the JPCZ polygon.",
        region="Original JPCZ polygon",
        time_window="event peak only",
        purpose="Peak local convergence in the Shinoda detection region.",
    ),
    FeatureDefinition(
        column_name="coastal_japan_mean_convergence_peak_925",
        units="1e-5 s^-1",
        meaning="Area-weighted mean convergence magnitude in the coastal-Japan characterization box at peak time.",
        calculation="Same convergence field as above, summarized in the coastal-Japan box.",
        region="Coastal Japan box",
        time_window="event peak only",
        purpose="Measures whether convergence is enhanced along the west coast of Japan.",
    ),
    FeatureDefinition(
        column_name="coastal_japan_max_convergence_peak_925",
        units="1e-5 s^-1",
        meaning="Maximum convergence magnitude in the coastal-Japan characterization box at peak time.",
        calculation="Maximum of the positive-only convergence field in the coastal-Japan box.",
        region="Coastal Japan box",
        time_window="event peak only",
        purpose="Captures the strongest coastal convergence core.",
    ),
    FeatureDefinition(
        column_name="coastal_to_jpcz_mean_convergence_ratio",
        units="unitless",
        meaning="Ratio of coastal-Japan mean convergence to JPCZ-polygon mean convergence.",
        calculation="coastal_japan_mean_convergence_peak_925 / jpcz_polygon_mean_convergence_peak_925",
        region="Coastal Japan vs JPCZ polygon",
        time_window="event peak only",
        purpose="Measures how coastal-enhanced the event is relative to the canonical JPCZ core.",
    ),
    FeatureDefinition(
        column_name="coastal_to_jpcz_max_convergence_ratio",
        units="unitless",
        meaning="Ratio of the coastal-Japan maximum convergence to the JPCZ-polygon maximum convergence.",
        calculation="coastal_japan_max_convergence_peak_925 / jpcz_polygon_max_convergence_peak_925",
        region="Coastal Japan vs JPCZ polygon",
        time_window="event peak only",
        purpose="Alternative coastal-enhancement metric based on local maxima instead of regional means.",
    ),
    FeatureDefinition(
        column_name="pacific_east_of_japan_mean_convergence_peak_925",
        units="1e-5 s^-1",
        meaning="Area-weighted mean convergence magnitude east of Japan at peak time.",
        calculation="Same convergence field as above, summarized in the Pacific characterization box.",
        region="Pacific east of Japan box",
        time_window="event peak only",
        purpose="Measures whether the event is coupled to stronger Pacific-side convergence.",
    ),
    FeatureDefinition(
        column_name="pacific_east_of_japan_max_convergence_peak_925",
        units="1e-5 s^-1",
        meaning="Maximum convergence magnitude east of Japan at peak time.",
        calculation="Maximum of the positive-only convergence field in the Pacific characterization box.",
        region="Pacific east of Japan box",
        time_window="event peak only",
        purpose="Captures the strongest Pacific-side convergence core.",
    ),
    FeatureDefinition(
        column_name="pacific_to_jpcz_mean_convergence_ratio",
        units="unitless",
        meaning="Ratio of Pacific-box mean convergence to JPCZ-polygon mean convergence.",
        calculation="pacific_east_of_japan_mean_convergence_peak_925 / jpcz_polygon_mean_convergence_peak_925",
        region="Pacific east of Japan vs JPCZ polygon",
        time_window="event peak only",
        purpose="Measures how strongly the event is coupled to Pacific-side convergence.",
    ),
    FeatureDefinition(
        column_name="pacific_to_jpcz_max_convergence_ratio",
        units="unitless",
        meaning="Ratio of the Pacific-box maximum convergence to the JPCZ-polygon maximum convergence.",
        calculation="pacific_east_of_japan_max_convergence_peak_925 / jpcz_polygon_max_convergence_peak_925",
        region="Pacific east of Japan vs JPCZ polygon",
        time_window="event peak only",
        purpose="Alternative Pacific-coupling metric based on local maxima instead of regional means.",
    ),
    FeatureDefinition(
        column_name="sea_of_japan_mean_vorticity_peak_925",
        units="1e-5 s^-1",
        meaning="Area-weighted mean relative vorticity in the Sea of Japan box at peak time.",
        calculation="Compute 925 hPa relative vorticity from ERA5 u and v and area-average it in the Sea of Japan box.",
        region="Sea of Japan box",
        time_window="event peak only",
        purpose="Quantifies circulation-centered forcing over the Sea of Japan.",
    ),
    FeatureDefinition(
        column_name="sea_of_japan_max_vorticity_peak_925",
        units="1e-5 s^-1",
        meaning="Maximum relative vorticity in the Sea of Japan box at peak time.",
        calculation="Maximum of the 925 hPa relative-vorticity field in the Sea of Japan box.",
        region="Sea of Japan box",
        time_window="event peak only",
        purpose="Captures the strongest low-level circulation center in the Sea of Japan.",
    ),
    FeatureDefinition(
        column_name="hokkaido_min_z850_anomaly_tminus12_to_tplus12",
        units="gpm",
        meaning="Most negative 850 hPa geopotential-height anomaly in the Hokkaido box over the t-12, t0, t+12 window.",
        calculation="Compute z850 anomaly as event-time z850 minus monthly climatological z850, then take the minimum box value over the three-time window.",
        region="Hokkaido box",
        time_window="t-12 h, t0, t+12 h",
        purpose="Quantifies passing-low or trough forcing near Hokkaido.",
    ),
    FeatureDefinition(
        column_name="hokkaido_mean_z850_anomaly_tminus12_to_tplus12",
        units="gpm",
        meaning="Most negative box-mean 850 hPa geopotential-height anomaly in the Hokkaido box over the t-12, t0, t+12 window.",
        calculation="Compute box-mean z850 anomaly at each offset time and save the most negative value.",
        region="Hokkaido box",
        time_window="t-12 h, t0, t+12 h",
        purpose="A more stable synoptic-forcing metric than the single gridpoint minimum.",
    ),
    FeatureDefinition(
        column_name="sea_of_japan_min_z850_anomaly_tminus12_to_tplus12",
        units="gpm",
        meaning="Most negative 850 hPa geopotential-height anomaly in the Sea of Japan box over the t-12, t0, t+12 window.",
        calculation="Same anomaly logic as above, but evaluated in the Sea of Japan box.",
        region="Sea of Japan box",
        time_window="t-12 h, t0, t+12 h",
        purpose="Quantifies whether the synoptic-height depression is centered over the Sea of Japan.",
    ),
    FeatureDefinition(
        column_name="front_box_max_temp_gradient_850_tminus12_to_tplus12",
        units="K (100 km)^-1",
        meaning="Largest 850 hPa horizontal temperature-gradient magnitude in the Hokkaido front box over the t-12, t0, t+12 window.",
        calculation="Compute |grad T850| and save the maximum box value across the three-time window.",
        region="Hokkaido front box",
        time_window="t-12 h, t0, t+12 h",
        purpose="Quantifies frontal or baroclinic forcing upstream of the JPCZ.",
    ),
    FeatureDefinition(
        column_name="pacific_box_max_temp_gradient_850_tminus12_to_tplus12",
        units="K (100 km)^-1",
        meaning="Largest 850 hPa horizontal temperature-gradient magnitude in the Pacific front box over the t-12, t0, t+12 window.",
        calculation="Compute |grad T850| and save the maximum Pacific-box value across the three-time window.",
        region="Pacific front box",
        time_window="t-12 h, t0, t+12 h",
        purpose="Quantifies frontal or baroclinic forcing east of Japan.",
    ),
)


def feature_definitions_dataframe() -> pd.DataFrame:
    """Return the objective subtype feature dictionary as a DataFrame."""
    return pd.DataFrame([asdict(defn) for defn in FEATURE_DEFINITIONS])


def compute_monthly_geopotential_height_climatology(
    ds: xr.Dataset,
    *,
    years: Iterable[int],
    months: Iterable[int] = (11, 12, 1, 2),
    domain: BoundingBox = EXTENDED_DOMAIN,
    level: int = 850,
    z_name: str = "geopotential",
) -> xr.DataArray:
    """Compute monthly mean 850 hPa geopotential height climatology."""
    monthly_windows = []

    for year in years:
        for month in months:
            start, end = month_window(year, month)
            subset = ds[[z_name]].sel(
                time=slice(start, end),
                longitude=slice(domain.lon_min, domain.lon_max),
                latitude=slice(domain.lat_max, domain.lat_min),
            )

            if "level" in subset.dims or "level" in subset.coords:
                subset = subset.sel(level=level)

            monthly_windows.append(subset[z_name] / 9.80665)

    combined = xr.concat(monthly_windows, dim="time")
    climatology = combined.groupby("time.month").mean("time").rename("monthly_z850_climatology")
    climatology.attrs["units"] = "gpm"
    climatology.attrs["display_units"] = "gpm"
    return climatology.load()


def build_objective_subtype_feature_table(
    ds: xr.Dataset,
    catalog_df: pd.DataFrame,
    *,
    z850_climatology: xr.DataArray,
    characterization_domain: BoundingBox = EXTENDED_DOMAIN,
    polygon_vertices: Sequence[tuple[float, float]] = JPCZ_POLYGON_VERTICES,
    coastal_box: BoundingBox = COASTAL_JAPAN_BOX,
    pacific_box: BoundingBox = PACIFIC_EAST_OF_JAPAN_BOX,
    hokkaido_box: BoundingBox = HOKKAIDO_BOX,
    sea_of_japan_box: BoundingBox = SEA_OF_JAPAN_BOX,
    front_box: BoundingBox = HOKKAIDO_FRONT_BOX,
    pacific_front_box: BoundingBox = PACIFIC_FRONT_BOX,
    offset_hours: Sequence[int] = (-12, 0, 12),
) -> pd.DataFrame:
    """Build the event-level objective subtype feature table."""
    catalog = catalog_df.copy()
    for column_name in ("event_start", "event_end", "event_peak"):
        if column_name in catalog.columns:
            catalog[column_name] = pd.to_datetime(catalog[column_name])

    geometry_925 = None
    records: list[dict[str, object]] = []

    for idx, row in catalog.iterrows():
        peak_snapshot_925 = load_event_peak_snapshot(
            ds,
            row["event_peak"],
            domain=characterization_domain,
            level=925,
        )

        if geometry_925 is None:
            geometry_925 = prepare_detection_geometry(
                peak_snapshot_925.longitude,
                peak_snapshot_925.latitude,
                polygon_vertices,
            )

        divergence_field = compute_divergence_field(
            peak_snapshot_925,
            dx=geometry_925.dx,
            dy=geometry_925.dy,
        )
        convergence_field = ((-divergence_field).clip(min=0.0) * 1e5).rename("convergence_925_display")
        vorticity_field = (
            compute_relative_vorticity_field(
                peak_snapshot_925,
                dx=geometry_925.dx,
                dy=geometry_925.dy,
            )
            * 1e5
        ).rename("relative_vorticity_925_display")

        jpcz_mean = _weighted_mean_masked(convergence_field, geometry_925.polygon_mask)
        jpcz_max = _masked_max(convergence_field, geometry_925.polygon_mask)
        coastal_mean = _box_weighted_mean(convergence_field, coastal_box)
        coastal_max = _box_max(convergence_field, coastal_box)
        pacific_mean = _box_weighted_mean(convergence_field, pacific_box)
        pacific_max = _box_max(convergence_field, pacific_box)
        soj_vort_mean = _box_weighted_mean(vorticity_field, sea_of_japan_box)
        soj_vort_max = _box_max(vorticity_field, sea_of_japan_box)

        hokkaido_min_anoms = []
        hokkaido_mean_anoms = []
        sea_of_japan_min_anoms = []
        frontality_max_values = []
        pacific_frontality_max_values = []

        for offset in offset_hours:
            synoptic_time = pd.Timestamp(row["event_peak"]) + pd.Timedelta(hours=offset)
            synoptic_snapshot_850 = load_offset_snapshot(
                ds,
                row["event_peak"],
                offset_hours=offset,
                variables=["geopotential", "temperature"],
                domain=characterization_domain,
                level=850,
            )

            z850 = compute_geopotential_height_field(synoptic_snapshot_850)
            z850_anomaly = (z850 - z850_climatology.sel(month=synoptic_time.month)).rename("z850_anomaly")

            hokkaido_min_anoms.append(_box_min(z850_anomaly, hokkaido_box))
            hokkaido_mean_anoms.append(_box_weighted_mean(z850_anomaly, hokkaido_box))
            sea_of_japan_min_anoms.append(_box_min(z850_anomaly, sea_of_japan_box))

            temp_grad = compute_temperature_gradient_magnitude(synoptic_snapshot_850)
            temp_grad_display = (
                temp_grad * float(temp_grad.attrs.get("display_scale_factor", 1.0))
            ).rename("temperature_gradient_display")
            frontality_max_values.append(_box_max(temp_grad_display, front_box))
            pacific_frontality_max_values.append(_box_max(temp_grad_display, pacific_front_box))

        record = row.to_dict()
        record.update(
            {
                "jpcz_polygon_mean_convergence_peak_925": jpcz_mean,
                "jpcz_polygon_max_convergence_peak_925": jpcz_max,
                "coastal_japan_mean_convergence_peak_925": coastal_mean,
                "coastal_japan_max_convergence_peak_925": coastal_max,
                "coastal_to_jpcz_mean_convergence_ratio": _safe_ratio(coastal_mean, jpcz_mean),
                "coastal_to_jpcz_max_convergence_ratio": _safe_ratio(coastal_max, jpcz_max),
                "pacific_east_of_japan_mean_convergence_peak_925": pacific_mean,
                "pacific_east_of_japan_max_convergence_peak_925": pacific_max,
                "pacific_to_jpcz_mean_convergence_ratio": _safe_ratio(pacific_mean, jpcz_mean),
                "pacific_to_jpcz_max_convergence_ratio": _safe_ratio(pacific_max, jpcz_max),
                "sea_of_japan_mean_vorticity_peak_925": soj_vort_mean,
                "sea_of_japan_max_vorticity_peak_925": soj_vort_max,
                "hokkaido_min_z850_anomaly_tminus12_to_tplus12": float(np.nanmin(hokkaido_min_anoms)),
                "hokkaido_mean_z850_anomaly_tminus12_to_tplus12": float(np.nanmin(hokkaido_mean_anoms)),
                "sea_of_japan_min_z850_anomaly_tminus12_to_tplus12": float(np.nanmin(sea_of_japan_min_anoms)),
                "front_box_max_temp_gradient_850_tminus12_to_tplus12": float(np.nanmax(frontality_max_values)),
                "pacific_box_max_temp_gradient_850_tminus12_to_tplus12": float(np.nanmax(pacific_frontality_max_values)),
            }
        )
        records.append(record)

    return pd.DataFrame(records, index=catalog.index)


def standardize_feature_table(
    feature_df: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Standardize selected feature columns using z scores."""
    subset = feature_df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    means = subset.mean(axis=0)
    stds = subset.std(axis=0).replace(0.0, np.nan)
    standardized = (subset - means) / stds
    return standardized, means, stds


def compute_pca_scores(
    standardized_df: pd.DataFrame,
    *,
    n_components: int = 3,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute PCA scores using singular value decomposition."""
    valid = standardized_df.dropna(axis=0, how="any")
    if valid.empty:
        raise ValueError("No complete rows are available for PCA.")

    matrix = valid.to_numpy(dtype=float)
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    components = vt[:n_components]
    scores = matrix @ components.T

    total_variance = (singular_values**2).sum()
    explained_variance_ratio = (singular_values[:n_components] ** 2) / total_variance

    score_df = pd.DataFrame(
        scores,
        index=valid.index,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    return score_df, explained_variance_ratio


def assign_hierarchical_clusters(
    standardized_df: pd.DataFrame,
    *,
    n_clusters: int = 4,
    method: str = "ward",
) -> pd.Series:
    """Assign hierarchical-clustering labels from a standardized feature table."""
    from scipy.cluster.hierarchy import fcluster, linkage

    valid = standardized_df.dropna(axis=0, how="any")
    if valid.empty:
        raise ValueError("No complete rows are available for clustering.")

    linkage_matrix = linkage(valid.to_numpy(dtype=float), method=method)
    cluster_ids = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    return pd.Series(cluster_ids, index=valid.index, name=f"cluster_{method}_{n_clusters}")


def compute_mean_silhouette_score(
    standardized_df: pd.DataFrame,
    cluster_labels: pd.Series,
) -> float:
    """Compute a simple mean silhouette score for one clustering solution."""
    from scipy.spatial.distance import pdist, squareform

    valid = standardized_df.dropna(axis=0, how="any")
    if valid.empty:
        return float("nan")

    labels = cluster_labels.reindex(valid.index)
    keep_mask = labels.notna()
    valid = valid.loc[keep_mask]
    labels = labels.loc[keep_mask]

    unique_labels = pd.Index(labels.unique())
    if len(unique_labels) < 2 or len(valid) < 2:
        return float("nan")

    distance_matrix = squareform(pdist(valid.to_numpy(dtype=float), metric="euclidean"))
    label_array = labels.to_numpy()
    silhouette_values: list[float] = []

    for row_idx, cluster_id in enumerate(label_array):
        same_cluster = label_array == cluster_id
        same_count = int(same_cluster.sum())

        if same_count <= 1:
            intra_cluster_distance = 0.0
        else:
            intra_cluster_distance = float(distance_matrix[row_idx, same_cluster].sum() / (same_count - 1))

        nearest_other_cluster_distance = float("inf")
        for other_cluster_id in unique_labels:
            if other_cluster_id == cluster_id:
                continue
            other_cluster = label_array == other_cluster_id
            other_distance = float(distance_matrix[row_idx, other_cluster].mean())
            nearest_other_cluster_distance = min(nearest_other_cluster_distance, other_distance)

        denominator = max(intra_cluster_distance, nearest_other_cluster_distance)
        if denominator == 0.0 or np.isinf(nearest_other_cluster_distance):
            silhouette_values.append(0.0)
        else:
            silhouette_values.append(
                (nearest_other_cluster_distance - intra_cluster_distance) / denominator
            )

    return float(np.mean(silhouette_values))


def evaluate_hierarchical_cluster_solutions(
    standardized_df: pd.DataFrame,
    *,
    cluster_counts: Sequence[int] = (2, 3, 4, 5, 6),
    method: str = "ward",
) -> pd.DataFrame:
    """Summarize clustering quality across a range of k values."""
    rows = []

    for n_clusters in cluster_counts:
        labels = assign_hierarchical_clusters(
            standardized_df,
            n_clusters=n_clusters,
            method=method,
        )
        cluster_sizes = labels.value_counts().sort_index()
        rows.append(
            {
                "n_clusters": int(n_clusters),
                "n_complete_rows": int(len(labels)),
                "mean_silhouette_score": compute_mean_silhouette_score(standardized_df, labels),
                "smallest_cluster_size": int(cluster_sizes.min()),
                "largest_cluster_size": int(cluster_sizes.max()),
                "singleton_cluster_count": int((cluster_sizes == 1).sum()),
            }
        )

    return pd.DataFrame(rows)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _weighted_mean_masked(field: xr.DataArray, mask: xr.DataArray) -> float:
    masked = field.where(mask)
    weights = build_coslat_weights(
        masked.latitude,
        masked.longitude,
        mask=mask,
    )
    total_weight = float(weights.sum().values)
    if total_weight == 0.0:
        return float("nan")
    return float(((masked.fillna(0.0) * weights).sum() / total_weight).values)


def _box_weighted_mean(field: xr.DataArray, box: BoundingBox) -> float:
    boxed = field.sel(
        longitude=slice(box.lon_min, box.lon_max),
        latitude=slice(box.lat_max, box.lat_min),
    )
    weights = build_coslat_weights(boxed.latitude, boxed.longitude)
    total_weight = float(weights.sum().values)
    if total_weight == 0.0:
        return float("nan")
    return float(((boxed.fillna(0.0) * weights).sum() / total_weight).values)


def _masked_max(field: xr.DataArray, mask: xr.DataArray) -> float:
    masked = field.where(mask)
    return float(masked.max(skipna=True).values)


def _box_max(field: xr.DataArray, box: BoundingBox) -> float:
    boxed = field.sel(
        longitude=slice(box.lon_min, box.lon_max),
        latitude=slice(box.lat_max, box.lat_min),
    )
    return float(boxed.max(skipna=True).values)


def _box_min(field: xr.DataArray, box: BoundingBox) -> float:
    boxed = field.sel(
        longitude=slice(box.lon_min, box.lon_max),
        latitude=slice(box.lat_max, box.lat_min),
    )
    return float(boxed.min(skipna=True).values)
