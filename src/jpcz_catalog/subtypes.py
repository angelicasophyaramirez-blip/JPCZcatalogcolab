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
    HOKKAIDO_BOX,
    HOKKAIDO_FRONT_BOX,
    JPCZ_POLYGON_VERTICES,
    OBJECTIVE_SUBTYPE_DOMAIN,
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
    formula: str
    calculation: str
    interpretation: str
    region: str
    time_window: str
    purpose: str


FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        column_name="jpcz_polygon_mean_divergence_peak_925",
        units="1e-5 s^-1",
        meaning="Area-weighted mean signed divergence in the original JPCZ polygon at peak time.",
        formula="mean_polygon(div925)",
        calculation="Compute 925 hPa divergence from ERA5 u and v and area-average the signed del dot U field inside the JPCZ polygon.",
        interpretation="More negative values mean stronger polygon-mean convergence; more positive values mean polygon-mean divergence.",
        region="Original JPCZ polygon",
        time_window="event peak only",
        purpose="Canonical signed divergence metric in the Shinoda detection region.",
    ),
    FeatureDefinition(
        column_name="jpcz_polygon_min_divergence_peak_925",
        units="1e-5 s^-1",
        meaning="Most negative signed divergence value in the original JPCZ polygon at peak time.",
        formula="min_polygon(div925)",
        calculation="Minimum of the signed divergence field inside the JPCZ polygon.",
        interpretation="More negative values mean a stronger local convergence center within the original JPCZ polygon.",
        region="Original JPCZ polygon",
        time_window="event peak only",
        purpose="Captures the strongest local convergence center in the Shinoda detection region without clipping divergence.",
    ),
    FeatureDefinition(
        column_name="coastal_japan_mean_divergence_peak_925",
        units="1e-5 s^-1",
        meaning="Area-weighted mean signed divergence in the coastal-Japan characterization box at peak time.",
        formula="mean_coast(div925)",
        calculation="Same signed divergence field as above, summarized in the coastal-Japan box.",
        interpretation="More negative values mean stronger coastal-Japan convergence; more positive values mean coastal-Japan divergence.",
        region="Coastal Japan box",
        time_window="event peak only",
        purpose="Measures whether the event is more convergent or divergent along the west coast of Japan.",
    ),
    FeatureDefinition(
        column_name="coastal_japan_min_divergence_peak_925",
        units="1e-5 s^-1",
        meaning="Most negative signed divergence value in the coastal-Japan characterization box at peak time.",
        formula="min_coast(div925)",
        calculation="Minimum of the signed divergence field in the coastal-Japan box.",
        interpretation="More negative values mean a stronger local coastal convergence center.",
        region="Coastal Japan box",
        time_window="event peak only",
        purpose="Captures the strongest coastal convergence center without clipping divergence elsewhere in the box.",
    ),
    FeatureDefinition(
        column_name="coastal_to_jpcz_mean_divergence_ratio",
        units="unitless",
        meaning="Ratio of coastal-Japan mean signed divergence to JPCZ-polygon mean signed divergence.",
        formula="coastal_japan_mean_divergence_peak_925 / jpcz_polygon_mean_divergence_peak_925",
        calculation="coastal_japan_mean_divergence_peak_925 / jpcz_polygon_mean_divergence_peak_925",
        interpretation="When both regional means are negative, values > 1 mean the coastal box is more convergent than the polygon mean, values between 0 and 1 mean it is less convergent, and negative values indicate opposite-signed regional means.",
        region="Coastal Japan vs JPCZ polygon",
        time_window="event peak only",
        purpose="Measures how the coastal signed-divergence mean compares with the canonical JPCZ signed-divergence mean.",
    ),
    FeatureDefinition(
        column_name="coastal_to_jpcz_min_divergence_ratio",
        units="unitless",
        meaning="Ratio of the coastal-Japan minimum signed divergence to the JPCZ-polygon minimum signed divergence.",
        formula="coastal_japan_min_divergence_peak_925 / jpcz_polygon_min_divergence_peak_925",
        calculation="coastal_japan_min_divergence_peak_925 / jpcz_polygon_min_divergence_peak_925",
        interpretation="When both extrema are negative, values > 1 mean the strongest local convergence is more coastal-enhanced than polygon-centered.",
        region="Coastal Japan vs JPCZ polygon",
        time_window="event peak only",
        purpose="Alternative coastal-enhancement metric based on local signed-divergence minima instead of regional means.",
    ),
    FeatureDefinition(
        column_name="pacific_east_of_japan_mean_divergence_peak_925",
        units="1e-5 s^-1",
        meaning="Area-weighted mean signed divergence east of Japan at peak time.",
        formula="mean_pacific(div925)",
        calculation="Same signed divergence field as above, summarized in the Pacific characterization box.",
        interpretation="More negative values mean stronger Pacific-side convergence; more positive values mean Pacific-side divergence.",
        region="Pacific east of Japan box",
        time_window="event peak only",
        purpose="Measures whether the event is coupled to stronger Pacific-side signed convergence/divergence anomalies.",
    ),
    FeatureDefinition(
        column_name="pacific_east_of_japan_min_divergence_peak_925",
        units="1e-5 s^-1",
        meaning="Most negative signed divergence value east of Japan at peak time.",
        formula="min_pacific(div925)",
        calculation="Minimum of the signed divergence field in the Pacific characterization box.",
        interpretation="More negative values mean a stronger local Pacific-side convergence center.",
        region="Pacific east of Japan box",
        time_window="event peak only",
        purpose="Captures the strongest Pacific-side convergence center without clipping divergence elsewhere in the box.",
    ),
    FeatureDefinition(
        column_name="pacific_to_jpcz_mean_divergence_ratio",
        units="unitless",
        meaning="Ratio of Pacific-box mean signed divergence to JPCZ-polygon mean signed divergence.",
        formula="pacific_east_of_japan_mean_divergence_peak_925 / jpcz_polygon_mean_divergence_peak_925",
        calculation="pacific_east_of_japan_mean_divergence_peak_925 / jpcz_polygon_mean_divergence_peak_925",
        interpretation="When both regional means are negative, values > 1 mean the Pacific box is more convergent than the polygon mean; negative values indicate opposite-signed regional means.",
        region="Pacific east of Japan vs JPCZ polygon",
        time_window="event peak only",
        purpose="Measures how strongly the event is coupled to Pacific-side signed convergence/divergence relative to the polygon mean.",
    ),
    FeatureDefinition(
        column_name="pacific_to_jpcz_min_divergence_ratio",
        units="unitless",
        meaning="Ratio of the Pacific-box minimum signed divergence to the JPCZ-polygon minimum signed divergence.",
        formula="pacific_east_of_japan_min_divergence_peak_925 / jpcz_polygon_min_divergence_peak_925",
        calculation="pacific_east_of_japan_min_divergence_peak_925 / jpcz_polygon_min_divergence_peak_925",
        interpretation="When both extrema are negative, values > 1 mean the strongest local convergence is more Pacific-enhanced than polygon-centered.",
        region="Pacific east of Japan vs JPCZ polygon",
        time_window="event peak only",
        purpose="Alternative Pacific-coupling metric based on local signed-divergence minima instead of regional means.",
    ),
    FeatureDefinition(
        column_name="sea_of_japan_mean_vorticity_peak_925",
        units="1e-5 s^-1",
        meaning="Area-weighted mean relative vorticity in the Sea of Japan box at peak time.",
        formula="mean_soj(zeta925), where zeta925 = dv/dx - du/dy",
        calculation="Compute 925 hPa relative vorticity from ERA5 u and v and area-average it in the Sea of Japan box.",
        interpretation="More positive values indicate stronger cyclonic low-level circulation over the Sea of Japan; values near zero indicate weaker circulation-centered forcing.",
        region="Sea of Japan box",
        time_window="event peak only",
        purpose="Quantifies circulation-centered forcing over the Sea of Japan.",
    ),
    FeatureDefinition(
        column_name="sea_of_japan_max_vorticity_peak_925",
        units="1e-5 s^-1",
        meaning="Maximum relative vorticity in the Sea of Japan box at peak time.",
        formula="max_soj(zeta925), where zeta925 = dv/dx - du/dy",
        calculation="Maximum of the 925 hPa relative-vorticity field in the Sea of Japan box.",
        interpretation="More positive values indicate a stronger local cyclonic vorticity center in the Sea of Japan box.",
        region="Sea of Japan box",
        time_window="event peak only",
        purpose="Captures the strongest low-level circulation center in the Sea of Japan.",
    ),
    FeatureDefinition(
        column_name="hokkaido_min_z850_anomaly_tminus12_to_tplus12",
        units="gpm",
        meaning="Most negative 850 hPa geopotential-height anomaly in the Hokkaido box over the t-12, t0, t+12 window.",
        formula="min_{t in [-12,0,+12]} min_hokkaido(z850_event(t) - z850_climatology(month(t)))",
        calculation="Compute z850 anomaly as event-time z850 minus monthly climatological z850, then take the minimum box value over the three-time window.",
        interpretation="More negative values indicate a deeper passing low or trough near Hokkaido; values closer to zero indicate weaker synoptic-height forcing.",
        region="Hokkaido box",
        time_window="t-12 h, t0, t+12 h",
        purpose="Quantifies passing-low or trough forcing near Hokkaido.",
    ),
    FeatureDefinition(
        column_name="hokkaido_mean_z850_anomaly_tminus12_to_tplus12",
        units="gpm",
        meaning="Most negative box-mean 850 hPa geopotential-height anomaly in the Hokkaido box over the t-12, t0, t+12 window.",
        formula="min_{t in [-12,0,+12]} mean_hokkaido(z850_event(t) - z850_climatology(month(t)))",
        calculation="Compute box-mean z850 anomaly at each offset time and save the most negative value.",
        interpretation="More negative values indicate broader and more spatially coherent synoptic-height depression near Hokkaido.",
        region="Hokkaido box",
        time_window="t-12 h, t0, t+12 h",
        purpose="A more stable synoptic-forcing metric than the single gridpoint minimum.",
    ),
    FeatureDefinition(
        column_name="sea_of_japan_min_z850_anomaly_tminus12_to_tplus12",
        units="gpm",
        meaning="Most negative 850 hPa geopotential-height anomaly in the Sea of Japan box over the t-12, t0, t+12 window.",
        formula="min_{t in [-12,0,+12]} min_soj(z850_event(t) - z850_climatology(month(t)))",
        calculation="Same anomaly logic as above, but evaluated in the Sea of Japan box.",
        interpretation="More negative values indicate a deeper synoptic-height depression centered over the Sea of Japan.",
        region="Sea of Japan box",
        time_window="t-12 h, t0, t+12 h",
        purpose="Quantifies whether the synoptic-height depression is centered over the Sea of Japan.",
    ),
    FeatureDefinition(
        column_name="front_box_max_temp_gradient_850_tminus12_to_tplus12",
        units="K (100 km)^-1",
        meaning="Largest 850 hPa horizontal temperature-gradient magnitude in the Hokkaido front box over the t-12, t0, t+12 window.",
        formula="max_{t in [-12,0,+12]} max_front(|grad T850(t)|), where |grad T850| = sqrt((dT/dx)^2 + (dT/dy)^2)",
        calculation="Compute |grad T850| and save the maximum box value across the three-time window.",
        interpretation="Larger values indicate stronger frontal or baroclinic forcing upstream of the JPCZ.",
        region="Hokkaido front box",
        time_window="t-12 h, t0, t+12 h",
        purpose="Quantifies frontal or baroclinic forcing upstream of the JPCZ.",
    ),
    FeatureDefinition(
        column_name="pacific_box_max_temp_gradient_850_tminus12_to_tplus12",
        units="K (100 km)^-1",
        meaning="Largest 850 hPa horizontal temperature-gradient magnitude in the Pacific front box over the t-12, t0, t+12 window.",
        formula="max_{t in [-12,0,+12]} max_pacific_front(|grad T850(t)|), where |grad T850| = sqrt((dT/dx)^2 + (dT/dy)^2)",
        calculation="Compute |grad T850| and save the maximum Pacific-box value across the three-time window.",
        interpretation="Larger values indicate stronger frontal or baroclinic forcing east of Japan on the Pacific side.",
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
    domain: BoundingBox = OBJECTIVE_SUBTYPE_DOMAIN,
    level: int = 850,
    z_name: str = "geopotential",
) -> xr.DataArray:
    """Compute monthly mean 850 hPa geopotential height climatology."""
    monthly_sums: dict[int, xr.DataArray] = {}
    monthly_counts: dict[int, int] = {}

    for month in months:
        month_sum = None
        month_count = 0

        for year in years:
            start, end = month_window(year, month)
            print(f"Climatology month {month:02d}: loading {year}-{month:02d}")
            subset = ds[[z_name]].sel(
                time=slice(start, end),
                longitude=slice(domain.lon_min, domain.lon_max),
                latitude=slice(domain.lat_max, domain.lat_min),
            )

            if "level" in subset.dims or "level" in subset.coords:
                subset = subset.sel(level=level)

            geopotential_height = subset[z_name] / 9.80665
            window_sum = geopotential_height.sum("time").load()
            window_count = int(geopotential_height.sizes["time"])

            if month_sum is None:
                month_sum = window_sum
            else:
                month_sum = month_sum + window_sum
            month_count += window_count

        if month_sum is None or month_count == 0:
            continue

        monthly_sums[int(month)] = month_sum
        monthly_counts[int(month)] = month_count

    climatology_slices = []
    for month in sorted(monthly_sums):
        month_mean = (monthly_sums[month] / monthly_counts[month]).expand_dims(month=[month])
        climatology_slices.append(month_mean)

    climatology = xr.concat(climatology_slices, dim="month").rename("monthly_z850_climatology")
    climatology.attrs["units"] = "gpm"
    climatology.attrs["display_units"] = "gpm"
    return climatology


def build_objective_subtype_feature_table(
    ds: xr.Dataset,
    catalog_df: pd.DataFrame,
    *,
    z850_climatology: xr.DataArray,
    characterization_domain: BoundingBox = OBJECTIVE_SUBTYPE_DOMAIN,
    polygon_vertices: Sequence[tuple[float, float]] = JPCZ_POLYGON_VERTICES,
    coastal_box: BoundingBox = COASTAL_JAPAN_BOX,
    pacific_box: BoundingBox = PACIFIC_EAST_OF_JAPAN_BOX,
    hokkaido_box: BoundingBox = HOKKAIDO_BOX,
    sea_of_japan_box: BoundingBox = SEA_OF_JAPAN_BOX,
    front_box: BoundingBox = HOKKAIDO_FRONT_BOX,
    pacific_front_box: BoundingBox = PACIFIC_FRONT_BOX,
    offset_hours: Sequence[int] = (-12, 0, 12),
    progress_every: int = 10,
) -> pd.DataFrame:
    """Build the event-level objective subtype feature table."""
    catalog = catalog_df.copy()
    for column_name in ("event_start", "event_end", "event_peak"):
        if column_name in catalog.columns:
            catalog[column_name] = pd.to_datetime(catalog[column_name])

    geometry_925 = None
    records: list[dict[str, object]] = []
    available_climatology_months = {
        int(month_value) for month_value in z850_climatology["month"].values.tolist()
    }

    total_events = len(catalog)

    for event_number, (idx, row) in enumerate(catalog.iterrows(), start=1):
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

        divergence_field = (
            compute_divergence_field(
            peak_snapshot_925,
            dx=geometry_925.dx,
            dy=geometry_925.dy,
            )
            * 1e5
        ).rename("divergence_925_display")
        vorticity_field = (
            compute_relative_vorticity_field(
                peak_snapshot_925,
                dx=geometry_925.dx,
                dy=geometry_925.dy,
            )
            * 1e5
        ).rename("relative_vorticity_925_display")

        jpcz_mean = _weighted_mean_masked(divergence_field, geometry_925.polygon_mask)
        jpcz_min = _masked_min(divergence_field, geometry_925.polygon_mask)
        coastal_mean = _box_weighted_mean(divergence_field, coastal_box)
        coastal_min = _box_min(divergence_field, coastal_box)
        pacific_mean = _box_weighted_mean(divergence_field, pacific_box)
        pacific_min = _box_min(divergence_field, pacific_box)
        soj_vort_mean = _box_weighted_mean(vorticity_field, sea_of_japan_box)
        soj_vort_max = _box_max(vorticity_field, sea_of_japan_box)

        hokkaido_min_anoms = []
        hokkaido_mean_anoms = []
        sea_of_japan_min_anoms = []
        frontality_max_values = []
        pacific_frontality_max_values = []

        for offset in offset_hours:
            synoptic_time = pd.Timestamp(row["event_peak"]) + pd.Timedelta(hours=offset)
            if synoptic_time.month not in available_climatology_months:
                raise KeyError(
                    "Monthly z850 climatology is missing month "
                    f"{synoptic_time.month}. Available months: "
                    f"{sorted(available_climatology_months)}"
                )
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
                "jpcz_polygon_mean_divergence_peak_925": jpcz_mean,
                "jpcz_polygon_min_divergence_peak_925": jpcz_min,
                "coastal_japan_mean_divergence_peak_925": coastal_mean,
                "coastal_japan_min_divergence_peak_925": coastal_min,
                "coastal_to_jpcz_mean_divergence_ratio": _safe_ratio(coastal_mean, jpcz_mean),
                "coastal_to_jpcz_min_divergence_ratio": _safe_ratio(coastal_min, jpcz_min),
                "pacific_east_of_japan_mean_divergence_peak_925": pacific_mean,
                "pacific_east_of_japan_min_divergence_peak_925": pacific_min,
                "pacific_to_jpcz_mean_divergence_ratio": _safe_ratio(pacific_mean, jpcz_mean),
                "pacific_to_jpcz_min_divergence_ratio": _safe_ratio(pacific_min, jpcz_min),
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

        if progress_every > 0 and (event_number % progress_every == 0 or event_number == total_events):
            print(
                f"Built objective subtype features for {event_number}/{total_events} events"
            )

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


def compute_permutation_silhouette_null(
    standardized_df: pd.DataFrame,
    *,
    n_clusters: int,
    method: str = "ward",
    n_permutations: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Estimate a null distribution for silhouette by permuting each feature column."""
    valid = standardized_df.dropna(axis=0, how="any")
    if valid.empty:
        raise ValueError("No complete rows are available for permutation testing.")

    rng = np.random.default_rng(random_seed)
    observed_labels = assign_hierarchical_clusters(
        valid,
        n_clusters=n_clusters,
        method=method,
    )
    observed_score = compute_mean_silhouette_score(valid, observed_labels)

    rows = [
        {
            "replicate": -1,
            "kind": "observed",
            "silhouette_score": float(observed_score),
        }
    ]

    for replicate in range(n_permutations):
        permuted = valid.copy()
        for column_name in permuted.columns:
            permuted[column_name] = rng.permutation(permuted[column_name].to_numpy())

        permuted_labels = assign_hierarchical_clusters(
            permuted,
            n_clusters=n_clusters,
            method=method,
        )
        permuted_score = compute_mean_silhouette_score(permuted, permuted_labels)
        rows.append(
            {
                "replicate": int(replicate),
                "kind": "permuted",
                "silhouette_score": float(permuted_score),
            }
        )

    return pd.DataFrame(rows)


def compute_resampled_cluster_stability(
    standardized_df: pd.DataFrame,
    *,
    n_clusters: int,
    method: str = "ward",
    n_resamples: int = 200,
    sample_fraction: float = 0.8,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Estimate clustering stability with repeated subsampling and adjusted Rand index."""
    from sklearn.metrics import adjusted_rand_score

    valid = standardized_df.dropna(axis=0, how="any")
    if valid.empty:
        raise ValueError("No complete rows are available for stability analysis.")

    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError("sample_fraction must be between 0 and 1.")

    baseline_labels = assign_hierarchical_clusters(
        valid,
        n_clusters=n_clusters,
        method=method,
    )

    rng = np.random.default_rng(random_seed)
    indices = valid.index.to_numpy()
    sample_size = max(n_clusters * 2, int(round(sample_fraction * len(indices))))
    sample_size = min(sample_size, len(indices))

    rows = []
    for replicate in range(n_resamples):
        sampled_indices = rng.choice(indices, size=sample_size, replace=False)
        subset = valid.loc[sampled_indices]
        subset_labels = assign_hierarchical_clusters(
            subset,
            n_clusters=n_clusters,
            method=method,
        )
        baseline_subset = baseline_labels.loc[subset.index]
        ari = adjusted_rand_score(
            baseline_subset.to_numpy(dtype=int),
            subset_labels.to_numpy(dtype=int),
        )
        subset_silhouette = compute_mean_silhouette_score(subset, subset_labels)
        rows.append(
            {
                "replicate": int(replicate),
                "n_sampled_events": int(sample_size),
                "adjusted_rand_index": float(ari),
                "subset_silhouette_score": float(subset_silhouette),
            }
        )

    return pd.DataFrame(rows)


def compute_cluster_kruskal_wallis_table(
    feature_df: pd.DataFrame,
    *,
    cluster_column: str,
    variables: Sequence[str],
) -> pd.DataFrame:
    """Run Kruskal-Wallis tests for a list of variables across cluster groups."""
    from scipy.stats import kruskal

    rows = []
    valid_clusters = feature_df.loc[feature_df[cluster_column].notna()].copy()
    if valid_clusters.empty:
        raise ValueError(f"No non-null labels found in {cluster_column}.")

    for variable in variables:
        subset = valid_clusters.loc[:, [cluster_column, variable]].copy()
        subset[variable] = pd.to_numeric(subset[variable], errors="coerce")
        subset = subset.dropna()
        cluster_ids = sorted(pd.unique(subset[cluster_column]))
        groups = [subset.loc[subset[cluster_column] == cluster_id, variable].to_numpy() for cluster_id in cluster_ids]
        groups = [group for group in groups if len(group) > 0]

        if len(groups) < 2:
            statistic = float("nan")
            p_value = float("nan")
        else:
            statistic, p_value = kruskal(*groups)

        rows.append(
            {
                "cluster_column": cluster_column,
                "variable": variable,
                "n_groups": int(len(groups)),
                "n_complete_rows": int(len(subset)),
                "kruskal_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

    return pd.DataFrame(rows)


def compute_pairwise_mannwhitney_table(
    feature_df: pd.DataFrame,
    *,
    cluster_column: str,
    variables: Sequence[str],
    cluster_pairs: Sequence[tuple[int | float, int | float]],
) -> pd.DataFrame:
    """Run pairwise Mann-Whitney U tests for selected cluster pairs and variables."""
    from scipy.stats import mannwhitneyu

    rows = []
    valid_clusters = feature_df.loc[feature_df[cluster_column].notna()].copy()
    if valid_clusters.empty:
        raise ValueError(f"No non-null labels found in {cluster_column}.")

    for variable in variables:
        subset = valid_clusters.loc[:, [cluster_column, variable]].copy()
        subset[variable] = pd.to_numeric(subset[variable], errors="coerce")
        subset = subset.dropna()

        for left_cluster, right_cluster in cluster_pairs:
            left_values = subset.loc[subset[cluster_column] == left_cluster, variable].to_numpy()
            right_values = subset.loc[subset[cluster_column] == right_cluster, variable].to_numpy()

            if len(left_values) == 0 or len(right_values) == 0:
                statistic = float("nan")
                p_value = float("nan")
            else:
                statistic, p_value = mannwhitneyu(
                    left_values,
                    right_values,
                    alternative="two-sided",
                )

            rows.append(
                {
                    "cluster_column": cluster_column,
                    "variable": variable,
                    "left_cluster": left_cluster,
                    "right_cluster": right_cluster,
                    "n_left": int(len(left_values)),
                    "n_right": int(len(right_values)),
                    "mannwhitney_u": float(statistic),
                    "p_value": float(p_value),
                    "left_median": float(np.nanmedian(left_values)) if len(left_values) else float("nan"),
                    "right_median": float(np.nanmedian(right_values)) if len(right_values) else float("nan"),
                }
            )

    return pd.DataFrame(rows)


def benjamini_hochberg_adjust(p_values: Sequence[float | int | np.floating]) -> np.ndarray:
    """Apply Benjamini-Hochberg false-discovery-rate adjustment."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return adjusted

    finite_values = values[finite_mask]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    n = len(ranked)

    scaled = ranked * n / np.arange(1, n + 1)
    monotonic = np.minimum.accumulate(scaled[::-1])[::-1]
    monotonic = np.clip(monotonic, 0.0, 1.0)

    restored = np.empty_like(monotonic)
    restored[order] = monotonic
    adjusted[finite_mask] = restored
    return adjusted


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
    valid_weights = weights.where(masked.notnull(), 0.0)
    total_weight = float(valid_weights.sum().values)
    if total_weight == 0.0:
        return float("nan")
    return float(((masked.fillna(0.0) * valid_weights).sum() / total_weight).values)


def _box_weighted_mean(field: xr.DataArray, box: BoundingBox) -> float:
    boxed = field.sel(
        longitude=slice(box.lon_min, box.lon_max),
        latitude=slice(box.lat_max, box.lat_min),
    )
    weights = build_coslat_weights(boxed.latitude, boxed.longitude)
    valid_weights = weights.where(boxed.notnull(), 0.0)
    total_weight = float(valid_weights.sum().values)
    if total_weight == 0.0:
        return float("nan")
    return float(((boxed.fillna(0.0) * valid_weights).sum() / total_weight).values)


def _masked_max(field: xr.DataArray, mask: xr.DataArray) -> float:
    masked = field.where(mask)
    return float(masked.max(skipna=True).values)


def _masked_min(field: xr.DataArray, mask: xr.DataArray) -> float:
    masked = field.where(mask)
    return float(masked.min(skipna=True).values)


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
