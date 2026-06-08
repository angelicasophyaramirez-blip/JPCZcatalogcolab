from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from jpcz_catalog.analysis import build_manual_verification_scaffold


DEFAULT_OBJECTIVE_LABEL = "weak_or_unclear"
OFFSHORE_LABEL = "offshore_jpcz_core"
COASTAL_LABEL = "coastal_interaction"
MIXED_LABEL = "mixed_transition"


def ensure_event_index(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a stable integer event index column."""
    catalog = catalog_df.copy()
    if "event_index" not in catalog.columns:
        catalog["event_index"] = catalog.index.astype(int)
    return catalog


def assign_objective_labels_from_thresholds(
    metrics_df: pd.DataFrame,
    *,
    polygon_qflux_min: float,
    polygon_div_max: float,
    coastal_qflux_split: float,
    coastal_div_max: float,
) -> pd.DataFrame:
    """Apply the simplified offshore/coastal regime rules to an event table."""
    labeled_df = metrics_df.copy()
    offshore_mask = (
        np.isfinite(labeled_df["polygon_qflux_850_mean"])
        & np.isfinite(labeled_df["polygon_div_925_mean"])
        & np.isfinite(labeled_df["coastal_qflux_850_mean"])
        & (labeled_df["polygon_qflux_850_mean"] >= polygon_qflux_min)
        & (labeled_df["polygon_div_925_mean"] <= polygon_div_max)
        & (labeled_df["coastal_qflux_850_mean"] < coastal_qflux_split)
    )
    coastal_mask = (
        np.isfinite(labeled_df["coastal_qflux_850_mean"])
        & np.isfinite(labeled_df["coastal_div_925_mean"])
        & (labeled_df["coastal_qflux_850_mean"] >= coastal_qflux_split)
        & (labeled_df["coastal_div_925_mean"] <= coastal_div_max)
    )
    labels = np.full(len(labeled_df), DEFAULT_OBJECTIVE_LABEL, dtype=object)
    labels[offshore_mask.values] = OFFSHORE_LABEL
    labels[coastal_mask.values] = COASTAL_LABEL
    labels[(offshore_mask & coastal_mask).values] = MIXED_LABEL
    labeled_df["offshore_rule_pass"] = offshore_mask.astype(bool)
    labeled_df["coastal_rule_pass"] = coastal_mask.astype(bool)
    labeled_df["objective_regime_label"] = labels
    return labeled_df


def collapse_label_sequence(
    labels: Iterable[str],
    *,
    drop_labels: tuple[str, ...] = (DEFAULT_OBJECTIVE_LABEL,),
) -> list[str]:
    """Drop weak labels and collapse consecutive duplicates."""
    collapsed: list[str] = []
    for raw_label in labels:
        if pd.isna(raw_label):
            continue
        label = str(raw_label)
        if label in drop_labels:
            continue
        if not collapsed or collapsed[-1] != label:
            collapsed.append(label)
    return collapsed


def classify_episode_regime_path(labels: Iterable[str]) -> str:
    """Reduce an event-label sequence to a compact episode path label."""
    clear_sequence = collapse_label_sequence(labels)
    if not clear_sequence:
        return "weak_only"

    unique_labels = set(clear_sequence)
    if unique_labels == {OFFSHORE_LABEL}:
        return "offshore_only"
    if unique_labels == {COASTAL_LABEL}:
        return "coastal_only"
    if unique_labels == {MIXED_LABEL}:
        return "mixed_event_only"
    if MIXED_LABEL in unique_labels:
        return "mixed_or_oscillating"

    first_offshore = next((idx for idx, label in enumerate(clear_sequence) if label == OFFSHORE_LABEL), None)
    first_coastal = next((idx for idx, label in enumerate(clear_sequence) if label == COASTAL_LABEL), None)
    if first_offshore is not None and first_coastal is not None:
        if first_offshore < first_coastal and OFFSHORE_LABEL not in clear_sequence[first_coastal + 1 :]:
            return "offshore_to_coastal"
        if first_coastal < first_offshore and COASTAL_LABEL not in clear_sequence[first_offshore + 1 :]:
            return "coastal_to_offshore"
    return "mixed_or_oscillating"


def assign_objective_episode_ids(
    labeled_df: pd.DataFrame,
    *,
    gap_hours: float,
    episode_prefix: str = "obj",
    event_start_column: str = "event_start",
    event_end_column: str = "event_end",
    event_peak_column: str = "event_peak",
) -> pd.DataFrame:
    """Group nearby event peaks into broader objective-regime episodes."""
    labeled = ensure_event_index(labeled_df)
    catalog = labeled.copy()

    for column in (event_start_column, event_end_column, event_peak_column):
        if column in catalog.columns:
            catalog[column] = pd.to_datetime(catalog[column])

    sort_columns = [column for column in (event_start_column, event_peak_column, "event_index") if column in catalog.columns]
    catalog = catalog.sort_values(sort_columns).reset_index(drop=True)

    episode_ids: list[str] = []
    gap_from_previous_hours: list[float] = []
    episode_counter = 0
    previous_end = None
    previous_peak = None

    for _, row in catalog.iterrows():
        current_start = row.get(event_start_column, pd.NaT)
        current_end = row.get(event_end_column, pd.NaT)
        current_peak = row.get(event_peak_column, pd.NaT)

        if pd.isna(current_start):
            current_start = current_peak
        if pd.isna(current_end):
            current_end = current_peak

        gap_hours_value = np.nan
        start_new_episode = previous_end is None and previous_peak is None
        if not start_new_episode:
            if previous_end is not None and current_start is not None and not pd.isna(previous_end) and not pd.isna(current_start):
                gap_hours_value = float((current_start - previous_end).total_seconds() / 3600.0)
            elif previous_peak is not None and current_peak is not None and not pd.isna(previous_peak) and not pd.isna(current_peak):
                gap_hours_value = float((current_peak - previous_peak).total_seconds() / 3600.0)
            start_new_episode = bool(pd.isna(gap_hours_value) or gap_hours_value > gap_hours)

        if start_new_episode:
            episode_counter += 1

        episode_ids.append(f"{episode_prefix}_{int(gap_hours):02d}h_{episode_counter:03d}")
        gap_from_previous_hours.append(gap_hours_value)
        previous_end = current_end
        previous_peak = current_peak

    catalog["objective_episode_id"] = episode_ids
    catalog["gap_from_previous_event_hours"] = gap_from_previous_hours
    catalog["objective_episode_gap_hours"] = float(gap_hours)
    return catalog


def summarize_objective_episodes(
    labeled_df: pd.DataFrame,
    *,
    gap_hours: float,
    label_column: str = "objective_regime_label",
    event_start_column: str = "event_start",
    event_end_column: str = "event_end",
    event_peak_column: str = "event_peak",
    event_peak_jst_column: str = "event_peak_jst",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return event-level and episode-level objective-regime timing summaries."""
    events_df = assign_objective_episode_ids(
        labeled_df,
        gap_hours=gap_hours,
        event_start_column=event_start_column,
        event_end_column=event_end_column,
        event_peak_column=event_peak_column,
    )

    summary_rows: list[dict[str, object]] = []
    for episode_id, episode_df in events_df.groupby("objective_episode_id", sort=False):
        episode_df = episode_df.sort_values(event_peak_column).copy()
        label_sequence = episode_df[label_column].fillna(DEFAULT_OBJECTIVE_LABEL).astype(str).tolist()
        clear_sequence = collapse_label_sequence(label_sequence)
        episode_path = classify_episode_regime_path(label_sequence)

        offshore_rows = episode_df.loc[episode_df[label_column] == OFFSHORE_LABEL]
        coastal_rows = episode_df.loc[episode_df[label_column] == COASTAL_LABEL]
        first_offshore_peak = offshore_rows[event_peak_column].min() if not offshore_rows.empty else pd.NaT
        first_coastal_peak = coastal_rows[event_peak_column].min() if not coastal_rows.empty else pd.NaT
        offshore_precedes_coastal = bool(
            pd.notna(first_offshore_peak)
            and pd.notna(first_coastal_peak)
            and first_offshore_peak < first_coastal_peak
        )
        offshore_to_coastal_lag_hours = (
            float((first_coastal_peak - first_offshore_peak).total_seconds() / 3600.0)
            if offshore_precedes_coastal
            else np.nan
        )
        coastal_precedes_offshore = bool(
            pd.notna(first_offshore_peak)
            and pd.notna(first_coastal_peak)
            and first_coastal_peak < first_offshore_peak
        )
        coastal_to_offshore_lag_hours = (
            float((first_offshore_peak - first_coastal_peak).total_seconds() / 3600.0)
            if coastal_precedes_offshore
            else np.nan
        )

        row = {
            "objective_episode_id": episode_id,
            "objective_episode_gap_hours": float(gap_hours),
            "objective_episode_regime_path": episode_path,
            "event_count": int(len(episode_df)),
            "clear_event_count": int(sum(label != DEFAULT_OBJECTIVE_LABEL for label in label_sequence)),
            "offshore_event_count": int((episode_df[label_column] == OFFSHORE_LABEL).sum()),
            "coastal_event_count": int((episode_df[label_column] == COASTAL_LABEL).sum()),
            "mixed_event_count": int((episode_df[label_column] == MIXED_LABEL).sum()),
            "weak_event_count": int((episode_df[label_column] == DEFAULT_OBJECTIVE_LABEL).sum()),
            "episode_start": episode_df[event_start_column].min() if event_start_column in episode_df.columns else pd.NaT,
            "episode_end": episode_df[event_end_column].max() if event_end_column in episode_df.columns else pd.NaT,
            "first_event_peak": episode_df[event_peak_column].min() if event_peak_column in episode_df.columns else pd.NaT,
            "last_event_peak": episode_df[event_peak_column].max() if event_peak_column in episode_df.columns else pd.NaT,
            "first_event_peak_jst": episode_df[event_peak_jst_column].min() if event_peak_jst_column in episode_df.columns else pd.NaT,
            "last_event_peak_jst": episode_df[event_peak_jst_column].max() if event_peak_jst_column in episode_df.columns else pd.NaT,
            "collapsed_clear_sequence": " -> ".join(clear_sequence) if clear_sequence else "weak_only",
            "offshore_precedes_coastal": offshore_precedes_coastal,
            "offshore_to_coastal_lag_hours": offshore_to_coastal_lag_hours,
            "coastal_precedes_offshore": coastal_precedes_offshore,
            "coastal_to_offshore_lag_hours": coastal_to_offshore_lag_hours,
            "first_offshore_peak": first_offshore_peak,
            "first_coastal_peak": first_coastal_peak,
        }
        summary_rows.append(row)

    episode_summary_df = pd.DataFrame(summary_rows)
    if not episode_summary_df.empty:
        episode_summary_df["episode_duration_hours"] = (
            episode_summary_df["episode_end"] - episode_summary_df["episode_start"]
        ).dt.total_seconds() / 3600.0

    event_level_df = events_df.merge(
        episode_summary_df[
            [
                "objective_episode_id",
                "objective_episode_regime_path",
                "offshore_precedes_coastal",
                "offshore_to_coastal_lag_hours",
                "coastal_precedes_offshore",
                "coastal_to_offshore_lag_hours",
            ]
        ],
        on="objective_episode_id",
        how="left",
    )
    return event_level_df, episode_summary_df


def summarize_padded_spell_windows(
    catalog_df: pd.DataFrame,
    *,
    gap_hours: float,
    padding_hours: float,
    episode_prefix: str = "catalog",
    event_start_column: str = "event_start",
    event_end_column: str = "event_end",
    event_peak_column: str = "event_peak",
    event_peak_jst_column: str = "event_peak_jst",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group the catalog into broader timing-only spells plus padded analysis windows."""
    spell_event_df = assign_objective_episode_ids(
        catalog_df,
        gap_hours=gap_hours,
        episode_prefix=episode_prefix,
        event_start_column=event_start_column,
        event_end_column=event_end_column,
        event_peak_column=event_peak_column,
    )

    padding_delta = pd.Timedelta(hours=float(padding_hours))
    summary_rows: list[dict[str, object]] = []
    for spell_id, spell_df in spell_event_df.groupby("objective_episode_id", sort=False):
        spell_df = spell_df.sort_values(event_peak_column).copy()
        spell_start = spell_df[event_start_column].min() if event_start_column in spell_df.columns else pd.NaT
        spell_end = spell_df[event_end_column].max() if event_end_column in spell_df.columns else pd.NaT
        first_peak = spell_df[event_peak_column].min() if event_peak_column in spell_df.columns else pd.NaT
        last_peak = spell_df[event_peak_column].max() if event_peak_column in spell_df.columns else pd.NaT
        analysis_start = spell_start - padding_delta if pd.notna(spell_start) else pd.NaT
        analysis_end = spell_end + padding_delta if pd.notna(spell_end) else pd.NaT

        summary_rows.append(
            {
                "catalog_spell_id": spell_id,
                "catalog_spell_gap_hours": float(gap_hours),
                "analysis_padding_hours": float(padding_hours),
                "event_count": int(len(spell_df)),
                "spell_start": spell_start,
                "spell_end": spell_end,
                "first_event_peak": first_peak,
                "last_event_peak": last_peak,
                "first_event_peak_jst": spell_df[event_peak_jst_column].min()
                if event_peak_jst_column in spell_df.columns
                else pd.NaT,
                "last_event_peak_jst": spell_df[event_peak_jst_column].max()
                if event_peak_jst_column in spell_df.columns
                else pd.NaT,
                "analysis_start": analysis_start,
                "analysis_end": analysis_end,
                "spell_duration_hours": (
                    float((spell_end - spell_start).total_seconds() / 3600.0)
                    if pd.notna(spell_start) and pd.notna(spell_end)
                    else np.nan
                ),
                "analysis_window_hours": (
                    float((analysis_end - analysis_start).total_seconds() / 3600.0)
                    if pd.notna(analysis_start) and pd.notna(analysis_end)
                    else np.nan
                ),
                "max_internal_gap_hours": float(spell_df["gap_from_previous_event_hours"].dropna().max())
                if spell_df["gap_from_previous_event_hours"].notna().any()
                else 0.0,
            }
        )

    spell_summary_df = pd.DataFrame(summary_rows)
    spell_event_df = spell_event_df.rename(columns={"objective_episode_id": "catalog_spell_id"})
    spell_event_df["catalog_spell_gap_hours"] = float(gap_hours)
    spell_event_df["analysis_padding_hours"] = float(padding_hours)
    return spell_event_df, spell_summary_df


def summarize_gap_sensitivity(
    labeled_df: pd.DataFrame,
    *,
    gap_hours_options: Iterable[float],
    label_column: str = "objective_regime_label",
) -> pd.DataFrame:
    """Compact summary showing how episode counts change with the gap rule."""
    rows: list[dict[str, object]] = []
    for gap_hours in gap_hours_options:
        _, episode_summary_df = summarize_objective_episodes(
            labeled_df,
            gap_hours=gap_hours,
            label_column=label_column,
        )
        path_counts = episode_summary_df["objective_episode_regime_path"].value_counts() if not episode_summary_df.empty else pd.Series(dtype=int)
        row = {
            "gap_hours": float(gap_hours),
            "episode_count": int(len(episode_summary_df)),
            "offshore_only_episodes": int(path_counts.get("offshore_only", 0)),
            "coastal_only_episodes": int(path_counts.get("coastal_only", 0)),
            "offshore_to_coastal_episodes": int(path_counts.get("offshore_to_coastal", 0)),
            "coastal_to_offshore_episodes": int(path_counts.get("coastal_to_offshore", 0)),
            "mixed_or_oscillating_episodes": int(path_counts.get("mixed_or_oscillating", 0)),
            "weak_only_episodes": int(path_counts.get("weak_only", 0)),
            "median_offshore_to_coastal_lag_hours": float(
                episode_summary_df.loc[
                    episode_summary_df["offshore_precedes_coastal"],
                    "offshore_to_coastal_lag_hours",
                ].median()
            )
            if not episode_summary_df.empty
            else np.nan,
            "median_coastal_to_offshore_lag_hours": float(
                episode_summary_df.loc[
                    episode_summary_df["coastal_precedes_offshore"],
                    "coastal_to_offshore_lag_hours",
                ].median()
            )
            if not episode_summary_df.empty
            else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_objective_manual_review_scaffold(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Extend the existing manual-review scaffold with regime/timing review columns."""
    scaffold = build_manual_verification_scaffold(catalog_df)
    defaults: dict[str, object] = {
        "manual_regime_label": "",
        "manual_episode_stage": "",
        "manual_transition_type": "",
        "coastal_impact_present_manual": "",
        "coastal_impact_intensity_manual": "",
        "coastal_impact_location_manual": "",
        "coastal_impact_notes": "",
        "transition_notes": "",
        "regime_confidence_manual": "",
    }
    for column_name, default_value in defaults.items():
        if column_name not in scaffold.columns:
            scaffold[column_name] = default_value
    return scaffold
