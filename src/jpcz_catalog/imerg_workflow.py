"""Checkpoint-first utilities shared by the IMERG event workflow notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


IMERG_FIRST_VALID_TIME = pd.Timestamp("2000-06-01 00:00:00")


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV atomically so an interrupted notebook never corrupts it."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)


def read_checkpoint(path: str | Path, *, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read a Drive checkpoint, or return an empty frame when it does not exist."""
    input_path = Path(path)
    if not input_path.exists():
        return pd.DataFrame()
    return pd.read_csv(input_path, parse_dates=list(parse_dates))


def merge_checkpoint(existing: pd.DataFrame, fresh: pd.DataFrame, *, key: str) -> pd.DataFrame:
    """Append new records and retain the newest record for each checkpoint key."""
    combined = fresh.copy() if existing.empty else pd.concat([existing, fresh], ignore_index=True)
    return combined.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)


def prepare_imerg_event_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Add the saved catalogued convergence and IMERG event windows to a merged catalog."""
    prepared = catalog.copy()
    for column in ("event_start", "event_end", "event_peak"):
        prepared[column] = pd.to_datetime(prepared[column])

    if "event_peak_D_1e5_s-1" in prepared:
        saved_divergence = pd.to_numeric(prepared["event_peak_D_1e5_s-1"], errors="coerce")
    elif "event_peak_D_s-1" in prepared:
        saved_divergence = pd.to_numeric(prepared["event_peak_D_s-1"], errors="coerce") * 1e5
    else:
        raise KeyError(
            "The merged catalog needs event_peak_D_1e5_s-1 or event_peak_D_s-1. "
            "Rerun Notebook 06 from the current Notebook 04 catalog."
        )

    prepared["jpcz_polygon_convergence_1e5_s-1"] = -saved_divergence
    prepared = prepared.sort_values("event_start").reset_index(drop=True)
    prepared["event_id"] = prepared["event_peak"].dt.strftime("%Y%m%dT%H%M")
    if prepared["event_id"].duplicated().any():
        raise ValueError("Merged catalog event peaks must be unique for IMERG checkpointing.")

    prepared["precip_window_start"] = prepared["event_start"] - pd.Timedelta(hours=11)
    prepared["precip_window_end_exclusive"] = prepared["event_end"] + pd.Timedelta(hours=1)
    prepared["precip_window_hours"] = (
        prepared["precip_window_end_exclusive"] - prepared["precip_window_start"]
    ).dt.total_seconds() / 3600
    return prepared.loc[prepared["precip_window_start"] >= IMERG_FIRST_VALID_TIME].copy()


def completed_event_ids(event_metrics: pd.DataFrame) -> set[str]:
    """Return only event IDs whose saved IMERG metric passed coverage checks."""
    if not {"event_id", "status"}.issubset(event_metrics.columns):
        return set()
    return set(event_metrics.loc[event_metrics["status"].eq("ok"), "event_id"].astype(str))


def write_event_plan(
    events: pd.DataFrame,
    event_metrics: pd.DataFrame,
    *,
    path: str | Path,
) -> pd.DataFrame:
    """Save the complete request inventory and its current completion status."""
    complete = completed_event_ids(event_metrics)
    columns = [
        "event_id",
        "event_start",
        "event_end",
        "event_peak",
        "duration_hours",
        "precip_window_start",
        "precip_window_end_exclusive",
        "precip_window_hours",
        "jpcz_polygon_convergence_1e5_s-1",
    ]
    plan = events[columns].copy()
    plan["imerg_product"] = "GPM_3IMERGHH V07 Final / Grid/precipitationCal"
    plan["collection_status"] = np.where(plan["event_id"].isin(complete), "complete", "pending")
    atomic_csv(plan, path)
    return plan


def event_precipitation_metrics(
    event: object,
    rates: pd.DataFrame,
    *,
    region_names: tuple[str, ...],
    minimum_coverage: float,
) -> dict[str, object]:
    """Turn checkpointed half-hourly regional IMERG rates into one event row."""
    expected = pd.date_range(
        event.precip_window_start,
        event.precip_window_end_exclusive,
        freq="30min",
        inclusive="left",
    )
    indexed = rates.set_index("time").sort_index() if "time" in rates else pd.DataFrame(index=pd.DatetimeIndex([]))
    window = indexed.reindex(expected)
    row: dict[str, object] = {
        "event_id": event.event_id,
        "event_peak": event.event_peak,
        "imerg_expected_halfhours": len(expected),
        "imerg_window_hours": len(expected) * 0.5,
    }
    complete = True
    for region in region_names:
        column = f"{region}_rate_mm_hr"
        valid = int(window[column].notna().sum()) if column in window else 0
        coverage = valid / len(expected)
        accumulation = (
            window[column].sum(skipna=True) * 0.5
            if coverage >= minimum_coverage and column in window
            else np.nan
        )
        row[f"{region}_imerg_valid_halfhours"] = valid
        row[f"{region}_imerg_coverage_fraction"] = coverage
        row[f"{region}_imerg_accumulation_mm"] = accumulation
        row[f"{region}_imerg_mean_rate_mm_hr"] = (
            accumulation / (len(expected) * 0.5) if pd.notna(accumulation) else np.nan
        )
        complete = complete and pd.notna(accumulation)
    row["status"] = "ok" if complete else "incomplete"
    return row


def association_statistics(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    region: str,
    measure: str,
) -> dict[str, object]:
    """Return Pearson and OLS summary statistics for one requested comparison."""
    if x_column not in frame or y_column not in frame:
        return {"region": region, "precipitation_measure": measure, "n": 0, "status": "data unavailable"}
    sample = frame[[x_column, y_column]].dropna()
    n = len(sample)
    if n < 4:
        return {"region": region, "precipitation_measure": measure, "n": n, "status": "need at least four complete events"}

    correlation = stats.pearsonr(sample[x_column], sample[y_column])
    regression = stats.linregress(sample[x_column], sample[y_column])
    fisher_z = np.arctanh(correlation.statistic)
    r_margin = stats.norm.ppf(0.975) / np.sqrt(n - 3)
    r_ci_low, r_ci_high = np.tanh([fisher_z - r_margin, fisher_z + r_margin])
    slope_margin = stats.t.ppf(0.975, n - 2) * regression.stderr
    return {
        "region": region,
        "precipitation_measure": measure,
        "n": n,
        "status": "ok",
        "pearson_r": correlation.statistic,
        "r_95ci_low": r_ci_low,
        "r_95ci_high": r_ci_high,
        "r_two_sided_p": correlation.pvalue,
        "slope": regression.slope,
        "slope_95ci_low": regression.slope - slope_margin,
        "slope_95ci_high": regression.slope + slope_margin,
        "slope_two_sided_p": regression.pvalue,
        "intercept": regression.intercept,
        "r_squared": regression.rvalue**2,
        "null_decision_alpha_0.05": "reject H0" if correlation.pvalue < 0.05 else "fail to reject H0",
    }
