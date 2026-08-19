"""Checkpoint-first utilities shared by the IMERG event workflow notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# NASA's current V07B archive extends into the TRMM era from January 1998.
IMERG_FIRST_VALID_TIME = pd.Timestamp("1998-01-01 00:00:00")
# GPM_3IMERGHH V07 Final is presently archived through September 2025. Keep
# this separate from the catalog time range: an event may be a valid ERA5 JPCZ
# event but unavailable for a Final-V07 precipitation comparison.
IMERG_FINAL_V07_END_EXCLUSIVE = pd.Timestamp("2025-10-01 00:00:00")


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
    """Save the Final-V07 request inventory and transparent completion status."""
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
    plan["imerg_product"] = "GPM_3IMERGHH V07 Final / Grid/precipitation (legacy fallback: precipitationCal)"
    plan["final_v07_availability"] = np.where(
        plan["precip_window_end_exclusive"] <= IMERG_FINAL_V07_END_EXCLUSIVE,
        "available",
        "not_available_final_v07",
    )
    plan["analysis_inclusion"] = np.where(
        plan["final_v07_availability"].eq("available"), "include", "exclude"
    )
    plan["collection_status"] = np.select(
        [
            plan["final_v07_availability"].eq("not_available_final_v07"),
            plan["event_id"].isin(complete),
        ],
        ["not_available_final_v07", "complete"],
        default="pending",
    )
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


def fixed_peak_12h_precipitation_metrics(
    events: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    region_names: tuple[str, ...],
    minimum_coverage: float,
) -> pd.DataFrame:
    """Calculate a fixed 12-hour IMERG accumulation matched to each D12 peak.

    D12 at a peak uses the 12 hourly timestamps from peak - 11 hours through
    peak. The equivalent half-hourly IMERG window has 24 samples beginning at
    peak - 11 hours and ending one hour after the peak timestamp (exclusive).
    This window is fully contained in the existing event collection window.
    """
    indexed = rates.set_index("time").sort_index() if "time" in rates else pd.DataFrame(index=pd.DatetimeIndex([]))
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        start = pd.Timestamp(event.event_peak) - pd.Timedelta(hours=11)
        end_exclusive = pd.Timestamp(event.event_peak) + pd.Timedelta(hours=1)
        expected = pd.date_range(start, end_exclusive, freq="30min", inclusive="left")
        window = indexed.reindex(expected)
        row: dict[str, object] = {
            "event_id": event.event_id,
            "event_peak": event.event_peak,
            "fixed12_window_start": start,
            "fixed12_window_end_exclusive": end_exclusive,
            "fixed12_expected_halfhours": len(expected),
            "fixed12_window_hours": len(expected) * 0.5,
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
            row[f"{region}_fixed12_imerg_valid_halfhours"] = valid
            row[f"{region}_fixed12_imerg_coverage_fraction"] = coverage
            row[f"{region}_fixed12_imerg_accumulation_mm"] = accumulation
            row[f"{region}_fixed12_imerg_mean_rate_mm_hr"] = (
                accumulation / 12 if pd.notna(accumulation) else np.nan
            )
            complete = complete and pd.notna(accumulation)
        row["fixed12_status"] = "ok" if complete else "incomplete"
        rows.append(row)
    return pd.DataFrame(rows)


def association_statistics(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    region: str,
    measure: str,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Return descriptive, Pearson, and OLS diagnostics for one comparison."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")
    if x_column not in frame or y_column not in frame:
        return {"region": region, "precipitation_measure": measure, "n": 0, "status": "data unavailable"}
    sample = frame[[x_column, y_column]].dropna()
    n = len(sample)
    if n < 4:
        return {"region": region, "precipitation_measure": measure, "n": n, "status": "need at least four complete events"}

    from scipy import stats

    x = sample[x_column].to_numpy(dtype=float)
    y = sample[y_column].to_numpy(dtype=float)
    correlation = stats.pearsonr(x, y)
    regression = stats.linregress(x, y)
    x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
    x_sd, y_sd = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    fitted = regression.intercept + regression.slope * x
    residuals = y - fitted
    sse = float(np.sum(residuals**2))
    regression_df = n - 2
    residual_standard_error = float(np.sqrt(sse / regression_df))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    x_centered_sum_squares = float(np.sum((x - x_mean) ** 2))
    intercept_standard_error = float(
        residual_standard_error * np.sqrt(1 / n + x_mean**2 / x_centered_sum_squares)
    )
    fisher_z = np.arctanh(correlation.statistic)
    fisher_z_standard_error = 1 / np.sqrt(n - 3)
    alpha = 1 - confidence_level
    r_margin = stats.norm.ppf(1 - alpha / 2) * fisher_z_standard_error
    r_ci_low, r_ci_high = np.tanh([fisher_z - r_margin, fisher_z + r_margin])
    slope_t_critical_95 = stats.t.ppf(1 - alpha / 2, regression_df)
    slope_margin = slope_t_critical_95 * regression.stderr
    return {
        "region": region,
        "precipitation_measure": measure,
        "predictor_column": x_column,
        "response_column": y_column,
        "n": n,
        "status": "ok",
        "confidence_level": confidence_level,
        "alpha": alpha,
        "x_mean": x_mean,
        "x_sample_sd": x_sd,
        "y_mean": y_mean,
        "y_sample_sd": y_sd,
        "pearson_r": correlation.statistic,
        "r_ci_method": "Fisher z; SE_z = 1/sqrt(n - 3)",
        "r_fisher_z": fisher_z,
        "r_fisher_z_standard_error": fisher_z_standard_error,
        "r_95ci_low": r_ci_low,
        "r_95ci_high": r_ci_high,
        "r_two_sided_p": correlation.pvalue,
        "slope": regression.slope,
        "slope_standard_error": regression.stderr,
        "slope_t_critical_95": slope_t_critical_95,
        "slope_95ci_low": regression.slope - slope_margin,
        "slope_95ci_high": regression.slope + slope_margin,
        "slope_two_sided_p": regression.pvalue,
        "intercept": regression.intercept,
        "intercept_standard_error": intercept_standard_error,
        "r_squared": regression.rvalue**2,
        "residual_sum_squares": sse,
        "residual_standard_error": residual_standard_error,
        "rmse": rmse,
        "regression_df": regression_df,
        "evidence_for_nonzero_association_alpha_0.05": "yes" if correlation.pvalue < 0.05 else "no",
    }
