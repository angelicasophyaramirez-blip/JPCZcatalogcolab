"""Safe, resumable helpers for a full-record historical JPCZ catalog.

This module deliberately does not modify the existing 2000--2025 catalog.
It creates a second catalog whose anomaly baseline is formed from complete
NDJF seasons, while allowing matching 2000--2025 monthly D12 checkpoints to
be reused as read-only inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .detect import (
    DetectionGeometry,
    compute_divergence_stack,
    compute_rolling_mean,
    count_events_from_threshold,
    threshold_from_std,
)


@dataclass(frozen=True)
class NDJFMonth:
    """One calendar month belonging to a complete NDJF season."""

    season_year: int
    year: int
    month: int


def complete_ndjf_months(
    first_season_year: int = 1940,
    last_season_year: int = 2025,
) -> tuple[NDJFMonth, ...]:
    """Return Nov--Feb records for complete NDJF seasons.

    ``season_year=1940`` means November--December 1940 followed by
    January--February 1941.  This intentionally avoids the partial
    January--February 1940 season, for which November--December 1939 are not
    available in ERA5.
    """
    if last_season_year < first_season_year:
        raise ValueError("last_season_year must be >= first_season_year")

    records: list[NDJFMonth] = []
    for season_year in range(first_season_year, last_season_year + 1):
        records.extend(
            [
                NDJFMonth(season_year, season_year, 11),
                NDJFMonth(season_year, season_year, 12),
                NDJFMonth(season_year, season_year + 1, 1),
                NDJFMonth(season_year, season_year + 1, 2),
            ]
        )
    return tuple(records)


def checkpoint_path_for_month(directory: str | Path, record: NDJFMonth) -> Path:
    """Return the standard D12 checkpoint filename for a calendar month."""
    return Path(directory) / f"{record.year}_{record.month:02d}_D.nc"


def find_existing_checkpoint(
    record: NDJFMonth,
    *,
    legacy_monthly_dir: str | Path,
    legacy_december_dir: str | Path | None = None,
) -> Path | None:
    """Find a compatible old checkpoint without writing to its directory."""
    monthly_path = checkpoint_path_for_month(legacy_monthly_dir, record)
    if monthly_path.exists():
        return monthly_path
    if record.month == 12 and legacy_december_dir is not None:
        december_path = Path(legacy_december_dir) / f"december_{record.year}_D.nc"
        if december_path.exists():
            return december_path
    return None


def write_inventory(
    records: Sequence[dict[str, object]], path: str | Path) -> pd.DataFrame:
    """Atomically persist the Drive-resume inventory and return it."""
    inventory_path = Path(path)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(records).sort_values(["season_year", "year", "month"])
    temporary_path = inventory_path.with_suffix(inventory_path.suffix + ".tmp")
    inventory.to_csv(temporary_path, index=False)
    temporary_path.replace(inventory_path)
    return inventory


def write_dataframe_atomic(frame: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Write a CSV through a temporary sibling file, then atomically replace it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    frame.to_csv(temporary_path, index=index)
    temporary_path.replace(destination)
    return destination


def initialize_historical_inventory(
    records: Sequence[NDJFMonth],
    *,
    historical_monthly_dir: str | Path,
    legacy_monthly_dir: str | Path,
    legacy_december_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    """Describe whether each month is reusable, already built, or pending.

    A historical-run checkpoint always takes precedence.  Existing catalog
    checkpoints are only *read*; this keeps the established catalog safe.
    """
    historical_dir = Path(historical_monthly_dir)
    rows: list[dict[str, object]] = []
    for record in records:
        new_path = checkpoint_path_for_month(historical_dir, record)
        legacy_path = find_existing_checkpoint(
            record,
            legacy_monthly_dir=legacy_monthly_dir,
            legacy_december_dir=legacy_december_dir,
        )
        if new_path.exists():
            status, source, selected_path = "complete", "historical_checkpoint", new_path
        elif legacy_path is not None:
            status, source, selected_path = "complete", "reused_legacy_checkpoint", legacy_path
        else:
            status, source, selected_path = "pending", "new_era5_calculation", new_path
        rows.append(
            {
                **asdict(record),
                "status": status,
                "source": source,
                "checkpoint_path": str(selected_path),
            }
        )
    return rows


def compute_polygon_mean_divergence_series_chunked(
    window_ds: xr.Dataset,
    *,
    geometry: DetectionGeometry,
    chunk_hours: int = 48,
    progress_label: str = "Monthly divergence",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate hourly polygon divergence in small, memory-safe time chunks.

    Only each short wind-data chunk and its divergence stack are loaded.  The
    small one-dimensional hourly polygon series is concatenated before the
    12-hour rolling mean is calculated, so the D12 calculation remains
    identical to a non-chunked monthly calculation.
    """
    if chunk_hours < 1:
        raise ValueError("chunk_hours must be at least one hour")
    if "time" not in window_ds.dims:
        raise ValueError("window_ds must have a time dimension")

    n_hours = int(window_ds.sizes["time"])
    hourly_chunks: list[xr.DataArray] = []
    for start in range(0, n_hours, chunk_hours):
        stop = min(start + chunk_hours, n_hours)
        chunk = window_ds.isel(time=slice(start, stop)).load()
        divergence = compute_divergence_stack(
            chunk,
            dx=geometry.dx,
            dy=geometry.dy,
        )
        hourly_chunk = (
            (divergence * geometry.weights).sum(dim=("latitude", "longitude"))
            / geometry.weights.sum(dim=("latitude", "longitude"))
        ).rename("polygon_mean_divergence_925hpa")
        hourly_chunk.attrs.update({"units": "s^-1", "display_units": "1e-5 s^-1"})
        hourly_chunks.append(hourly_chunk)
        print(f"{progress_label}: {stop}/{n_hours} hourly fields complete")
        del divergence, chunk

    hourly = xr.concat(hourly_chunks, dim="time").sortby("time")
    D_12h = compute_rolling_mean(hourly, window=12, min_periods=12, name="D_12h")
    return hourly, D_12h


def build_events_from_monthly_checkpoints(
    inventory: pd.DataFrame,
    *,
    required_months: Iterable[int] = (11, 12, 1, 2),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recalculate full-record thresholds and raw events from D12 files.

    The inventory must be complete.  Thresholds are computed afresh from all
    complete-season D12 values; no threshold from the established catalog is
    reused.
    """
    required_months = tuple(required_months)
    pending = inventory.loc[inventory["status"] != "complete"]
    if not pending.empty:
        raise RuntimeError(
            f"Historical D12 inventory is incomplete: {len(pending)} month(s) pending."
        )

    records: list[dict[str, object]] = []
    for row in inventory.sort_values(["season_year", "year", "month"]).itertuples(index=False):
        path = Path(row.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Inventory lists a missing D12 checkpoint: {path}")
        D = xr.open_dataarray(path).load().dropna("time")
        records.append(
            {
                "season_year": int(row.season_year),
                "year": int(row.year),
                "month": int(row.month),
                "D": D,
            }
        )

    threshold_rows: list[dict[str, float | int]] = []
    thresholds: dict[int, float] = {}
    for month in required_months:
        D_month = xr.concat(
            [record["D"] for record in records if record["month"] == month], dim="time"
        ).sortby("time")
        mean_value, std_value, threshold = threshold_from_std(D_month, n_std=2.0)
        thresholds[month] = threshold
        threshold_rows.append(
            {
                "month": month,
                "D_mean_s-1": mean_value,
                "D_std_s-1": std_value,
                "threshold_s-1": threshold,
                "threshold_1e5_s-1": threshold * 1e5,
                "valid_D12_hours": int(D_month.sizes["time"]),
            }
        )

    event_frames: list[pd.DataFrame] = []
    for record in records:
        events = count_events_from_threshold(record["D"], thresholds[record["month"]]).copy()
        if events.empty:
            continue
        events["year"] = record["year"]
        events["month"] = record["month"]
        events["season_year"] = record["season_year"]
        events["detection_threshold_s-1"] = thresholds[record["month"]]
        events["detection_threshold_1e5_s-1"] = thresholds[record["month"]] * 1e5
        event_frames.append(events)

    events_df = (
        pd.concat(event_frames, ignore_index=True).sort_values("event_peak").reset_index(drop=True)
        if event_frames
        else pd.DataFrame()
    )
    thresholds_df = pd.DataFrame(threshold_rows).sort_values("month").reset_index(drop=True)
    return events_df, thresholds_df


def compare_merged_catalogs(
    established_catalog: pd.DataFrame,
    historical_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match merged episodes by temporal overlap and report catalog changes.

    An established episode is considered retained if any historical-baseline
    merged episode overlaps its inclusive start--end interval.  This avoids
    requiring identical peak hours when a changed threshold broadens or
    shortens the same physical episode.
    """
    required = {"event_start", "event_end", "event_peak"}
    for name, catalog in (("established", established_catalog), ("historical", historical_catalog)):
        missing = required.difference(catalog.columns)
        if missing:
            raise ValueError(f"{name} catalog is missing required columns: {sorted(missing)}")

    old = established_catalog.copy().reset_index(drop=True)
    new = historical_catalog.copy().reset_index(drop=True)
    for frame in (old, new):
        for column in required:
            frame[column] = pd.to_datetime(frame[column])
    old = old.sort_values("event_start").reset_index(drop=True)
    new = new.sort_values("event_start").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    old_match_count = np.zeros(len(old), dtype=int)
    new_match_count = np.zeros(len(new), dtype=int)
    for old_index, old_event in old.iterrows():
        overlapping = new.loc[
            (new["event_start"] <= old_event["event_end"])
            & (new["event_end"] >= old_event["event_start"])
        ]
        for new_index, new_event in overlapping.iterrows():
            overlap_start = max(old_event["event_start"], new_event["event_start"])
            overlap_end = min(old_event["event_end"], new_event["event_end"])
            overlap_hours = int((overlap_end - overlap_start).total_seconds() / 3600) + 1
            rows.append(
                {
                    "comparison_status": "shared_temporal_overlap",
                    "established_event_id": old_index + 1,
                    "historical_event_id": new_index + 1,
                    "established_start": old_event["event_start"],
                    "established_end": old_event["event_end"],
                    "established_peak": old_event["event_peak"],
                    "historical_start": new_event["event_start"],
                    "historical_end": new_event["event_end"],
                    "historical_peak": new_event["event_peak"],
                    "overlap_hours": overlap_hours,
                    "peak_offset_hours": (new_event["event_peak"] - old_event["event_peak"]).total_seconds() / 3600,
                }
            )
            old_match_count[old_index] += 1
            new_match_count[new_index] += 1

    for old_index, old_event in old.iterrows():
        if old_match_count[old_index] == 0:
            rows.append(
                {
                    "comparison_status": "established_only_not_flagged_historical",
                    "established_event_id": old_index + 1,
                    "historical_event_id": pd.NA,
                    "established_start": old_event["event_start"],
                    "established_end": old_event["event_end"],
                    "established_peak": old_event["event_peak"],
                    "historical_start": pd.NaT,
                    "historical_end": pd.NaT,
                    "historical_peak": pd.NaT,
                    "overlap_hours": 0,
                    "peak_offset_hours": np.nan,
                }
            )

    for new_index, new_event in new.iterrows():
        if new_match_count[new_index] == 0:
            rows.append(
                {
                    "comparison_status": "historical_only_newly_flagged",
                    "established_event_id": pd.NA,
                    "historical_event_id": new_index + 1,
                    "established_start": pd.NaT,
                    "established_end": pd.NaT,
                    "established_peak": pd.NaT,
                    "historical_start": new_event["event_start"],
                    "historical_end": new_event["event_end"],
                    "historical_peak": new_event["event_peak"],
                    "overlap_hours": 0,
                    "peak_offset_hours": np.nan,
                }
            )

    comparison = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {"metric": "established_merged_episodes", "value": len(old)},
            {"metric": "historical_merged_episodes", "value": len(new)},
            {
                "metric": "established_episodes_retained_by_temporal_overlap",
                "value": int((old_match_count > 0).sum()),
            },
            {
                "metric": "established_episodes_not_flagged_historical",
                "value": int((old_match_count == 0).sum()),
            },
            {
                "metric": "historical_only_newly_flagged_episodes",
                "value": int((new_match_count == 0).sum()),
            },
        ]
    )
    return comparison, summary
