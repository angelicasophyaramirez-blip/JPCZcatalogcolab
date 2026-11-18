"""Onset-aligned verification of WN2 against ERA5 for the two JPCZ regions."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import xarray as xr

from .weathernext_atlas import (
    FIELD_PRESETS, LEAD_HOURS, POLYGONS, diagnostic_domain, diagnostic_settings,
    derive_field, forecast_requests, iso_utc, select_forecast, utc_naive,
)

ERA5_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VERIFICATION_DOMAIN = (125., 145., 33., 47.)
ENSEMBLE_SIZE = 64
DIVERGENCE_SPEC = FIELD_PRESETS["divergence"]
DIVERGENCE_COLUMN = "divergence_1e5_s-1"
RANK_GROUPS = ("1–13", "14–26", "27–39", "40–52", "53–65")
CHECKPOINT_VERSION = 1


def polygon_divergence(field):
    """Area-weighted means on the native grid, independently for each polygon."""
    from matplotlib.path import Path as PolygonPath

    lon, lat = np.meshgrid(field.lon.values, field.lat.values)
    points = np.column_stack((lon.ravel(), lat.ravel()))
    means, counts = {}, {}
    for region, vertices in POLYGONS.items():
        mask = PolygonPath(vertices).contains_points(points).reshape(lon.shape)
        weights = np.cos(np.deg2rad(lat[mask]))
        values = field.values[..., mask]
        if not mask.any() or not np.isfinite(values).all():
            raise ValueError(f"{region} lacks a complete finite set of divergence cells.")
        means[region] = np.average(values, weights=weights, axis=-1)
        counts[region] = int(mask.sum())
    return means, counts


@lru_cache(maxsize=1)
def open_verification_era5():
    return xr.open_zarr(ERA5_STORE, chunks={}, storage_options={"token": "anon"})


def era5_polygon_values(valid_time, era5):
    """Derive divergence from the exact ERA5 925-hPa wind snapshot."""
    target = utc_naive(valid_time)
    winds = era5[list(DIVERGENCE_SPEC.variables)].sel(time=target, level=DIVERGENCE_SPEC.level)
    winds = winds.rename({"latitude": "lat", "longitude": "lon"})
    west, east, south, north = diagnostic_domain(winds, VERIFICATION_DOMAIN)
    winds = winds.sortby("lat").sortby("lon").sel(lat=slice(south, north), lon=slice(west, east))
    if any(abs(actual - expected) > 1e-5 for actual, expected in zip(
            (float(winds.lon.min()), float(winds.lon.max()), float(winds.lat.min()), float(winds.lat.max())),
            (west, east, south, north))):
        raise ValueError("ERA5 grid does not cover the verification domain and its derivative halo.")
    winds = winds.transpose("lat", "lon")
    field = derive_field(winds, DIVERGENCE_SPEC, VERIFICATION_DOMAIN).value
    means, counts = polygon_divergence(field)
    return {"valid_time": iso_utc(target), "source_store": ERA5_STORE, "polygon_cells": counts,
            "values": {region: float(value) for region, value in means.items()}}


def forecast_polygon_values(archive, request, progress=print):
    """Read groups of native WN2 samples and immediately reduce each field to two means."""
    store, ds = archive.run(request["init_time"])
    labels = [str(value) for value in ds.sample.values.tolist()]
    if len(labels) != ENSEMBLE_SIZE or len(set(labels)) != ENSEMBLE_SIZE:
        raise ValueError(f"Expected {ENSEMBLE_SIZE} distinct WN2 member labels, found {len(set(labels))}.")
    snap = select_forecast(
        ds, request["init_time"], request["lead_hours"], None, DIVERGENCE_SPEC.variables,
        DIVERGENCE_SPEC.level, diagnostic_domain(ds, VERIFICATION_DOMAIN),
    )
    winds = snap[DIVERGENCE_SPEC.variables[0]]
    sizes = winds.chunksizes.get("sample", (4,) * (ENSEMBLE_SIZE // 4))
    values = {region: [] for region in POLYGONS}
    start = 0
    for size in sizes:
        end = start + size
        progress(f"  Reading {request['lead_days']}-day members {start + 1}–{end} of {ENSEMBLE_SIZE}")
        block = snap.isel(sample=slice(start, end))
        field = derive_field(block, DIVERGENCE_SPEC, VERIFICATION_DOMAIN).value
        means, counts = polygon_divergence(field)
        for region in POLYGONS:
            values[region].extend(means[region].tolist())
        start = end
    return {"init_time": iso_utc(request["init_time"]), "valid_time": iso_utc(request["valid_time"]),
            "lead_hours": request["lead_hours"], "source_store": store, "members": labels,
            "polygon_cells": counts, "values": values}


def checkpoint_identity(archive):
    """Reuse verified values only with the same fields, polygons and source configuration."""
    return {
        "version": CHECKPOINT_VERSION, "era5_store": ERA5_STORE, "wn2_root": archive.root,
        "wn2_overrides": archive.overrides, "domain": VERIFICATION_DOMAIN, "polygons": POLYGONS,
        "pressure_hpa": DIVERGENCE_SPEC.level, "ensemble_size": ENSEMBLE_SIZE,
        "diagnostic": diagnostic_settings(None, DIVERGENCE_SPEC, VERIFICATION_DOMAIN),
    }


def atomic_json_file(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            json.dump(payload, temporary, allow_nan=False, indent=2)
        except Exception:
            temporary.close()
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(path)


def checkpoint(path, identity, calculate):
    if path.is_file():
        try:
            saved = json.loads(path.read_text())
            if saved["request"] == identity:
                return saved["data"], True
        except (KeyError, ValueError, TypeError):
            pass
    data = calculate()
    atomic_json_file(path, {"request": identity, "data": data})
    return data, False


def observation_rank(members, reference, case_key, *, tie_seed=37):
    """Return the ERA5 rank, its tied interval, and its position relative to the ensemble."""
    values = np.asarray(members, dtype=np.float64)
    if values.shape != (ENSEMBLE_SIZE,) or not np.isfinite(values).all() or not np.isfinite(reference):
        raise ValueError("Ranking requires 64 finite member values and one finite ERA5 value.")
    less = int(np.count_nonzero(values < reference))
    equal = int(np.count_nonzero(values == reference))
    lower, upper = less + 1, less + equal + 1
    key = f"{tie_seed}|{case_key}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    rank = lower if lower == upper else int(rng.integers(lower, upper + 1))
    return {
        "era5_rank": rank, "rank_min_if_tied": lower, "rank_max_if_tied": upper,
        "rank_group": RANK_GROUPS[(rank - 1) // 13],
        "below_member_min": bool(reference < values.min()),
        "above_member_max": bool(reference > values.max()),
    }


def case_identity(row):
    return {
        "event_number": int(row.event_number), "review_key": row.review_key,
        "event_start_utc": iso_utc(row.event_start), "event_end_utc": iso_utc(row.event_end),
        "event_peak_utc": iso_utc(row.event_peak), "valid_time_utc": iso_utc(row.target_valid_time),
        "target_minus_onset_hours": float(row.onset_offset_hours),
        "catalog_peak_D12_1e5_s-1": row["event_peak_D_1e5_s-1"],
        "vortex_label": row.get("vortex_plot_label", ""),
        "vortex_scan_status": row.get("vortex_scan_status", ""),
    }


def run_ensemble_verification(catalog, archive, output_dir, *, era5_reader=open_verification_era5, progress=print):
    """Checkpoint one ERA5 target and each WN2 run/lead independently."""
    identity = checkpoint_identity(archive)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(output_dir) / "checkpoints" / signature
    member_rows, reference_rows, rank_rows, status_rows = [], [], [], []

    for _, row in catalog.iterrows():
        event = case_identity(row)
        target = utc_naive(row.target_valid_time)
        stamp = target.strftime("%Y%m%dT%H%MZ")
        progress(f"Event {event['event_number']} | verification {event['valid_time_utc']}")
        reference = None
        try:
            reference, cached = checkpoint(
                cache / f"era5_{stamp}.json", {"valid_time": event["valid_time_utc"]},
                lambda: era5_polygon_values(target, era5_reader()),
            )
            reference_rows.extend([
                {**event, "polygon": region, "pressure_hpa": DIVERGENCE_SPEC.level,
                 DIVERGENCE_COLUMN: reference["values"][region],
                 "polygon_cells": reference["polygon_cells"][region], "source_store": reference["source_store"]}
                for region in POLYGONS
            ])
            ref_status = "cached" if cached else "computed"
            ref_error = ""
        except Exception as error:
            reference = None
            ref_status = "failed"
            ref_error = f"{type(error).__name__}: {error}"
            progress(f"  ERA5 unavailable: {ref_error}")

        event_ranks = []
        for request in forecast_requests(row).to_dict("records"):
            lead = request["lead_hours"]
            timing = {"lead_hours": lead, "lead_days": request["lead_days"], "init_time_utc": iso_utc(request["init_time"])}
            status = {**event, **timing, "era5_status": ref_status, "era5_error": ref_error}
            try:
                forecast, cached = checkpoint(
                    cache / f"wn2_{stamp}_lead{lead:03d}h.json",
                    {"init_time": timing["init_time_utc"], "lead_hours": lead, "valid_time": event["valid_time_utc"]},
                    lambda: forecast_polygon_values(archive, request, progress),
                )
                members = forecast["members"]
                if len(members) != ENSEMBLE_SIZE or len(set(members)) != ENSEMBLE_SIZE:
                    raise ValueError("Forecast checkpoint does not contain 64 distinct members.")
                case_values, case_ranks = [], []
                for region in POLYGONS:
                    values = np.asarray(forecast["values"][region], dtype=float)
                    if values.shape != (ENSEMBLE_SIZE,) or not np.isfinite(values).all():
                        raise ValueError(f"{region} does not contain 64 finite member values.")
                    info = {**event, **timing, "polygon": region, "pressure_hpa": DIVERGENCE_SPEC.level}
                    case_values.extend({**info, "member": member, DIVERGENCE_COLUMN: value,
                                        "polygon_cells": forecast["polygon_cells"][region],
                                        "source_store": forecast["source_store"]} for member, value in zip(members, values))
                    if reference is not None:
                        observed = reference["values"][region]
                        rank = observation_rank(values, observed, f"{row.review_key}|{lead}|{region}")
                        case_ranks.append({**info, "member_count": len(members),
                                           "era5_divergence_1e5_s-1": observed,
                                           "member_min_1e5_s-1": values.min(), "member_mean_1e5_s-1": values.mean(),
                                           "member_max_1e5_s-1": values.max(), **rank})
                member_rows.extend(case_values)
                event_ranks.extend(case_ranks)
                status.update(wn2_status="cached" if cached else "computed", wn2_error="")
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                status.update(wn2_status="failed", wn2_error=message)
                progress(f"  {request['lead_days']}-day forecast unavailable: {message}")
            status_rows.append(status)
        common_sample = len(event_ranks) == len(LEAD_HOURS) * len(POLYGONS)
        for record in event_ranks:
            record["in_common_sample"] = common_sample
        rank_rows.extend(event_ranks)
    return {
        "case_ranks": pd.DataFrame(rank_rows),
        "member_values": pd.DataFrame(member_rows),
        "era5_reference": pd.DataFrame(reference_rows),
        "batch_status": pd.DataFrame(status_rows),
    }, identity


def plot_rank_histograms(ranks):
    """Use five equally sized groups of ranks, with the same cases at each lead."""
    import matplotlib.pyplot as plt

    if ranks.empty or not ranks.in_common_sample.any():
        raise ValueError("No event has complete ERA5 and all three WN2 leads.")
    complete = ranks.loc[ranks.in_common_sample]
    n_events = complete.review_key.nunique()
    largest_bin = complete.groupby(["polygon", "lead_hours", "rank_group"]).size().max()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True, layout="constrained")
    for r, region in enumerate(POLYGONS):
        for c, lead in enumerate(LEAD_HOURS):
            ax = axes[r, c]
            matched = complete.loc[(complete.polygon == region) & (complete.lead_hours == lead)]
            counts = matched.rank_group.value_counts().reindex(RANK_GROUPS, fill_value=0)
            ax.bar(RANK_GROUPS, counts, color="#277486", width=.8)
            ax.axhline(n_events / len(RANK_GROUPS), color=".25", linestyle="--", linewidth=1)
            ax.set_title(f"{region} · {lead // 24}-day lead")
            ax.set_xlabel("ERA5 rank (most to least convergent)")
            ax.text(.03, .96, f"Outside ensemble range: {matched.below_member_min.sum()} below, "
                    f"{matched.above_member_max.sum()} above", transform=ax.transAxes, va="top", fontsize=8)
            ax.set_ylim(0, 1.22 * max(largest_bin, n_events / len(RANK_GROUPS)))
            ax.yaxis.get_major_locator().set_params(integer=True)
            if c == 0:
                ax.set_ylabel("Number of JPCZ cases")
    fig.suptitle(f"925-hPa divergence at JPCZ onset · {n_events} matched {'case' if n_events == 1 else 'cases'}"
                 f"{' (pilot)' if n_events == 1 else ''}\n"
                 "ERA5 ranks among 64 WeatherNext 2 members · 13 ranks per bar · dashed line = uniform ranks")
    return fig


def export_verification(tables, identity, destination):
    """Write the results and a workbook for inspecting the computed ranks in Drive."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json_file(destination / "calculation_settings.json", identity)
    for name, frame in tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    with NamedTemporaryFile(suffix=".xlsx", dir=destination, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with pd.ExcelWriter(temporary_path, engine="xlsxwriter",
                            engine_kwargs={"options": {"strings_to_formulas": False, "strings_to_urls": False}}) as writer:
            for name, frame in tables.items():
                sheet = name.replace("_", " ").capitalize()
                frame.to_excel(writer, sheet_name=sheet, index=False)
                worksheet = writer.sheets[sheet]
                worksheet.freeze_panes(1, 0)
                if len(frame.columns):
                    worksheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
                    for index, column in enumerate(frame.columns):
                        width = max(len(column) + 2, 14)
                        if not frame.empty:
                            width = max(width, int(frame[column].head(100).astype(str).str.len().max()) + 2)
                        worksheet.set_column(index, index, min(width, 55))
        temporary_path.replace(destination / "ensemble_verification.xlsx")
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
