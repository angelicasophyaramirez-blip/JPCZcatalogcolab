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
import zarr

from .weathernext_atlas import (
    FIELD_PRESETS, LEAD_HOURS, POLYGONS, diagnostic_domain, diagnostic_settings,
    derive_field, forecast_requests, iso_utc, select_forecast, utc_naive, vortex_classification,
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
    # Creating Dask graphs for all decades and variables can exceed Colab RAM
    # before any field is read. Keep backend indexing lazy through subsetting.
    return xr.open_zarr(ERA5_STORE, chunks=None, storage_options={"token": "anon"})


def era5_polygon_values(valid_time, era5, *, allow_era5t=False):
    """Derive divergence from the exact ERA5 925-hPa wind snapshot."""
    target = utc_naive(valid_time)
    stable = era5.attrs.get('valid_time_stop')
    preliminary = era5.attrs.get('valid_time_stop_era5t')
    product = 'ERA5'
    if stable and target > utc_naive(stable):
        if not allow_era5t or not preliminary or target > utc_naive(preliminary):
            raise ValueError(f'ERA5 pending for {iso_utc(target)}; final through {stable}; '
                             f'ERA5T through {preliminary}. No nearest time is substituted.')
        product = 'ERA5T'
    winds = era5[list(DIVERGENCE_SPEC.variables)].sel(time=target, level=DIVERGENCE_SPEC.level)
    winds = winds.rename({"latitude": "lat", "longitude": "lon"})
    west, east, south, north = diagnostic_domain(winds, VERIFICATION_DOMAIN)
    latitude = slice(south, north) if winds.lat[0] < winds.lat[-1] else slice(north, south)
    longitude = slice(west, east) if winds.lon[0] < winds.lon[-1] else slice(east, west)
    winds = winds.sel(lat=latitude, lon=longitude)
    if any(abs(actual - expected) > 1e-5 for actual, expected in zip(
            (float(winds.lon.min()), float(winds.lon.max()), float(winds.lat.min()), float(winds.lat.max())),
            (west, east, south, north))):
        raise ValueError("ERA5 grid does not cover the verification domain and its derivative halo.")
    # Decode one wind variable at a time, only after selecting the region.
    with zarr.config.set({"async.concurrency": 1, "threading.max_workers": 2}):
        for name in DIVERGENCE_SPEC.variables:
            winds[name] = winds[name].load(scheduler="synchronous")
    winds = winds.transpose("lat", "lon").sortby("lat").sortby("lon")
    field = derive_field(winds, DIVERGENCE_SPEC, VERIFICATION_DOMAIN).value
    means, counts = polygon_divergence(field)
    return {"valid_time": iso_utc(target), "source_store": ERA5_STORE, "polygon_cells": counts,
            "reference_product": product, "archive_last_updated": str(era5.attrs.get('last_updated', '')),
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
        with zarr.config.set({"async.concurrency": 1, "threading.max_workers": 2}):
            for name in DIVERGENCE_SPEC.variables:
                block[name] = block[name].load(scheduler="synchronous")
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
        "bin_number": (rank - 1) // 13 + 1,
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
        "vortex_classification": row.get('vortex_classification', vortex_classification(row)),
    }


def run_ensemble_verification(catalog, archive, output_dir, *, era5_reader=open_verification_era5,
                              allow_era5t=False, progress=print):
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
            progress("  ERA5: opening the reference archive and reading one 925-hPa wind snapshot")
            reference_path = cache / f"era5_{stamp}.json"
            # Preliminary data can change. Refresh it on every run, including when it
            # has become final. Forecast checkpoints remain independent and reusable.
            if reference_path.is_file():
                try:
                    saved = json.loads(reference_path.read_text())
                    if saved['data'].get('reference_product') == 'ERA5T':
                        reference_path.unlink()
                except (KeyError, ValueError, TypeError):
                    pass
            reference, cached = checkpoint(
                reference_path, {"valid_time": event["valid_time_utc"]},
                lambda: era5_polygon_values(target, era5_reader(), allow_era5t=allow_era5t),
            )
            reference_rows.extend([
                {**event, "polygon": region, "pressure_hpa": DIVERGENCE_SPEC.level,
                 DIVERGENCE_COLUMN: reference["values"][region],
                 "polygon_cells": reference["polygon_cells"][region], "source_store": reference["source_store"],
                 "reference_product": reference.get('reference_product', 'ERA5')}
                for region in POLYGONS
            ])
            ref_status = "cached" if cached else "computed"
            ref_error = ""
            progress(f"  ERA5 {ref_status}: polygon means saved")
        except Exception as error:
            reference = None
            ref_status = "failed"
            ref_error = f"{type(error).__name__}: {error}"
            progress(f"  ERA5 unavailable: {ref_error}")

        event_ranks = []
        for request in forecast_requests(row).to_dict("records"):
            lead = request["lead_hours"]
            timing = {"lead_hours": lead, "lead_days": request["lead_days"], "init_time_utc": iso_utc(request["init_time"])}
            status = {**event, **timing, "era5_status": ref_status, "era5_error": ref_error,
                      "reference_product": reference.get('reference_product', 'ERA5') if reference else 'pending'}
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
                                           "reference_product": reference.get('reference_product', 'ERA5'),
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


def rank_histogram_counts(ranks, *, grouped=True):
    """Count every rank, including empty bins, using common cases at each lead."""
    if ranks.empty or not ranks.in_common_sample.any():
        raise ValueError("No event has complete ERA5 and all three WN2 leads.")
    complete = ranks.loc[ranks.in_common_sample]
    n_events = complete.review_key.nunique()
    bin_width = 13 if grouped else 1
    records = []
    for region in POLYGONS:
        for lead in LEAD_HOURS:
            matched = complete.loc[(complete.polygon == region) & (complete.lead_hours == lead)]
            counts = np.bincount(matched.era5_rank.to_numpy(dtype=int), minlength=ENSEMBLE_SIZE + 2)[1:]
            counts = counts.reshape(-1, bin_width).sum(axis=1)
            records.extend({
                "polygon": region, "lead_days": lead // 24, "matched_events": n_events,
                "bin_number": index + 1,
                "rank_from": index * bin_width + 1, "rank_through": (index + 1) * bin_width,
                "case_count": int(count), "uniform_expected_count": n_events * bin_width / (ENSEMBLE_SIZE + 1),
            } for index, count in enumerate(counts))
    return pd.DataFrame(records)


def plot_rank_histograms(ranks, *, grouped=True, title_suffix='All events'):
    """Show either all 65 ranks or five groups, with the same cases at each lead."""
    import matplotlib.pyplot as plt

    bins = rank_histogram_counts(ranks, grouped=grouped)
    complete = ranks.loc[ranks.in_common_sample]
    n_events = complete.review_key.nunique()
    largest_bin = bins.case_count.max()
    expectation = bins.uniform_expected_count.iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, layout="constrained")
    for r, region in enumerate(POLYGONS):
        for c, lead in enumerate(LEAD_HOURS):
            ax = axes[r, c]
            matched = complete.loc[(complete.polygon == region) & (complete.lead_hours == lead)]
            panel_bins = bins.loc[(bins.polygon == region) & (bins.lead_days == lead // 24)]
            positions = ([f"Bin {index + 1}\nRanks {label}" for index, label in enumerate(RANK_GROUPS)]
                         if grouped else np.arange(1, ENSEMBLE_SIZE + 2))
            ax.bar(positions, panel_bins.case_count, color="#277486", width=.8)
            ax.axhline(expectation, color=".25", linestyle="--", linewidth=1)
            ax.set_title(f"{region} · {lead // 24}-day lead · N = {len(matched)}")
            if not grouped:
                ax.set_xticks([1, 9, 17, 25, 33, 41, 49, 57, 65])
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel("ERA5 rank position among 64 forecast values")
            ax.text(.03, .96, f"Outside ensemble range: {matched.below_member_min.sum()} below, "
                    f"{matched.above_member_max.sum()} above", transform=ax.transAxes, va="top", fontsize=8)
            ax.set_ylim(0, 1.22 * max(largest_bin, expectation))
            ax.yaxis.get_major_locator().set_params(integer=True)
            if c == 0:
                ax.set_ylabel("Number of events")
    fig.suptitle(f"{title_suffix} · 925-hPa polygon-average divergence at the onset target · {n_events} matched "
                 f"{'event' if n_events == 1 else 'events'}\n"
                 f"{'13 ranks' if grouped else '1 rank'} per bar · dashed line = uniform rank reference\n"
                 "Lower rank = ERA5 more convergent relative to forecasts · Higher rank = ERA5 less convergent")
    return fig


def ordered_verification_values(members, ranks):
    """Pool 64 forecast values and ERA5, preserving the exact saved tie-broken rank."""
    keys = ["review_key", "polygon", "lead_hours"]
    columns = [*keys, "event_number", "valid_time_utc", "init_time_utc", "lead_days", "pressure_hpa",
               "in_common_sample", "position", "bin_number", "rank_range", "source", "member", DIVERGENCE_COLUMN]
    if ranks.empty:
        return pd.DataFrame(columns=columns)
    if ranks.duplicated(keys).any():
        raise ValueError("Duplicate event/polygon/lead ranks cannot be exported.")
    groups = {key: group for key, group in members.groupby(keys, sort=False)}
    records = []
    for rank in ranks.to_dict("records"):
        key = tuple(rank[name] for name in keys)
        group = groups.get(key)
        if group is None or len(group) != ENSEMBLE_SIZE or group.member.astype(str).nunique() != ENSEMBLE_SIZE:
            raise ValueError(f"Ordered values require 64 distinct members for {key}.")
        values = group[DIVERGENCE_COLUMN].to_numpy(dtype=float)
        observed = rank["era5_divergence_1e5_s-1"]
        if not np.isfinite(values).all() or not np.isfinite(observed):
            raise ValueError(f"Ordered values contain missing or nonfinite divergence for {key}.")
        position = int(rank["era5_rank"])
        lower = int((values < observed).sum()) + 1
        upper = lower + int((values == observed).sum())
        if not lower <= position <= upper:
            raise ValueError(f"Saved ERA5 rank disagrees with the member values for {key}.")
        ordered = group.sort_values(DIVERGENCE_COLUMN, kind="stable")
        entries = [
            {"source": "WN2", "member": str(item["member"]), DIVERGENCE_COLUMN: float(item[DIVERGENCE_COLUMN])}
            for item in ordered.to_dict("records")
        ]
        entries.insert(position - 1, {"source": "ERA5", "member": "", DIVERGENCE_COLUMN: float(observed)})
        info = {name: rank[name] for name in columns[:9]}
        for index, entry in enumerate(entries):
            bin_index = index // 13
            records.append({**info, "position": index + 1, "bin_number": bin_index + 1,
                            "rank_range": RANK_GROUPS[bin_index], **entry})
    return pd.DataFrame(records, columns=columns)


def write_ordered_bins(writer, ordered, ranks):
    """Show thirteen entries in each of five bin columns for every verifiable case."""
    sheet = writer.book.add_worksheet("Ordered bins")
    sheet.hide_gridlines(2)
    sheet.set_zoom(85)
    sheet.freeze_panes(4, 0)
    sheet.set_landscape()
    sheet.fit_to_pages(1, 0)
    book = writer.book
    title_format = book.add_format({"bold": True, "font_size": 13, "font_color": "#17324D"})
    note_format = book.add_format({"text_wrap": True, "valign": "top"})
    bin_format = book.add_format({"bold": True, "bg_color": "#DDECF1", "align": "center"})
    header_format = book.add_format({"bold": True, "text_wrap": True, "bottom": 1})
    value_format = book.add_format({"num_format": "0.000000"})
    era5_value_format = book.add_format({"num_format": "0.000000", "bold": True, "bg_color": "#DDF0DE"})
    era5_label_format = book.add_format({"bold": True, "bg_color": "#DDF0DE"})
    sheet.merge_range(0, 0, 0, 14, "Ordered 925-hPa polygon-average divergence", title_format)
    sheet.merge_range(1, 0, 2, 14,
                      "Each block contains 64 WN2 member values and one ERA5 value, sorted from lowest divergence "
                      "(strongest convergence) to highest divergence. ERA5 is labeled and highlighted. "
                      "Values are in 10⁻⁵ s⁻¹. Histogram counts use only ERA5's saved position.", note_format)
    for bin_index in range(5):
        start = bin_index * 3
        sheet.set_column(start, start, 8)
        sheet.set_column(start + 1, start + 1, 20)
        sheet.set_column(start + 2, start + 2, 16, value_format)
    if ordered.empty:
        sheet.write(4, 0, "No event/polygon/lead has both ERA5 and all 64 members.")
        return
    keys = ["review_key", "polygon", "lead_hours"]
    rank_lookup = {tuple(item[name] for name in keys): item for item in ranks.to_dict("records")}
    row = 4
    for key, group in ordered.groupby(keys, sort=False):
        case = rank_lookup[key]
        sheet.merge_range(row, 0, row, 14,
                          f"Event {case['event_number']} · {case['polygon']} · {case['lead_days']}-day lead · 925 hPa",
                          title_format)
        sheet.merge_range(row + 1, 0, row + 1, 14,
                          f"Valid {case['valid_time_utc']} · Initialized {case['init_time_utc']} · "
                          f"ERA5 rank {case['era5_rank']} of 65 · "
                          f"{'Included in all-lead histograms' if case['in_common_sample'] else 'Incomplete all-lead comparison'}")
        for bin_index, label in enumerate(RANK_GROUPS):
            start = bin_index * 3
            sheet.merge_range(row + 2, start, row + 2, start + 2,
                              f"Bin {bin_index + 1} · ranks {label}", bin_format)
            for offset, text in enumerate(("Position", "Source", "D (10⁻⁵ s⁻¹)")):
                sheet.write(row + 3, start + offset, text, header_format)
        for item in group.to_dict("records"):
            cell_row = row + 4 + (item["position"] - 1) % 13
            cell_col = (item["bin_number"] - 1) * 3
            label_format = era5_label_format if item["source"] == "ERA5" else None
            number_format = era5_value_format if item["source"] == "ERA5" else value_format
            sheet.write_number(cell_row, cell_col, item["position"], label_format)
            source = "ERA5" if item["source"] == "ERA5" else f"WN2 sample {item['member']}"
            sheet.write_string(cell_row, cell_col + 1, source, label_format)
            sheet.write_number(cell_row, cell_col + 2, item[DIVERGENCE_COLUMN], number_format)
        row += 19
    sheet.print_area(0, 0, row - 1, 14)


def export_verification(tables, identity, destination):
    """Write the results and a workbook for inspecting the computed ranks in Drive."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tables = dict(tables)
    ranks = tables["case_ranks"]
    tables["ordered_values"] = ordered_verification_values(tables["member_values"], ranks)
    # A previously complete run must not leave count tables that describe an old sample.
    for name in ('rank_bin_counts', 'rank_group_counts', 'bin_counts'):
        (destination / f'{name}.csv').unlink(missing_ok=True)
    if not ranks.empty and ranks.in_common_sample.any():
        tables["rank_bin_counts"] = rank_histogram_counts(ranks, grouped=False)
        tables["rank_group_counts"] = rank_histogram_counts(ranks, grouped=True)
        tables["bin_counts"] = tables["rank_group_counts"].pivot(
            index=["polygon", "lead_days", "matched_events"], columns="bin_number", values="case_count"
        ).rename(columns={index + 1: f"Bin {index + 1} · ranks {label}"
                          for index, label in enumerate(RANK_GROUPS)}).reset_index()
    atomic_json_file(destination / "calculation_settings.json", identity)
    for name, frame in tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    with NamedTemporaryFile(suffix=".xlsx", dir=destination, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with pd.ExcelWriter(temporary_path, engine="xlsxwriter",
                            engine_kwargs={"options": {"strings_to_formulas": False, "strings_to_urls": False}}) as writer:
            era5_format = writer.book.add_format({"bold": True, "bg_color": "#DDF0DE"})
            for name, frame in tables.items():
                sheet = name.replace("_", " ").capitalize()
                frame.to_excel(writer, sheet_name=sheet, index=False)
                worksheet = writer.sheets[sheet]
                worksheet.freeze_panes(1, 0)
                worksheet.hide_gridlines(2)
                worksheet.set_column(0, max(0, len(frame.columns) - 1), 18)
                if len(frame.columns):
                    worksheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
                    for index, column in enumerate(frame.columns):
                        width = max(len(column) + 2, 14)
                        if not frame.empty:
                            width = max(width, int(frame[column].head(100).astype(str).str.len().max()) + 2)
                        worksheet.set_column(index, index, min(width, 55))
                if name == "ordered_values" and not frame.empty:
                    from xlsxwriter.utility import xl_col_to_name
                    source_col = xl_col_to_name(frame.columns.get_loc("source"))
                    worksheet.conditional_format(1, 0, len(frame), len(frame.columns) - 1,
                        {"type": "formula", "criteria": "=$" + source_col + '2="ERA5"', "format": era5_format})
                if name == "case_ranks":
                    write_ordered_bins(writer, tables["ordered_values"], ranks)
        temporary_path.replace(destination / "ensemble_verification.xlsx")
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def verification_subset(tables, group='All'):
    """Use completed vortex screens for strata; retain unchecked cases as Unknown."""
    ranks = tables['case_ranks'].copy()
    if ranks.empty:
        return ranks
    classification = ranks.get('vortex_classification', pd.Series('', index=ranks.index))
    checked = ranks.vortex_scan_status.eq('complete')
    groups = np.select([checked & classification.eq('Vortex'), checked & classification.eq('Non-vortex')],
                       ['Vortex', 'Non-vortex'], default='Unknown')
    ranks['verification_group'] = groups
    if group not in ('All', 'Vortex', 'Non-vortex', 'Unknown'):
        raise ValueError('Choose All, Vortex, Non-vortex, or Unknown.')
    return ranks if group == 'All' else ranks.loc[ranks.verification_group.eq(group)].copy()


def scheduled_verification_catalog(start, end, *, interval_hours=24, months=(11, 12, 1, 2)):
    """Outcome-independent, half-open UTC schedule including crossings and non-crossings."""
    start, end = utc_naive(start), utc_naive(end)
    if start >= end or interval_hours not in (6, 12, 24):
        raise ValueError('Use start < exclusive end and a 6, 12, or 24 hour schedule.')
    if start != start.floor('6h') or end != end.floor('6h'):
        raise ValueError('Schedule endpoints must be exact six-hour UTC times.')
    times = pd.date_range(start, end, freq=f'{interval_hours}h', inclusive='left')
    times = times[times.month.isin(months)]
    if times.empty:
        raise ValueError('The schedule contains no winter verification targets.')
    frame = pd.DataFrame({'event_start': times, 'event_end': times, 'event_peak': times,
                          'event_number': np.arange(1, len(times) + 1), 'target_valid_time': times,
                          'onset_offset_hours': 0., 'event_peak_D_1e5_s-1': np.nan,
                          'vortex_plot_label': '', 'vortex_scan_status': 'not_applicable'})
    frame['review_key'] = [f'scheduled|{iso_utc(t)}' for t in times]
    return frame


def reliability_probabilities(members, ranks, thresholds, *, sample='event_conditioned'):
    """P(D925 snapshot < threshold) from all 64 members and binary ERA5 outcome.

    This is instantaneous polygon convergence, not the hourly trailing-D12 JPCZ detector.
    Only the common all-lead sample enters, exactly as in the rank histograms.
    """
    if sample not in ('event_conditioned', 'scheduled_winter'):
        raise ValueError('Unknown reliability sampling design.')
    if set(thresholds) != set(POLYGONS) or not np.isfinite(list(thresholds.values())).all():
        raise ValueError('Supply one finite snapshot threshold (10^-5 s^-1) per polygon.')
    keys = ['review_key', 'polygon', 'lead_hours']
    columns = [*keys, 'lead_days', 'valid_time_utc', 'event_number', 'vortex_label', 'vortex_scan_status',
               'sample', 'threshold_1e5_s-1', 'member_count', 'forecast_probability', 'observed', 'brier_score']
    if ranks.empty:
        return pd.DataFrame(columns=columns)
    if ranks.duplicated(keys).any():
        raise ValueError('Duplicate verification cases would overweight reliability.')
    groups = {key: frame for key, frame in members.groupby(keys, sort=False)}
    rows = []
    for row in ranks.loc[ranks.in_common_sample].to_dict('records'):
        key = tuple(row[name] for name in keys)
        frame = groups.get(key)
        if frame is None or len(frame) != ENSEMBLE_SIZE or frame.member.astype(str).nunique() != ENSEMBLE_SIZE:
            raise ValueError(f'Reliability requires 64 distinct members for {key}.')
        values = frame[DIVERGENCE_COLUMN].to_numpy(dtype=float)
        observed_value = float(row['era5_divergence_1e5_s-1'])
        if not np.isfinite(values).all() or not np.isfinite(observed_value):
            raise ValueError('Reliability inputs must be finite.')
        threshold = float(thresholds[row['polygon']])
        probability = float(np.mean(values < threshold))
        observed = int(observed_value < threshold)
        rows.append({**{name: row.get(name, '') for name in columns[:8]},
                     'sample': sample, 'threshold_1e5_s-1': threshold, 'member_count': ENSEMBLE_SIZE,
                     'forecast_probability': probability, 'observed': observed,
                     'brier_score': (probability - observed)**2})
    return pd.DataFrame(rows, columns=columns)


def reliability_bin_counts(probabilities, *, n_bins=10):
    """Fixed [left,right) bins, final bin closed at 1, with empty bins retained."""
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError('Use at least two probability bins.')
    rows = []
    for region in POLYGONS:
        for lead in LEAD_HOURS:
            subset = probabilities.loc[probabilities.polygon.eq(region) & probabilities.lead_hours.eq(lead)]
            if subset.empty:
                continue
            p = subset.forecast_probability.to_numpy(dtype=float)
            y = subset.observed.to_numpy(dtype=float)
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isin(y, (0, 1)).all():
                raise ValueError('Need probabilities in [0,1] and binary observed outcomes.')
            bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
            for index in range(n_bins):
                mask = bins == index
                n, hits = int(mask.sum()), int(y[mask].sum())
                frequency = hits / n if n else np.nan
                # Pointwise Wilson 95% binomial intervals; serial dependence is not modeled.
                z = 1.959963984540054
                denominator = 1 + z*z/n if n else np.nan
                center = (frequency + z*z/(2*n)) / denominator if n else np.nan
                half = z * np.sqrt(frequency*(1-frequency)/n + z*z/(4*n*n)) / denominator if n else np.nan
                rows.append({'polygon': region, 'lead_days': lead // 24, 'sample': subset['sample'].iloc[0],
                             'threshold_1e5_s-1': subset['threshold_1e5_s-1'].iloc[0],
                             'bin_number': index + 1, 'probability_from': index/n_bins,
                             'probability_through': (index+1)/n_bins, 'case_count': n, 'observed_count': hits,
                             'mean_forecast_probability': p[mask].mean() if n else np.nan,
                             'observed_frequency': frequency, 'wilson_lower': max(0., center-half) if n else np.nan,
                             'wilson_upper': min(1., center+half) if n else np.nan,
                             'panel_cases': len(subset), 'base_rate': float(y.mean()),
                             'brier_score': float(subset.brier_score.mean())})
    return pd.DataFrame(rows, columns=[
        'polygon', 'lead_days', 'sample', 'threshold_1e5_s-1', 'bin_number',
        'probability_from', 'probability_through', 'case_count', 'observed_count',
        'mean_forecast_probability', 'observed_frequency', 'wilson_lower', 'wilson_upper',
        'panel_cases', 'base_rate', 'brier_score'])


def plot_reliability_diagrams(probabilities, *, n_bins=10, title_suffix='All'):
    """Six reliability panels and inset probability-count histograms."""
    import matplotlib.pyplot as plt

    if probabilities.empty:
        raise ValueError('No complete cases for reliability.')
    counts = reliability_bin_counts(probabilities, n_bins=n_bins)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    for r, region in enumerate(POLYGONS):
        for c, lead in enumerate(LEAD_HOURS):
            ax = axes[r, c]
            bins = counts.loc[counts.polygon.eq(region) & counts.lead_days.eq(lead // 24)]
            if bins.empty:
                ax.set_title(f'{region} · {lead // 24}-day · No complete cases')
                continue
            filled = bins.loc[bins.case_count.gt(0)]
            x, y = filled.mean_forecast_probability, filled.observed_frequency
            ax.plot([0, 1], [0, 1], '--', color='.4', linewidth=1)
            ax.axhline(bins.base_rate.iloc[0], color='.6', linestyle=':', linewidth=1)
            ax.errorbar(x, y, yerr=np.maximum(0, np.array([y-filled.wilson_lower, filled.wilson_upper-y])),
                        marker='o', color='#277486', capsize=3)
            for px, py, n in zip(x, y, filled.case_count):
                ax.annotate(f'n={n}', (px, py), xytext=(4, 5), textcoords='offset points', fontsize=7)
            ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel='Forecast probability',
                   ylabel='Observed ERA5 frequency',
                   title=f'{region} · {lead // 24}-day · N={int(bins.panel_cases.iloc[0])}\n'
                         f'D < {bins["threshold_1e5_s-1"].iloc[0]:g} × 10⁻⁵ s⁻¹ · BS={bins.brier_score.iloc[0]:.3f}')
            inset = ax.inset_axes([.58, .10, .35, .22])
            inset.bar((bins.probability_from+bins.probability_through)/2, bins.case_count,
                      width=.9/n_bins, color='#92b8c1')
            inset.set(xlim=(0, 1), title='Forecast counts')
            inset.tick_params(labelsize=6)
            inset.title.set_fontsize(7)
    design = probabilities['sample'].iloc[0]
    description = ('Event-conditioned convergence reliability (ERA5-selected events)'
                   if design == 'event_conditioned' else 'Scheduled winter convergence reliability (crossings and non-crossings)')
    fig.suptitle(f'{description} · {title_suffix}\n'
                 'Instantaneous 925-hPa polygon mean; 64 members; common all-lead cases\n'
                 'Dashed: perfect reliability · dotted: sample base rate · bars: pointwise Wilson 95% (dependence ignored)')
    return fig


def export_reliability(tables, destination, thresholds, *, sample='event_conditioned', group='All', n_bins=10):
    """Refresh probability tables and plots from the current sample, never append old aggregates."""
    import matplotlib.pyplot as plt

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    ranks = verification_subset(tables, group)
    probabilities = reliability_probabilities(tables['member_values'], ranks, thresholds, sample=sample)
    bins = reliability_bin_counts(probabilities, n_bins=n_bins)
    probabilities.to_csv(destination / 'reliability_probabilities.csv', index=False)
    bins.to_csv(destination / 'reliability_bins.csv', index=False)
    if not probabilities.empty:
        summary = probabilities.groupby(['polygon', 'lead_days'], as_index=False).agg(
            cases=('observed', 'size'), base_rate=('observed', 'mean'), brier_score=('brier_score', 'mean'))
        summary.to_csv(destination / 'reliability_summary.csv', index=False)
        fig = plot_reliability_diagrams(probabilities, n_bins=n_bins, title_suffix=group)
        for extension in ('png', 'pdf'):
            fig.savefig(destination / f'reliability_diagrams.{extension}', dpi=180)
        plt.close(fig)
    else:
        # Remove plots from a previous successful run if no current cases remain.
        for name in ('reliability_summary.csv', 'reliability_diagrams.png', 'reliability_diagrams.pdf'):
            (destination / name).unlink(missing_ok=True)
    atomic_json_file(destination / 'reliability_settings.json', {
        'sample': sample, 'group': group, 'thresholds_1e5_s-1': thresholds, 'n_bins': n_bins,
        'criterion': 'instantaneous polygon divergence strictly less than threshold',
        'common_sample': 'ERA5 + 64 WN2 members at every lead and both polygons',
        'not_catalog_D12_event_probability': True,
    })
    return probabilities, bins
