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
    compact_source_load, convert_field,
    forecast_input_group,
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
            raise ValueError(f"{region} lacks a complete finite set of field cells.")
        means[region] = np.average(values, weights=weights, axis=-1)
        counts[region] = int(mask.sum())
    return means, counts


@lru_cache(maxsize=1)
def open_verification_era5():
    # Creating Dask graphs for all decades and variables can exceed Colab RAM
    # before any field is read. Keep backend indexing lazy through subsetting.
    return xr.open_zarr(ERA5_STORE, chunks=None, storage_options={"token": "anon"})


def era5_polygon_values(valid_time, era5, *, allow_era5t=False, spec=None):
    """Evaluate the selected field at the exact ERA5 time and pressure level."""
    spec = spec or DIVERGENCE_SPEC
    target = utc_naive(valid_time)
    stable = era5.attrs.get('valid_time_stop')
    preliminary = era5.attrs.get('valid_time_stop_era5t')
    product = 'ERA5'
    if stable and target > utc_naive(stable):
        if not allow_era5t or not preliminary or target > utc_naive(preliminary):
            raise ValueError(f'ERA5 pending for {iso_utc(target)}; final through {stable}; '
                             f'ERA5T through {preliminary}. No nearest time is substituted.')
        product = 'ERA5T'
    winds = era5[list(spec.variables)].sel(time=target)
    if spec.level is not None:
        winds = winds.sel(level=spec.level)
    winds = winds.rename({"latitude": "lat", "longitude": "lon"})
    west, east, south, north = (diagnostic_domain(winds, VERIFICATION_DOMAIN)
                                if spec.key == 'divergence' else VERIFICATION_DOMAIN)
    latitude = slice(south, north) if winds.lat[0] < winds.lat[-1] else slice(north, south)
    longitude = slice(west, east) if winds.lon[0] < winds.lon[-1] else slice(east, west)
    winds = winds.sel(lat=latitude, lon=longitude)
    if any(abs(actual - expected) > 1e-5 for actual, expected in zip(
            (float(winds.lon.min()), float(winds.lon.max()), float(winds.lat.min()), float(winds.lat.max())),
            (west, east, south, north))):
        raise ValueError("ERA5 grid does not cover the verification domain and its derivative halo.")
    # Decode one wind variable at a time, only after selecting the region.
    winds = compact_source_load(winds)
    winds = winds.transpose("lat", "lon").sortby("lat").sortby("lon")
    field = (derive_field(winds, spec, VERIFICATION_DOMAIN) if spec.key == 'divergence'
             else convert_field(winds, spec)).value
    means, counts = polygon_divergence(field)
    return {"valid_time": iso_utc(target), "source_store": ERA5_STORE, "polygon_cells": counts,
            "reference_product": product, "archive_last_updated": str(era5.attrs.get('last_updated', '')),
            "values": {region: float(value) for region, value in means.items()}}


def forecast_polygon_values(archive, request, progress=print, *, spec=None, group_cache=None,
                            member_start=0, member_stop=64):
    """Read groups of native WN2 samples and immediately reduce each field to two means."""
    spec = spec or DIVERGENCE_SPEC
    store, ds = archive.run(request["init_time"])
    labels = [str(value) for value in ds.sample.values.tolist()]
    if len(labels) != ENSEMBLE_SIZE or len(set(labels)) != ENSEMBLE_SIZE:
        raise ValueError(f"Expected {ENSEMBLE_SIZE} distinct WN2 member labels, found {len(set(labels))}.")
    snap = select_forecast(
        ds, request["init_time"], request["lead_hours"], None, spec.variables,
        spec.level, diagnostic_domain(ds, VERIFICATION_DOMAIN) if spec.key == 'divergence' else VERIFICATION_DOMAIN,
    )
    # Source groups are at most four members; never decode all 64 together.
    if not (0<=member_start<member_stop<=ENSEMBLE_SIZE and member_start%4==0 and member_stop%4==0):
        raise ValueError('Member ranges must cover complete native four-member groups.')
    sizes = (4,) * ((member_stop-member_start)//4)
    values = {region: [] for region in POLYGONS}
    start = member_start
    for size in sizes:
        end = start + size
        progress(f"  Reading {request['lead_days']}-day members {start + 1}–{end} of {ENSEMBLE_SIZE}")
        identity = json.loads(json.dumps(dict(store=store, init_time=iso_utc(request['init_time']),
            lead_hours=int(request['lead_hours']), members=labels[start:end], field=spec.key, level=spec.level,
            units=spec.units, domain=VERIFICATION_DOMAIN, polygons=POLYGONS,
            diagnostic=diagnostic_settings(ds, spec, VERIFICATION_DOMAIN)),sort_keys=True))
        path = Path(group_cache)/f'group_{start:02d}_{end:02d}.json' if group_cache else None
        def calculate_group():
            block = forecast_input_group(archive,store,ds,request,spec,
                diagnostic_domain(ds,VERIFICATION_DOMAIN),start,end)
            field = (derive_field(block, spec, VERIFICATION_DOMAIN) if spec.key == 'divergence'
                     else convert_field(block, spec)).value
            means, counts = polygon_divergence(field)
            return dict(values={region:np.asarray(value).tolist() for region,value in means.items()},
                        polygon_cells=counts)
        group, _ = checkpoint(path, identity, calculate_group) if path else (calculate_group(), False)
        counts = group['polygon_cells']
        for region in POLYGONS:
            if len(group['values'][region]) != size or not np.isfinite(group['values'][region]).all():
                raise ValueError('Invalid saved member-group values.')
            values[region].extend(group['values'][region])
        start = end
    return {"init_time": iso_utc(request["init_time"]), "valid_time": iso_utc(request["valid_time"]),
            "lead_hours": request["lead_hours"], "source_store": store, "members": labels[member_start:member_stop],
            "polygon_cells": counts, "values": values}


def checkpoint_identity(archive, spec=None):
    """Reuse verified values only with the same fields, polygons and source configuration."""
    spec = spec or DIVERGENCE_SPEC
    result = {
        "version": CHECKPOINT_VERSION, "era5_store": ERA5_STORE, "wn2_root": archive.root,
        "wn2_overrides": archive.overrides, "domain": VERIFICATION_DOMAIN, "polygons": POLYGONS,
        "pressure_hpa": spec.level, "ensemble_size": ENSEMBLE_SIZE,
        "diagnostic": diagnostic_settings(None, spec, VERIFICATION_DOMAIN),
    }
    if spec != DIVERGENCE_SPEC:
        result.update(field=spec.key, units=spec.units)
    return result


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
                              allow_era5t=False, progress=print, spec=None, job_runner=None):
    """Checkpoint one ERA5 target and each WN2 run/lead independently."""
    spec = spec or DIVERGENCE_SPEC
    identity = checkpoint_identity(archive, spec)
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
            progress(f"  ERA5: reading one {spec.title} snapshot (or reusing its checkpoint)")
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
                lambda: (job_runner('reference', dict(valid_time=iso_utc(target), allow_era5t=allow_era5t, spec=spec.__dict__))
                         if job_runner else era5_polygon_values(target, era5_reader(), allow_era5t=allow_era5t, spec=spec)),
            )
            reference_rows.extend([
                {**event, "polygon": region, "pressure_hpa": spec.level,
                 'field':spec.key, 'field_title':spec.title, 'units':spec.units,
                 'value':reference['values'][region],
                 **({DIVERGENCE_COLUMN:reference['values'][region]} if spec.key=='divergence' else {}),
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
                    lambda: (job_runner('forecast',dict(request={k:iso_utc(v) if isinstance(v,pd.Timestamp) else v
                        for k,v in request.items()}, spec=spec.__dict__, group_cache=str(cache/f'groups_{stamp}_{lead:03d}h')))
                        if job_runner else forecast_polygon_values(archive, request, progress, spec=spec,
                            group_cache=cache/f'groups_{stamp}_{lead:03d}h')),
                )
                members = forecast["members"]
                if len(members) != ENSEMBLE_SIZE or len(set(members)) != ENSEMBLE_SIZE:
                    raise ValueError("Forecast checkpoint does not contain 64 distinct members.")
                case_values, case_ranks = [], []
                for region in POLYGONS:
                    values = np.asarray(forecast["values"][region], dtype=float)
                    if values.shape != (ENSEMBLE_SIZE,) or not np.isfinite(values).all():
                        raise ValueError(f"{region} does not contain 64 finite member values.")
                    info = {**event, **timing, "polygon": region, "pressure_hpa": spec.level,
                            'field':spec.key, 'field_title':spec.title, 'units':spec.units}
                    case_values.extend({**info, "member": member, 'value':value,
                                        **({DIVERGENCE_COLUMN:value} if spec.key=='divergence' else {}),
                                        "polygon_cells": forecast["polygon_cells"][region],
                                        "source_store": forecast["source_store"]} for member, value in zip(members, values))
                    if reference is not None:
                        observed = reference["values"][region]
                        rank = observation_rank(values, observed, f"{row.review_key}|{lead}|{region}")
                        case_ranks.append({**info, "member_count": len(members),
                                           "reference_product": reference.get('reference_product', 'ERA5'),
                                           'era5_value':observed, 'member_min':values.min(),
                                           'member_mean':values.mean(), 'member_max':values.max(),
                                           **({'era5_divergence_1e5_s-1':observed,
                                               'member_min_1e5_s-1':values.min(), 'member_mean_1e5_s-1':values.mean(),
                                               'member_max_1e5_s-1':values.max()} if spec.key=='divergence' else {}), **rank})
                member_rows.extend(case_values)
                event_ranks.extend(case_ranks)
                status.update(wn2_status="cached" if cached else "computed", wn2_error="")
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                status.update(wn2_status="failed", wn2_error=message)
                progress(f"  {request['lead_days']}-day forecast unavailable: {message}")
            status_rows.append({**status,'field':spec.key,'field_title':spec.title,'pressure_hpa':spec.level,'units':spec.units})
            # Keep a lightweight progress table even if the kernel stops during the next lead.
            progress_path = Path(output_dir)/'progress'/signature/'batch_status.csv'
            progress_path.parent.mkdir(parents=True,exist_ok=True)
            temporary = progress_path.with_suffix('.tmp.csv')
            pd.DataFrame(status_rows).to_csv(temporary,index=False)
            temporary.replace(progress_path)
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
    if 'field' in ranks and ranks['field'].nunique()>1:
        raise ValueError('Plot each verification field separately; do not pool different variables.')
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
    field_title = str(ranks.field_title.iloc[0]) if 'field_title' in ranks else '925-hPa horizontal divergence'
    convergence = 'field' not in ranks or ranks['field'].eq('divergence').all()
    fig.suptitle(f"{title_suffix} · {field_title} polygon mean at the onset target · {n_events} matched "
                 f"{'event' if n_events == 1 else 'events'}\n"
                 f"{'13 ranks' if grouped else '1 rank'} per bar · dashed line = uniform rank reference\n"
                 + ("Lower rank = ERA5 more convergent relative to forecasts · Higher rank = ERA5 less convergent"
                    if convergence else 'Lower rank = ERA5 lower relative to forecasts · Higher rank = ERA5 higher'))
    return fig


def ordered_verification_values(members, ranks):
    """Pool 64 forecast values and ERA5, preserving the exact saved tie-broken rank."""
    keys = ["review_key", "polygon", "lead_hours"]
    columns = [*keys, "event_number", "valid_time_utc", "init_time_utc", "lead_days", "pressure_hpa",
               "in_common_sample", "position", "bin_number", "rank_range", "source", "member", DIVERGENCE_COLUMN,
               'value','field','field_title','units','vortex_label','vortex_scan_status','vortex_classification']
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
        value_column = 'value' if 'value' in group else DIVERGENCE_COLUMN
        values = group[value_column].to_numpy(dtype=float)
        observed = rank['era5_value'] if 'era5_value' in rank else rank['era5_divergence_1e5_s-1']
        if not np.isfinite(values).all() or not np.isfinite(observed):
            raise ValueError(f"Ordered values contain missing or nonfinite divergence for {key}.")
        position = int(rank["era5_rank"])
        lower = int((values < observed).sum()) + 1
        upper = lower + int((values == observed).sum())
        if not lower <= position <= upper:
            raise ValueError(f"Saved ERA5 rank disagrees with the member values for {key}.")
        ordered = group.sort_values(value_column, kind="stable")
        entries = [
            {"source": "WN2", "member": str(item["member"]), 'value':float(item[value_column])}
            for item in ordered.to_dict("records")
        ]
        entries.insert(position - 1, {"source": "ERA5", "member": "", 'value':float(observed)})
        info = {name: rank[name] for name in columns[:9]}
        info.update(field=rank.get('field','divergence'),field_title=rank.get('field_title','925-hPa horizontal divergence'),
                    units=rank.get('units','10⁻⁵ s⁻¹'),vortex_label=rank.get('vortex_label',''),
                    vortex_scan_status=rank.get('vortex_scan_status',''),
                    vortex_classification=rank.get('vortex_classification',vortex_classification(rank)))
        for index, entry in enumerate(entries):
            if info['field']=='divergence':entry[DIVERGENCE_COLUMN]=entry['value']
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
    title = str(ranks.field_title.iloc[0]) if len(ranks) and 'field_title' in ranks else '925-hPa polygon-average divergence'
    units = str(ranks.units.iloc[0]) if len(ranks) and 'units' in ranks else '10⁻⁵ s⁻¹'
    convergence_note = (' Negative divergence is convergence.' if 'field' not in ranks
                        or (not ranks.empty and ranks.field.eq('divergence').all()) else '')
    sheet.merge_range(0, 0, 0, 14, 'Ordered '+title, title_format)
    sheet.merge_range(1, 0, 2, 14,
                      'Each block contains 64 WN2 values and one ERA5 value, sorted from lowest to highest. '
                      f'ERA5 is highlighted. Values are in {units}.{convergence_note} '
                      "Histogram counts use only ERA5's saved position.", note_format)
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
                          f"Event {case['event_number']} · {case['polygon']} · {case['lead_days']}-day lead · {case['pressure_hpa']} hPa",
                          title_format)
        sheet.merge_range(row + 1, 0, row + 1, 14,
                          f"Valid {case['valid_time_utc']} · Initialized {case['init_time_utc']} · "
                          f"ERA5 rank {case['era5_rank']} of 65 · "
                          f"{'Included in all-lead histograms' if case['in_common_sample'] else 'Incomplete all-lead comparison'}")
        for bin_index, label in enumerate(RANK_GROUPS):
            start = bin_index * 3
            sheet.merge_range(row + 2, start, row + 2, start + 2,
                              f"Bin {bin_index + 1} · ranks {label}", bin_format)
            for offset, text in enumerate(("Position", "Source", f'Value ({units})')):
                sheet.write(row + 3, start + offset, text, header_format)
        for item in group.to_dict("records"):
            cell_row = row + 4 + (item["position"] - 1) % 13
            cell_col = (item["bin_number"] - 1) * 3
            label_format = era5_label_format if item["source"] == "ERA5" else None
            number_format = era5_value_format if item["source"] == "ERA5" else value_format
            sheet.write_number(cell_row, cell_col, item["position"], label_format)
            source = "ERA5" if item["source"] == "ERA5" else f"WN2 sample {item['member']}"
            sheet.write_string(cell_row, cell_col + 1, source, label_format)
            sheet.write_number(cell_row, cell_col + 2, item['value'] if 'value' in item else item[DIVERGENCE_COLUMN], number_format)
        row += 19
    sheet.print_area(0, 0, row - 1, 14)


def export_verification(tables, identity, destination):
    """Write the results and a workbook for inspecting the computed ranks in Drive."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tables = dict(tables)
    ranks = tables["case_ranks"]
    tables["ordered_values"] = ordered_verification_values(tables["member_values"], ranks)
    tables['crps_cases'] = continuous_ranked_scores(tables)
    tables['crps_summary'] = continuous_score_summary(tables['crps_cases'])
    tables['event_accounting'] = verification_event_accounting(tables)
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


def empirical_ensemble_crps(members, reference):
    """CRPS of the equally weighted empirical CDF, in the original variable units.

    mean|X-y| - (1/(2*M**2))*sum_ij|Xi-Xj|. The sorted form uses O(M log M)
    work and O(M) memory. This scores the finite ensemble CDF shown in the proposal;
    it is not the independence-dependent 'fair' estimator of an underlying distribution.
    """
    values=np.asarray(members,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or not np.isfinite(reference):
        raise ValueError('CRPS requires finite members and a finite verifying value.')
    count=len(values)
    first=float(np.mean(np.abs(values-reference)))
    centered=np.sort(values-values[0])
    second=float(np.dot(2*np.arange(1,count+1)-count-1,centered)/(count*count))
    score=first-second
    if score < -1e-12*max(1.,first,second):raise ArithmeticError('Negative empirical CRPS beyond rounding tolerance.')
    return max(0.,score)


def continuous_ranked_scores(tables, *, group='All'):
    """Score saved polygon means only, retaining the common all-lead event sample."""
    ranks=verification_subset(tables,group)
    columns=['event_number','review_key','valid_time_utc','init_time_utc','polygon','lead_hours','lead_days',
             'field','field_title','pressure_hpa','units','vortex_label','vortex_scan_status',
             'vortex_classification','reference_product','member_count','era5_value','crps',
             'ensemble_mean','ensemble_median','ensemble_min','ensemble_max','in_common_sample']
    if ranks.empty:return pd.DataFrame(columns=columns)
    keys=['review_key','polygon','lead_hours']
    if ranks.duplicated(keys).any():raise ValueError('Duplicate cases would overweight CRPS.')
    members=tables['member_values']
    groups={key:frame for key,frame in members.groupby(keys,sort=False)}
    rows=[]
    for row in ranks.loc[ranks.in_common_sample].to_dict('records'):
        frame=groups.get(tuple(row[key] for key in keys))
        if frame is None or len(frame)!=ENSEMBLE_SIZE or frame.member.astype(str).nunique()!=ENSEMBLE_SIZE:
            raise ValueError('CRPS requires all 64 distinct native ensemble members for every scored case.')
        for name in ('field','pressure_hpa','units'):
            if name in frame and name in row and not frame[name].eq(row[name]).all():
                raise ValueError(f'CRPS member/reference {name} metadata differs.')
        values=frame['value' if 'value' in frame else DIVERGENCE_COLUMN].to_numpy(dtype=float)
        observed=float(row['era5_value'] if 'era5_value' in row else row['era5_divergence_1e5_s-1'])
        info={name:row.get(name,'') for name in columns[:15]}
        info.update(field=row.get('field','divergence'),field_title=row.get('field_title','925-hPa divergence'),
                    pressure_hpa=row.get('pressure_hpa',925),units=row.get('units','10⁻⁵ s⁻¹'),
                    reference_product=row.get('reference_product','ERA5'))
        rows.append(dict(**info,member_count=ENSEMBLE_SIZE,era5_value=observed,
            crps=empirical_ensemble_crps(values,observed),ensemble_mean=float(values.mean()),
            ensemble_median=float(np.median(values)),ensemble_min=float(values.min()),
            ensemble_max=float(values.max()),in_common_sample=True))
    return pd.DataFrame(rows,columns=columns)


def continuous_score_summary(scores):
    columns=['field','pressure_hpa','units','polygon','lead_days','cases','mean_crps','median_crps',
             'q25_crps','q75_crps','min_crps','max_crps','era5t_cases','valid_time_start','valid_time_end']
    if scores.empty:return pd.DataFrame(columns=columns)
    rows=[]
    for key,frame in scores.groupby(columns[:5],sort=False):
        times=pd.to_datetime(frame.valid_time_utc,utc=True)
        rows.append(dict(zip(columns[:5],key),cases=len(frame),mean_crps=float(frame.crps.mean()),
            median_crps=float(frame.crps.median()),q25_crps=float(frame.crps.quantile(.25)),
            q75_crps=float(frame.crps.quantile(.75)),min_crps=float(frame.crps.min()),max_crps=float(frame.crps.max()),
            era5t_cases=int(frame.reference_product.eq('ERA5T').sum()),valid_time_start=iso_utc(times.min()),
            valid_time_end=iso_utc(times.max())))
    return pd.DataFrame(rows,columns=columns)


def verification_event_accounting(tables):
    """One row per selected event: membership, usable leads and full retrieval errors."""
    status=tables.get('batch_status',pd.DataFrame())
    ranks=tables['case_ranks']
    base=status if not status.empty else ranks
    columns=['event_number','review_key','valid_time_utc','field','pressure_hpa','vortex_classification',
             'used_in_common_sample','wn2_complete_leads','expected_leads','era5_status','wn2_errors','era5_errors']
    if base.empty:return pd.DataFrame(columns=columns)
    complete=set(ranks.loc[ranks.in_common_sample,'review_key']) if not ranks.empty else set()
    rows=[]
    for key,frame in base.groupby('review_key',sort=False):
        first=frame.iloc[0]
        def joined(name):return ' | '.join(dict.fromkeys(str(x) for x in frame.get(name,[]) if pd.notna(x) and str(x)))
        good=frame.loc[frame.wn2_status.isin(['cached','computed','complete']),'lead_days'].nunique() if 'wn2_status' in frame else frame.lead_days.nunique()
        rows.append({**{name:first.get(name,'') for name in columns[:6]},'review_key':key,
            'used_in_common_sample':key in complete,'wn2_complete_leads':good,'expected_leads':len(LEAD_HOURS),
            'era5_status':joined('era5_status'),'wn2_errors':joined('wn2_error'),'era5_errors':joined('era5_error')})
    return pd.DataFrame(rows,columns=columns)


def plot_crps_summary(scores, *, title_suffix='All'):
    """Lead-time comparison and statistics from stored scalar values, without field reads."""
    import matplotlib.pyplot as plt
    if scores.empty:raise ValueError('No complete cases for CRPS.')
    if scores[['field','pressure_hpa','units']].drop_duplicates().shape[0]!=1:
        raise ValueError('Plot each CRPS variable and level separately.')
    summary=continuous_score_summary(scores);first=scores.iloc[0]
    fig=plt.figure(figsize=(13,8.5))
    grid=fig.add_gridspec(2,2,height_ratios=[3.2,1.7],left=.08,right=.97,top=.79,bottom=.10,hspace=.27,wspace=.28)
    for column,region in enumerate(POLYGONS):
        ax=fig.add_subplot(grid[0,column]);table_ax=fig.add_subplot(grid[1,column])
        region_summary=summary.loc[summary.polygon.eq(region)].set_index('lead_days').reindex([1,5,10])
        samples=[scores.loc[scores.polygon.eq(region)&scores.lead_days.eq(lead),'crps'].to_numpy() for lead in (1,5,10)]
        ax.boxplot(samples,positions=[1,5,10],widths=.9,patch_artist=True,
            boxprops={'facecolor':'#dcebea','edgecolor':'#277f84'},medianprops={'color':'#b06c19','linewidth':1.6},
            flierprops={'marker':'.','markerfacecolor':'#6e8087','markeredgecolor':'#6e8087','markersize':4})
        ax.plot([1,5,10],region_summary.mean_crps,'o-',color='#087d78',linewidth=2,label='Mean CRPS')
        ax.set(xticks=[1,5,10],xticklabels=['1 day','5 days','10 days'],xlim=(0,11),ylim=(0,None),
               xlabel='Forecast lead',ylabel=f'CRPS ({first.units})',title=region)
        ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True);ax.legend(frameon=False,loc='upper left')
        table_ax.axis('off')
        rows=[['Usable events',*[str(int(x)) for x in region_summary.cases]],
              ['Mean CRPS',*[f'{x:.3f}' for x in region_summary.mean_crps]],
              ['Median CRPS',*[f'{x:.3f}' for x in region_summary.median_crps]],
              ['25th–75th percentile',*[f'{a:.3f}–{b:.3f}' for a,b in zip(region_summary.q25_crps,region_summary.q75_crps)]],
              ['Preliminary ERA5T cases',*[str(int(x)) for x in region_summary.era5t_cases]]]
        table=table_ax.table(cellText=rows,colLabels=['Statistic','1 day','5 days','10 days'],
            colWidths=[.43,.19,.19,.19],cellLoc='left',bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(10)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor('#d7e0e7');cell.set_linewidth(.6)
            if r==0:cell.set_facecolor('#e6eef3');cell.set_text_props(weight='bold')
            elif r%2==0:cell.set_facecolor('#f5f8fa')
    fig.suptitle(f'Continuous ranked probability score · {title_suffix}\n{first.field_title}\n'
                 'All 64 members · same complete events at every lead · lower is better',fontsize=14,y=.975,linespacing=1.5)
    times=pd.to_datetime(scores.valid_time_utc,utc=True)
    fig.text(.08,.025,f'Verifying dates: {times.min():%Y-%m-%d} to {times.max():%Y-%m-%d}. '
             'Boxes: middle 50% of case scores; orange line: median; green line: mean.\n'
             'CRPS scores polygon means. CRPSS is reported separately against a complete independent reference.',fontsize=9)
    return fig


def export_continuous_scores(tables, destination, *, group='All'):
    import matplotlib.pyplot as plt
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    scores=continuous_ranked_scores(tables,group=group);summary=continuous_score_summary(scores)
    scores.to_csv(destination/'crps_cases.csv',index=False);summary.to_csv(destination/'crps_summary.csv',index=False)
    if scores.empty:
        for suffix in ('png','pdf'):(destination/f'crps_summary.{suffix}').unlink(missing_ok=True)
    else:
        fig=plot_crps_summary(scores,title_suffix=group)
        for suffix in ('png','pdf'):fig.savefig(destination/f'crps_summary.{suffix}',dpi=180)
        plt.close(fig)
    atomic_json_file(destination/'crps_settings.json',dict(group=group,ensemble_members=64,
        estimator='empirical equally weighted ensemble CDF',units='same units as each selected field',
        formula='mean(abs(X-y)) - sum_ij(abs(X_i-X_j))/(2*M*M)',
        sample='common all-lead complete events; instantaneous polygon means',
        crpss_status='reported separately; see crpss/crpss_status.json for reference readiness',
        source='https://scoringrules.readthedocs.io/en/latest/crps_estimators.html'))
    return scores,summary


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


def reliability_probabilities(members, ranks, thresholds, *, sample='event_conditioned', direction='less'):
    """A selected-field threshold probability from all 64 members and binary ERA5 outcome.

    This is instantaneous polygon convergence, not the hourly trailing-D12 JPCZ detector.
    Only the common all-lead sample enters, exactly as in the rank histograms.
    """
    if sample not in ('event_conditioned', 'scheduled_winter'):
        raise ValueError('Unknown reliability sampling design.')
    if set(thresholds) != set(POLYGONS) or not np.isfinite(list(thresholds.values())).all():
        raise ValueError('Supply one finite snapshot threshold in the selected field units per polygon.')
    if direction not in ('less','greater'):raise ValueError('Choose less or greater for the probability threshold.')
    keys = ['review_key', 'polygon', 'lead_hours']
    columns = [*keys, 'lead_days', 'valid_time_utc', 'event_number', 'vortex_label', 'vortex_scan_status',
               'sample', 'threshold_1e5_s-1', 'member_count', 'forecast_probability', 'observed', 'brier_score',
               'threshold','direction','field','field_title','units','pressure_hpa']
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
        values = frame['value' if 'value' in frame else DIVERGENCE_COLUMN].to_numpy(dtype=float)
        observed_value = float(row['era5_value'] if 'era5_value' in row else row['era5_divergence_1e5_s-1'])
        if not np.isfinite(values).all() or not np.isfinite(observed_value):
            raise ValueError('Reliability inputs must be finite.')
        threshold = float(thresholds[row['polygon']])
        probability = float(np.mean(values < threshold if direction=='less' else values > threshold))
        observed = int(observed_value < threshold if direction=='less' else observed_value > threshold)
        rows.append({**{name: row.get(name, '') for name in columns[:8]},
                     'sample': sample, 'threshold_1e5_s-1': threshold if row.get('field','divergence')=='divergence' else np.nan,
                     'member_count': ENSEMBLE_SIZE,
                     'forecast_probability': probability, 'observed': observed,
                     'brier_score': (probability - observed)**2, 'threshold':threshold,'direction':direction,
                     'field':row.get('field','divergence'),'field_title':row.get('field_title','925-hPa divergence'),
                     'units':row.get('units','10⁻⁵ s⁻¹'),'pressure_hpa':row.get('pressure_hpa',925)})
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
                             'threshold':subset['threshold'].iloc[0] if 'threshold' in subset else subset['threshold_1e5_s-1'].iloc[0],
                             'direction':subset['direction'].iloc[0] if 'direction' in subset else 'less',
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
        'panel_cases', 'base_rate', 'brier_score','threshold','direction'])


def reliability_panel_statistics(probabilities, *, n_bins=10):
    """Numbers printed below the diagrams, with the same case/bin accounting as the curve."""
    columns=['polygon','lead_days','cases','observed_crossings','observed_non_crossings',
             'base_rate','mean_forecast_probability','brier_score','populated_bins',
             'probability_bins','valid_time_start','valid_time_end','field','pressure_hpa',
             'units','threshold','direction','ensemble_members']
    if probabilities.empty:return pd.DataFrame(columns=columns)
    counts=reliability_bin_counts(probabilities,n_bins=n_bins)
    rows=[]
    for (region,lead),frame in probabilities.groupby(['polygon','lead_days'],sort=False):
        bins=counts.loc[counts.polygon.eq(region)&counts.lead_days.eq(lead)]
        times=pd.to_datetime(frame.valid_time_utc,utc=True)
        hits=int(frame.observed.sum())
        rows.append(dict(polygon=region,lead_days=int(lead),cases=len(frame),observed_crossings=hits,
            observed_non_crossings=len(frame)-hits,base_rate=float(frame.observed.mean()),
            mean_forecast_probability=float(frame.forecast_probability.mean()),
            brier_score=float(frame.brier_score.mean()),populated_bins=int(bins.case_count.gt(0).sum()),
            probability_bins=n_bins,valid_time_start=iso_utc(times.min()),valid_time_end=iso_utc(times.max()),
            field=frame.field.iloc[0],pressure_hpa=int(frame.pressure_hpa.iloc[0]),units=frame.units.iloc[0],
            threshold=float(frame.threshold.iloc[0]),direction=frame.direction.iloc[0],
            ensemble_members=int(frame.member_count.iloc[0])))
    return pd.DataFrame(rows,columns=columns)


def _report_statistics_table(ax, rows, *, widths=(.66,.34)):
    ax.axis('off')
    table=ax.table(cellText=rows,colLabels=['Statistic','Value'],colWidths=list(widths),
                   cellLoc='left',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#d7e0e7');cell.set_linewidth(.6);cell.PAD=.08
        if r==0:cell.set_facecolor('#e6eef3');cell.set_text_props(weight='bold',color='#17324d')
        elif r%2==0:cell.set_facecolor('#f5f8fa')
    return table


def plot_reliability_diagrams(probabilities, *, n_bins=10, title_suffix='All', region=None,
                              show_intervals=False, lead_hours=None):
    """Readable percent axes and a separate statistics table beneath every panel."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    if probabilities.empty:raise ValueError('No complete cases for reliability.')
    regions=list(POLYGONS) if region is None else [region]
    if any(item not in POLYGONS for item in regions):raise ValueError('Unknown verification polygon.')
    leads=list(LEAD_HOURS) if lead_hours is None else [lead_hours]
    if any(lead not in LEAD_HOURS for lead in leads):raise ValueError('Choose an exact configured forecast lead.')
    single=len(leads)==1
    counts=reliability_bin_counts(probabilities,n_bins=n_bins)
    statistics=reliability_panel_statistics(probabilities,n_bins=n_bins)
    fig=plt.figure(figsize=(8 if single else 16,(10.5 if single else 7.7)*len(regions)))
    grid=fig.add_gridspec(2*len(regions),len(leads),height_ratios=[3.5,1.65]*len(regions),
                          left=.07,right=.975,bottom=.09 if len(regions)==1 else .055,
                          top=(.72 if single else .80) if len(regions)==1 else .89,wspace=.29,hspace=.32)
    for r,polygon in enumerate(regions):
        for c,lead in enumerate(leads):
            ax=fig.add_subplot(grid[2*r,c]);info_ax=fig.add_subplot(grid[2*r+1,c])
            bins=counts.loc[counts.polygon.eq(polygon)&counts.lead_days.eq(lead//24)]
            ax.plot([0,100],[0,100],'--',color='#46515a',linewidth=1.4,zorder=1)
            ax.set(xlim=(-2,102),ylim=(-2,102),xlabel='Forecast probability',ylabel='Observed frequency (ERA5)',
                   title=f'{polygon} · {lead//24}-day lead')
            ax.set_aspect('equal',adjustable='box');ax.set_xticks(range(0,101,20));ax.set_yticks(range(0,101,20))
            ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
            ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
            ax.tick_params(labelsize=10);ax.title.set_fontsize(12);ax.grid(color='#dfe5e9',linewidth=.6)
            if bins.empty:
                ax.text(.5,.5,'No complete cases',ha='center',transform=ax.transAxes)
                _report_statistics_table(info_ax,[['Usable cases','0']]);continue
            # NaNs leave gaps at empty probability bins; no smoothing or invented frequencies.
            ax.plot(100*bins.mean_forecast_probability,100*bins.observed_frequency,
                    marker='o',markersize=6,color='#087d78',linewidth=2,zorder=3)
            ax.axhline(100*bins.base_rate.iloc[0],color='#b06c19',linestyle=':',linewidth=1.4,zorder=2)
            if show_intervals:
                filled=bins.loc[bins.case_count.gt(0)]
                y=100*filled.observed_frequency
                ax.errorbar(100*filled.mean_forecast_probability,y,
                    yerr=np.maximum(0,np.array([y-100*filled.wilson_lower,100*filled.wilson_upper-y])),
                    fmt='none',ecolor='#087d78',alpha=.4,capsize=3,zorder=2)
            stat=statistics.loc[statistics.polygon.eq(polygon)&statistics.lead_days.eq(lead//24)].iloc[0]
            _report_statistics_table(info_ax,[['Usable events',str(int(stat.cases))],
                ['ERA5 threshold crossings',f'{int(stat.observed_crossings)} / {int(stat.cases)}'],
                ['Observed frequency',f'{stat.base_rate:.1%}'],
                ['Mean forecast probability',f'{stat.mean_forecast_probability:.1%}'],
                ['Brier score (lower is better)',f'{stat.brier_score:.3f}'],
                ['Populated probability bins',f'{int(stat.populated_bins)} / {n_bins}']])
    first=probabilities.iloc[0];operator='<' if first.direction=='less' else '>'
    design='ERA5-event-conditioned sample' if first['sample']=='event_conditioned' else 'Scheduled winter sample'
    threshold_label=(f'{first.threshold:g} × {first.units}' if first.field=='divergence'
                     else f'{first.threshold:g} {first.units}')
    criterion=f'Criterion: polygon mean {operator} {threshold_label}'
    header=(f'{criterion}\n64 members · {design}' if single else f'{criterion} · 64 members · {design}')
    fig.suptitle(f'Reliability · {title_suffix}\n{first.field_title}\n{header}',
                 fontsize=14,y=.985,linespacing=1.5)
    handles=[Line2D([],[],color='#087d78',marker='o',linewidth=2,label='Observed reliability'),
             Line2D([],[],color='#46515a',linestyle='--',label='Perfect reliability'),
             Line2D([],[],color='#b06c19',linestyle=':',label='Observed frequency in this sample')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.5,(.795 if single else .865) if len(regions)==1 else .935),
               ncol=1 if single else 3,frameon=False,fontsize=10)
    times=pd.to_datetime(probabilities.valid_time_utc,utc=True)
    caption=(f'Verifying dates: {times.min():%Y-%m-%d} to {times.max():%Y-%m-%d}. '
             + ('\n' if single else '')
             + f'Each event contributes once per panel. Bins: {100/n_bins:g} percentage points.\n'
             + 'Above the diagonal: probabilities too low; below: probabilities too high. '
             + ('\n' if single else '')
             + 'Counts and pointwise Wilson 95% intervals are saved separately.')
    fig.text(.07,.02,caption,fontsize=9,color='#344858')
    return fig


def plot_forecast_probability_counts(probabilities, *, n_bins=10, title_suffix='All'):
    """Sharpness/counts companion figure, with no inset inside reliability axes."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter,MaxNLocator
    if probabilities.empty:raise ValueError('No complete cases for forecast counts.')
    counts=reliability_bin_counts(probabilities,n_bins=n_bins)
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    for r,region in enumerate(POLYGONS):
        for c,lead in enumerate(LEAD_HOURS):
            ax=axes[r,c];bins=counts.loc[counts.polygon.eq(region)&counts.lead_days.eq(lead//24)]
            ax.bar(50*(bins.probability_from+bins.probability_through),bins.case_count,
                   width=90/n_bins,color='#277f84',edgecolor='white')
            ax.set(xlim=(0,100),ylim=(0,None),xlabel='Forecast probability bin',ylabel='Number of events',
                   title=f'{region} · {lead//24}-day lead · N={int(bins.case_count.sum())}')
            ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'Forecast probability counts · {title_suffix}\n'
                 'One forecast probability per event, polygon and lead. Empty bins contain no events.',fontsize=14)
    return fig


def export_reliability(tables, destination, thresholds, *, sample='event_conditioned', group='All', n_bins=10,
                       direction='less', show_intervals=False):
    """Refresh probability tables and plots from the current sample, never append old aggregates."""
    import matplotlib.pyplot as plt

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    ranks = verification_subset(tables, group)
    probabilities = reliability_probabilities(tables['member_values'], ranks, thresholds, sample=sample,direction=direction)
    bins = reliability_bin_counts(probabilities, n_bins=n_bins)
    probabilities.to_csv(destination / 'reliability_probabilities.csv', index=False)
    bins.to_csv(destination / 'reliability_bins.csv', index=False)
    if not probabilities.empty:
        summary = reliability_panel_statistics(probabilities,n_bins=n_bins)
        summary.to_csv(destination / 'reliability_summary.csv', index=False)
        figures=[('reliability_diagrams',None,None),('reliability_jpcz_polygon','JPCZ polygon',None),
                 ('reliability_coastal_wedge','Coastal wedge',None)]
        figures.extend((f'reliability_{tag}_lead{lead//24:02d}d',polygon,lead)
                       for tag,polygon in (('jpcz_polygon','JPCZ polygon'),('coastal_wedge','Coastal wedge'))
                       for lead in LEAD_HOURS)
        for stem,region,lead in figures:
            fig=plot_reliability_diagrams(probabilities,n_bins=n_bins,title_suffix=group,region=region,
                                          show_intervals=show_intervals,lead_hours=lead)
            for extension in ('png','pdf'):fig.savefig(destination/f'{stem}.{extension}',dpi=180)
            plt.close(fig)
        fig=plot_forecast_probability_counts(probabilities,n_bins=n_bins,title_suffix=group)
        for extension in ('png','pdf'):fig.savefig(destination/f'forecast_probability_counts.{extension}',dpi=180)
        plt.close(fig)
    else:
        # Remove plots from a previous successful run if no current cases remain.
        for name in ('reliability_summary.csv',*[f'{stem}.{ext}' for stem in
            ('reliability_diagrams','reliability_jpcz_polygon','reliability_coastal_wedge','forecast_probability_counts',
             *[f'reliability_{tag}_lead{lead//24:02d}d' for tag in ('jpcz_polygon','coastal_wedge') for lead in LEAD_HOURS])
            for ext in ('png','pdf')]):
            (destination / name).unlink(missing_ok=True)
    atomic_json_file(destination / 'reliability_settings.json', {
        'sample': sample, 'group': group, 'thresholds':thresholds,'direction':direction, 'n_bins': n_bins,
        'criterion': f'instantaneous polygon mean strictly {direction} than threshold',
        'common_sample': 'ERA5 + 64 WN2 members at every lead and both polygons',
        'not_catalog_D12_event_probability': True, 'plot_intervals':show_intervals,
        'intervals':'pointwise Wilson 95%; serial dependence not modeled',
    })
    return probabilities, bins


def display_saved_verification_image(path, *, max_width=960):
    """Fit a saved PNG to the actual notebook width instead of creating horizontal overflow."""
    import base64
    from IPython.display import HTML,display
    encoded=base64.b64encode(Path(path).read_bytes()).decode('ascii')
    display(HTML(f'<img src="data:image/png;base64,{encoded}" alt="Saved verification figure" '
                 f'style="width:100%;max-width:{int(max_width)}px;height:auto;display:block;margin:0 auto;"/>'))
