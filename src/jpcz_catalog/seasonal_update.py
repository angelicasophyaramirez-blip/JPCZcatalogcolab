"""Extend the established catalog without changing its anomaly baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from .config import JPCZ_POLYGON_VERTICES, ARCO_ERA5_ZARR_STORE
from .detect import count_events_from_threshold, merge_nearby_events, prepare_detection_geometry, threshold_from_std
from .era5 import subset_era5_window
from .historical_catalog import (
    NDJFMonth, compute_polygon_mean_divergence_series_chunked, find_existing_checkpoint,
    validate_d12_checkpoint, write_dataframe_atomic, write_dataarray_atomic,
)

DATE_COLUMNS = ('event_start', 'event_end', 'event_peak')
RAW_NAME = 'jpcz_catalog_ndjf.csv'
MERGED_NAME = 'jpcz_catalog_ndjf_merged_12h.csv'
DETECTOR = {
    'version': 'established_shinoda_era5_frozen_monthly_v1', 'pressure_hpa': 925,
    'polygon': JPCZ_POLYGON_VERTICES, 'weighting': 'cos(latitude)',
    'divergence': 'MetPy divergence; lat_lon_grid_deltas; established Notebook 04',
    'rolling_hours': 12, 'rolling_label': 'trailing window end',
    'month_boundary': '11-hour warmup each calendar month, matching Notebook 04',
    'criterion': 'D12 < frozen month-specific baseline mean - 2 population std',
    'merge_gap_hours': 12, 'season_months': [11, 12, 1, 2],
}


def read_events(path):
    frame = pd.read_csv(path)
    for name in DATE_COLUMNS:
        frame[name] = pd.to_datetime(frame[name], utc=True, format='mixed').dt.tz_localize(None)
    if frame[list(DATE_COLUMNS)].isna().any().any():
        raise ValueError(f'Missing event timestamps in {path}')
    if frame.duplicated(list(DATE_COLUMNS)).any():
        raise ValueError(f'Duplicate event identities in {path}')
    return frame


def event_keys(frame):
    return frame[list(DATE_COLUMNS)].apply(lambda row: '|'.join(t.isoformat() for t in row), axis=1)


def inspect_catalog(root):
    """Inspect actual Drive files; never infer the user's count from repository CSVs."""
    root = Path(root)
    raw, merged = read_events(root / RAW_NAME), read_events(root / MERGED_NAME)
    missing = [r.event_start for r in raw.itertuples() if not
               ((merged.event_start <= r.event_start) & (merged.event_end >= r.event_end)).any()]
    if missing:
        raise ValueError('Saved raw and merged catalogs disagree. Run the current Notebook 06 before extending.')
    expected = merge_nearby_events(raw, max_gap_hours=12)
    if set(event_keys(expected)) != set(event_keys(merged)):
        raise ValueError('The saved merged catalog does not match a 12-hour merge of the saved raw catalog. '
                         'Review the Notebook 06 version/settings before extending.')
    last = raw.event_end.max()
    if pd.isna(last):
        raise ValueError('The established catalog is empty; build it with Notebook 04 first.')
    record = NDJFMonth(last.year, last.year, last.month)
    path = find_existing_checkpoint(record, legacy_monthly_dir=root / 'ndjf_monthly',
                                    legacy_december_dir=root / 'december_yearly')
    valid, reason = validate_d12_checkpoint(path, record) if path else (False, 'not found')
    return {'raw_events': len(raw), 'merged_events': len(merged),
            'first_peak': str(merged.event_peak.min()), 'last_peak': str(merged.event_peak.max()),
            'last_event_month_complete': valid, 'checkpoint_note': reason,
            'suggested_update_start': str((last.to_period('M') + 1).start_time.date()) if valid else None,
            'detector': DETECTOR}


def frozen_thresholds(root):
    """Use Notebook 04 thresholds, or reconstruct them from its complete saved months."""
    root = Path(root)
    path = root / 'ndjf_monthly_thresholds.csv'
    if path.exists():
        frame = pd.read_csv(path)
        source = str(path)
    else:
        # Only years in the saved established raw catalog, never the new winter.
        raw = read_events(root / RAW_NAME)
        years = sorted(raw.event_start.dt.year.unique())
        rows = []
        for month in (11, 12, 1, 2):
            series = []
            for year in years:
                record = NDJFMonth(int(year), int(year), month)
                saved = find_existing_checkpoint(record, legacy_monthly_dir=root / 'ndjf_monthly',
                                                  legacy_december_dir=root / 'december_yearly')
                valid, _ = validate_d12_checkpoint(saved, record) if saved else (False, '')
                if not valid:
                    raise FileNotFoundError(
                        f'Missing complete baseline checkpoint {year}-{month:02d}. Restore '
                        'ndjf_monthly_thresholds.csv from the catalog run; do not fit thresholds to this winter.')
                with xr.open_dataarray(saved) as array:
                    series.append(array.load())
            mean, std, value = threshold_from_std(xr.concat(series, dim='time'))
            rows.append({'month': month, 'D_mean_s-1': mean, 'D_std_s-1': std, 'threshold_s-1': value})
        frame, source = pd.DataFrame(rows), 'reconstructed from complete established-catalog years: ' + str(years)
    if frame.month.duplicated().any() or set(frame.month) != {11, 12, 1, 2}:
        raise ValueError('Need exactly one threshold for each NDJF calendar month.')
    if not np.isfinite(frame['threshold_s-1']).all():
        raise ValueError('Thresholds must be finite.')
    raw = read_events(root / RAW_NAME)
    if 'detection_threshold_s-1' in raw:
        recorded = pd.to_numeric(raw['detection_threshold_s-1'], errors='raise')
        expected = raw.event_start.dt.month.map(frame.set_index('month')['threshold_s-1'])
        if not np.isfinite(recorded).all() or not np.isclose(recorded, expected, rtol=1e-10, atol=1e-14).all():
            raise ValueError('Threshold file disagrees with thresholds recorded in the established raw catalog. '
                             'Restore the matching Notebook 04 threshold table before extending.')
    return frame, source


def initialize_update(root, update_start=None):
    """Create an immutable baseline backup once; subsequent runs use that same baseline."""
    root = Path(root)
    folder = root / 'seasonal_updates'
    manifest_path = folder / 'baseline.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if update_start and pd.Timestamp(update_start) != pd.Timestamp(manifest['update_start']):
            raise ValueError('Keep the saved update_start when extending this same rolling catalog.')
        # Detect accidental edits to the frozen threshold record.
        payload = (folder / 'frozen_thresholds.csv').read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest['threshold_sha256']:
            raise ValueError('Frozen thresholds have changed; restore the original baseline copy.')
        if hashlib.sha256((folder / ('baseline_' + RAW_NAME)).read_bytes()).hexdigest() != manifest['baseline_raw_sha256']:
            raise ValueError('Frozen baseline raw catalog has changed; restore its original copy.')
        return manifest
    audit = inspect_catalog(root)
    update_start = update_start or audit['suggested_update_start']
    if not update_start:
        raise ValueError('Cannot infer catalog coverage from the last event alone. Enter UPDATE_START_UTC '
                         'after verifying the end of your completed Notebook 04 run.')
    start = pd.Timestamp(update_start)
    raw = read_events(root / RAW_NAME)
    if start != start.to_period('M').start_time or start <= raw.event_end.max():
        raise ValueError('Start must be a calendar-month boundary after all established raw events.')
    thresholds, source = frozen_thresholds(root)
    folder.mkdir(parents=True, exist_ok=True)
    for name in (RAW_NAME, MERGED_NAME):
        backup = folder / ('baseline_' + name)
        if not backup.exists():
            shutil.copy2(root / name, backup)
    write_dataframe_atomic(thresholds, folder / 'frozen_thresholds.csv')
    manifest = {'detector': DETECTOR, 'update_start': start.isoformat(), 'threshold_source': source,
                'threshold_sha256': hashlib.sha256((folder / 'frozen_thresholds.csv').read_bytes()).hexdigest(),
                'baseline_raw_sha256': hashlib.sha256((folder / ('baseline_' + RAW_NAME)).read_bytes()).hexdigest(),
                'baseline_merged_events': audit['merged_events'], 'era5_store': ARCO_ERA5_ZARR_STORE}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def availability(ds, allow_era5t=False):
    """ARCO coordinates may extend past populated data; trust published inclusive stops."""
    stable = ds.attrs.get('valid_time_stop')
    preliminary = ds.attrs.get('valid_time_stop_era5t') if allow_era5t else None
    stops = [pd.to_datetime(s, utc=True).tz_localize(None) for s in (stable, preliminary) if s]
    if not stops:
        raise ValueError('Archive has no valid_time_stop metadata; refusing to read unpopulated times.')
    return max(stops).floor('h')


def update_catalog(root, ds, *, update_start=None, requested_end=None, allow_era5t=True, progress=print):
    """Scan only the extension, checkpoint monthly scalars, then refresh the shared CSVs.

    requested_end is inclusive. A pending tail is reported, not filled or counted as no events.
    Preliminary months are re-read on each run and once more when finalized.
    """
    root = Path(root)
    manifest = initialize_update(root, update_start)
    folder = root / 'seasonal_updates'
    start = pd.Timestamp(manifest['update_start'])
    available = availability(ds, allow_era5t)
    requested = pd.to_datetime(requested_end or pd.Timestamp.now(tz='UTC').floor('h'), utc=True).tz_localize(None)
    stop = min(requested, available)
    stable_stop = pd.to_datetime(ds.attrs['valid_time_stop'], utc=True).tz_localize(None)
    thresholds = pd.read_csv(folder / 'frozen_thresholds.csv').set_index('month')['threshold_s-1']
    baseline = read_events(folder / ('baseline_' + RAW_NAME))
    # Do not erase later extensions if somebody requests an earlier end date on a rerun.
    previous = folder / 'scan_manifest.json'
    if previous.exists() and stop < pd.Timestamp(json.loads(previous.read_text())['scanned_through']):
        raise ValueError('Requested/available end precedes an existing scan. Keep or extend the saved end date.')
    if stop < start:
        return {'status': 'pending_era5', 'available_through': str(available), 'requested_through': str(requested),
                'pending_after': str(available), 'merged_events': len(read_events(root / MERGED_NAME))}
    event_frames, inventory = [], []
    for period in pd.period_range(start, stop, freq='M'):
        if period.month not in thresholds.index:
            continue
        begin = period.start_time
        end = min(period.end_time.floor('h'), stop)
        if end < begin + pd.Timedelta(hours=11):
            continue
        product = 'ERA5' if end <= stable_stop else 'ERA5T'
        stamp = str(period)
        path = folder / 'monthly' / f'{stamp}_D12.nc'
        cached = None
        if path.exists():
            with xr.open_dataarray(path) as saved:
                if (saved.attrs.get('source_product') == 'ERA5' and product == 'ERA5'
                        and saved.attrs.get('scanned_through') == end.isoformat()):
                    cached = saved.load()
        if cached is None:
            progress(f'{stamp}: {begin} to {end} ({product})')
            winds = subset_era5_window(ds, str(begin), str(end), variables=('u_component_of_wind', 'v_component_of_wind'))
            expected = pd.date_range(begin, end, freq='h')
            if not pd.DatetimeIndex(winds.time.values).equals(expected):
                raise ValueError(f'{stamp}: missing hourly winds; catalog was not updated.')
            geometry = prepare_detection_geometry(winds.longitude, winds.latitude, JPCZ_POLYGON_VERTICES)
            # Final data resume; preliminary scalar values must be refreshed when the archive is revised.
            partial = folder / 'hourly' / f'{stamp}.nc' if product == 'ERA5' else None
            hourly, d12 = compute_polygon_mean_divergence_series_chunked(
                winds, geometry=geometry, chunk_hours=6, progress_every_hours=24,
                progress_label=stamp, hourly_checkpoint_path=partial)
            if not np.isfinite(hourly.values).all():
                raise ValueError(f'{stamp}: nonfinite winds/divergence; catalog was not updated.')
            cached = d12.dropna('time')
            cached.attrs.update(source_product=product, scanned_through=end.isoformat())
            write_dataarray_atomic(cached, path)
        events = count_events_from_threshold(cached, float(thresholds[period.month]))
        if not events.empty:
            events['year'], events['month'] = period.year, period.month
            # Existing notebook labels winters by their Jan/Feb year.
            events['season_year'] = period.year + (period.month in (11, 12))
            events['detection_threshold_s-1'] = float(thresholds[period.month])
            events['detection_threshold_1e5_s-1'] = float(thresholds[period.month]) * 1e5
            events['reference_product'] = product
            events['catalog_provisional'] = (events.event_end + pd.Timedelta(hours=12) >= stop) | (product == 'ERA5T')
            event_frames.append(events)
        inventory.append({'month': stamp, 'scanned_through': str(end), 'source_product': product,
                          'events': len(events), 'checkpoint': str(path)})
    extension = pd.concat(event_frames, ignore_index=True) if event_frames else baseline.iloc[:0].copy()
    combined = pd.concat([baseline, extension], ignore_index=True).sort_values('event_start').reset_index(drop=True)
    merged = merge_nearby_events(combined, max_gap_hours=12)
    # Keep all existing display numbers on exact identities; assign new numbers only to changed/new events.
    old = read_events(root / MERGED_NAME)
    old_keys = event_keys(old)
    numbers = pd.to_numeric(old.get('event_number', pd.Series(np.arange(1, len(old) + 1))), errors='coerce')
    if numbers.isna().any() or not numbers.is_unique or (numbers <= 0).any():
        numbers = pd.Series(np.arange(1, len(old) + 1))
    lookup = dict(zip(old_keys, numbers.astype(int)))
    next_number = int(numbers.max()) + 1
    allocated = []
    for key in event_keys(merged):
        if key not in lookup:
            lookup[key] = next_number
            next_number += 1
        allocated.append(lookup[key])
    merged['event_number'] = allocated
    merged['review_key'] = event_keys(merged)
    # Propagate a provisional fragment to its entire merged episode.
    extension_provisional = extension.loc[extension.get('catalog_provisional', pd.Series(False, index=extension.index)).eq(True)]
    merged['catalog_provisional'] = [bool(((extension_provisional.event_start <= row.event_end) &
                                          (extension_provisional.event_end >= row.event_start)).any())
                                     for row in merged.itertuples()]
    changes = merged.copy()
    changes['update_status'] = np.where(changes.review_key.isin(old_keys), 'unchanged', 'new_or_revised')
    # Stage both files and a ledger before promotion. The immutable baseline remains available for recovery.
    write_dataframe_atomic(combined, folder / 'candidate_raw.csv')
    write_dataframe_atomic(merged, folder / 'candidate_merged.csv')
    write_dataframe_atomic(changes, folder / 'event_changes.csv')
    write_dataframe_atomic(pd.DataFrame(inventory), folder / 'monthly_scan_status.csv')
    write_dataframe_atomic(combined, root / RAW_NAME)
    write_dataframe_atomic(merged, root / MERGED_NAME)
    result = {'status': 'updated' if requested <= available else 'updated_with_pending_era5_tail',
              'scanned_through': stop.isoformat(), 'available_through': available.isoformat(),
              'requested_through': requested.isoformat(), 'merged_events': len(merged),
              'new_or_revised': int(changes.update_status.ne('unchanged').sum()),
              'provisional_events': int(merged.catalog_provisional.sum()), 'allow_era5t': allow_era5t,
              'detector': manifest['detector']}
    previous.write_text(json.dumps(result, indent=2) + '\n')
    progress(str(result))
    return result
