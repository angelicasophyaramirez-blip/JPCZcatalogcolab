"""Frozen, resumable historical ERA5 reference; no WN2 forecasts are requested."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ensemble_verification import POLYGONS, atomic_json_file, checkpoint, checkpoint_identity
from .weathernext_atlas import normalise_catalog, iso_utc


def climatology_schedule(catalog_path, *, kind, start='2000-01-01', end='2022-01-01'):
    """A fixed historical JPCZ-event sample or every NDJF synoptic hour; end is exclusive."""
    start = pd.Timestamp(start, tz='UTC'); end = pd.Timestamp(end, tz='UTC')
    if end <= start: raise ValueError('Reference end must follow its start.')
    if kind == 'Historical JPCZ cases':
        frame = normalise_catalog(pd.read_csv(catalog_path))
        onset = pd.to_datetime(frame.event_start, utc=True)
        frame = frame.loc[onset.ge(start) & onset.lt(end) & onset.dt.month.isin([11, 12, 1, 2])].copy()
        frame['valid_time_utc'] = frame.target_valid_time.map(iso_utc)
        frame['reference_key'] = frame.review_key
        frame = frame[['reference_key', 'event_number', 'valid_time_utc']]
    elif kind == 'All winter conditions':
        times = pd.date_range(start, end, freq='6h', inclusive='left')
        times = times[times.month.isin([11, 12, 1, 2])]
        frame = pd.DataFrame({'reference_key': [f'winter|{t.isoformat()}' for t in times],
                              'event_number': pd.NA, 'valid_time_utc': [iso_utc(t) for t in times]})
    else: raise ValueError('Choose Historical JPCZ cases or All winter conditions.')
    if frame.empty: raise ValueError('No reference targets in the selected period.')
    if frame.reference_key.duplicated().any() or frame.valid_time_utc.duplicated().any():
        raise ValueError('Reference targets must be unique; duplicate targets would overweight a weather state.')
    times = pd.to_datetime(frame.valid_time_utc, utc=True)
    if not (times.ge(start) & times.lt(end)).all(): raise ValueError('Reference target lies outside its frozen period.')
    return frame.sort_values('valid_time_utc').reset_index(drop=True)


def climatology_definition(schedule, calculation, *, kind, start, end):
    """Archive the sampling and spatial/derivative conventions before any skill evaluation."""
    definition = dict(kind=kind, start_utc=iso_utc(start), end_utc_exclusive=iso_utc(end),
        weighting='one target, one equal weight', expected_targets=len(schedule),
        season='NDJF', matching='pooled historical distribution; no test-outcome tuning',
        target_rule='nearest 00/06/12/18 UTC on onset date, ties earlier' if kind=='Historical JPCZ cases' else 'every 6-hourly NDJF target',
        calculation={key: calculation.get(key) for key in ('era5_store', 'domain', 'polygons', 'pressure_hpa', 'diagnostic')},
        field=calculation.get('field', 'divergence'),
        schedule=schedule[['reference_key', 'valid_time_utc']].to_dict('records'))
    definition = json.loads(json.dumps(definition, sort_keys=True))
    signature = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:16]
    return definition, signature


def build_saved_climatology(schedule, destination, definition, spec, *, job_runner=None,
                            download_missing=False, limit=25, shared_reference_cache=None, progress=print):
    """Save each target immediately; decode weather only in the existing bounded worker.

    Preview/reopen mode reads JSON/scalar caches only. A complete frozen schedule is
    required by CRPSS; failures and unattempted dates remain explicit rather than zero.
    """
    if download_missing and job_runner is None: raise ValueError('Missing-reference retrieval requires the bounded ERA5 worker.')
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError('Reference cases per run must be a positive integer or None.')
    if len(schedule) != definition['expected_targets'] or schedule[['reference_key', 'valid_time_utc']].to_dict('records') != definition['schedule']:
        raise ValueError('The reference schedule changed after its definition was frozen.')
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    settings_path = destination/'reference_settings.json'
    if settings_path.is_file():
        old = json.loads(settings_path.read_text())
        if any(old.get(key) != value for key, value in definition.items()):
            raise ValueError('Use a new reference folder for changed sampling or calculation definitions.')
    schedule.to_csv(destination/'reference_schedule.csv', index=False)
    statuses = []; values = []; attempted = 0
    field_identity = dict(field=spec.key, pressure_hpa=spec.level, units=spec.units,
        calculation=definition['calculation'])

    def checked(data, target):
        if data.get('valid_time') != target or data.get('reference_product', 'ERA5') != 'ERA5':
            raise ValueError('Reference needs exact-time final ERA5.')
        if set(data.get('values', {})) != set(POLYGONS) or not np.isfinite(list(data['values'].values())).all():
            raise ValueError('Reference target lacks finite means for both polygons.')
        if data.get('source_store') != definition['calculation']['era5_store']:
            raise ValueError('Reference came from a different ERA5 archive.')
        return data

    def save_progress():
        done = len({row['reference_key'] for row in values})
        pd.DataFrame(values, columns=['reference_key', 'event_number', 'valid_time_utc', 'polygon',
            'field', 'field_title', 'pressure_hpa', 'units', 'value', 'reference_product']).to_csv(
                destination/'.reference_values.tmp.csv', index=False)
        (destination/'.reference_values.tmp.csv').replace(destination/'reference_values.csv')
        pd.DataFrame(statuses).to_csv(destination/'reference_status.csv', index=False)
        atomic_json_file(settings_path, dict(**definition, completed_targets=done,
            complete=done==len(schedule), pending_targets=len(schedule)-done,
            independent_reference='all reference dates precede evaluated forecast initializations; checked again when scoring'))

    # Collect valid cached targets first. A small retrieval limit never hides completed
    # targets later in the schedule after a runtime restart.
    pending = []
    for row in schedule.itertuples():
        target = row.valid_time_utc; stamp = pd.Timestamp(target).strftime('%Y%m%dT%H%MZ')
        path = destination/'targets'/f'era5_{stamp}.json'
        identity = dict(**field_identity, valid_time=target)
        data = None
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved.get('request') == identity:
                try: data = checked(saved['data'], target)
                except (KeyError, ValueError): pass
        if data is None and shared_reference_cache is not None:
            shared = Path(shared_reference_cache)/f'era5_{stamp}.json'
            if shared.is_file():
                saved = json.loads(shared.read_text())
                if saved.get('request') == {'valid_time': target}:
                    try:
                        data = checked(saved['data'], target)
                        atomic_json_file(path, dict(request=identity, data=data))
                    except (KeyError, ValueError): pass
        statuses.append(dict(reference_key=row.reference_key, valid_time_utc=target, status='cached' if data else 'pending', error=''))
        if data is None:
            pending.append((row, path, identity, len(statuses)-1))
        else:
            for region in POLYGONS:
                values.append(dict(reference_key=row.reference_key, event_number=row.event_number,
                    valid_time_utc=target, polygon=region, field=spec.key, field_title=spec.title,
                    pressure_hpa=spec.level, units=spec.units, value=data['values'][region], reference_product='ERA5'))
    save_progress()
    for row, path, identity, status_index in pending:
        if not download_missing or (limit is not None and attempted >= limit): break
        attempted += 1
        progress(f'Climatology ERA5 {attempted}: {row.valid_time_utc}; one target in a bounded worker')
        try:
            data = checked(job_runner('reference', dict(valid_time=row.valid_time_utc, allow_era5t=False, spec=spec.__dict__)), row.valid_time_utc)
            atomic_json_file(path, dict(request=identity, data=data))
            for region in POLYGONS:
                values.append(dict(reference_key=row.reference_key, event_number=row.event_number,
                    valid_time_utc=row.valid_time_utc, polygon=region, field=spec.key, field_title=spec.title,
                    pressure_hpa=spec.level, units=spec.units, value=data['values'][region], reference_product='ERA5'))
            statuses[status_index]['status'] = 'computed'
        except Exception as exc:
            statuses[status_index].update(status='failed', error=f'{type(exc).__name__}: {exc}')
            progress(statuses[status_index]['error'])
        save_progress()
    result = json.loads(settings_path.read_text())
    progress(f"Reference: {result['completed_targets']}/{result['expected_targets']} saved targets; complete={result['complete']}")
    return result
