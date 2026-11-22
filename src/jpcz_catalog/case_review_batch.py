"""Resumable three-time Notebook 35 exports using its existing case reader and plots."""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BATCH_EXPORT_VERSION = 1
BATCH_DATE_COLUMNS = ('event_start', 'event_end', 'event_peak')


class EarthdataLoginRequired(RuntimeError):
    """A session problem must stop the batch, rather than produce missing-data pages."""
    requires_earthdata_login = True


def require_earthdata_auth(auth):
    """Check the Auth returned by the notebook's explicit, non-persistent login cell."""
    if not getattr(auth, 'authenticated', False):
        raise EarthdataLoginRequired(
            'NASA Earthdata login is missing in this runtime. Run the cell under '
            '"NASA authentication for new IMERG map requests" and wait for '
            '"NASA Earthdata authenticated", then rerun the batch with the same settings. '
            'Saved exports and cached data will be reused; unfinished phases will resume. '
            'Use BATCH_RETRY_INCOMPLETE = True only to retry saved data gaps.')


def batch_json_value(value):
    if isinstance(value, dict):
        return {str(k): batch_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [batch_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return batch_json_value(value.item())
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if value is None or value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def batch_atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(batch_json_value(value), indent=2, allow_nan=False))
    temporary.replace(path)


def batch_atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def batch_digest(value):
    encoded = json.dumps(batch_json_value(value), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def batch_screen_signature(row, settings):
    """Reloaded catalog rows may omit a signature already fixed by reader settings."""
    value = row.get('vortex_screen_signature', '')
    if isinstance(value, str) and value:
        return value
    value = (settings or {}).get('vortex_signature', '')
    return value if isinstance(value, str) else ''


def batch_request_matches(saved, current):
    # Older rows sometimes omitted this duplicate of config.reader_settings.vortex_signature.
    # Preserve every other comparison, including the complete reader/settings identity.
    def canonical(request):
        result = dict(request)
        settings = result.get('config', {}).get('reader_settings', {})
        result['vortex_screen_signature'] = batch_screen_signature(result, settings)
        return result
    return canonical(saved) == canonical(current)


def batch_data_status(metadata):
    """Unavailable satellite archives are explicit, permanent gaps; transient gaps retry."""
    rain = metadata.get('imerg_status') == 'complete'
    dynamics = metadata.get('era5_map_status', 'complete') == 'complete'
    total = metadata.get('accumulation_status') in ('complete', 'not_started')
    satellite = (metadata.get('satellite_status') == 'available_daily_mosaic'
                 and metadata.get('cloud_rgb_satellite_status') == 'available_daily_mosaic')
    archive_gap = all('predates archive availability' in str(metadata.get(name, '')).lower()
                      for name in ('satellite_error', 'cloud_rgb_satellite_error'))
    if dynamics and rain and total and (satellite or archive_gap):
        return 'complete_archive_gap' if archive_gap else 'complete'
    return 'partial_data'


def batch_combine_event_pdf(paths, destination):
    """Merge only one event's pages at a time, keeping memory bounded."""
    from pypdf import PdfWriter
    temporary = destination.with_name('.' + destination.stem + '.tmp.pdf')
    writer = PdfWriter()
    try:
        for path in paths:
            writer.append(str(path))
        with temporary.open('wb') as stream:
            writer.write(stream)
        temporary.replace(destination)
    finally:
        writer.close()
        temporary.unlink(missing_ok=True)


def batch_export_phase(row, files, *, metadata, load_case, render_cases, hours, offset,
                       satellite_platform, satellite_day_offset, accumulation_mode, dpi, progress=print):
    """Use the same reader/renderer for an in-process or isolated phase export."""
    import matplotlib.pyplot as plt
    figure = None
    figures_before = set(plt.get_fignums())
    try:
        try:
            case = load_case(row, hours=hours, offset=offset,
                satellite_platform=satellite_platform, satellite_day_offset=satellite_day_offset,
                accumulation_mode=accumulation_mode)
            metadata.update(case['metadata'])
            dynamics = case.get('dynamics')
            if dynamics is not None:
                names = ('divergence', 'vorticity', 'u', 'v', 'mslp')
                missing = [name for name in names if name not in dynamics
                           or not np.isfinite(dynamics[name].values).any()]
                if missing:
                    raise ValueError('ERA5 maps unavailable for: ' + ', '.join(missing))
                metadata['era5_map_status'] = ('complete' if all(
                    np.isfinite(dynamics[name].values).all() for name in names) else 'partial')
            figure = render_cases([case])
            status = batch_data_status(metadata)
        except Exception as error:
            if isinstance(error, MemoryError) or getattr(error, 'requires_earthdata_login', False):
                raise
            for figure_id in set(plt.get_fignums()) - figures_before:
                plt.close(figure_id)
            status = 'failed'
            metadata['error'] = f'{type(error).__name__}: {error}'
            progress('  ' + metadata['error'])
            figure = plt.figure(figsize=(18, 10.5), layout='constrained')
            axis = figure.add_subplot()
            axis.axis('off')
            axis.text(.5, .65, f'Event {int(row.event_number)} · {offset:+d} h from peak\n'
                      f"{metadata['view_hour_utc']} UTC", ha='center', va='center', fontsize=18,
                      transform=axis.transAxes)
            axis.text(.5, .4, 'Case data unavailable; this phase will be retried.\n' + metadata['error'],
                      ha='center', va='center', wrap=True, fontsize=12, transform=axis.transAxes)
        for extension in ('png', 'pdf'):
            destination = Path(files[extension])
            temporary = destination.with_name('.' + destination.stem + '.tmp.' + extension)
            figure.savefig(temporary, format=extension, dpi=dpi, bbox_inches='tight')
            temporary.replace(destination)
        return metadata, status
    finally:
        for figure_id in set(plt.get_fignums()) - figures_before:
            plt.close(figure_id)


def run_case_review_batch(frame, output_dir, *, load_case, render_cases, label_for,
                          refresh_event=None, release_memory=None, check_auth=None, export_phase=None,
                          hours=1, limit=None, offsets=(-12, 0, 12),
                          satellite_platform='Auto', satellite_day_offset=0,
                          accumulation_mode='event_start', settings=None, dpi=180,
                          retry_incomplete=False, progress=print):
    """Resume missing/changed exports; retry saved partial products only when requested.

    Per event: three PNG/PDF/metadata sets plus a three-page PDF and combined CSV.
    The root manifest/CSV always describes the supplied current inventory, not old folders.
    Source data caches and the event catalog are owned by the existing notebook readers.
    """
    import matplotlib.pyplot as plt
    try:
        import pypdf
    except ModuleNotFoundError as error:
        raise RuntimeError('Run Notebook 35 installation cell first to install pypdf, then retry batch.') from error
    if hours not in (1, 12) or tuple(offsets) != (-12, 0, 12):
        raise ValueError('Choose 1 or 12 hours; the batch phases are always -12, 0, +12 hours.')
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError('Use EVENTS_PER_RUN = None for all remaining events, or a positive integer.')
    if satellite_platform not in ('Auto', 'Terra', 'Aqua') or satellite_day_offset not in (-1, 0, 1):
        raise ValueError('Choose Auto/Terra/Aqua and a satellite day offset of -1, 0, or +1.')
    if accumulation_mode not in ('event_start', 'window'):
        raise ValueError('Choose event_start or window accumulation.')
    if not set((*BATCH_DATE_COLUMNS, 'review_key', 'event_number')).issubset(frame):
        raise ValueError('Load the current Notebook 35 catalog before running batch.')
    if frame.review_key.duplicated().any():
        raise ValueError('Duplicate event identities in batch inventory.')
    # Before source reads or checkpoint changes, fail clearly if the login cell was skipped.
    if check_auth:
        check_auth()
    config = dict(export_version=BATCH_EXPORT_VERSION, hours=hours, offsets=list(offsets),
                  satellite_platform=satellite_platform, satellite_day_offset=satellite_day_offset,
                  accumulation_mode=accumulation_mode, dpi=dpi, reader_settings=settings or {})
    root = Path(output_dir) / 'batch_evolution' / f'{hours}h_{satellite_platform}_{batch_digest(config)}'
    root.mkdir(parents=True, exist_ok=True)
    batch_atomic_json(config, root / 'batch_settings.json')
    batch_atomic_json(dict(run_status='running', selected_events=len(frame),
                           requested_phases=len(frame)*3), root / 'batch_summary.json')
    progress(f'Batch folder: {root}')
    progress(f'{len(frame)} events; three times per event; satellite mode {satellite_platform}.')
    jobs, attempted_events, reused_phases = [], 0, 0

    for _, original in frame.iterrows():
        row = refresh_event(original.copy()) if refresh_event else original.copy()
        event_dir = root / f'event_{int(row.event_number):04d}_{batch_digest(str(row.review_key))}'
        event_dir.mkdir(exist_ok=True)
        event_jobs, touched_event = [], False
        for offset in offsets:
            if refresh_event:
                row = refresh_event(row)
            phase = 'peak' if offset == 0 else ('minus12h' if offset < 0 else 'plus12h')
            stem = event_dir / phase
            state_path = stem.with_suffix('.json')
            files = {'png': stem.with_suffix('.png'), 'pdf': stem.with_suffix('.pdf'),
                     'metadata_csv': event_dir / f'{phase}_metadata.csv'}
            def request_identity():
                return batch_json_value(dict(config=config, review_key=str(row.review_key),
                    event_number=int(row.event_number), offset_hours=offset,
                    event_start=row.event_start, event_end=row.event_end, event_peak=row.event_peak,
                    label=label_for(row), vortex_scan_status=row.get('vortex_scan_status', ''),
                    vortex_screen_signature=batch_screen_signature(row, settings)))
            identity = request_identity()
            saved = None
            if state_path.exists() and all(path.is_file() and path.stat().st_size for path in files.values()):
                try:
                    candidate = json.loads(state_path.read_text())
                    if batch_request_matches(candidate['request'], identity):
                        saved = candidate
                except (KeyError, ValueError, OSError):
                    pass
            wrote_phase = False
            if saved and (saved['record']['status'] in ('complete', 'complete_archive_gap')
                          or (not retry_incomplete and saved['record']['status'] in ('partial_data', 'failed'))):
                record = saved['record']
                reused_phases += 1
                if reused_phases % 75 == 0:
                    progress(f'Reused {reused_phases} saved phases; no maps redrawn. '
                             f'Checking event {int(row.event_number)} of {len(frame)}.')
            elif limit is not None and not touched_event and attempted_events >= limit:
                # Inventory still includes saved partial products and work left for later runs.
                record = saved['record'] if saved else dict(review_key=str(row.review_key),
                    event_number=int(row.event_number), offset_hours=offset, status='not_attempted',
                    **{name: str(path) for name, path in files.items()})
            else:
                if not touched_event:
                    touched_event = True
                    attempted_events += 1
                hour = pd.Timestamp(row.event_peak) + pd.Timedelta(hours=offset)
                progress(f'Event {int(row.event_number)} | {offset:+d} h | {hour} UTC | {satellite_platform}')
                metadata = {name: row[name] for name in BATCH_DATE_COLUMNS}
                metadata.update(review_key=str(row.review_key), event_number=int(row.event_number),
                                offset_hours=offset, view_hour_utc=hour,
                                hours=hours, accumulation_mode=accumulation_mode,
                                satellite_day_offset=satellite_day_offset,
                                catalog_provisional=row.get('catalog_provisional', False),
                                satellite_requested_platform=satellite_platform)
                try:
                    if check_auth:
                        check_auth()
                    options = dict(metadata=metadata, hours=hours, offset=offset,
                        satellite_platform=satellite_platform, satellite_day_offset=satellite_day_offset,
                        accumulation_mode=accumulation_mode, dpi=dpi)
                    if export_phase:
                        metadata, status = export_phase(row, files, **options)
                    else:
                        metadata, status = batch_export_phase(row, files, load_case=load_case,
                            render_cases=render_cases, progress=progress, **options)
                except Exception as error:
                    authentication = getattr(error, 'requires_earthdata_login', False)
                    batch_atomic_json(dict(run_status='paused_authentication' if authentication else 'paused_worker',
                        selected_events=len(frame), attempted_events=attempted_events,
                        requested_phases=len(frame)*3, saved_phases=len(jobs),
                        interrupted_event=int(row.event_number), interrupted_offset_hours=offset,
                        error=str(error)), root / 'batch_summary.json')
                    (release_memory or gc.collect)()
                    raise
                # A first case read can finish the event's previously unchecked vortex screen.
                for name in ('vortex_detected', 'vortex_scan_status', 'vortex_screen_signature'):
                    if name in metadata:
                        row[name] = metadata[name]
                identity = request_identity()
                metadata['batch_status'] = status
                batch_atomic_csv(pd.DataFrame([metadata]), files['metadata_csv'])
                record = batch_json_value(dict(**metadata, status=status,
                    **{name: str(path) for name, path in files.items()}))
                batch_atomic_json(dict(request=identity, record=record), state_path)
                wrote_phase = True
                (release_memory or gc.collect)()
            event_jobs.append(record)
            jobs.append(record)
            # Update after each phase, including a partial/failed phase, so a disconnect loses little.
            if wrote_phase:
                batch_atomic_csv(pd.DataFrame(jobs), root / 'batch_manifest.csv')
                batch_atomic_csv(pd.DataFrame(jobs), root / 'all_cases_metadata.csv')
        # Each phase has its own outputs. This PDF groups all three times for one event.
        if all(Path(job.get('pdf', '')).is_file() and job['status'] != 'not_attempted' for job in event_jobs):
            event_pdf = event_dir / 'all_three_times.pdf'
            sources = [Path(job['pdf']) for job in event_jobs]
            if (not event_pdf.exists() or event_pdf.stat().st_mtime_ns < max(path.stat().st_mtime_ns for path in sources)):
                batch_combine_event_pdf(sources, event_pdf)
            for job in event_jobs:
                job['event_pdf'] = str(event_pdf)
                job['event_metadata_csv'] = str(event_dir / 'all_three_times_metadata.csv')
            batch_atomic_csv(pd.DataFrame(event_jobs), event_dir / 'all_three_times_metadata.csv')
            (release_memory or gc.collect)()
    result = pd.DataFrame(jobs)
    batch_atomic_csv(result, root / 'batch_manifest.csv')
    batch_atomic_csv(result, root / 'all_cases_metadata.csv')
    counts = result.status.value_counts().to_dict() if not result.empty else {}
    batch_atomic_json(dict(run_status='finished', selected_events=len(frame), attempted_events=attempted_events,
                           requested_phases=len(frame)*3, reused_phases=reused_phases,
                           retry_incomplete=retry_incomplete, statuses=counts), root / 'batch_summary.json')
    progress(f'Attempted {attempted_events} events this run. Phase status: {counts}.')
    progress(f'All PDF/PNG/CSV exports are together in {root}')
    result.attrs['output_dir'] = str(root)
    return result
