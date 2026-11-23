"""Drive-only, resumable event reports. No meteorological retrieval or calculation."""
from __future__ import annotations

import copy
import gc
import hashlib
from io import BytesIO
import json
from pathlib import Path

import pandas as pd

REPORT_VERSION = 1
DATES = ('event_start', 'event_end', 'event_peak')
PHASES = (-12, 0, 12)
LEADS = (24, 120, 240)
ASSET_COLUMNS = ['event_key', 'event_number', *DATES, 'source', 'field', 'level_hpa',
                 'domain', 'definition', 'units', 'offset_hours', 'lead_hours', 'member',
                 'valid_time', 'init_time', 'status', 'figure', 'kind', 'metadata_file']
FIELD_NAMES = {'divergence': 'Horizontal divergence', 'ertel_pv': 'Ertel potential vorticity',
               'height': 'Geopotential height', 'geopotential': 'Geopotential',
               'winds': 'Wind speed / vectors', 'u': 'Zonal wind', 'v': 'Meridional wind',
               'onshore': 'Onshore wind component'}


def report_event_key(row):
    """Dates, not mutable catalog row/event numbers, identify an event."""
    values = [pd.to_datetime(row[name], utc=True, errors='raise') for name in DATES]
    if any(pd.isna(value) for value in values):
        raise ValueError('All three event dates are required.')
    if not values[0] <= values[2] <= values[1]:
        raise ValueError('Expected start <= peak <= end.')
    return '|'.join(value.isoformat() for value in values)


def _text(value):
    return '' if value is None or (not isinstance(value, (dict, list, tuple)) and pd.isna(value)) else str(value)


def _number(value, default=None):
    return default if not _text(value) else int(float(value))


def _domain(value):
    if not _text(value):
        return ''
    if isinstance(value, str):
        try:
            value = json.loads(value.replace('(', '[').replace(')', ']'))
        except ValueError:
            try:
                value = [float(item) for item in value.split(',')]
            except ValueError:
                return value
    if isinstance(value, (tuple, list)):
        return ','.join(f'{float(item):g}' for item in value)
    return _text(value)


def _atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(value, path):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def _asset(row, source, field, level, figure, metadata, **extra):
    result = dict.fromkeys(ASSET_COLUMNS, '')
    result.update({name: _text(row.get(name)) for name in DATES})
    result.update(event_key=report_event_key(row), event_number=_number(row.get('event_number')),
                  source=source, field=field, level_hpa=_number(level), figure=str(figure),
                  metadata_file=str(metadata), kind='standalone', status=_text(row.get('status')) or 'saved')
    result.update(extra)
    return result


def _validate_timing(row):
    """Prevent a saved map being assigned an unsupported offset or target label."""
    valid = pd.to_datetime(row.get('valid_time'), utc=True, errors='coerce')
    if pd.isna(valid):
        raise ValueError('A saved figure needs its actual UTC valid_time.')
    if row['source'] == 'ERA5':
        offset = _number(row.get('offset_hours'))
        if offset not in PHASES:
            raise ValueError('ERA5 offset_hours must be -12, 0 or 12.')
        expected = pd.to_datetime(row['event_peak'], utc=True) + pd.Timedelta(hours=offset)
        if valid != expected:
            raise ValueError('ERA5 valid_time does not match its peak-relative offset.')
    else:
        lead = _number(row.get('lead_hours'))
        init = pd.to_datetime(row.get('init_time'), utc=True, errors='coerce')
        if lead not in LEADS or pd.isna(init) or init + pd.Timedelta(hours=lead) != valid:
            raise ValueError('WN2 initialization + lead must equal its actual valid_time (24/120/240-hour leads).')
        if not _text(row.get('member')):
            raise ValueError('WN2 figures require their saved member/ensemble-summary label.')


def _saved_figure(stem):
    for extension in ('.pdf', '.png'):
        path = Path(stem).with_suffix(extension)
        if path.is_file() and path.stat().st_size:
            return path
    return Path(stem).with_suffix('.pdf')  # Keep a missing-file entry for the audit.


def scan_saved_figures(era5_root, wn2_root, manifest_paths=()):
    """Read known export metadata, never guess field/level from image filenames.

    Additional producers can write figure_manifest.csv using ASSET_COLUMNS, with
    figure paths relative to that manifest. Explicit manifests take precedence.
    """
    records, issues = [], []
    era5_root, wn2_root = Path(era5_root), Path(wn2_root)
    legacy = list(era5_root.glob('batch_evolution/*/event_*/*_metadata.csv'))
    legacy += list(era5_root.glob('*_metadata.csv'))
    for path in sorted(legacy):
        if path.name in ('all_three_times_metadata.csv', 'all_cases_metadata.csv'):
            continue
        try:
            table = pd.read_csv(path)
            # Interactive comparisons may contain several cases in one image.
            if len(table) != 1 or not set(DATES).issubset(table.columns):
                issues.append(dict(metadata_file=str(path), reason='Not a one-case phase export; skipped.'))
                continue
            row = table.iloc[0].to_dict()
            figure = _saved_figure(path.with_name(path.name.removesuffix('_metadata.csv')))
            offset = _number(row.get('offset_hours'), 0)
            if offset not in PHASES:
                continue
            hours = _number(row.get('hours'), 1)
            domain = row.get('domain', '')
            state_path = path.with_name(path.name.removesuffix('_metadata.csv') + '.json')
            if not _text(domain) and state_path.is_file():
                state = json.loads(state_path.read_text())
                domain = state.get('request', {}).get('config', {}).get('reader_settings', {}).get('domain', '')
            records.append(_asset(row, 'ERA5', 'divergence', 925, figure, path,
                offset_hours=offset, valid_time=_text(row.get('view_hour_utc')),
                units='10^-5 s^-1', domain=_domain(domain),
                definition='instantaneous' if hours == 1 else f'{hours}-hour mean',
                status=_text(row.get('era5_map_status')) or _text(row.get('batch_status')) or 'saved',
                kind='notebook35_six_panel', priority=0))
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            issues.append(dict(metadata_file=str(path), reason=str(error)))
    for path in sorted(wn2_root.glob('figures/event_*/forecast_metadata.csv')):
        try:
            for row in pd.read_csv(path).to_dict('records'):
                stem = path.parent / f"row{_number(row['row_index']) + 1}_{row['field']}_lead{_number(row['lead_hours']):03d}h"
                domain = _domain(row.get('domain', ''))
                records.append(_asset(row, 'WN2', row['field'], row.get('level_hpa'),
                    _saved_figure(stem), path, lead_hours=_number(row['lead_hours']),
                    member=_text(row.get('member')), valid_time=_text(row.get('valid_time')),
                    init_time=_text(row.get('init_time')), units=_text(row.get('units')),
                    domain=_text(domain), definition='instantaneous', priority=0))
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            issues.append(dict(metadata_file=str(path), reason=str(error)))
    manifests = {Path(p) for p in manifest_paths}
    for root in (era5_root, wn2_root):
        manifests.update(root.rglob('figure_manifest.csv'))
    for path in sorted(manifests):
        try:
            for row in pd.read_csv(path).to_dict('records'):
                source = _text(row['source']).upper()
                if source not in ('ERA5', 'WN2'):
                    raise ValueError('source must be ERA5 or WN2')
                figure = Path(str(row['figure']))
                if not figure.is_absolute():
                    figure = path.parent / figure
                asset = _asset(row, source, str(row['field']), row.get('level_hpa'), figure, path)
                for key in ASSET_COLUMNS:
                    if key in row and key not in ('event_key', 'figure', 'metadata_file') and _text(row[key]):
                        asset[key] = row[key]
                asset.update(source=source, priority=1)
                asset['offset_hours'] = _number(asset.get('offset_hours'))
                asset['lead_hours'] = _number(asset.get('lead_hours'))
                asset['level_hpa'] = _number(asset.get('level_hpa'))
                records.append(asset)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            issues.append(dict(metadata_file=str(path), reason=str(error)))
    frame = pd.DataFrame(records, columns=[*ASSET_COLUMNS, 'priority'])
    if len(frame):
        frame['file_exists'] = frame.figure.map(lambda p: Path(p).is_file())
        frame['mtime_ns'] = frame.figure.map(lambda p: Path(p).stat().st_mtime_ns if Path(p).is_file() else 0)
    else:
        frame['file_exists'] = pd.Series(dtype=bool)
        frame['mtime_ns'] = pd.Series(dtype='int64')
    return frame, pd.DataFrame(issues, columns=['metadata_file', 'reason'])


def prepare_report_inventory(catalog, assets, *, start=None, end=None, fields=None,
                             event_numbers=None, members=None, era5_hours=1):
    """Select events from the catalog, including those without any saved figures."""
    events = catalog.copy()
    events['event_key'] = events.apply(report_event_key, axis=1)
    if events.event_key.duplicated().any() or events.event_number.duplicated().any():
        raise ValueError('Catalog contains duplicate event identities or numbers.')
    peaks = pd.to_datetime(events.event_peak, utc=True)
    if start:
        events = events.loc[peaks >= pd.to_datetime(start, utc=True)]
    if end:
        events = events.loc[peaks < pd.to_datetime(end, utc=True)]
    if event_numbers is not None:
        events = events.loc[events.event_number.isin(event_numbers)]
    events = events.assign(_peak=pd.to_datetime(events.event_peak, utc=True)).sort_values('_peak', kind='stable').drop(columns='_peak').reset_index(drop=True)
    selected = assets.loc[assets.event_key.isin(events.event_key)].copy()
    if members is not None:
        selected = selected.loc[selected.source.eq('ERA5') | selected.member.astype(str).isin(map(str, members))]
    # Keep forecast targets and time averages distinct. Legacy 35 selection is explicit.
    if era5_hours is not None:
        definition = 'instantaneous' if era5_hours == 1 else f'{era5_hours}-hour mean'
        selected = selected.loc[~selected.kind.eq('notebook35_six_panel') | selected.definition.eq(definition)]
    if fields is not None:
        allowed = {(str(field), _number(level)) for field, level in fields}
        selected = selected.loc[[ (str(r.field), _number(r.level_hpa)) in allowed for r in selected.itertuples() ]]
    selected['catalog_event_number'] = selected.event_key.map(events.set_index('event_key').event_number)
    for column in ('domain', 'definition', 'member'):
        selected[column] = selected[column].map(_domain if column == 'domain' else _text)
    for column in ('valid_time', 'init_time'):
        selected[column] = selected[column].map(lambda value: pd.to_datetime(value, utc=True).isoformat() if _text(value) else '')
    # Latest valid file wins only within the same physical/timing/member selection.
    keys = ['event_key', 'source', 'field', 'level_hpa', 'domain', 'definition',
            'offset_hours', 'lead_hours', 'member', 'valid_time', 'init_time']
    if len(selected):
        for row in selected.to_dict('records'):
            _validate_timing(row)
        selected = selected.sort_values(['file_exists', 'priority', 'mtime_ns', 'figure'], kind='stable')
        selected = selected.drop_duplicates(keys, keep='last')
    return events, selected.reset_index(drop=True)


def _fingerprint(path):
    path = Path(path)
    if not path.is_file():
        return [str(path), 'missing']
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


def notebook35_divergence_crop(path):
    """Validate the known six-panel template and retain its divergence/colorbar.

    Returns PDF coordinates. Unrecognized layouts are shown intact, never given
    a guessed scientific crop. Text anchors establish the panel's lower edge.
    """
    import pdfplumber
    with pdfplumber.open(path) as document:
        if len(document.pages) != 1:
            return None
        page = document.pages[0]
        text = page.extract_text() or ''
        if not all(term in text for term in ('925-hPa divergence', 'IMERG mean rate', 'MODIS')):
            return None
        width, height = page.width, page.height
        if not 1.4 < width / height < 2.1:
            return None
        anchors = [w for w in page.extract_words() if w['text'].startswith('Negative/blue:')
                   and width / 3 < (w['x0'] + w['x1']) / 2 < 2 * width / 3
                   and .3 * height < w['top'] < .6 * height]
        if len(anchors) != 1:
            return None
        bottom = height - anchors[0]['bottom'] - 6
        return (width / 3, bottom, width * 2 / 3, height)


def _safe(value):
    # Headers use built-in PDF fonts; scientific source glyphs remain untouched.
    return _text(value).replace('\u2212', '-').encode('latin-1', 'replace').decode('latin-1')


def _wrapped(canvas, text, x, y, width, size=10, leading=14, max_lines=4):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    canvas.setFont('Helvetica', size)
    lines, current = [], ''
    for word in _safe(text).split():
        candidate = f'{current} {word}'.strip()
        if stringWidth(candidate, 'Helvetica', size) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    for line in lines[:max_lines]:
        canvas.drawString(x, y, line)
        y -= leading
    return y


def field_title(field, level):
    return (f'{int(level)}-hPa ' if level is not None and not pd.isna(level) else '') + FIELD_NAMES.get(field, str(field))


def _source_page(canvas, event, title, source, panels, page_number, page_size):
    """Draw only furniture; source figures are overlaid losslessly as PDFs below."""
    width, height = page_size
    canvas.setFillColorRGB(.08, .17, .23)
    canvas.rect(0, height - 116, width, 116, fill=1, stroke=0)
    canvas.setFillColorRGB(1, 1, 1)
    canvas.setFont('Helvetica-Bold', 22)
    canvas.drawString(35, height - 40, _safe(f"Event {int(event['event_number'])} | {title}"))
    canvas.setFont('Helvetica', 12)
    canvas.drawString(35, height - 65, _safe(f"Peak {event['event_peak']} UTC | {source}"))
    _wrapped(canvas, f"Start {event['event_start']} UTC | End {event['event_end']} UTC", 35, height - 88, width - 70, 11)
    canvas.setFillColorRGB(.10, .23, .28)
    overlays = []
    gutter, margin = 20, 35
    slot_width = (width - 2 * margin - 2 * gutter) / 3
    for i, (label, row) in enumerate(panels):
        x = margin + i * (slot_width + gutter)
        canvas.setFont('Helvetica-Bold', 15)
        canvas.drawString(x, height - 149, _safe(label))
        if row is None:
            canvas.setFillColorRGB(.95, .96, .97)
            canvas.rect(x, 210, slot_width, height - 395, fill=1, stroke=0)
            canvas.setFillColorRGB(.35, .4, .44)
            _wrapped(canvas, 'No saved figure for this selection. Run the source exporter when ready, then rerun Notebook 39.',
                     x + 18, height / 2 + 30, slot_width - 36, 13, 20)
            continue
        reason = ''
        figure = Path(row['figure'])
        crop = None
        if not figure.is_file():
            reason = 'Saved metadata exists, but the figure file is missing.'
        elif row['kind'] == 'notebook35_six_panel':
            crop = notebook35_divergence_crop(figure) if figure.suffix.lower() == '.pdf' else None
            if crop is None:
                reason = 'Source overview shown intact: layout could not be safely isolated.'
        # Reserve the bottom third for provenance, keeping labels out of maps.
        rect = (x, 210, slot_width, height - 395)
        if figure.is_file():
            overlays.append((figure, crop, rect))
        detail = f"Valid {row.get('valid_time') or 'not recorded'} UTC"
        detail += f" | Status: {row.get('status') or 'not recorded'}"
        if source.startswith('WN2'):
            detail += f" | Init {row.get('init_time') or 'not recorded'} UTC"
        detail += f" | Units: {row.get('units') or 'see source figure'}"
        detail += f" | Domain: {row.get('domain') or 'see source figure'}"
        detail += f" | {row.get('definition') or 'see source metadata'}"
        _wrapped(canvas, detail, x, 179, slot_width, 10, 14, 6)
        if reason:
            _wrapped(canvas, reason, x, 83, slot_width, 9, 12, 3)
    canvas.setFillColorRGB(.35, .4, .44)
    _wrapped(canvas, 'Saved figures only. ERA5 offsets are relative to the catalog peak. WN2 leads refer to its recorded forecast target; these times may differ.',
             35, 40, width - 110, 10, 13, 2)
    canvas.drawRightString(width - 35, 23, str(page_number))
    return overlays


def _overlay(page, figure, crop, rectangle):
    from pypdf import PdfReader, Transformation
    from pypdf.generic import RectangleObject
    from reportlab.pdfgen.canvas import Canvas
    if figure.suffix.lower() == '.pdf':
        reader = PdfReader(figure)
        if len(reader.pages) != 1:
            raise ValueError(f'Expected a standalone one-page figure: {figure}')
        source = copy.copy(reader.pages[0])
        if source.rotation:
            source.transfer_rotation_to_content()
        if crop:
            source.cropbox = RectangleObject(crop)
        box = source.cropbox
        left, bottom, width, height = float(box.left), float(box.bottom), float(box.width), float(box.height)
    else:
        from PIL import Image
        with Image.open(figure) as image:
            width, height = image.size
        stream = BytesIO()
        canvas = Canvas(stream, pagesize=(width, height))
        canvas.drawImage(str(figure), 0, 0, width, height, preserveAspectRatio=True, mask='auto')
        canvas.save()
        source = PdfReader(stream).pages[0]
        left, bottom = 0, 0
    x, y, slot_width, slot_height = rectangle
    scale = min(slot_width / width, slot_height / height)
    transform = Transformation().translate(-left, -bottom).scale(scale).translate(
        x + (slot_width - width * scale) / 2, y + (slot_height - height * scale) / 2)
    page.merge_transformed_page(source, transform, over=True)


def _groups(frame):
    groups = {(r.field, _number(r.level_hpa)) for r in frame.itertuples()}
    order = {'divergence': 0, 'ertel_pv': 1, 'height': 2, 'geopotential': 3, 'winds': 4, 'u': 5, 'v': 6, 'onshore': 7}
    return sorted(groups, key=lambda item: (order.get(item[0], 99), item[0], -(item[1] or 0)))


def build_event_report(event, frame, destination, *, required_fields=None):
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.pagesizes import A3, landscape
    from reportlab.pdfgen.canvas import Canvas
    groups = _groups(frame)
    if required_fields is not None:
        groups = _groups(pd.DataFrame([dict(field=f, level_hpa=l) for f, l in required_fields]))
    if not groups:
        groups = [('divergence', 925)]
    writer, audit, bookmarks = PdfWriter(), [], []
    for field, level in groups:
        same_level = frame.level_hpa.fillna(-1).astype(float).eq(level if level is not None else -1)
        group = frame.loc[frame.field.eq(field) & same_level]
        title = field_title(field, level)
        bookmarks.append((title, len(writer.pages)))
        # Different domains/averaging definitions are separate labelled pages.
        for source in ('ERA5', 'WN2'):
            subset = group.loc[group.source.eq(source)]
            variants = list(subset.groupby(['domain', 'definition', 'member'], dropna=False, sort=True)) if len(subset) else [(('', '', ''), subset)]
            for (domain, definition, member), variant in variants:
                source_title = source + (f' | member {member}' if source == 'WN2' and _text(member) else '')
                slots = PHASES if source == 'ERA5' else LEADS
                panels = []
                for slot in slots:
                    column = 'offset_hours' if source == 'ERA5' else 'lead_hours'
                    matches = variant.loc[pd.to_numeric(variant[column], errors='coerce').eq(slot)]
                    # Preserve every different target/initialization instead of silently choosing one.
                    if len(matches) > 1:
                        raise ValueError(f"Ambiguous {source} figures for {event['event_number']}, {title}, {slot}. Narrow the manifest/roots to one target.")
                    row = matches.iloc[0].to_dict() if len(matches) else None
                    label = ('Peak' if slot == 0 else f'{slot:+d} h from peak') if source == 'ERA5' else f'{slot // 24}-day forecast lead'
                    panels.append((label, row))
                    audit.append(dict(event_number=event['event_number'], event_key=event['event_key'], field=field,
                        level_hpa=level, source=source, selection=label, domain=domain, definition=definition,
                        member=member, figure=row['figure'] if row else '',
                        availability='saved' if row and Path(row['figure']).is_file() else 'missing',
                        source_status=row.get('status', '') if row else '',
                        report_page=len(writer.pages) + 1))
                stream = BytesIO()
                canvas = Canvas(stream, pagesize=landscape(A3))
                overlays = _source_page(canvas, event, title, source_title, panels, len(writer.pages) + 1, landscape(A3))
                canvas.save()
                page = writer.add_page(PdfReader(stream).pages[0])
                for figure, crop, rectangle in overlays:
                    _overlay(page, figure, crop, rectangle)
                page.compress_content_streams()
    for title, page in bookmarks:
        writer.add_outline_item(title, page)
    writer.add_metadata({'/Title': f"JPCZ event {event['event_number']} - saved ERA5 and WN2 maps",
                         '/Subject': 'Saved-figure comparison; no meteorological recalculation'})
    temporary = Path(destination).with_suffix('.tmp.pdf')
    with temporary.open('wb') as stream:
        writer.write(stream)
    temporary.replace(destination)
    writer.close()
    return pd.DataFrame(audit)


def _merge_reports(rows, destination):
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    for row in rows:
        writer.append(str(row['pdf']), outline_item=f"Event {row['event_number']} | {row['event_peak']}")
    temporary = destination.with_suffix('.tmp.pdf')
    with temporary.open('wb') as stream:
        writer.write(stream)
    writer.close()
    # Verify the finalized file is readable before replacing the prior report.
    expected = sum(int(row['pages']) for row in rows)
    if len(PdfReader(temporary).pages) != expected:
        raise ValueError('Combined report page count differs from its source PDFs.')
    temporary.replace(destination)


def build_figure_report(events, assets, output_dir, *, required_fields=None,
                        events_per_volume=10, combine=True, combined_limit_mb=350, progress=print):
    """Checkpoint each event; create bounded volumes and an optional full PDF.

    Original source files are read-only. All output replacement is atomic. An
    interrupted rerun reuses event PDFs only when source stats AND metadata match.
    """
    from pypdf import PdfReader
    if events.empty:
        raise ValueError('No catalog events selected. Check the dates and event filter.')
    if assets.empty or not assets.figure.map(lambda p: Path(p).is_file()).any():
        raise ValueError('No saved figure files match this selection. Check Drive roots, metadata issues and field/date filters before building.')
    if events_per_volume < 1:
        raise ValueError('events_per_volume must be positive.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_root = output_dir / 'events'
    event_root.mkdir(exist_ok=True)
    report_fields = required_fields if required_fields is not None else (_groups(assets) or [('divergence', 925)])
    index, audits = [], []
    for event in events.to_dict('records'):
        frame = assets.loc[assets.event_key.eq(event['event_key'])].copy()
        token = _digest(event['event_key'])[:10]
        stem = event_root / f"event_{int(event['event_number']):04d}_{token}"
        pdf, state, audit_path = stem.with_suffix('.pdf'), stem.with_suffix('.json'), stem.with_suffix('.csv')
        signature = _digest(dict(version=REPORT_VERSION, event=event, fields=report_fields,
            assets=frame.fillna('').to_dict('records'), files=[_fingerprint(p) for p in frame.figure]))
        reuse = False
        if pdf.is_file() and state.is_file() and audit_path.is_file():
            try:
                saved = json.loads(state.read_text())
                reuse = saved['signature'] == signature and len(PdfReader(pdf).pages) == saved['pages']
            except (OSError, ValueError, KeyError):
                pass
        if reuse:
            audit = pd.read_csv(audit_path)
        else:
            audit = build_event_report(event, frame, pdf, required_fields=report_fields)
            _atomic_csv(audit, audit_path)
            _atomic_json(dict(signature=signature, pages=len(PdfReader(pdf).pages)), state)
        pages = len(PdfReader(pdf).pages)
        index.append(dict(event_number=int(event['event_number']), event_peak=event['event_peak'],
            event_key=event['event_key'], pdf=str(pdf), pages=pages, status='reused' if reuse else 'built',
            missing_slots=int(audit.availability.eq('missing').sum())))
        audits.append(audit)
        _atomic_csv(pd.DataFrame(index), output_dir / 'event_report_index.csv')
        progress(f"Event {event['event_number']}: {'reused' if reuse else 'built'} {pages} pages; {index[-1]['missing_slots']} missing figure slots")
        gc.collect()
    report_audit = pd.concat(audits, ignore_index=True)
    _atomic_csv(report_audit, output_dir / 'figure_coverage.csv')
    _atomic_csv(report_audit.loc[report_audit.availability.eq('missing')], output_dir / 'missing_figures.csv')
    _atomic_csv(assets, output_dir / 'selected_figure_inventory.csv')
    volumes = []
    for start in range(0, len(index), events_per_volume):
        rows = index[start:start + events_per_volume]
        path = output_dir / f"volume_{start // events_per_volume + 1:03d}.pdf"
        signature = _digest([_fingerprint(row['pdf']) for row in rows])
        marker = path.with_suffix('.json')
        if not path.is_file() or not marker.is_file() or json.loads(marker.read_text()).get('signature') != signature:
            _merge_reports(rows, path)
            _atomic_json(dict(signature=signature), marker)
        volumes.append(str(path))
        gc.collect()
    full = output_dir / 'all_events.pdf'
    full_created = False
    total_mb = sum(Path(row['pdf']).stat().st_size for row in index) / 1024**2
    if combine and total_mb <= combined_limit_mb:
        signature = _digest([_fingerprint(row['pdf']) for row in index])
        marker = full.with_suffix('.json')
        if not full.is_file() or not marker.is_file() or json.loads(marker.read_text()).get('signature') != signature:
            _merge_reports(index, full)
            _atomic_json(dict(signature=signature), marker)
        full_created = True
    summary = dict(events=len(index), pages=sum(row['pages'] for row in index),
        missing_slots=int(report_audit.availability.eq('missing').sum()), volumes=volumes,
        combined_pdf=str(full) if full_created else None, source_pdf_mb=round(total_mb, 1),
        combined_note='Created' if full_created else 'Use volumes: combined export disabled or exceeds size limit.',
        output_dir=str(output_dir))
    _atomic_json(summary, output_dir / 'report_summary.json')
    progress(json.dumps(summary, indent=2))
    return pd.DataFrame(index), report_audit, summary
