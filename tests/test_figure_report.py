"""Saved-file integration tests: no cloud login or weather data access."""
import json
from pathlib import Path
import sys

import pandas as pd
from pypdf import PdfReader
import pytest
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jpcz_catalog.figure_report import (
    report_event_key, scan_saved_figures, prepare_report_inventory,
    build_figure_report, notebook35_divergence_crop,
)


def catalog():
    return pd.DataFrame([dict(event_number=i + 276, event_start=f'2026-01-{10+i:02d} 01:00:00',
        event_peak=f'2026-01-{10+i:02d} 04:00:00', event_end=f'2026-01-{10+i:02d} 09:00:00') for i in range(2)])


def pdf(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    c = Canvas(str(path), pagesize=(500, 430))
    c.setFont('Helvetica', 15)
    c.drawString(35, 380, text)
    c.rect(35, 70, 430, 270)
    c.drawString(35, 40, 'Synthetic map fixture; not meteorological data')
    c.save()


def fixtures(tmp_path):
    era, wn = tmp_path / 'era5', tmp_path / 'wn2'
    era.mkdir(); wn.mkdir()
    row = catalog().iloc[0].to_dict()
    manifest = []
    for offset in (-12, 0, 12):
        figure = era / f'div_{offset}.pdf'
        pdf(figure, f'Synthetic ERA5 {offset:+d}')
        manifest.append(dict(**row, source='ERA5', field='divergence', level_hpa=925,
            figure=figure.name, offset_hours=offset, valid_time=str(pd.Timestamp(row['event_peak']) + pd.Timedelta(hours=offset)),
            units='10^-5 s^-1', domain='125,145,33,47', definition='instantaneous', status='complete'))
    pd.DataFrame(manifest).to_csv(era / 'figure_manifest.csv', index=False)
    folder = wn / 'figures' / 'event_0276_member0_fixture'
    folder.mkdir(parents=True)
    metadata = []
    for lead in (24, 120, 240):
        figure = folder / f'row1_divergence_lead{lead:03d}h.pdf'
        pdf(figure, f'Synthetic WN2 member 0 lead {lead}')
        metadata.append(dict(**row, row_index=0, member='0', field='divergence', level_hpa=925,
            lead_hours=lead, valid_time='2026-01-10 00:00:00',
            init_time=str(pd.Timestamp('2026-01-10') - pd.Timedelta(hours=lead)),
            units='10^-5 s^-1', domain='[125.0, 145.0, 33.0, 47.0]', status='complete'))
    pd.DataFrame(metadata).to_csv(folder / 'forecast_metadata.csv', index=False)
    return era, wn


def test_event_identity_uses_dates_and_utc_not_event_number():
    row = catalog().iloc[0].to_dict()
    other = dict(row, event_number=1, event_start='2026-01-10T01:00:00Z')
    assert report_event_key(row) == report_event_key(other)
    other['event_end'] = '2026-01-10T10:00:00Z'
    assert report_event_key(row) != report_event_key(other)
    with pytest.raises(ValueError):
        report_event_key(dict(row, event_peak='2026-01-12'))


def test_native_exports_and_manifest_join_current_catalog(tmp_path):
    era, wn = fixtures(tmp_path)
    assets, issues = scan_saved_figures(era, wn)
    assert len(assets) == 6 and issues.empty
    assert set(assets.domain) == {'125,145,33,47'}
    current = catalog(); current.loc[0, 'event_number'] = 77
    events, chosen = prepare_report_inventory(current, assets, start='2026-01-01T00:00:00Z')
    assert len(events) == 2 and len(chosen) == 6
    assert set(chosen.catalog_event_number) == {77}
    assert len(prepare_report_inventory(current, assets, fields=[('ertel_pv', 500)])[1]) == 0
    assert len(prepare_report_inventory(current, assets, members=['9'])[1]) == 3


def test_report_grouping_missing_cases_resume_and_new_levels(tmp_path):
    era, wn = fixtures(tmp_path)
    assets, _ = scan_saved_figures(era, wn)
    events, chosen = prepare_report_inventory(catalog(), assets)
    output = tmp_path / 'report'
    index, audit, summary = build_figure_report(events, chosen, output, progress=lambda _: None, events_per_volume=1)
    assert summary['events'] == 2 and summary['pages'] == 4 and summary['missing_slots'] == 6
    assert index.status.eq('built').all()
    assert audit.iloc[:3].source.eq('ERA5').all()
    assert audit.iloc[3:6].source.eq('WN2').all()
    assert list(audit.selection[:3]) == ['-12 h from peak', 'Peak', '+12 h from peak']
    document = PdfReader(summary['combined_pdf'])
    assert len(document.pages) == 4 and document.outline
    text = '\n'.join(page.extract_text() for page in document.pages)
    assert 'member 0' in text and 'No saved figure' in text and 'Synthetic WN2' in text
    before = Path(index.iloc[0].pdf).stat().st_mtime_ns
    second, _, _ = build_figure_report(events, chosen, output, progress=lambda _: None, events_per_volume=1)
    assert second.status.eq('reused').all()
    assert Path(index.iloc[0].pdf).stat().st_mtime_ns == before
    # Add a new field at a distinct pressure level; all cases gain its section/gaps.
    manifest = pd.read_csv(era / 'figure_manifest.csv')
    pv = manifest.iloc[0].to_dict(); pv.update(field='ertel_pv', level_hpa=500, units='PVU')
    pd.concat([manifest, pd.DataFrame([pv])]).to_csv(era / 'figure_manifest.csv', index=False)
    assets, _ = scan_saved_figures(era, wn)
    events, chosen = prepare_report_inventory(catalog(), assets)
    third, audit, summary = build_figure_report(events, chosen, output, progress=lambda _: None, events_per_volume=1)
    assert third.status.eq('built').all() and summary['pages'] == 8
    assert set(audit.level_hpa) == {500, 925}
    assert summary['missing_slots'] == 17


def test_overwrite_changed_source_rebuilds_only_affected_event(tmp_path):
    era, wn = fixtures(tmp_path)
    assets, _ = scan_saved_figures(era, wn)
    events, chosen = prepare_report_inventory(catalog(), assets)
    build_figure_report(events, chosen, tmp_path / 'report', progress=lambda _: None)
    pdf(era / 'div_0.pdf', 'Updated scientific source figure fixture')
    # Even a retained inventory detects file stat changes on the next build.
    index, _, _ = build_figure_report(events, chosen, tmp_path / 'report', progress=lambda _: None)
    assert list(index.status) == ['built', 'reused']


def test_safe_crop_and_unknown_layout_fallback(tmp_path):
    known = ROOT / 'outputs/case_review_build/validation_output/synthetic_running_total_validation.pdf'
    if known.exists():
        crop = notebook35_divergence_crop(known)
        assert crop and crop[0] < crop[2] and crop[1] < crop[3]
    unknown = tmp_path / 'unrecognized.pdf'; pdf(unknown, 'Unrecognized layout')
    assert notebook35_divergence_crop(unknown) is None


def test_different_targets_are_rejected_instead_of_silently_relabelled(tmp_path):
    era, wn = fixtures(tmp_path)
    assets, _ = scan_saved_figures(era, wn)
    duplicate = assets.loc[assets.source.eq('WN2')].iloc[[0]].copy()
    duplicate['valid_time'] = '2026-01-10 06:00:00'
    duplicate['init_time'] = '2026-01-09 06:00:00'
    events, chosen = prepare_report_inventory(catalog(), pd.concat([assets, duplicate], ignore_index=True))
    with pytest.raises(ValueError, match='Ambiguous WN2'):
        build_figure_report(events, chosen, tmp_path / 'report', progress=lambda _: None)


def test_combined_size_limit_keeps_volumes_and_reports_no_current_full_pdf(tmp_path):
    era, wn = fixtures(tmp_path)
    assets, _ = scan_saved_figures(era, wn)
    events, chosen = prepare_report_inventory(catalog(), assets)
    _, _, summary = build_figure_report(events, chosen, tmp_path / 'report', progress=lambda _: None, combined_limit_mb=0)
    assert summary['combined_pdf'] is None
    assert all(Path(p).is_file() for p in summary['volumes'])


def test_notebook_is_standalone_and_has_no_weather_retrieval():
    import ast
    notebook = json.loads((ROOT / 'notebooks/39_saved_event_figure_report.ipynb').read_text())
    source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    ast.parse(source)
    assert 'def build_figure_report' in source
    assert not any(term in source for term in ('earthaccess', 'open_zarr', 'gcsfs', 'weathernext_atlas import'))


def test_legacy35_discovery_and_timing_guard(tmp_path):
    era, wn = tmp_path / 'era', tmp_path / 'wn'
    folder = era / 'batch_evolution' / '1h_Auto_fixture' / 'event_0276_fixture'
    folder.mkdir(parents=True)
    row = catalog().iloc[0].to_dict()
    row.update(offset_hours=0, hours=1, view_hour_utc=row['event_peak'],
               era5_map_status='complete', batch_status='partial_data')
    pd.DataFrame([row]).to_csv(folder / 'peak_metadata.csv', index=False)
    pdf(folder / 'peak.pdf', 'Synthetic source overview')
    assets, issues = scan_saved_figures(era, wn)
    assert issues.empty and len(assets) == 1
    assert assets.iloc[0].kind == 'notebook35_six_panel'
    assert assets.iloc[0].status == 'complete'  # IMERG gap does not mean ERA5 is missing.
    prepare_report_inventory(catalog(), assets)
    assets['valid_time'] = '2026-01-10 05:00:00'
    with pytest.raises(ValueError, match='peak-relative offset'):
        prepare_report_inventory(catalog(), assets)


def test_zero_saved_files_stops_before_creating_empty_report(tmp_path):
    assets, _ = scan_saved_figures(tmp_path / 'era', tmp_path / 'wn')
    events, chosen = prepare_report_inventory(catalog(), assets)
    with pytest.raises(ValueError, match='No saved figure files'):
        build_figure_report(events, chosen, tmp_path / 'report')
