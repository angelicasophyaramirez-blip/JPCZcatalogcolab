"""Matched reference scoring, frozen-schedule resumption and all-event PDF coverage."""
from copy import deepcopy
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from pypdf import PdfReader
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from jpcz_catalog import continuous_skill as s
from jpcz_catalog import climatology_reference as c
from jpcz_catalog import ensemble_verification as v
from jpcz_catalog import weathernext_atlas as a
from test_reliability import fixtures


def scores():
    members, ranks = fixtures()
    ranks['init_time_utc'] = [v.iso_utc(pd.Timestamp(row.valid_time_utc)-pd.Timedelta(hours=row.lead_hours))
                             for row in ranks.itertuples()]
    return v.continuous_ranked_scores({'member_values': members, 'case_ranks': ranks})


def reference_inputs(tmp_path):
    path = tmp_path/'catalog.csv'
    pd.DataFrame([dict(event_number=i, event_start=f'{year}-01-13T01:00',
        event_end=f'{year}-01-13T12:00', event_peak=f'{year}-01-13T06:00')
        for i, year in enumerate([2001, 2002, 2023], 1)]).to_csv(path, index=False)
    schedule = c.climatology_schedule(path, kind='Historical JPCZ cases')
    calculation = v.checkpoint_identity(type('Archive', (), {'root': 'synthetic', 'overrides': {}})())
    definition, signature = c.climatology_definition(schedule, calculation,
        kind='Historical JPCZ cases', start='2000-01-01', end='2022-01-01')
    def job(kind, payload):
        assert kind == 'reference' and payload['allow_era5t'] is False
        return dict(valid_time=payload['valid_time'], source_store=v.ERA5_STORE, reference_product='ERA5',
                    values={region: (-2. if '2001' in payload['valid_time'] else 0.) for region in v.POLYGONS})
    return schedule, calculation, definition, signature, job


def test_crpss_uses_ratio_of_means_and_matches_cases(tmp_path):
    schedule, calculation, definition, _, job = reference_inputs(tmp_path)
    c.build_saved_climatology(schedule, tmp_path/'reference', definition, v.DIVERGENCE_SPEC,
        job_runner=job, download_missing=True, limit=None, progress=lambda _: None)
    import json
    settings = json.loads((tmp_path/'reference/reference_settings.json').read_text())
    reference = pd.read_csv(tmp_path/'reference/reference_values.csv')
    original = scores()
    # Make reference CRPS vary by case, so averaging individual ratios gives a
    # demonstrably different answer. Event 3's deterministic -2 forecast vs 4 scores 6.
    original.loc[original.event_number.eq(3), ['era5_value', 'crps']] = [4., 6.]
    cases, summary = s.matched_climatology_skill(original, reference, settings, calculation)
    np.testing.assert_allclose(cases.loc[cases.event_number.eq(3), 'reference_crps'], 4.5)
    np.testing.assert_allclose(cases.loc[~cases.event_number.eq(3), 'reference_crps'], .5)
    np.testing.assert_allclose(summary.mean_model_crps, 6.5/3)
    np.testing.assert_allclose(summary.mean_reference_crps, 5.5/3)
    np.testing.assert_allclose(summary.crpss_percent, 100*(1-6.5/5.5))
    assert not np.isclose(summary.crpss_percent.iloc[0], 100*(1-(cases.crps/cases.reference_crps).mean()))
    assert len(cases) == 18 and len(summary) == 6 and summary.cases.eq(3).all()
    assert 'reference_crps' not in original
    pd.testing.assert_series_equal(cases.crps_improvement, cases.reference_crps-cases.crps, check_names=False)
    changed = deepcopy(settings); changed['complete'] = False
    with pytest.raises(ValueError, match='incomplete'): s.matched_climatology_skill(original, reference, changed, calculation)
    changed = deepcopy(settings); changed['calculation']['pressure_hpa'] = 500
    with pytest.raises(ValueError, match='pressure'): s.matched_climatology_skill(original, reference, changed, calculation)
    overlap = original.copy(); overlap['valid_time_utc'] = '2001-01-20T00:00:00Z'
    overlap['init_time_utc'] = [v.iso_utc(pd.Timestamp(row.valid_time_utc)-pd.Timedelta(hours=row.lead_hours))
                              for row in overlap.itertuples()]
    with pytest.raises(ValueError, match='before'): s.matched_climatology_skill(overlap, reference, settings, calculation)
    with pytest.raises(ValueError, match='Duplicate'): s.matched_climatology_skill(original, pd.concat([reference, reference.iloc[:1]]), settings, calculation)
    missing = reference.iloc[:-1]
    with pytest.raises(ValueError, match='inventory'): s.matched_climatology_skill(original, missing, settings, calculation)


def test_reference_preview_and_resume_never_repeat_completed_reads(tmp_path):
    schedule, _, definition, _, job = reference_inputs(tmp_path)
    destination = tmp_path/'reference'; calls = []
    def runner(kind, payload):
        calls.append(payload['valid_time']); return job(kind, payload)
    preview = c.build_saved_climatology(schedule, destination, definition, v.DIVERGENCE_SPEC, progress=lambda _: None)
    assert not calls and preview['completed_targets'] == 0 and not preview['complete']
    first = c.build_saved_climatology(schedule, destination, definition, v.DIVERGENCE_SPEC,
        download_missing=True, limit=1, job_runner=runner, progress=lambda _: None)
    assert len(calls) == 1 and first['completed_targets'] == 1 and not first['complete']
    second = c.build_saved_climatology(schedule, destination, definition, v.DIVERGENCE_SPEC,
        download_missing=True, limit=1, job_runner=runner, progress=lambda _: None)
    assert len(calls) == 2 and len(set(calls)) == 2 and second['complete']
    c.build_saved_climatology(schedule, destination, definition, v.DIVERGENCE_SPEC,
        download_missing=True, job_runner=lambda *args: pytest.fail('Re-downloaded completed reference'), progress=lambda _: None)
    assert len(pd.read_csv(destination/'reference_values.csv')) == 4
    changed = deepcopy(definition); changed['kind'] = 'All winter conditions'
    with pytest.raises(ValueError, match='new reference folder'):
        c.build_saved_climatology(schedule, destination, changed, v.DIVERGENCE_SPEC)


def test_reference_failure_is_pending_and_skill_cannot_use_partial_values(tmp_path):
    schedule, calculation, definition, _, job = reference_inputs(tmp_path)
    def runner(kind, payload):
        if '2001' in payload['valid_time']: raise RuntimeError('Missing historical ERA5')
        return job(kind, payload)
    status = c.build_saved_climatology(schedule, tmp_path/'ref', definition, v.DIVERGENCE_SPEC,
        download_missing=True, job_runner=runner, limit=None, progress=lambda _: None)
    assert status['completed_targets'] == 1 and not status['complete']
    destination = tmp_path/'skill'; destination.mkdir(); (destination/'crpss_summary.csv').write_text('stale')
    state, cases, summary = s.export_crpss_report(scores(), tmp_path/'ref', destination, calculation)
    assert state['status'] == 'reference_incomplete' and cases is None and summary is None
    assert not (destination/'crpss_summary.csv').exists()
    assert pd.read_csv(tmp_path/'ref/reference_status.csv').status.tolist() == ['failed', 'computed']


def test_all_winter_schedule_and_reference_zero_denominator(tmp_path):
    schedule = c.climatology_schedule('unused', kind='All winter conditions', start='2020-02-28', end='2020-03-02')
    assert len(schedule) == 8  # leap day included, March excluded
    schedule, calculation, definition, _, job = reference_inputs(tmp_path)
    c.build_saved_climatology(schedule, tmp_path/'ref', definition, v.DIVERGENCE_SPEC,
        download_missing=True, job_runner=job, limit=None, progress=lambda _: None)
    import json
    settings = json.loads((tmp_path/'ref/reference_settings.json').read_text())
    reference = pd.read_csv(tmp_path/'ref/reference_values.csv'); reference['value'] = 0.
    model = scores(); model['era5_value'] = 0.
    _, summary = s.matched_climatology_skill(model, reference, settings, calculation)
    assert summary.crpss_percent.isna().all()
    assert summary.status.eq('undefined_zero_reference_crps').all()


def test_every_case_has_one_pdf_page_and_unchanged_figures_are_reused(tmp_path):
    model = scores(); original = model.copy(deep=True)
    index, wide = s.export_event_crps_reports(model, tmp_path/'cases')
    assert len(index) == len(wide) == 3 and index.status.eq('exported').all()
    assert len(PdfReader(tmp_path/'cases/all_events_crps.pdf').pages) == 3
    assert len(pd.read_csv(tmp_path/'cases/all_event_scores.csv')) == 18
    assert pd.ExcelFile(tmp_path/'cases/all_event_scores.xlsx').sheet_names == ['All events']
    stamp = Path(index.event_png.iloc[0]).stat().st_mtime_ns
    repeated, _ = s.export_event_crps_reports(model, tmp_path/'cases')
    assert repeated.status.eq('cached').all() and Path(index.event_png.iloc[0]).stat().st_mtime_ns == stamp
    pd.testing.assert_frame_equal(model, original)
    changed = model.copy(); changed.loc[0, 'crps'] += .25
    updated, _ = s.export_event_crps_reports(changed, tmp_path/'cases')
    assert updated.status.value_counts().to_dict() == {'cached': 2, 'exported': 1}
    with pytest.raises(ValueError, match='both polygons'): s.validate_saved_case_scores(model.iloc[:-1])
    with pytest.raises(ValueError, match='Duplicate'): s.validate_saved_case_scores(pd.concat([model, model.iloc[:1]]))


def test_complete_skill_exports_paired_values_and_summary(tmp_path):
    schedule, calculation, definition, _, job = reference_inputs(tmp_path)
    c.build_saved_climatology(schedule, tmp_path/'ref', definition, v.DIVERGENCE_SPEC,
        download_missing=True, job_runner=job, limit=None, progress=lambda _: None)
    state, cases, summary = s.export_crpss_report(scores(), tmp_path/'ref', tmp_path/'skill', calculation)
    assert state['status'] == 'complete' and len(summary) == 6
    assert len(pd.read_csv(tmp_path/'skill/crpss_cases.csv')) == 18
    assert len(PdfReader(tmp_path/'skill/crpss_summary.pdf').pages) == 1
    index, wide = s.export_event_crps_reports(cases, tmp_path/'paired')
    assert len(index) == 3
    assert len([x for x in wide if x.endswith('reference CRPS')]) == 6
    assert len([x for x in wide if x.endswith('CRPS improvement')]) == 6
