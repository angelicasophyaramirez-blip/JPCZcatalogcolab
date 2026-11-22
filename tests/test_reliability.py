"""Probability accounting, missing-data handling, and outcome-independent schedules."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jpcz_catalog import ensemble_verification as v


def fixtures():
    members, ranks = [], []
    # 0, 0.5, and 1 forecast probabilities, with strict equality treated as non-crossing.
    for event, values, reference in [(1, np.zeros(64), 0.),
                                     (2, np.repeat([-2., 0.], 32), -2.),
                                     (3, np.full(64, -2.), 0.)]:
        for region in v.POLYGONS:
            for lead in v.LEAD_HOURS:
                info = dict(review_key=f'event-{event}', polygon=region, lead_hours=lead, lead_days=lead//24,
                            event_number=event, valid_time_utc='2026-11-19T00:00:00Z',
                            vortex_label='Non-vortex-associated', vortex_scan_status='not_scanned',
                            vortex_classification='Vortex not checked', in_common_sample=True)
                ranks.append({**info, 'era5_divergence_1e5_s-1': reference})
                members.extend({**info, 'member': str(i), v.DIVERGENCE_COLUMN: value}
                               for i, value in enumerate(values))
    return pd.DataFrame(members), pd.DataFrame(ranks)


def test_probabilities_edges_empty_bins_and_scores(tmp_path):
    members, ranks = fixtures()
    thresholds = {region: -1. for region in v.POLYGONS}
    p = v.reliability_probabilities(members, ranks, thresholds)
    assert len(p) == 18
    for _, group in p.groupby(['polygon', 'lead_hours']):
        assert group.forecast_probability.tolist() == [0., .5, 1.]
        assert group.observed.tolist() == [0, 1, 0]
        np.testing.assert_allclose(group.brier_score, [0., .25, 1.])
    bins = v.reliability_bin_counts(p, n_bins=10)
    for _, group in bins.groupby(['polygon', 'lead_days']):
        assert group.case_count.sum() == 3
        assert group.observed_count.sum() == 1
        assert group.case_count.tolist() == [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        assert group.loc[group.case_count.eq(0), 'observed_frequency'].isna().all()
        assert group.brier_score.iloc[0] == pytest.approx(1.25/3)
    # Equality to the threshold does not count as a crossing in either data source.
    equal = v.reliability_probabilities(members, ranks, {region: 0. for region in v.POLYGONS})
    assert equal.loc[equal.event_number.eq(1), 'forecast_probability'].eq(0).all()
    assert equal.loc[equal.event_number.eq(3), 'observed'].eq(0).all()
    fig = v.plot_reliability_diagrams(p)
    assert len(fig.axes) == 6
    fig.savefig(tmp_path / 'reliability.png')
    plt.close(fig)
    tables = {'member_values': members, 'case_ranks': ranks}
    v.export_reliability(tables, tmp_path, thresholds)
    assert (tmp_path / 'reliability_diagrams.pdf').stat().st_size > 0
    tables['case_ranks'] = ranks.iloc[:0]
    v.export_reliability(tables, tmp_path, thresholds)
    assert not (tmp_path / 'reliability_diagrams.pdf').exists()


def test_missing_cases_are_excluded_and_unchecked_is_not_nonvortex():
    members, ranks = fixtures()
    ranks.loc[ranks.event_number.eq(3), 'in_common_sample'] = False
    p = v.reliability_probabilities(members, ranks, {region: -1. for region in v.POLYGONS})
    assert len(p) == 12
    assert v.verification_subset({'case_ranks': ranks}, 'Non-vortex').empty
    assert len(v.verification_subset({'case_ranks': ranks}, 'Unknown')) == 18
    ranks.loc[ranks.event_number.eq(1), ['vortex_scan_status', 'vortex_classification']] = ['complete', 'Non-vortex']
    assert len(v.verification_subset({'case_ranks': ranks}, 'Non-vortex')) == 6
    # A completed screen with an unparseable verdict is also Unknown.
    ranks.loc[ranks.event_number.eq(2), 'vortex_scan_status'] = 'complete'
    assert len(v.verification_subset({'case_ranks': ranks}, 'Non-vortex')) == 6
    with pytest.raises(ValueError, match='64 distinct members'):
        v.reliability_probabilities(members.iloc[1:], ranks, {region: -1. for region in v.POLYGONS})
    with pytest.raises(ValueError, match='Duplicate'):
        v.reliability_probabilities(members, pd.concat([ranks, ranks.iloc[:1]]),
                                    {region: -1. for region in v.POLYGONS})


def test_schedule_includes_non_events_and_exact_nov19_requests():
    from jpcz_catalog.weathernext_atlas import forecast_requests
    catalog = v.scheduled_verification_catalog('2026-10-31', '2026-11-21')
    assert len(catalog) == 20
    assert catalog.event_start.min() == pd.Timestamp('2026-11-01')
    assert catalog.event_start.max() == pd.Timestamp('2026-11-20')
    event = catalog.loc[catalog.event_start.eq(pd.Timestamp('2026-11-19'))].iloc[0]
    requests = forecast_requests(event)
    assert requests.init_time.tolist() == [pd.Timestamp('2026-11-18'), pd.Timestamp('2026-11-14'),
                                          pd.Timestamp('2026-11-09')]
    assert catalog.review_key.is_unique
    with pytest.raises(ValueError, match='six-hour'):
        v.scheduled_verification_catalog('2026-11-19T01:00', '2026-11-21')


def test_unpopulated_era5_times_are_rejected_before_wind_reads():
    ds = xr.Dataset(attrs={'valid_time_stop': '2026-08-31T23:00Z',
                           'valid_time_stop_era5t': '2026-11-14T23:00Z'})
    with pytest.raises(ValueError, match='ERA5 pending'):
        v.era5_polygon_values('2026-11-19', ds, allow_era5t=True)
    with pytest.raises(ValueError, match='ERA5 pending'):
        v.era5_polygon_values('2026-11-13', ds, allow_era5t=False)


def test_added_event_and_revised_era5t_refresh_plots_without_redownloading_wn2(tmp_path, monkeypatch):
    from jpcz_catalog.weathernext_atlas import normalise_catalog
    dates = pd.to_datetime(['2026-11-01', '2026-11-10', '2026-11-19'])
    catalog = normalise_catalog(pd.DataFrame({'event_start': dates, 'event_peak': dates, 'event_end': dates}))
    class Archive:
        root = 'test-wn2'
        overrides = {}
    archive = Archive()
    forecast_calls, reference_calls = [], []
    revised = False
    def reference(time, ds, **kwargs):
        reference_calls.append(time)
        value = 20. if revised else -1.
        return {'values': {region: value for region in v.POLYGONS},
                'polygon_cells': {region: 10 for region in v.POLYGONS},
                'source_store': 'test-era5', 'reference_product':
                ('ERA5T' if time == dates[1] and not revised else 'ERA5')}
    def forecast(archive, request, progress, **kwargs):
        forecast_calls.append(request)
        return {'members': [str(i) for i in range(64)],
                'values': {region: np.arange(64).tolist() for region in v.POLYGONS},
                'polygon_cells': {region: 10 for region in v.POLYGONS}, 'source_store': 'test-wn2'}
    monkeypatch.setattr(v, 'era5_polygon_values', reference)
    monkeypatch.setattr(v, 'forecast_polygon_values', forecast)
    first, _ = v.run_ensemble_verification(catalog.iloc[:2], archive, tmp_path,
                                           era5_reader=lambda: None, progress=lambda _: None)
    assert len(forecast_calls) == 6
    assert first['case_ranks'].era5_rank.eq(1).all()
    revised = True
    forecast_calls.clear()
    reference_calls.clear()
    second, _ = v.run_ensemble_verification(catalog, archive, tmp_path,
                                            era5_reader=lambda: None, progress=lambda _: None)
    assert len(forecast_calls) == 3  # only the new event's three leads
    assert reference_calls == [dates[1], dates[2]]  # stable reference reused, preliminary refreshed
    ranks = second['case_ranks']
    assert len(ranks) == 18
    assert ranks.loc[ranks.event_number.eq(1), 'era5_rank'].eq(1).all()
    assert ranks.loc[ranks.event_number.eq(2), 'era5_rank'].between(21, 22).all()
    counts = v.rank_histogram_counts(ranks)
    assert counts.groupby(['polygon', 'lead_days']).case_count.sum().eq(3).all()
    thresholds = {region: 10. for region in v.POLYGONS}
    p, bins = v.export_reliability(second, tmp_path / 'reliability', thresholds)
    assert p.groupby(['polygon', 'lead_hours']).size().eq(3).all()
    assert p.loc[p.event_number.eq(1), 'observed'].eq(1).all()
    assert p.loc[p.event_number.isin([2, 3]), 'observed'].eq(0).all()
    assert bins.groupby(['polygon', 'lead_days']).case_count.sum().eq(3).all()
    # Removing a revised/deleted event rebuilds counts rather than adding prior aggregates.
    third, _ = v.run_ensemble_verification(catalog.iloc[[0, 2]], archive, tmp_path,
                                           era5_reader=lambda: None, progress=lambda _: None)
    assert v.rank_histogram_counts(third['case_ranks']).matched_events.eq(2).all()
