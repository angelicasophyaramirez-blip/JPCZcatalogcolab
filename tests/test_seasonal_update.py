"""Regression checks for the rolling established-detector catalog, without cloud reads."""
import ast
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jpcz_catalog import seasonal_update as update


def seed_catalog(root):
    baseline = pd.DataFrame([dict(event_start='2025-02-02T00:00:00', event_end='2025-02-02T03:00:00',
        event_peak='2025-02-02T01:00:00', event_peak_D_s_1=-3e-5,
        duration_hours=4, event_number=275)])
    baseline = baseline.rename(columns={'event_peak_D_s_1': 'event_peak_D_s-1'})
    for name in (update.RAW_NAME, update.MERGED_NAME):
        baseline.to_csv(root / name, index=False)
    pd.DataFrame({'month': [11, 12, 1, 2], 'threshold_s-1': [-2e-5]*4}).to_csv(
        root / 'ndjf_monthly_thresholds.csv', index=False)


def fake_archive(stop, *, stable='2026-11-30T23:00:00'):
    times = pd.date_range('2026-11-01', stop, freq='h')
    return xr.Dataset(coords={'time': times}, attrs={
        'valid_time_stop': stable, 'valid_time_stop_era5t': stop})


def test_new_event_extends_without_duplicates_and_preserves_baseline(tmp_path, monkeypatch):
    seed_catalog(tmp_path)
    original = (tmp_path / update.RAW_NAME).read_bytes()
    calls = []
    def subset(ds, start, end, **kwargs):
        return xr.Dataset(coords={'time': pd.date_range(start, end, freq='h'),
                                  'longitude': [128, 130, 132], 'latitude': [40, 39, 38]})
    def scalar_calculation(winds, **kwargs):
        calls.append(str(winds.time.values[-1]))
        times = pd.DatetimeIndex(winds.time.values)
        # One ongoing November 19 event that will lengthen on a later scan.
        values = np.where(times >= pd.Timestamp('2026-11-19T12:00'), -3e-5, 0.)
        hourly = xr.DataArray(values, coords={'time': times}, dims='time')
        return hourly, hourly.rolling(time=12, min_periods=12).mean()
    monkeypatch.setattr(update, 'subset_era5_window', subset)
    monkeypatch.setattr(update, 'prepare_detection_geometry', lambda *a: None)
    monkeypatch.setattr(update, 'compute_polygon_mean_divergence_series_chunked', scalar_calculation)
    first = update.update_catalog(tmp_path, fake_archive('2026-11-19T23:00:00'),
        update_start='2026-11-01', requested_end='2026-11-19T23:00:00', progress=lambda _: None)
    assert first['merged_events'] == 2
    merged = update.read_events(tmp_path / update.MERGED_NAME)
    assert merged.event_number.tolist() == [275, 276]
    assert merged.catalog_provisional.tolist() == [False, True]
    assert (tmp_path / 'seasonal_updates' / ('baseline_' + update.RAW_NAME)).read_bytes() == original
    frozen = (tmp_path / 'seasonal_updates/frozen_thresholds.csv').read_bytes()
    # Repeating the exact stable request uses the monthly cache, and never duplicates rows.
    again = update.update_catalog(tmp_path, fake_archive('2026-11-19T23:00:00'),
        requested_end='2026-11-19T23:00:00', progress=lambda _: None)
    assert again['new_or_revised'] == 0
    assert len(calls) == 1
    later = update.update_catalog(tmp_path, fake_archive('2026-11-20T23:00:00'),
        requested_end='2026-11-20T23:00:00', progress=lambda _: None)
    assert later['merged_events'] == 2
    assert later['new_or_revised'] == 1
    merged = update.read_events(tmp_path / update.MERGED_NAME)
    assert merged.event_number.tolist() == [275, 277]  # revised window needs a fresh review
    assert merged.event_end.iloc[1] == pd.Timestamp('2026-11-20T23:00:00')
    assert (tmp_path / 'seasonal_updates/frozen_thresholds.csv').read_bytes() == frozen
    assert update.read_events(tmp_path / update.RAW_NAME).event_start.is_unique
    with pytest.raises(ValueError, match='precedes an existing scan'):
        update.update_catalog(tmp_path, fake_archive('2026-11-19T23:00:00'),
            requested_end='2026-11-19T23:00:00', progress=lambda _: None)


def test_pending_data_never_erases_the_catalog(tmp_path):
    seed_catalog(tmp_path)
    original = (tmp_path / update.MERGED_NAME).read_bytes()
    ds = fake_archive('2026-10-31T23:00:00', stable='2026-08-31T23:00:00')
    result = update.update_catalog(tmp_path, ds, update_start='2026-11-01',
        requested_end='2026-11-19T23:00:00')
    assert result['status'] == 'pending_era5'
    assert (tmp_path / update.MERGED_NAME).read_bytes() == original
    assert update.availability(ds, allow_era5t=True) == pd.Timestamp('2026-10-31T23:00:00')
    assert update.availability(ds, allow_era5t=False) == pd.Timestamp('2026-08-31T23:00:00')
    with pytest.raises(ValueError, match='metadata'):
        update.availability(xr.Dataset(coords={'time': pd.date_range('2026-01-01', periods=100)}))


def test_frozen_threshold_edits_and_overlap_are_rejected(tmp_path):
    seed_catalog(tmp_path)
    with pytest.raises(ValueError, match='after all established'):
        update.initialize_update(tmp_path, '2025-02-01')
    update.initialize_update(tmp_path, '2026-11-01')
    with pytest.raises(ValueError, match='saved update_start'):
        update.initialize_update(tmp_path, '2026-12-01')
    (tmp_path / 'seasonal_updates/frozen_thresholds.csv').write_text('changed')
    with pytest.raises(ValueError, match='thresholds have changed'):
        update.initialize_update(tmp_path)


def test_standalone_notebooks_compile_and_update_bundle_loads():
    for number in ('35', '36', '37', '38'):
        path = next((ROOT / 'notebooks').glob(number + '*.ipynb'))
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for item in notebook.cells:
            if item.cell_type == 'code':
                ast.parse(item.source)
                if number == '38' and item.source.startswith('# @title 3'):
                    namespace = {'__name__': '__main__'}
                    exec(compile(item.source, 'notebook38_helpers', 'exec'), namespace)
                    assert callable(namespace['update_catalog'])


def test_preliminary_month_is_refreshed_until_finalized(tmp_path, monkeypatch):
    seed_catalog(tmp_path)
    calls = []
    values = [0.]
    def subset(ds, start, end, **kwargs):
        return xr.Dataset(coords={'time': pd.date_range(start, end, freq='h'),
                                  'longitude': [128, 130, 132], 'latitude': [40, 39, 38]})
    def calculate(winds, **kwargs):
        calls.append(1)
        hourly = xr.DataArray(np.full(winds.sizes['time'], values[0]), coords={'time': winds.time}, dims='time')
        return hourly, hourly.rolling(time=12, min_periods=12).mean()
    monkeypatch.setattr(update, 'subset_era5_window', subset)
    monkeypatch.setattr(update, 'prepare_detection_geometry', lambda *a: None)
    monkeypatch.setattr(update, 'compute_polygon_mean_divergence_series_chunked', calculate)
    args = dict(update_start='2026-11-01', requested_end='2026-11-02T23:00', progress=lambda _: None)
    preliminary = fake_archive('2026-11-02T23:00', stable='2026-08-31T23:00')
    first = update.update_catalog(tmp_path, preliminary, **args)
    assert first['merged_events'] == 1  # no crossing
    values[0] = -3e-5
    second = update.update_catalog(tmp_path, preliminary, **args)
    assert second['merged_events'] == 2  # revised ERA5T now crosses
    finalized = fake_archive('2026-11-02T23:00', stable='2026-11-02T23:00')
    update.update_catalog(tmp_path, finalized, **args)
    update.update_catalog(tmp_path, finalized, **args)
    assert len(calls) == 3  # prelim twice, final once, then stable cache


def test_actual_drive_threshold_provenance_is_checked(tmp_path):
    seed_catalog(tmp_path)
    raw = pd.read_csv(tmp_path / update.RAW_NAME)
    raw['detection_threshold_s-1'] = -4e-5
    raw.to_csv(tmp_path / update.RAW_NAME, index=False)
    with pytest.raises(ValueError, match='Threshold file disagrees'):
        update.initialize_update(tmp_path, '2026-11-01')
    assert not (tmp_path / 'seasonal_updates/baseline.json').exists()


def test_inspection_does_not_assume_a_merge_version(tmp_path):
    seed_catalog(tmp_path)
    merged = pd.read_csv(tmp_path / update.MERGED_NAME)
    merged['event_end'] = '2025-02-02T09:00:00'
    merged.to_csv(tmp_path / update.MERGED_NAME, index=False)
    with pytest.raises(ValueError, match='does not match a 12-hour merge'):
        update.inspect_catalog(tmp_path)


def test_real_chunked_calculation_resumes_after_crash_and_matches_notebook04(tmp_path, monkeypatch):
    """Exercise real MetPy and NetCDF checkpoints, with cloud winds replaced by a small local grid."""
    from jpcz_catalog import historical_catalog as history
    from jpcz_catalog.detect import compute_polygon_mean_divergence_series, prepare_detection_geometry
    seed_catalog(tmp_path)
    times = pd.date_range('2026-11-01', periods=54, freq='h')
    lon, lat = np.arange(128., 138.1, .5), np.arange(42., 34.9, -.5)
    shape = (len(times), len(lat), len(lon))
    amplitude = (2. + np.sin(np.arange(len(times)) / 4.))[:, None, None]
    u = np.broadcast_to(-8. * amplitude * (lon - 132.)[None, None, :], shape).copy()
    v = np.broadcast_to(.2 * amplitude * (lat - 38.)[None, :, None], shape).copy()
    winds = xr.Dataset({'u_component_of_wind': (('time', 'latitude', 'longitude'), u),
                        'v_component_of_wind': (('time', 'latitude', 'longitude'), v)},
                       coords={'time': times, 'latitude': lat, 'longitude': lon})
    geometry = prepare_detection_geometry(winds.longitude, winds.latitude, update.JPCZ_POLYGON_VERTICES)
    expected_hourly, expected_d12, _ = compute_polygon_mean_divergence_series(winds.copy(deep=True), geometry=geometry)
    archive = fake_archive(str(times[-1]), stable='2026-08-31T23:00')
    archive.attrs['last_updated'] = '2026-11-10T12:00:00Z'
    monkeypatch.setattr(update, 'subset_era5_window', lambda ds, start, end, **kw: winds.sel(time=slice(start, end)))
    real_divergence = history.mpcalc.divergence
    first_batches = []
    def fail_after_checkpoint(u, *args, **kwargs):
        first_batches.append(len(u))
        if len(first_batches) == 5:
            raise RuntimeError('simulated runtime crash')
        return real_divergence(u, *args, **kwargs)
    monkeypatch.setattr(history.mpcalc, 'divergence', fail_after_checkpoint)
    args = dict(update_start='2026-11-01', requested_end=str(times[-1]), progress=lambda _: None)
    original_catalog = (tmp_path / update.MERGED_NAME).read_bytes()
    with pytest.raises(RuntimeError, match='simulated runtime crash'):
        update.update_catalog(tmp_path, archive, **args)
    assert first_batches == [6]*5
    assert (tmp_path / update.MERGED_NAME).read_bytes() == original_catalog
    saved = list((tmp_path / 'seasonal_updates/hourly').glob('era5t_*/*.nc'))
    assert len(saved) == 1
    with xr.open_dataarray(saved[0]) as partial:
        assert partial.sizes['time'] == 24
        np.testing.assert_allclose(partial.values, expected_hourly.values[:24], rtol=1e-12, atol=1e-15)
    resumed_batches = []
    def count_remaining(u, *args, **kwargs):
        resumed_batches.append(len(u))
        return real_divergence(u, *args, **kwargs)
    monkeypatch.setattr(history.mpcalc, 'divergence', count_remaining)
    update.update_catalog(tmp_path, archive, **args)
    assert resumed_batches == [6]*5  # only hours 24..53, not all 54 again
    with xr.open_dataarray(tmp_path / 'seasonal_updates/monthly/2026-11_D12.nc') as saved_d12:
        np.testing.assert_allclose(saved_d12.values, expected_d12.dropna('time').values, rtol=1e-12, atol=1e-15)
        assert saved_d12.attrs['source_product'] == 'ERA5T'
    # A completed month in the same archive revision needs no wind calculation.
    update.update_catalog(tmp_path, archive, **args)
    assert len(resumed_batches) == 5
    # A changed publication must recompute earlier hours even if availability is unchanged.
    archive.attrs['last_updated'] = '2026-11-10T18:00:00Z'
    update.update_catalog(tmp_path, archive, **args)
    assert len(resumed_batches) == 14  # fresh 54 hours, nine six-hour batches
    assert len(list((tmp_path / 'seasonal_updates/hourly').glob('era5t_*/*.nc'))) == 2
    # Finalization cannot reuse preliminary wind calculations.
    archive.attrs['valid_time_stop'] = str(times[-1])
    update.update_catalog(tmp_path, archive, **args)
    assert len(resumed_batches) == 23
    with xr.open_dataarray(tmp_path / 'seasonal_updates/monthly/2026-11_D12.nc') as saved_d12:
        assert saved_d12.attrs['source_product'] == 'ERA5'


def test_chunk_checkpoint_interval_is_validated_before_any_drive_mutation(tmp_path):
    with pytest.raises(ValueError, match='divisible'):
        update.update_catalog(tmp_path, xr.Dataset(), chunk_hours=5, checkpoint_every_hours=24)
    assert not (tmp_path / 'seasonal_updates').exists()


def local_wind_archive(hours=54):
    times = pd.date_range('2026-11-01', periods=hours, freq='h')
    lon, lat = np.arange(128., 138.1, .5), np.arange(42., 34.9, -.5)
    shape = (hours, len(lat), len(lon))
    u = np.broadcast_to(-8. * (lon - 132.)[None, None, :], shape).copy()
    winds = xr.Dataset({'u_component_of_wind': (('time', 'latitude', 'longitude'), u),
                        'v_component_of_wind': (('time', 'latitude', 'longitude'), np.zeros(shape))},
                       coords={'time': times, 'latitude': lat, 'longitude': lon})
    archive = fake_archive(str(times[-1]), stable='2026-08-31T23:00')
    archive.attrs['last_updated'] = '2026-11-21T18:21:05Z'
    return winds, archive


def test_missing_interior_wind_hour_saves_good_prefix_and_retries_without_bridging(tmp_path, monkeypatch):
    from jpcz_catalog import historical_catalog as history
    from jpcz_catalog.detect import compute_polygon_mean_divergence_series, prepare_detection_geometry
    seed_catalog(tmp_path)
    winds, archive = local_wind_archive()
    original_winds = winds.copy(deep=True)
    geometry = prepare_detection_geometry(winds.longitude, winds.latitude, update.JPCZ_POLYGON_VERTICES)
    expected_hourly, expected_d12, _ = compute_polygon_mean_divergence_series(original_winds, geometry=geometry)
    winds['u_component_of_wind'][31, :, :] = np.nan
    winds['v_component_of_wind'][31, :, :] = np.nan
    monkeypatch.setattr(update, 'subset_era5_window', lambda ds, start, end, **kw: winds.sel(time=slice(start, end)))
    args = dict(update_start='2026-11-01', requested_end=str(winds.time.values[-1]), progress=lambda _: None)
    first = update.update_catalog(tmp_path, archive, **args)
    assert first['status'] == 'updated_with_pending_wind_data'
    assert first['scanned_through'] == pd.Timestamp(winds.time.values[30]).isoformat()
    assert first['first_missing_time_utc'] == pd.Timestamp(winds.time.values[31]).isoformat() + 'Z'
    assert first['pending_data']['u_nonfinite_regional_cells'] > 0
    with xr.open_dataarray(next((tmp_path / 'seasonal_updates/hourly').glob('era5t_*/*.nc'))) as partial:
        assert partial.sizes['time'] == 31  # includes good hours since the 24-hour checkpoint
        assert np.isfinite(partial.values).all()
    with xr.open_dataarray(tmp_path / 'seasonal_updates/monthly/2026-11_D12.nc') as d12:
        assert d12.sizes['time'] == 20
        assert pd.Timestamp(d12.time.values[-1]) == pd.Timestamp(winds.time.values[30])
        np.testing.assert_allclose(d12.values, expected_d12.dropna('time').values[:20])
    current = update.read_events(tmp_path / update.MERGED_NAME)
    assert current.event_end.max() <= pd.Timestamp(winds.time.values[30])
    assert (tmp_path / 'seasonal_updates/pending_data.csv').is_file()
    real_divergence = history.mpcalc.divergence
    batches = []
    def count(u, *args, **kwargs):
        batches.append(len(u))
        return real_divergence(u, *args, **kwargs)
    monkeypatch.setattr(history.mpcalc, 'divergence', count)
    update.update_catalog(tmp_path, archive, **args)
    assert batches == [6]  # retries the hole, rather than reading all earlier hours again
    winds['u_component_of_wind'][31, :, :] = original_winds['u_component_of_wind'][31].values
    winds['v_component_of_wind'][31, :, :] = original_winds['v_component_of_wind'][31].values
    batches.clear()
    filled = update.update_catalog(tmp_path, archive, **args)
    assert batches == [6, 6, 6, 5]
    assert filled['status'] == 'updated'
    assert filled['first_missing_time_utc'] is None
    assert not (tmp_path / 'seasonal_updates/pending_data.csv').exists()
    with xr.open_dataarray(tmp_path / 'seasonal_updates/monthly/2026-11_D12.nc') as d12:
        np.testing.assert_allclose(d12.values, expected_d12.dropna('time').values)


@pytest.mark.parametrize('missing_index', [0, 5])
def test_missing_winds_before_first_d12_preserve_catalog(tmp_path, monkeypatch, missing_index):
    seed_catalog(tmp_path)
    winds, archive = local_wind_archive(hours=18)
    winds['u_component_of_wind'][missing_index, :, :] = np.nan
    monkeypatch.setattr(update, 'subset_era5_window', lambda ds, start, end, **kw: winds.sel(time=slice(start, end)))
    original = (tmp_path / update.MERGED_NAME).read_bytes()
    result = update.update_catalog(tmp_path, archive, update_start='2026-11-01',
                                  requested_end=str(winds.time.values[-1]), progress=lambda _: None)
    assert result['status'] == 'pending_wind_data_catalog_unchanged'
    assert not result['catalog_updated']
    assert (tmp_path / update.MERGED_NAME).read_bytes() == original
    assert not (tmp_path / 'seasonal_updates/scan_manifest.json').exists()


def test_new_revision_gap_cannot_reduce_existing_catalog_coverage(tmp_path, monkeypatch):
    seed_catalog(tmp_path)
    winds, archive = local_wind_archive()
    monkeypatch.setattr(update, 'subset_era5_window', lambda ds, start, end, **kw: winds.sel(time=slice(start, end)))
    args = dict(update_start='2026-11-01', requested_end=str(winds.time.values[-1]), progress=lambda _: None)
    update.update_catalog(tmp_path, archive, **args)
    original = (tmp_path / update.MERGED_NAME).read_bytes()
    manifest = (tmp_path / 'seasonal_updates/scan_manifest.json').read_bytes()
    winds['u_component_of_wind'][31, :, :] = np.nan
    archive.attrs['last_updated'] = '2026-11-21T20:00:00Z'
    result = update.update_catalog(tmp_path, archive, **args)
    assert result['status'] == 'pending_wind_data_catalog_unchanged'
    assert (tmp_path / update.MERGED_NAME).read_bytes() == original
    assert (tmp_path / 'seasonal_updates/scan_manifest.json').read_bytes() == manifest


def test_old_nonfinite_checkpoint_tail_is_salvaged_and_irrelevant_nans_do_not_poison_mean(tmp_path):
    from jpcz_catalog.historical_catalog import compute_polygon_mean_divergence_series_chunked
    from jpcz_catalog.detect import compute_polygon_mean_divergence_series, prepare_detection_geometry
    winds, _ = local_wind_archive(hours=30)
    geometry = prepare_detection_geometry(winds.longitude, winds.latitude, update.JPCZ_POLYGON_VERTICES)
    hourly, d12, _ = compute_polygon_mean_divergence_series(winds.copy(deep=True), geometry=geometry)
    old = hourly.copy(deep=True)
    old.values[7:] = np.nan
    path = tmp_path / 'old_hourly.nc'
    old.to_netcdf(path)
    # This corner is far outside the polygon and all its required derivative stencils.
    winds['u_component_of_wind'][15, 0, 0] = np.nan
    actual_hourly, actual_d12 = compute_polygon_mean_divergence_series_chunked(
        winds, geometry=geometry, chunk_hours=6, progress_every_hours=24, hourly_checkpoint_path=path)
    np.testing.assert_allclose(actual_hourly.values, hourly.values)
    np.testing.assert_allclose(actual_d12.values, d12.values, equal_nan=True)
    with xr.open_dataarray(path) as saved:
        assert len(saved.time) == 30
        assert np.isfinite(saved.values).all()


def test_finite_wind_numerical_failure_is_not_misreported_as_pending_archive(tmp_path, monkeypatch):
    from jpcz_catalog import historical_catalog as history
    seed_catalog(tmp_path)
    winds, archive = local_wind_archive(hours=18)
    monkeypatch.setattr(update, 'subset_era5_window', lambda ds, start, end, **kw: winds.sel(time=slice(start, end)))
    class FailedCalculation:
        magnitude = np.full((6, winds.sizes['latitude'], winds.sizes['longitude']), np.nan)
    monkeypatch.setattr(history.mpcalc, 'divergence', lambda *a, **kw: FailedCalculation())
    original = (tmp_path / update.MERGED_NAME).read_bytes()
    with pytest.raises(history.IncompleteDivergenceError, match='finite_winds'):
        update.update_catalog(tmp_path, archive, update_start='2026-11-01',
                              requested_end=str(winds.time.values[-1]), progress=lambda _: None)
    assert (tmp_path / update.MERGED_NAME).read_bytes() == original
