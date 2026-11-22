"""Three-phase exports, real PDFs, Terra/Aqua fallback, and checkpointed retries without cloud data."""
import ast
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jpcz_catalog.case_review_batch import (
    EarthdataLoginRequired, require_earthdata_auth, run_case_review_batch,
)


def cases(count=2):
    dates = pd.date_range('2018-02-03T20:00', periods=count, freq='7D')
    frame = pd.DataFrame({'event_peak': dates, 'event_start': dates - pd.Timedelta(hours=3),
                          'event_end': dates + pd.Timedelta(hours=4), 'event_number': range(1, count+1),
                          'vortex_scan_status': 'complete', 'vortex_detected': True,
                          'vortex_screen_signature': 'screen-v1'})
    frame['review_key'] = frame[['event_start', 'event_end', 'event_peak']].apply(
        lambda row: '|'.join(t.isoformat() for t in row), axis=1)
    return frame


def batch_api():
    nb = json.loads((ROOT / 'notebooks/35_imerg_manual_case_review.ipynb').read_text())
    text = next(''.join(c['source']) for c in nb['cells'] if 'def get_case(' in ''.join(c['source']))
    function = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'load_satellite_pair')
    ns = {'json': json}
    attempts, reads, renders = [], [], []
    def satellite(hour, platform, day_offset, composite):
        attempts.append((pd.Timestamp(hour), platform, composite))
        available = platform == 'Aqua'
        return (np.ones((4, 6, 3)) if available else None,
                {'satellite_status': 'available_daily_mosaic' if available else 'unavailable',
                 'satellite_platform': f'MODIS {platform}',
                 'satellite_error': '' if available else 'synthetic blank Terra response',
                 'satellite_mosaic_date_utc': str(pd.Timestamp(hour).date())})
    ns['load_visible_satellite'] = satellite
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'notebook35_satellite', 'exec'), ns)
    def load(row, **kwargs):
        reads.append((int(row.event_number), kwargs.copy()))
        hour = row.event_peak + pd.Timedelta(hours=kwargs['offset'])
        image, info, cloud, cloud_info = ns['load_satellite_pair'](
            hour, platform=kwargs['satellite_platform'], day_offset=kwargs['satellite_day_offset'])
        metadata = dict(event_start=row.event_start, event_end=row.event_end, event_peak=row.event_peak,
                        event_number=int(row.event_number), review_key=row.review_key,
                        view_hour_utc=hour, offset_hours=kwargs['offset'], imerg_status='complete',
                        accumulation_status='not_started' if hour < row.event_start else 'complete',
                        label='Vortex-associated', vortex_detected=True, vortex_scan_status='complete',
                        vortex_screen_signature='screen-v1', **info)
        metadata.update({'cloud_rgb_'+key: value for key, value in cloud_info.items()})
        return {'metadata': metadata}
    def render(items):
        metadata = items[0]['metadata']
        renders.append(metadata.copy())
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(.1, .5, f"Event {metadata['event_number']}\nOffset {metadata['offset_hours']:+d} h\n"
                + metadata['satellite_platform'])
        return fig
    return dict(load_case=load, render_cases=render, label_for=lambda row: 'Vortex-associated'), reads, attempts, renders


def test_all_events_three_times_auto_satellite_and_grouped_pdf_csv(tmp_path):
    frame = cases()
    original = frame.copy(deep=True)
    api, reads, attempts, renders = batch_api()
    result = run_case_review_batch(frame, tmp_path, progress=lambda _: None, **api)
    assert len(result) == 6
    assert result.status.eq('complete').all()
    assert [args['offset'] for _, args in reads] == [-12, 0, 12]*2
    assert all(args['satellite_platform'] == 'Auto' for _, args in reads)
    assert result.satellite_platform.eq('MODIS Aqua').all()
    assert [platform for _, platform, _ in attempts] == ['Terra', 'Terra', 'Aqua', 'Aqua']*6
    assert [pd.Timestamp(item['view_hour_utc']) for item in renders[:3]] == [
        frame.event_peak.iloc[0] + pd.Timedelta(hours=offset) for offset in (-12, 0, 12)]
    root = Path(result.attrs['output_dir'])
    assert len(pd.read_csv(root / 'all_cases_metadata.csv')) == 6
    assert json.loads((root / 'batch_summary.json').read_text())['selected_events'] == 2
    for _, job in result.iterrows():
        assert all(Path(job[name]).is_file() for name in ('png', 'pdf', 'metadata_csv', 'event_pdf', 'event_metadata_csv'))
        assert len(pd.read_csv(job.metadata_csv)) == 1
        assert len(PdfReader(job.pdf).pages) == 1
    for pdf in result.event_pdf.unique():
        pages = PdfReader(pdf).pages
        assert len(pages) == 3
        assert all(f'Offset {offset:+d} h' in page.extract_text() for offset, page in zip((-12, 0, 12), pages))
    pd.testing.assert_frame_equal(frame, original)


def test_pilot_limit_counts_whole_events_and_resume_skips_all_existing_formats(tmp_path):
    api, reads, _, _ = batch_api()
    first = run_case_review_batch(cases(), tmp_path, limit=1, progress=lambda _: None, **api)
    assert len(reads) == 3
    assert first.status.tolist() == ['complete']*3 + ['not_attempted']*3
    reads.clear()
    second = run_case_review_batch(cases(), tmp_path, limit=1, progress=lambda _: None, **api)
    assert len(reads) == 3 and all(event == 2 for event, _ in reads)
    assert second.status.eq('complete').all()
    reads.clear()
    run_case_review_batch(cases(), tmp_path, progress=lambda _: None, **api)
    assert reads == []
    # A missing PDF cannot be mistaken for a completed PNG-only export.
    Path(second.pdf.iloc[1]).unlink()
    again = run_case_review_batch(cases(), tmp_path, progress=lambda _: None, **api)
    assert len(reads) == 1 and reads[0][1]['offset'] == 0
    assert len(PdfReader(again.event_pdf.iloc[0]).pages) == 3


def test_failed_phase_gets_labeled_page_later_cases_continue_and_retry_is_local(tmp_path):
    api, reads, _, _ = batch_api()
    real_load = api['load_case']
    def fail_peak(row, **kwargs):
        if row.event_number == 1 and kwargs['offset'] == 0:
            raise OSError('synthetic missing ERA5 winds')
        return real_load(row, **kwargs)
    api['load_case'] = fail_peak
    first = run_case_review_batch(cases(), tmp_path, progress=lambda _: None, **api)
    assert first.status.tolist() == ['complete', 'failed', 'complete'] + ['complete']*3
    assert len(PdfReader(first.event_pdf.iloc[0]).pages) == 3
    assert 'Case data unavailable' in PdfReader(first.pdf.iloc[1]).pages[0].extract_text()
    api['load_case'] = real_load
    reads.clear()
    second = run_case_review_batch(cases(), tmp_path, progress=lambda _: None, **api)
    assert len(reads) == 1 and reads[0][0] == 1 and reads[0][1]['offset'] == 0
    assert second.status.eq('complete').all()


def test_permanent_pre_modis_gap_is_saved_once_but_transient_gaps_retry(tmp_path):
    api, reads, _, _ = batch_api()
    real_load = api['load_case']
    def missing_satellite(row, **kwargs):
        case = real_load(row, **kwargs)
        case['metadata'].update(satellite_status='unavailable', cloud_rgb_satellite_status='unavailable',
            satellite_error='Terra true-color imagery predates archive availability.',
            cloud_rgb_satellite_error='Terra true-color imagery predates archive availability.')
        return case
    api['load_case'] = missing_satellite
    first = run_case_review_batch(cases(1), tmp_path, progress=lambda _: None, **api)
    assert first.status.eq('complete_archive_gap').all()
    reads.clear()
    run_case_review_batch(cases(1), tmp_path, progress=lambda _: None, **api)
    assert not reads
    # Different requested settings create a separate resumable atlas.
    api['load_case'] = real_load
    second = run_case_review_batch(cases(1), tmp_path, hours=12, progress=lambda _: None, **api)
    assert len(reads) == 3
    assert first.attrs['output_dir'] != second.attrs['output_dir']


def test_notebook_batch_defaults_all_events_and_suppresses_inline_plot_flood():
    nb = json.loads((ROOT / 'notebooks/35_imerg_manual_case_review.ipynb').read_text())
    core = next(''.join(c['source']) for c in nb['cells'] if 'def get_case(' in ''.join(c['source']))
    assert 'def plot_cases(cases, stem=None, *, show=True):' in core
    assert 'plot_cases(cases, show=False)' in core
    batch = next(''.join(c['source']) for c in nb['cells']
                 if c['cell_type'] == 'code' and 'RUN_BATCH = ' in ''.join(c['source']))
    namespace = {}
    exec(compile(batch, 'notebook35_batch_settings', 'exec'), namespace)
    assert namespace['EVENTS_PER_RUN'] is None
    assert namespace['BATCH_SATELLITE_PLATFORM'] == 'Auto'
    assert not namespace['RUN_BATCH']


def test_actual_notebook_maps_export_all_three_times_without_inline_displays(tmp_path, monkeypatch):
    import cartopy.crs as ccrs
    from matplotlib import colors
    import textwrap
    import xarray as xr
    from test_vortex_figure_labels import notebook35_functions
    from jpcz_catalog.weathernext_atlas import POLYGONS
    _, ns = notebook35_functions()
    def decorate(axis):
        axis.set_extent((125, 145, 33, 47), crs=ccrs.PlateCarree())
        for name, vertices in POLYGONS.items():
            closed = [*vertices, vertices[0]]
            axis.plot(*zip(*closed), transform=ccrs.PlateCarree(), label=name, linewidth=1)
    ns.update(plt=plt, ccrs=ccrs, colors=colors, textwrap=textwrap, decorate=decorate,
              DOMAIN=(125, 145, 33, 47),
              annotate_mslp_lows=lambda *args: None,
              PRECIP_LEVELS=[0, .5, 1, 2, 4, 8], DIV_LEVELS=[-10, -5, -2, 0, 2, 5, 10],
              VORT_LEVELS=[-10, -5, -2, 0, 2, 5, 10], ACCUM_LEVELS=[0, 5, 10, 25, 50, 100])
    api, _, _, _ = batch_api()
    simple_load = api['load_case']
    lat, lon = np.linspace(33, 47, 20), np.linspace(125, 145, 24)
    coords = {'latitude': lat, 'longitude': lon}
    gradient = np.linspace(0, 1, len(lon))[None, :] + np.linspace(0, 1, len(lat))[:, None]
    array = xr.DataArray(gradient, coords=coords, dims=('latitude', 'longitude'))
    def real_case(row, **kwargs):
        case = simple_load(row, **kwargs)
        hour = row.event_peak + pd.Timedelta(hours=kwargs['offset'])
        total_hours = max(0., (hour+pd.Timedelta(hours=1)-row.event_start)/pd.Timedelta(hours=1))
        meta = case['metadata']
        meta.update(hours=kwargs['hours'], window_start_utc=hour, window_end_exclusive_utc=hour+pd.Timedelta(hours=1),
                    accumulation_start_utc=row.event_start, accumulation_end_exclusive_utc=hour+pd.Timedelta(hours=1),
                    accumulation_mode='event_start', accumulation_hours=total_hours,
                    satellite_mosaic_date_utc=str((hour+pd.Timedelta(hours=9)).date()))
        image = np.stack([gradient/2, gradient[::-1]/2, np.ones_like(gradient)*.4], axis=-1)
        case.update(row=row, dynamics=xr.Dataset(dict(divergence=-array-1, vorticity=array,
                    u=array*2, v=array, mslp=1008+array*8)), precipitation=array,
                    precipitation_accumulation=array*total_hours if total_hours else None,
                    satellite_image=image, cloud_rgb_image=image[..., ::-1])
        return case
    def no_inline(*args, **kwargs):
        raise AssertionError('Batch must not retain full figures in notebook output')
    monkeypatch.setattr(plt, 'show', no_inline)
    api.update(load_case=real_case, render_cases=lambda items: ns['plot_cases'](items, show=False))
    result = run_case_review_batch(cases(1), tmp_path, dpi=100, progress=lambda _: None, **api)
    assert result.status.eq('complete').all()
    assert len(PdfReader(result.event_pdf.iloc[0]).pages) == 3
    assert all(Path(path).stat().st_size > 10000 for path in result.png)
    assert all('MODIS Aqua' in PdfReader(path).pages[0].extract_text() for path in result.pdf)
    assert not plt.get_fignums()


def test_partial_inputs_are_retried_and_do_not_stop_other_events(tmp_path):
    api, reads, _, _ = batch_api()
    real_load = api['load_case']
    def missing_rate(row, **kwargs):
        case = real_load(row, **kwargs)
        if row.event_number == 1:
            case['metadata']['imerg_status'] = 'unavailable'
        return case
    api['load_case'] = missing_rate
    first = run_case_review_batch(cases(), tmp_path, progress=lambda _: None, **api)
    assert first.status.tolist() == ['partial_data']*3 + ['complete']*3
    assert len(reads) == 6
    api['load_case'] = real_load
    reads.clear()
    second = run_case_review_batch(cases(), tmp_path, progress=lambda _: None, **api)
    assert len(reads) == 3 and all(number == 1 for number, _ in reads)
    assert second.status.eq('complete').all()


def test_embedded_batch_wrapper_calls_the_same_three_phase_exporter(tmp_path):
    import jpcz_catalog.case_review_batch as module
    nb = json.loads((ROOT / 'notebooks/35_imerg_manual_case_review.ipynb').read_text())
    source = next(''.join(c['source']) for c in nb['cells'] if 'def get_case(' in ''.join(c['source']))
    wrapper = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'run_batch')
    api, reads, _, _ = batch_api()
    namespace = dict(module.__dict__)
    namespace.update(OUTPUT_DIR=tmp_path, get_case=api['load_case'],
        auth=type('Auth', (), {'authenticated': True})(),
        release_review_memory=lambda: None,
        plot_cases=lambda items, show: api['render_cases'](items), vortex_plot_label=api['label_for'],
        attach_vortex_flags=lambda frame: frame.copy(), DOMAIN=(125,145,33,47), POLYGONS={},
        ERA5_STORE='test-era5', IMERG_PRODUCT='test-imerg', IMERG_VERSION='07',
        MIN_PIXEL_TEMPORAL_COVERAGE=.9, VORTEX_LABEL_SCHEME='test-labels', vortex_signature=lambda: 'test-screen')
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), 'notebook35_batch_wrapper', 'exec'), namespace)
    result = namespace['run_batch'](cases(1))
    assert len(result) == 3 and result.status.eq('complete').all()
    assert [args['offset'] for _, args in reads] == [-12, 0, 12]


@pytest.mark.parametrize('auth', [None, object(), type('Auth', (), {'authenticated': False})()])
def test_missing_login_stops_before_reads_or_checkpoint_changes(tmp_path, auth):
    api, reads, _, renders = batch_api()
    with pytest.raises(EarthdataLoginRequired, match='NASA authentication'):
        run_case_review_batch(cases(), tmp_path, check_auth=lambda: require_earthdata_auth(auth), **api)
    assert not reads and not renders
    assert not list(tmp_path.iterdir())


def test_login_failure_mid_batch_preserves_exports_and_resumes_remaining_phases(tmp_path):
    api, reads, _, _ = batch_api()
    real_load = api['load_case']
    def loses_login(row, **kwargs):
        if kwargs['offset'] == 0:
            require_earthdata_auth(None)
        return real_load(row, **kwargs)
    api['load_case'] = loses_login
    with pytest.raises(EarthdataLoginRequired):
        run_case_review_batch(cases(1), tmp_path, progress=lambda _: None, **api)
    root = next((tmp_path / 'batch_evolution').iterdir())
    manifest = pd.read_csv(root / 'batch_manifest.csv')
    assert manifest.offset_hours.tolist() == [-12]
    assert manifest.status.tolist() == ['complete']
    saved_png = Path(manifest.png.iloc[0])
    before = saved_png.stat().st_mtime_ns
    assert not list(root.glob('event_*/peak.pdf'))
    assert json.loads((root / 'batch_summary.json').read_text())['run_status'] == 'paused_authentication'
    api['load_case'] = real_load
    reads.clear()
    result = run_case_review_batch(cases(1), tmp_path, progress=lambda _: None, **api)
    assert [kwargs['offset'] for _, kwargs in reads] == [0, 12]
    assert result.status.eq('complete').all()
    assert saved_png.stat().st_mtime_ns == before


@pytest.mark.parametrize('failed_request', ['rate', 'running_total'])
def test_actual_case_reader_propagates_login_failures(failed_request):
    notebook = json.loads((ROOT / 'notebooks/35_imerg_manual_case_review.ipynb').read_text())
    source = next(''.join(c['source']) for c in notebook['cells'] if 'def get_case(' in ''.join(c['source']))
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'get_case')
    row = cases(1).iloc[0]
    def fail_login(*args):
        require_earthdata_auth(None)
    namespace = dict(np=np, pd=pd, re=__import__('re'), DATE_COLUMNS=('event_start', 'event_end', 'event_peak'),
        VORTEX_LABEL_SCHEME='test', POLYGONS={},
        accumulation_bounds=lambda *args: (row.event_start, row.event_end),
        load_era5=lambda *args: None,
        load_imerg=fail_login if failed_request == 'rate' else lambda *args: (None, None, {'imerg_status':'complete'}),
        load_imerg_interval=fail_login,
        window_bounds=lambda hour, hours: (hour, hour + pd.Timedelta(hours=hours)),
        vortex_case_metadata=lambda *args: {}, vortex_plot_label=lambda *args: 'test')
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'notebook35_get_case', 'exec'), namespace)
    with pytest.raises(EarthdataLoginRequired):
        namespace['get_case'](row)
