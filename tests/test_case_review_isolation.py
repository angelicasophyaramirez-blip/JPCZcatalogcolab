"""Fresh processes, bounded retained buffers, failed-worker recovery, and private login handoff."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jpcz_catalog import case_review_batch as batch
from jpcz_catalog import case_review_isolation as isolation
from test_case_review_batch import cases


def worker_archive(tmp_path):
    # Retain a deliberately large buffer until process exit, as a decoder/cache might do.
    # No NASA or cloud calls are made by this synthetic integration fixture.
    helpers = (ROOT / 'src/jpcz_catalog/case_review_batch.py').read_text() + '''
import os, time
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DATE_COLUMNS = BATCH_DATE_COLUMNS
KEPT = []
class FakeEarthaccess:
    @staticmethod
    def login(strategy, persist):
        assert strategy == 'environment' and persist is False
        assert os.environ.get('EARTHDATA_TOKEN')
        return SimpleNamespace(authenticated=True)
earthaccess = FakeEarthaccess()
def get_case(row, **options):
    assert auth.authenticated
    assert 'EARTHDATA_TOKEN' not in os.environ
    if (CACHE_DIR/'fail_peak').exists() and options['offset'] == 0:
        raise SystemExit(7)
    if (CACHE_DIR/'memory_error').exists():
        raise MemoryError('synthetic decoder allocation failure')
    if (CACHE_DIR/'sleep').exists():
        time.sleep(30)
    KEPT.append(np.ones(64*1024**2, dtype=np.uint8))
    return {'metadata':dict(imerg_status='complete', accumulation_status='complete',
        satellite_status='available_daily_mosaic',cloud_rgb_satellite_status='available_daily_mosaic',
        satellite_platform='MODIS Aqua', view_hour_utc=row.event_peak+pd.Timedelta(hours=options['offset']))}
def plot_cases(cases, show):
    assert show is False
    fig, ax = plt.subplots(figsize=(3,2))
    ax.text(.1,.5,'Synthetic phase')
    return fig
'''
    payload = BytesIO()
    with zipfile.ZipFile(payload, 'w') as archive:
        archive.writestr('helpers.py', helpers)
        archive.writestr('worker.py', (ROOT / 'src/jpcz_catalog/case_review_worker.py').read_text())
    path = tmp_path / 'worker.zip'
    path.write_bytes(payload.getvalue())
    return path, hashlib.sha256(payload.getvalue()).hexdigest()


def exporter(tmp_path, **kwargs):
    archive, digest = worker_archive(tmp_path)
    namespace = dict(batch.__dict__, CACHE_DIR=tmp_path,
        auth=SimpleNamespace(authenticated=True),
        earthaccess=SimpleNamespace(get_edl_token=lambda: {'access_token':'test-secret-not-for-files'}))
    return isolation.make_review_phase_exporter(namespace, archive_path=archive, digest=digest, **kwargs)


def run(tmp_path, export, count=1):
    def forbid(*args, **kwargs):
        raise AssertionError('Parent process must not load or render case data')
    return batch.run_case_review_batch(cases(count), tmp_path, load_case=forbid, render_cases=forbid,
        label_for=lambda row:'Vortex-associated', export_phase=export, progress=lambda _:None)


def test_fresh_worker_each_phase_no_parent_arrays_and_same_checkpoints(tmp_path):
    result = run(tmp_path, exporter(tmp_path), count=2)
    assert result.status.eq('complete').all()
    assert result.worker_pid.nunique() == 6
    # Six retained 64-MiB buffers would accumulate 384 MiB in a persistent worker.
    assert result.worker_peak_rss_mib.max() - result.worker_peak_rss_mib.min() < 96
    assert len(result) == 6 and result.event_pdf.map(lambda x:Path(x).exists()).all()
    def forbid(*args, **kwargs):
        raise AssertionError('Completed phases must be reused without starting another worker')
    resumed = run(tmp_path, forbid, count=2)
    assert resumed.worker_pid.tolist() == result.worker_pid.tolist()
    for file in tmp_path.rglob('*'):
        if file.is_file():
            assert b'test-secret-not-for-files' not in file.read_bytes()


def test_failed_worker_does_not_complete_phase_and_resume_preserves_previous_png(tmp_path):
    flag = tmp_path / 'fail_peak'
    flag.touch()
    with pytest.raises(isolation.ReviewWorkerError, match='exit 7'):
        run(tmp_path, exporter(tmp_path))
    root = next((tmp_path / 'batch_evolution').iterdir())
    manifest = pd.read_csv(root / 'batch_manifest.csv')
    assert manifest.offset_hours.tolist() == [-12]
    saved = Path(manifest.png.iloc[0])
    before = saved.stat().st_mtime_ns
    assert json.loads((root / 'batch_summary.json').read_text())['run_status'] == 'paused_worker'
    assert not list(root.glob('event_*/peak.json'))
    flag.unlink()
    resumed = run(tmp_path, exporter(tmp_path))
    assert resumed.status.eq('complete').all()
    assert saved.stat().st_mtime_ns == before


def test_memory_error_pauses_without_failed_placeholder_or_completion_record(tmp_path):
    (tmp_path / 'memory_error').touch()
    with pytest.raises(isolation.ReviewWorkerError, match='out of memory'):
        run(tmp_path, exporter(tmp_path))
    assert not list(tmp_path.glob('batch_evolution/**/minus12h.json'))
    assert not list(tmp_path.glob('batch_evolution/**/*.png'))


def test_timeout_and_interrupt_terminate_child(tmp_path, monkeypatch):
    (tmp_path / 'sleep').touch()
    children = []
    original = isolation.subprocess.Popen
    def capture(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(isolation.subprocess, 'Popen', capture)
    with pytest.raises(isolation.ReviewWorkerError, match='timed out'):
        run(tmp_path, exporter(tmp_path, timeout=.1))
    assert all(child.poll() is not None for child in children)
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(isolation.select, 'select', interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, exporter(tmp_path))
    assert len(children) == 2 and all(child.poll() is not None for child in children)


def test_archive_checksum_is_checked_before_executing_code(tmp_path):
    path, digest = worker_archive(tmp_path)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(isolation.ReviewWorkerError, match='checksum mismatch'):
        isolation.prepare_review_worker(digest, '', path)


def test_embedded_bundle_matches_current_reader_renderer_and_math():
    import ast
    nb = json.loads((ROOT/'notebooks/35_imerg_manual_case_review.ipynb').read_text())
    source = next(''.join(c['source']) for c in nb['cells'] if 'def get_case(' in ''.join(c['source']))
    assignments = {node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ('REVIEW_WORKER_SHA256', 'REVIEW_WORKER_URL')}
    path = ROOT/'notebooks/support'/assignments['REVIEW_WORKER_URL'].rsplit('/',1)[-1]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == assignments['REVIEW_WORKER_SHA256']
    with zipfile.ZipFile(path) as archive:
        assert archive.read('helpers.py').decode() == source.split('# --- Isolated batch map phases ---')[0].rstrip()+'\n'
        assert archive.read('worker.py').decode() == (ROOT/'src/jpcz_catalog/case_review_worker.py').read_text()
