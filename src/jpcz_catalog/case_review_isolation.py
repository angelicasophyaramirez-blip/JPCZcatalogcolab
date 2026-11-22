"""Run each Notebook 35 phase in a fresh interpreter, exchanging metadata only."""
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
import zipfile
from io import BytesIO

REVIEW_WORKER_SHA256 = ''  # Filled by the notebook updater from the exact helper source.
REVIEW_WORKER_URL = ''
REVIEW_WORKER_SETTINGS = (
    'DRIVE_ROOT', 'OUTPUT_DIR', 'CACHE_DIR', 'REVIEW_PATH', 'DOMAIN', 'POLYGONS',
    'ERA5_STORE', 'IMERG_PRODUCT', 'IMERG_VERSION', 'MIN_PIXEL_TEMPORAL_COVERAGE',
    'VORTEX_SCREEN_SETTINGS', 'VORTEX_LABEL_SCHEME', 'SHOW_CYCLONE_CENTERS',
    'PRECIP_LEVELS', 'ACCUM_LEVELS', 'DIV_LEVELS', 'VORT_LEVELS',
)


class ReviewWorkerError(RuntimeError):
    """Stop without declaring a phase complete if its isolated process fails."""


def prepare_review_worker(digest, url, archive_path=None):
    """Install only two known files from a content-verified, versioned archive."""
    root = Path(tempfile.gettempdir()) / 'jpcz_review_workers' / digest
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'runtime.zip'
    if archive_path is not None:
        payload = Path(archive_path).read_bytes()
    elif archive.exists():
        payload = archive.read_bytes()
    else:
        with urlopen(url, timeout=60) as response:
            payload = response.read()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ReviewWorkerError('Case-review worker checksum mismatch. Reload the updated notebook.')
    archive.write_bytes(payload)
    with zipfile.ZipFile(BytesIO(payload)) as bundle:
        if set(bundle.namelist()) != {'helpers.py', 'worker.py'}:
            raise ReviewWorkerError('Unexpected case-review worker archive contents.')
        for name in ('helpers.py', 'worker.py'):
            (root / name).write_bytes(bundle.read(name))
    return root / 'worker.py'


def make_review_phase_exporter(namespace, *, archive_path=None, digest=None, url=None, timeout=1200):
    """Create a lazy exporter; the parent never receives arrays or Matplotlib figures.

    Authentication is handed through stdin in memory, never command arguments or files.
    Each child uses the exact existing reader, Auto fallback, renderer, and Drive caches.
    """
    worker = None

    def export(row, files, **options):
        nonlocal worker
        namespace['require_earthdata_auth'](namespace.get('auth'))
        token = namespace['earthaccess'].get_edl_token()
        if isinstance(token, dict):
            token = token.get('access_token')
        if not isinstance(token, str) or not token:
            raise namespace['EarthdataLoginRequired']('Rerun the NASA authentication cell, then resume batch.')
        if worker is None:
            worker = prepare_review_worker(digest or REVIEW_WORKER_SHA256,
                url or REVIEW_WORKER_URL, archive_path)
        settings = {key: namespace[key] for key in REVIEW_WORKER_SETTINGS if key in namespace}
        payload = namespace['batch_json_value'](dict(row=row.to_dict(), files=files,
            options=options, settings=settings))
        payload['earthdata_token'] = token
        environment = os.environ.copy()
        environment.update(MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        # No credentials are inherited from the parent environment: stdin supplies this session only.
        for key in ('EARTHDATA_TOKEN', 'EARTHDATA_USERNAME', 'EARTHDATA_PASSWORD'):
            environment.pop(key, None)
        with tempfile.TemporaryDirectory(prefix='jpcz_review_reply_') as temporary:
            reply_path = Path(temporary) / 'reply.json'
            payload['reply_path'] = str(reply_path)
            process = subprocess.Popen([sys.executable, '-u', str(worker)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=environment)
            try:
                process.stdin.write(json.dumps(payload, allow_nan=False))
                process.stdin.close()
                del payload, token
                deadline = time.monotonic() + timeout
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        raise ReviewWorkerError('Map worker timed out. Saved phases remain reusable; rerun batch to retry.')
                    readable, _, _ = select.select([process.stdout], [], [], 1)
                    if readable:
                        line = process.stdout.readline()
                        if line:
                            print(line, end='', flush=True)
                for line in process.stdout:
                    print(line, end='', flush=True)
                if process.returncode or not reply_path.exists():
                    raise ReviewWorkerError(f'Map worker stopped (exit {process.returncode}). '
                        'The batch paused without marking this phase complete. Saved exports remain reusable.')
                reply = json.loads(reply_path.read_text())
                if reply.get('authentication_required'):
                    raise namespace['EarthdataLoginRequired'](reply['error'])
                if 'error' in reply:
                    raise ReviewWorkerError(reply['error'])
                return reply['metadata'], reply['status']
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                for stream in (process.stdin, process.stdout):
                    if stream and not stream.closed:
                        stream.close()
    return export
