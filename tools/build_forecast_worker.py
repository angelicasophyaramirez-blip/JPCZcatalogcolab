"""Create the deterministic, checksum-verified Notebook 36/37 worker bundle."""
import hashlib
from io import BytesIO
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def worker_bootstrap_source():
    package=ROOT/'src/jpcz_catalog'
    files={'worker.py':(package/'forecast_batch_worker.py').read_text(),'jpcz_catalog/__init__.py':'',
           'jpcz_catalog/weathernext_atlas.py':(package/'weathernext_atlas.py').read_text(),
           'jpcz_catalog/ensemble_verification.py':(package/'ensemble_verification.py').read_text()}
    payload=BytesIO()
    with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,source in sorted(files.items()):
            entry=zipfile.ZipInfo(name,(2026,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,source.encode())
    data=payload.getvalue();digest=hashlib.sha256(data).hexdigest()
    relative=f'notebooks/support/forecast_worker_{digest[:16]}.zip'
    (ROOT/relative).write_bytes(data)
    source=(package/'forecast_batch_runtime.py').read_text()
    source+=f'\nFORECAST_WORKER_SHA256 = {digest!r}\n'
    source+='FORECAST_WORKER_URL = '+repr('https://raw.githubusercontent.com/angelicasophyaramirez-blip/'
        'JPCZcatalogcolab/codex/notebook36-weathernext2/'+relative)+'\n'
    return source
