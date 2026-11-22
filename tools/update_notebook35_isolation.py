"""Bundle exact Notebook 35 helpers and enable a fresh interpreter for every phase."""
import ast
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / 'notebooks/35_imerg_manual_case_review.ipynb'
notebook = json.loads(target.read_text())
marker = '# --- Isolated batch map phases ---'
cell = next(cell for cell in notebook['cells'] if 'def get_case(' in ''.join(cell.get('source', [])))
source = ''.join(cell['source']).split(marker, 1)[0].rstrip() + '\n'
buffer = BytesIO()
with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for name, text in [('helpers.py', source),
                       ('worker.py', (ROOT / 'src/jpcz_catalog/case_review_worker.py').read_text())]:
        info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, text.encode())
payload = buffer.getvalue()
digest = hashlib.sha256(payload).hexdigest()
relative = f'notebooks/support/notebook35_worker_{digest[:16]}.zip'
archive_path = ROOT / relative
archive_path.parent.mkdir(exist_ok=True)
archive_path.write_bytes(payload)
source += '\n' + marker + '\n' + (ROOT / 'src/jpcz_catalog/case_review_isolation.py').read_text()
source += f'\nREVIEW_WORKER_SHA256 = {digest!r}\n'
source += 'REVIEW_WORKER_URL = ' + repr(
    'https://raw.githubusercontent.com/angelicasophyaramirez-blip/JPCZcatalogcolab/'
    'codex/notebook36-weathernext2/' + relative) + '\n'
ast.parse(source)
cell['source'] = source.splitlines(keepends=True)
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
print(f'{relative}: {len(payload)} bytes, SHA256 {digest}')
