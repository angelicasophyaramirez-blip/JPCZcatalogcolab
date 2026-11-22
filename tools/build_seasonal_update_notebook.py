"""Build the Colab update notebook and its small, versioned detector helper bundle."""
import ast
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'notebooks/38_update_winter_catalog.ipynb'

def cell(kind, source):
    source = textwrap.dedent(source).strip() + '\n'
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
    return result

sources = {'__init__': '"""Versioned Notebook 38 detector helpers."""\n'}
sources.update({name: (ROOT / 'src/jpcz_catalog' / f'{name}.py').read_text()
                for name in ('config', 'masks', 'detect', 'era5', 'historical_catalog', 'seasonal_update')})
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
    for name, source in sources.items():
        ast.parse(source)
        info = zipfile.ZipInfo(f'jpcz_update_runtime/{name}.py', date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        bundle.writestr(info, source)
bundle_bytes = buffer.getvalue()
bundle_hash = hashlib.sha256(bundle_bytes).hexdigest()
bundle_name = f'notebook38_runtime_{bundle_hash[:16]}.zip'
support_dir = ROOT / 'notebooks' / 'support'
support_dir.mkdir(exist_ok=True)
(support_dir / bundle_name).write_bytes(bundle_bytes)

loader = '''# @title 3 · Load detector helpers
import hashlib
import importlib
import sys
import tempfile
import urllib.request
from pathlib import Path

print('Loading detector helpers...', flush=True)
HELPER_URL = (
    'https://raw.githubusercontent.com/angelicasophyaramirez-blip/'
    'JPCZcatalogcolab/codex/notebook36-weathernext2/notebooks/support/'
    'BUNDLE_NAME'
)
HELPER_SHA256 = 'BUNDLE_HASH'
helper_file = Path(tempfile.gettempdir()) / 'BUNDLE_NAME'
if not helper_file.exists() or hashlib.sha256(helper_file.read_bytes()).hexdigest() != HELPER_SHA256:
    try:
        with urllib.request.urlopen(HELPER_URL, timeout=30) as response:
            helper_bytes = response.read()
    except Exception as error:
        raise RuntimeError('Could not download detector helpers. Check the connection and rerun cell 3.') from error
    if hashlib.sha256(helper_bytes).hexdigest() != HELPER_SHA256:
        raise RuntimeError('Detector helper verification failed. Reopen the updated notebook and rerun cell 3.')
    helper_file.write_bytes(helper_bytes)
sys.path.insert(0, str(helper_file))
for name in list(sys.modules):
    if name == 'jpcz_update_runtime' or name.startswith('jpcz_update_runtime.'):
        del sys.modules[name]
importlib.invalidate_caches()
from jpcz_update_runtime.seasonal_update import inspect_catalog, initialize_update, update_catalog
from jpcz_update_runtime.era5 import open_arco_era5
print('Detector helpers ready. Continue to cell 4.', flush=True)
'''.replace('BUNDLE_NAME', bundle_name).replace('BUNDLE_HASH', bundle_hash)

cells = [
cell('markdown', '''
# Notebook 38 · Find new winter JPCZ events and update the shared catalog

**This is the update entry point for Notebooks 35, 36, and 37.** Start here to search
for new events; you do not need to rerun the historical catalog or the old map batches.
Cell 3 loads a small versioned helper file and verifies its checksum. An old repository
clone or the repository's `main` settings cannot change the selected detector.

The established workflow is [Notebook 04: detector](https://colab.research.google.com/github/angelicasophyaramirez-blip/JPCZcatalogcolab/blob/codex/notebook36-weathernext2/notebooks/04_ndjf_catalog.ipynb)
then [Notebook 06: 12-hour fragment merge](https://colab.research.google.com/github/angelicasophyaramirez-blip/JPCZcatalogcolab/blob/codex/notebook36-weathernext2/notebooks/06_gap_merge_catalog.ipynb).
**Notebook 34 is a separate historical-baseline experiment**, not the rolling extension.

This version uses hourly **925-hPa ERA5 winds → MetPy grid divergence → cosine-latitude
polygon mean → trailing 12-hour mean → D12 strictly below a frozen month-specific
mean − 2 population standard deviations → 12-hour fragment merge**. It preserves Notebook
04's 11-hour warmup at each month boundary. These are this project's Shinoda-style ERA5
implementation choices. Vortex labels are a separate review in Notebook 35, not a detection filter.

Your actual Drive CSV supplies the count (whether it is 275 or another number). Old events
and threshold files are backed up before the first update. New thresholds are never fitted to
the new winter. Winds stay lazy until regional, 925-hPa, short time batches are selected.
Scans save compact scalar progress to Drive within each month. Only continuously verified
hours enter the updated catalog; archive gaps and later dates remain explicitly pending.
'''),
cell('code', '''
# @title 1 · Install dependencies
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'numpy>=1.26,<2.4', 'pandas>=2.2,<3', 'xarray>=2026.4,<2027',
    'zarr>=3.1,<4', 'dask[array]>=2025.1,<2027', 'gcsfs>=2025.3,<2027',
    'matplotlib>=3.9,<3.11', 'metpy>=1.6,<2', 'h5netcdf', 'scipy'], check=True)
'''),
cell('code', '''
# @title 2 · Mount Drive and set the update window
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import pandas as pd
from IPython.display import display
DRIVE_ROOT = Path('/content/drive/MyDrive/JPCZcatalog_outputs')
UPDATE_START_UTC = ''  # Optional first-run month boundary; otherwise infer from a complete saved month.
UPDATE_THROUGH_UTC = ''  # Inclusive UTC hour; blank means now. Example: '2026-11-19T23:00:00Z'.
ALLOW_ERA5T = True  # Preliminary reference; check the archive revision before reusing calculations.
CHUNK_HOURS = 6  # Lower to 1 or 2 for a smaller processing batch.
CHECKPOINT_EVERY_HOURS = 24  # Data hours, not wall-clock hours; use a multiple of CHUNK_HOURS.
'''),
cell('code', loader),
cell('markdown', '''
## Inspect before scanning

The preview reads only your Drive inventory. It prints the actual count, latest event,
and whether the last event's month has a complete detector checkpoint. The last event alone
cannot prove data coverage. If inference is unavailable, enter the first unscanned calendar
month in `UPDATE_START_UTC` after checking your saved Notebook 04 coverage.

Keep the saved start date on subsequent runs; extend only the end. If the threshold CSV is
missing, recovery requires every complete saved NDJF month in the established catalog years.
If those files are absent, restore `ndjf_monthly_thresholds.csv` from the original run.
'''),
cell('code', '''
# @title 4 · Inspect the established inventory and archive availability
import json
audit = inspect_catalog(DRIVE_ROOT)
print(json.dumps(audit, indent=2))
era5 = open_arco_era5(chunks=None)
print({key: era5.attrs.get(key) for key in ('valid_time_start', 'valid_time_stop',
                                          'valid_time_stop_era5t', 'last_updated')})
'''),
cell('markdown', '''
## Scan and save new events

Run this cell to update `jpcz_catalog_ndjf.csv` and `jpcz_catalog_ndjf_merged_12h.csv`,
the same filenames used by 35–37. The immutable original copies and frozen detector settings
are in `seasonal_updates/`. `event_changes.csv` identifies new/revised episodes.
Existing event numbers survive exact start/end/peak matches. A changed episode gets a new
number and needs a new review. Unchecked events remain an **Unknown** verification stratum.
New rows do not inherit the old monsoon or higher-vorticity classifications.

An event still touching the scan boundary, or one detected from ERA5T, is marked provisional.
Its end/peak may change on a later scan. The next run reconstructs the extension and merges
it with the frozen baseline, rather than appending duplicate rows.

ARCO's published data stop limits what can be requested; individual wind chunks can still
be missing inside that range. Coordinates and metadata alone do not prove fields are populated.
An event on November 19 will wait for reference data; missing days are reported
as pending. [ARCO documentation](https://github.com/google-research/arco-era5#overview) describes
roughly one week of ERA5T delay and about three months for final ERA5.

**RAM and crash recovery:** the default processes six data hours at a time, limits Zarr
read concurrency, and saves hourly polygon means to Drive after every 24 processed data
hours (and at the end of a partial month). This interval is data coverage, not 24 hours
of runtime. Completed monthly D12 series, frozen thresholds, backups, and CSVs also persist
under `JPCZcatalog_outputs/seasonal_updates/`. After a disconnect, reconnect Drive and rerun
the notebook. It validates saved progress and resumes from the last checkpoint.

For ERA5T, resuming requires the same published `last_updated` and availability metadata.
A changed archive revision uses fresh partial files; final ERA5 also replaces preliminary
calculations. If `last_updated` is absent, progress is still saved, but it cannot safely be
reused in another run. No full month of wind maps is retained in memory. Actual peak RAM
still depends on the archive's compressed source chunks and the Colab runtime.

**Missing wind hours:** the scan stops at the first nonfinite polygon calculation caused
by missing winds, saves the preceding finite hourly means immediately, and records the
exact time and counts in `seasonal_updates/pending_data.csv`. A sufficiently long continuous
prefix can update the catalog; remaining dates stay pending. A gap is never filled with zero,
skipped, or crossed by a rolling mean. If fewer than 12 good hours are available, or a changed
archive would reduce previously saved coverage, the existing catalog is kept. A numerical
failure with finite wind inputs still raises an error for investigation.

Rerun to retry the missing hour after the archive is filled. Checkpoints from the earlier
notebook version containing a nonfinite tail are automatically trimmed to their finite
prefix. Keep your Drive backup/checkpoint folders; no manual deletion is needed.
'''),
cell('code', '''
# @title 5 · Run the seasonal update and save the shared catalog
# Reopen metadata on every retry so newly published/revised hours are visible.
if 'era5' in globals():
    era5.close()
era5 = open_arco_era5(chunks=None)
manifest = initialize_update(DRIVE_ROOT, UPDATE_START_UTC or None)
print('Frozen update start:', manifest['update_start'])
print('Threshold source:', manifest['threshold_source'])
result = update_catalog(DRIVE_ROOT, era5, update_start=UPDATE_START_UTC or None,
                        requested_end=UPDATE_THROUGH_UTC or None, allow_era5t=ALLOW_ERA5T,
                        chunk_hours=CHUNK_HOURS, checkpoint_every_hours=CHECKPOINT_EVERY_HOURS)
print(json.dumps(result, indent=2))
status_path = DRIVE_ROOT / 'seasonal_updates/monthly_scan_status.csv'
if status_path.exists():
    display(pd.read_csv(status_path))
pending_path = DRIVE_ROOT / 'seasonal_updates/pending_data.csv'
if pending_path.exists():
    print('These dates are pending input data; they are not verified non-events:')
    display(pd.read_csv(pending_path))
'''),
cell('code', '''
# @title 6 · Search updated events by UTC date
SEARCH_START_UTC = ''  # Example: '2026-11-19'
SEARCH_END_UTC = ''  # Exclusive UTC boundary; e.g. '2026-11-20' for the 19th only.
from jpcz_update_runtime.seasonal_update import read_events
current = read_events(DRIVE_ROOT / 'jpcz_catalog_ndjf_merged_12h.csv')
found = current.copy()
if SEARCH_START_UTC:
    found = found.loc[found.event_end.ge(pd.Timestamp(SEARCH_START_UTC))]
if SEARCH_END_UTC:
    found = found.loc[found.event_start.lt(pd.Timestamp(SEARCH_END_UTC))]
print(f'{len(current)} total events; {len(found)} match the date search.')
columns = ['event_number', 'event_start', 'event_peak', 'event_end',
           'event_peak_D_1e5_s-1', 'catalog_provisional']
display(found[[column for column in columns if column in found]])
changes = DRIVE_ROOT / 'seasonal_updates/event_changes.csv'
if changes.exists():
    updates = pd.read_csv(changes)
    display(updates.loc[updates.update_status.ne('unchanged')])
'''),
cell('markdown', '''
## Carry the new cases into the figures

1. **Notebook 35:** rerun the catalog cell or click **Refresh catalog**. Screen new cases for
   vortex association; missing/failed screens stay Unknown in Notebook 37's strata. IMERG
   Final and satellite data have their own availability; they do not gate ensemble verification.
2. **Notebook 36:** rerun the inventory cell, then recreate the browser. Select a new event and
   check data. Exact 1-, 5-, and 10-day runs are requested relative to its onset target.
3. **Notebook 37:** choose **Catalog**, leave the end date open, reload the inventory, and run
   verification. Completed WN2 checkpoints are reused, missing runs are retried, ERA5T is
   refreshed, and all ranks/figures are regenerated from the current catalog. Use the regular
   winter schedule for broader convergence reliability; it must also include non-events.

If WN2 does not contain a new date, it stays missing in the status table until that exact
WN2 initialization/lead becomes available. An optional explicit WN2 store override is supported.
A different WeatherNext model is not substituted. Updating CSVs does not run Colab notebooks
in the background: rerun these stages when you want new figures.
'''),
]
for index, item in enumerate(cells):
    item['id'] = f'jpcz-update-{index:02d}'
    if item['cell_type'] == 'code' and ''.join(item['source']).startswith('# @title 3'):
        item['metadata']['cellView'] = 'form'
TARGET.write_text(json.dumps({'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {'colab': {'name': TARGET.name, 'provenance': []},
                 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                 'language_info': {'name': 'python', 'version': '3.12'}},
    'cells': cells}, indent=1, ensure_ascii=False) + '\n')
print(TARGET)
print(f'Helper bundle: {bundle_name} ({len(bundle_bytes):,} bytes)')
