"""Build a standalone, versioned extension of the established Notebook 04/06 detector."""
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'notebooks/38_update_winter_catalog.ipynb'

def cell(kind, source):
    source = textwrap.dedent(source).strip() + '\n'
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
    return result

sources = {name: (ROOT / 'src/jpcz_catalog' / f'{name}.py').read_text()
           for name in ('config', 'masks', 'detect', 'era5', 'historical_catalog', 'seasonal_update')}
loader = '''# @title 3 · Load the embedded detector (no Git branch checkout required)
import sys, types
sources = SOURCES
package_name = 'jpcz_update_runtime'
package = types.ModuleType(package_name)
package.__path__ = []
sys.modules[package_name] = package
for name, source in sources.items():
    qualified = package_name + '.' + name
    module = types.ModuleType(qualified)
    module.__package__ = package_name
    sys.modules[qualified] = module
    exec(compile(source, qualified, 'exec'), module.__dict__)
from jpcz_update_runtime.seasonal_update import inspect_catalog, initialize_update, update_catalog
from jpcz_update_runtime.era5 import open_arco_era5
print('Established hourly MetPy / trailing-D12 / frozen monthly thresholds / 12-hour merge loaded.')
'''.replace('SOURCES', repr(sources))

cells = [
cell('markdown', '''
# Notebook 38 · Find new winter JPCZ events and update the shared catalog

**This is the update entry point for Notebooks 35, 36, and 37.** Start here to search
for new events; you do not need to rerun the historical catalog or the old map batches.
The detector is embedded, so an old clone or the repository's `main` settings cannot change it.

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
the new winter. Scans checkpoint compact scalar series; a failed month cannot promote a partial catalog.
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
ALLOW_ERA5T = True  # Preliminary reference, explicitly marked and refreshed on rerun.
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

ARCO's published data stop controls what can be read. Coordinates alone do not prove fields
are populated. An event on November 19 will wait for reference data; missing days are reported
as pending. [ARCO documentation](https://github.com/google-research/arco-era5#overview) describes
roughly one week of ERA5T delay and about three months for final ERA5. Preliminary months
are reread on each run, and again when final data replace them.
'''),
cell('code', '''
# @title 5 · Run the seasonal update and save the shared catalog
manifest = initialize_update(DRIVE_ROOT, UPDATE_START_UTC or None)
print('Frozen update start:', manifest['update_start'])
print('Threshold source:', manifest['threshold_source'])
result = update_catalog(DRIVE_ROOT, era5, update_start=UPDATE_START_UTC or None,
                        requested_end=UPDATE_THROUGH_UTC or None, allow_era5t=ALLOW_ERA5T)
print(json.dumps(result, indent=2))
status_path = DRIVE_ROOT / 'seasonal_updates/monthly_scan_status.csv'
if status_path.exists():
    display(pd.read_csv(status_path))
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
TARGET.write_text(json.dumps({'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {'colab': {'name': TARGET.name, 'provenance': []},
                 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                 'language_info': {'name': 'python', 'version': '3.12'}},
    'cells': cells}, indent=1, ensure_ascii=False) + '\n')
print(TARGET)
