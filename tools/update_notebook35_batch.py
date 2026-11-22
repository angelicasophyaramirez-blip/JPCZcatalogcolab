"""Install the resumable all-event, three-time atlas into standalone Notebook 35."""
import ast
import json
import re
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'notebooks/35_imerg_manual_case_review.ipynb'
notebook = json.loads(TARGET.read_text())
module = (ROOT / 'src/jpcz_catalog/case_review_batch.py').read_text()
marker = '# --- Resumable three-time batch exports ---'
wrapper = '''def run_batch(frame, hours=1, limit=None, offsets=(-12, 0, 12), satellite_platform='Auto',
              satellite_day_offset=0, accumulation_mode='event_start'):
    return run_case_review_batch(
        frame, OUTPUT_DIR, load_case=get_case,
        check_auth=lambda: require_earthdata_auth(globals().get('auth')),
        export_phase=make_review_phase_exporter(globals()),
        render_cases=lambda cases: plot_cases(cases, show=False), label_for=vortex_plot_label,
        refresh_event=lambda row: attach_vortex_flags(pd.DataFrame([row])).iloc[0],
        hours=hours, limit=limit, offsets=offsets, satellite_platform=satellite_platform,
        satellite_day_offset=satellite_day_offset, accumulation_mode=accumulation_mode,
        settings=dict(domain=DOMAIN, polygons=POLYGONS, era5_store=ERA5_STORE,
                      imerg_product=IMERG_PRODUCT, imerg_version=IMERG_VERSION,
                      imerg_min_coverage=MIN_PIXEL_TEMPORAL_COVERAGE,
                      label_scheme=VORTEX_LABEL_SCHEME, vortex_signature=vortex_signature()))'''

for item in notebook['cells']:
    source = ''.join(item.get('source', []))
    if item['cell_type'] == 'code' and 'def get_case(' in source:
        source = source.split(marker, 1)[0].rstrip() + '\n'
        node = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'run_batch')
        source = source.replace(ast.get_source_segment(source, node), wrapper, 1)
        # Protect individual browser requests too, but allow fully cached IMERG reads offline.
        loader = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'load_imerg_interval')
        old_loader = ast.get_source_segment(source, loader)
        guarded_loader = old_loader.replace(
            "    if missing:\n        print(f'IMERG:",
            "    if missing:\n        require_earthdata_auth(globals().get('auth'))\n        print(f'IMERG:", 1)
        source = source.replace(old_loader, guarded_loader, 1)
        case_node = next(node for node in ast.parse(source).body
                         if isinstance(node, ast.FunctionDef) and node.name == 'get_case')
        old_case = ast.get_source_segment(source, case_node)
        guarded_case = old_case
        for indent in ('    ', '            '):
            anchor = indent + 'except Exception as error:\n'
            old_check = indent + "    if getattr(error, 'requires_earthdata_login', False):\n" + indent + '        raise\n'
            check = indent + "    if isinstance(error, MemoryError) or getattr(error, 'requires_earthdata_login', False):\n" + indent + '        raise\n'
            # Idempotent: never swallow a session failure as an ordinary data gap.
            guarded_case = re.sub('^' + re.escape(anchor + old_check), anchor, guarded_case, flags=re.M)
            guarded_case = re.sub('^' + re.escape(anchor + check), anchor, guarded_case, flags=re.M)
            guarded_case = re.sub('^' + re.escape(anchor), anchor + check, guarded_case, flags=re.M)
        source = source.replace(old_case, guarded_case, 1)
        source = source.replace('def plot_cases(cases, stem=None):',
                                'def plot_cases(cases, stem=None, *, show=True):')
        source = source.replace('\n    plt.show()\n    return fig', '\n    if show:\n        plt.show()\n    return fig')
        source += '\n\n' + marker + '\n' + module
        if not source.startswith('# @title'):
            source = '# @title Load case-review helpers (run unchanged)\n' + source
        item['metadata']['cellView'] = 'form'
    elif item['cell_type'] == 'code' and "'earthaccess', 'ipywidgets'" in source:
        source = source.replace(", 'pypdf>=5,<7'", '')
        source = source.replace("'earthaccess', 'ipywidgets'", "'earthaccess', 'ipywidgets', 'pypdf>=5,<7'")
    elif item['cell_type'] == 'code' and 'RUN_BATCH = ' in source:
        source = textwrap.dedent('''
        # @title Export all current events at −12 h, peak, and +12 h
        RUN_BATCH = False  # Change to True, then run this cell to export the whole catalog.
        EVENTS_PER_RUN = None  # None = all remaining events; use 1 for a one-event pilot.
        BATCH_WINDOW_HOURS = 1  # 1 = hourly maps; 12 = trailing 12-hour divergence/rate at each phase.
        BATCH_SATELLITE_PLATFORM = 'Auto'  # Same as the browser: Terra first, then Aqua if unavailable.
        BATCH_SATELLITE_DAY_OFFSET = 0  # Same Japanese calendar date as each weather time.
        BATCH_ACCUMULATION_MODE = 'event_start'  # Running total from event start; 'window' uses the selected window.
        if RUN_BATCH:
            reload_catalog_inventory()
            batch_results = run_batch(
                catalog, hours=BATCH_WINDOW_HOURS, limit=EVENTS_PER_RUN,
                satellite_platform=BATCH_SATELLITE_PLATFORM,
                satellite_day_offset=BATCH_SATELLITE_DAY_OFFSET,
                accumulation_mode=BATCH_ACCUMULATION_MODE)
            print('Batch exports:', batch_results.attrs['output_dir'])
            columns = ['event_number', 'offset_hours', 'view_hour_utc', 'satellite_platform', 'status', 'event_pdf']
            display(batch_results[[column for column in columns if column in batch_results]])
        else:
            print('Batch is off. Set RUN_BATCH = True to export all cases at −12 h, peak, and +12 h.')
        ''').strip() + '\n'
    elif item['cell_type'] == 'markdown' and source.startswith(('## Optional resumable atlas', '## Resumable batch:')):
        source = textwrap.dedent('''
        ## Resumable batch: every event at −12 h, peak, and +12 h

        Set **RUN_BATCH = True** below. The default **EVENTS_PER_RUN = None** processes all
        remaining current catalog events. A pilot limit of 1 processes all three times for
        one event. Times are relative to each event's peak; the browser slider does not set them.

        Run the **NASA authentication** cell in every new runtime first, and wait for its
        authenticated message. Missing login stops the batch before data reads; sign in and
        rerun with the same settings to reuse completed exports and retry partial phases.

        **Auto satellite selection matches the browser:** try Terra, then Aqua on the same
        requested date if Terra imagery is unavailable/blank. Titles and CSVs identify the
        selected platform. Dates before both archives retain an explicit unavailable panel.
        Daily daytime mosaics provide context rather than exact −12/0/+12-hour satellite views.

        All exports stay together under **MyDrive/JPCZcatalog_outputs/imerg_manual_case_review/batch_evolution/**.
        A folder for the chosen settings contains a folder for each event, with:

        - **all_three_times.pdf:** three pages in order: −12 h, peak, +12 h.
        - **minus12h / peak / plus12h:** a six-panel PNG, single-page PDF, and metadata CSV for each time.
        - **all_three_times_metadata.csv:** the three records for that event.

        The batch folder also contains **all_cases_metadata.csv**, **batch_manifest.csv**,
        **batch_summary.json**, and the settings. Inspect the manifest for missing data or failures.
        Phase PDFs remain present as labeled error pages if a phase cannot be loaded; they are
        marked failed and retried on the next run. No missing data are represented as zero.

        Matching completed phase exports are reused. Partial/failed phases retry while the batch
        continues to later events. Progress is saved after every phase. A disconnect can be
        resumed by running setup and this cell with the same settings and Drive folder.
        Underlying ERA5, IMERG, satellite, and vortex caches are reused. One figure is rendered
        at a time in a fresh Python process. That process exits after saving its PNG/PDF,
        releasing reader and plotting memory before the next phase. The notebook keeps only
        metadata and checkpoints. A worker failure pauses the batch without marking the phase
        complete; cached inputs and completed exports remain reusable.
        CSVs and figures describe the catalog, without changing it or its saved reviews.
        ''').strip() + '\n'
    if item['cell_type'] == 'code':
        ast.parse(source)
    item['source'] = source.splitlines(keepends=True)
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
print(TARGET)
memory_updater = ROOT / 'tools/update_notebook35_memory.py'
if memory_updater.exists():
    subprocess.run([sys.executable, str(memory_updater)], check=True)
subprocess.run([sys.executable, str(ROOT / 'tools/update_notebook35_isolation.py')], check=True)
