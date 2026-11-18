"""Build the standalone Colab notebook for JPCZ ensemble verification."""

import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/jpcz_catalog"
TARGET = ROOT / "notebooks/37_weathernext2_ensemble_verification.ipynb"


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
        if source.startswith("# @title"):
            result["metadata"]["cellView"] = "form"
    return result


atlas = (PACKAGE / "weathernext_atlas.py").read_text().split("# --- Interactive browser ---", 1)[0]
verification = (PACKAGE / "ensemble_verification.py").read_text()
# Notebook 37 loads these functions in its preceding cell.
imports = next(node for node in ast.parse(verification).body
               if isinstance(node, ast.ImportFrom) and node.module == "weathernext_atlas")
verification = verification.replace(ast.get_source_segment(verification, imports), "", 1)

cells = [
    cell("markdown", """
    # Notebook 37 · WeatherNext 2 JPCZ ensemble verification

    Compare **all 64 members** with ERA5 for **925-hPa horizontal divergence** in the offshore
    JPCZ polygon and coastal wedge, at **1, 5, and 10 days**. Negative values indicate
    convergence. The two polygons are evaluated independently, including their overlap.

    **Catalog** mode processes all selected events. **Pilot** mode processes one event and
    reports six ERA5 ranks. Each event contributes one rank per polygon and lead. Synced Drive
    checkpoints allow each completed ERA5 target and forecast lead to be reused after a
    runtime disconnect.

    Run sections 1–5 in this notebook. Its archive reader and rank calculation are included
    below. Each helper cell prints a confirmation when it finishes.

    The catalog is the same CSV Notebook 35 reads. Notebook 36 supplies the archive reader and
    onset timing rule. You can run this notebook directly using that CSV and your approved
    WeatherNext account. The plotting batches in Notebooks 35 and 36 are optional.

    [WeatherNext 2 data and attribution](https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_weathernext_2_0_0)
    · [Rank histogram method, Gneiting et al. (2007)](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf)
    · [Interpretation and case selection, Hamill (2001)](https://doi.org/10.1175/1520-0493%282001%29129%3C0550%3AIORHFV%3E2.0.CO%3B2)
    """),
    cell("code", """
    # @title 1 · Install dependencies in a fresh CPU runtime
    import subprocess
    import sys

    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'numpy>=1.26,<2.4', 'pandas>=2.2,<3', 'xarray>=2026.4,<2027',
                    'zarr>=3.1,<4', 'gcsfs>=2025.3,<2027', 'dask[array]>=2025.1,<2027',
                    'matplotlib>=3.9,<3.11', 'xlsxwriter>=3.2,<4'], check=True)
    """),
    cell("code", """
    # @title 2 · Connect Google Drive and authenticate for WeatherNext
    from google.colab import auth, drive

    drive.mount('/content/drive')
    auth.authenticate_user()
    import google.auth
    credentials, detected_project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform'])
    print('Drive mounted and credentials acquired. Use the approved account for WN2.')
    """),
    cell("markdown", """
    ## 3 · Choose the cases

    The study dates match Notebook 36’s provisional 2023-onward window by default. The
    operational model’s training cutoff has not been verified. Pilot event **0** selects the
    first event in the study window. You can enter an event number that worked in Notebook 36
    instead. The inventory displayed below includes the numbers and timing offsets.

    The target is the nearest 00, 06, 12, or 18 UTC on the original onset’s UTC date, with
    ties going earlier. ERA5 is evaluated at that exact time. The catalog’s peak D12 is kept
    as context. If required by your Cloud account, enter your existing project ID.
    """),
    cell("code", """
    # @title 3a · Settings
    from pathlib import Path

    RUN_MODE = 'Catalog'  # @param ["Pilot", "Catalog"]
    PILOT_EVENT_NUMBER = 0  # @param {type:"integer"}
    PROJECT_ID = ''  # @param {type:"string"}
    DRIVE_ROOT = Path('/content/drive/MyDrive/JPCZcatalog_outputs')
    CATALOG_PATH = DRIVE_ROOT / 'jpcz_catalog_ndjf_merged_12h.csv'
    NOTEBOOK35_DIR = DRIVE_ROOT / 'imerg_manual_case_review'
    OUTPUT_DIR = DRIVE_ROOT / 'weathernext2_ensemble_verification'
    STUDY_START_UTC = '2023-01-01'  # @param {type:"date"}
    STUDY_END_UTC = ''  # @param {type:"string"} # Optional exclusive UTC date.
    RUN_OVERRIDES = {}
    """),
    cell("code", "# @title 3b · Load WeatherNext archive helpers (in this notebook)\n" + atlas
         + "\nprint('Archive helpers loaded. Continue to section 3c.')\n"),
    cell("code", "# @title 3c · Load rank and Excel helpers (in this notebook)\n" + verification
         + "\nprint('Verification helpers loaded. Continue to section 3d.')\n"),
    cell("code", """
    # @title 3d · Preview the selected catalog
    from IPython.display import display

    if not callable(globals().get('read_catalog')) or not callable(globals().get('run_ensemble_verification')):
        raise RuntimeError(
            'Run sections 3b and 3c in this Notebook 37 runtime, then rerun section 3d. '
            'Each helper cell prints a confirmation when it finishes.')
    catalog, selection_report = read_catalog(
        CATALOG_PATH, NOTEBOOK35_DIR, STUDY_START_UTC, STUDY_END_UTC or None)
    if catalog.empty:
        raise ValueError('The current catalog has no events in the configured study window.')
    if RUN_MODE not in ('Pilot', 'Catalog'):
        raise ValueError('Choose Pilot or Catalog.')
    if RUN_MODE == 'Pilot':
        selected = catalog.head(1) if PILOT_EVENT_NUMBER == 0 else catalog.loc[
            catalog.event_number.eq(PILOT_EVENT_NUMBER)]
        if selected.empty:
            raise ValueError('Pilot event not found in this study window. Pick a listed event number.')
        result_directory = OUTPUT_DIR / f'pilot_event_{int(selected.event_number.iloc[0]):04d}'
    else:
        selected = catalog
        result_directory = OUTPUT_DIR / 'catalog'
    archive = GoogleCloudArchive(credentials, PROJECT_ID or detected_project or '',
                                 overrides=RUN_OVERRIDES, lazy_indexing=True)
    print(f'{len(catalog)} events in the study window. {RUN_MODE} will process {len(selected)}.')
    if RUN_MODE == 'Pilot':
        print('Pilot contributes ONE ERA5 rank to each histogram. Choose Catalog to aggregate all selected events.')
    print('Result folder:', result_directory)
    display(catalog[['event_number', 'event_start', 'event_peak', 'target_valid_time',
                     'onset_offset_hours', 'vortex_plot_label']])
    """),
    cell("markdown", """
    ## 4 · Run the verification

    Both inputs use the same spherical divergence calculation with a two-grid-cell halo,
    and the same cosine-latitude weighting and polygon boundaries on their native grids.
    The fixed 925-hPa fields include any below-ground pressure-level values supplied by the
    datasets. The coastal wedge results should be interpreted with that in mind. The notebook
    reads WN2 members in source chunk groups and saves two means per member and lead.
    Archive arrays use lazy indexing. The notebook selects the exact time, pressure level,
    and region before reading one wind component at a time, with limited read concurrency.

    Spatial averages use every finite grid cell inside each polygon with area weights.
    Negative divergence represents convergence. Positive and negative values retain their
    signs when averaged and ranked, so the resulting value measures the net convergence
    or divergence of the whole polygon.

    Each ERA5 rank is one plus the number of member values below ERA5, with seeded random
    tie-breaking. Rank 1 means ERA5 was at the most convergent end and rank 65 at the least
    convergent end. The exact out-of-range flags are recorded separately from the rank.

    The batch status records missing or failed inputs. Rerunning retries those inputs and
    reuses completed checkpoints. Cases with ERA5 and all three forecast leads contribute
    to the six histograms so that the lead comparison uses the same events.

    After a runtime restart, run sections 1–3 again before section 4. Wait for the
    confirmations from both helper cells, 3b and 3c, and the catalog preview in 3d.
    Completed checkpoints are stored in Drive.
    """),
    cell("code", """
    # @title 4 · Read the selected forecasts and calculate ranks
    missing_setup = []
    if not callable(globals().get('read_catalog')):
        missing_setup.append('3b · Load WeatherNext archive helpers')
    if not callable(globals().get('run_ensemble_verification')):
        missing_setup.append('3c · Load rank and Excel helpers')
    if any(name not in globals() for name in ('selected', 'archive', 'OUTPUT_DIR')):
        missing_setup.append('3d · Preview the selected catalog')
    if missing_setup:
        raise RuntimeError(
            'Notebook 37 setup is incomplete in this runtime. Run these cells in order, '
            'wait for each to finish, then rerun section 4.\\n' + '\\n'.join(missing_setup)
            + '\\nIf the runtime restarted, first rerun sections 1, 2, and 3a. '
            'Completed Drive checkpoints will be reused.')
    tables, calculation_settings = run_ensemble_verification(selected, archive, OUTPUT_DIR)
    display(tables['batch_status'][['event_number', 'lead_days', 'era5_status', 'wn2_status',
                                    'era5_error', 'wn2_error']])
    ranks = tables['case_ranks']
    complete_events = 0 if ranks.empty else ranks.loc[ranks.in_common_sample, 'review_key'].nunique()
    print(f'{len(selected)} selected events; {complete_events} complete events contribute to EACH histogram.')
    print(f'{len(selected) - complete_events} events lack at least one required input. Details appear in Batch status.')
    if not ranks.empty:
        display(ranks[['event_number', 'valid_time_utc', 'polygon', 'lead_days', 'member_count',
                       'era5_divergence_1e5_s-1', 'era5_rank', 'in_common_sample']])
    """),
    cell("markdown", """
    ## 5 · Save and inspect results

    The Excel workbook opens in Excel or Google Sheets. The same tables are available as
    CSVs. The plots save as PNG and PDF. Pilots and the catalog have separate result folders.

    **Case ranks** lists ERA5's value, exact rank, bin number, and ensemble minimum, mean,
    and maximum for each event, polygon, and lead. **Member values** retains each forecast
    value and native member ID. **ERA5 reference** retains the verifying values.
    **Ordered values** is a flat table with all 65 entries per verifiable case, sorted from
    most negative divergence to most positive divergence. **Ordered bins** presents the same
    numbers in five columns of 13 entries, with ERA5 explicitly labeled and highlighted.
    **Bin counts** gives each polygon/lead's event count in each numbered bin.
    Values are signed divergence in 10⁻⁵ s⁻¹. Exact rankings use full numerical precision.
    Ties retain the same reproducibly chosen ERA5 position as the graphs.

    Both plot versions are exported. **65 ranks** shows all possible positions of ERA5
    within the ensemble. **5 groups** combines 13 consecutive ranks per bar for readability
    with a small event sample. Both use the same calculated ranks and the same set of
    complete events. The dashed line shows the expected counts for a uniform rank
    distribution. The counts in every panel sum to its displayed N.

    A single pilot event populates just one bar per panel. Catalog histograms describe
    cases selected around ERA5-detected JPCZ events.
    """),
    cell("code", """
    # @title 5 · Save workbook, tables, and six-panel rank histogram
    import matplotlib.pyplot as plt
    RANK_DISPLAY = '65 ranks'  # @param ["65 ranks", "5 groups"]

    tables['event_selection'] = selection_report
    calculation_settings = {
        **calculation_settings, 'catalog': str(CATALOG_PATH), 'study_start_utc': STUDY_START_UTC,
        'study_end_utc_exclusive': STUDY_END_UTC or None, 'run_mode': RUN_MODE,
        'selected_events': len(selected), 'lead_hours': list(LEAD_HOURS), 'tie_seed': 37,
        'training_cutoff_status': 'operational archive training cutoff unverified',
        'wn2_data_reference': (
            'Google LLC WeatherNext 2 historic experimental data (CC BY 4.0). '
            'https://developers.google.com/earth-engine/datasets/catalog/'
            'projects_gcp-public-data-weathernext_assets_weathernext_2_0_0'
        ),
    }
    export_verification(tables, calculation_settings, result_directory)
    print('Saved', result_directory / 'ensemble_verification.xlsx')
    print('CSV tables and both rank-histogram figures are saved in the same folder.')
    if not ranks.empty and ranks.in_common_sample.any():
        for grouped, filename in ((False, 'rank_histograms_65ranks'), (True, 'rank_histograms')):
            figure = plot_rank_histograms(ranks, grouped=grouped)
            for extension in ('png', 'pdf'):
                figure.savefig(result_directory / f'{filename}.{extension}', dpi=180)
            if grouped == (RANK_DISPLAY == '5 groups'):
                display(figure)
            plt.close(figure)
    else:
        print('No complete event for all three leads. Check Batch status, then retry section 4.')
    """),
    cell("markdown", """
    ## Follow-up · Reliability diagrams

    Reliability will compare a forecast probability of crossing a specified convergence
    threshold with a binary ERA5 outcome. Use independently scheduled winter times that
    include crossings and non-crossings. The ERA5 catalog detector uses a 12-hour average
    of hourly values. Before applying its criterion to WN2, check how well a six-hourly
    approximation reproduces the hourly ERA5 decisions. The rank workflow above uses a
    matched snapshot and has no convergence threshold.

    Forecast data: Google WeatherNext 2 historical experimental data, CC BY 4.0.
    Reference: ERA5 reanalysis via Google Research ARCO-ERA5.
    """),
]

for i, item in enumerate(cells):
    item["id"] = f"wn2-verify-{i:02d}"
notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"colab": {"name": TARGET.name, "provenance": []},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "cells": cells,
}
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(TARGET)
