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

    **New winter events:** start with [Notebook 38 · update and search the catalog](https://colab.research.google.com/github/angelicasophyaramirez-blip/JPCZcatalogcolab/blob/codex/notebook36-weathernext2/notebooks/38_update_winter_catalog.ipynb).
    It extends the established Notebook 04/06 detector using frozen thresholds, then writes
    the same Drive CSV used here. Reload the catalog and rerun this notebook to update
    the ranks and reliability plots. Counts always come from that file; no 275-event limit.

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
    ALLOW_ERA5T = True  # Use and label preliminary references; refresh them on every rerun.
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
    # Clear stale archive metadata/listings when retrying a newly published date.
    open_verification_era5.cache_clear()
    archive.clear()
    tables, calculation_settings = run_ensemble_verification(
        selected, archive, OUTPUT_DIR, allow_era5t=ALLOW_ERA5T)
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
        'allow_era5t': ALLOW_ERA5T,
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
        for stem in ('rank_histograms_65ranks', 'rank_histograms'):
            for extension in ('png', 'pdf'):
                (result_directory / f'{stem}.{extension}').unlink(missing_ok=True)
        print('No complete event for all three leads. Check Batch status, then retry section 4.')
    """),
    cell("markdown", """
    ## 6 · Reliability diagrams and vortex strata

    For each polygon and lead, forecast probability is the fraction of the **64 members**
    whose instantaneous 925-hPa polygon-mean divergence is **strictly below** the configured
    threshold. ERA5 supplies a binary crossing/non-crossing at the same valid time.
    The threshold below is an **editable exploratory snapshot criterion**, in 10⁻⁵ s⁻¹.
    Choose it before evaluating the forecast sample. It is **not** the catalog's hourly
    trailing-D12 threshold, and these probabilities are not probabilities of the complete
    JPCZ detector event or vortex occurrence. Testing a six-hourly approximation to D12
    remains a separate scientific step.

    These first diagrams use the **ERA5-event-conditioned sample**, the same complete
    all-lead cases as the rank histograms. They describe convergence calibration among
    selected events, not winter-wide occurrence calibration. Section 7 supplies the
    independent winter schedule including non-events for broader assessment.

    Fixed probability bins include 0 and 1; empty bins remain in the saved tables but
    are not plotted as zero occurrence. Each diagram includes sample counts, base rate,
    Brier score, and forecast-count inset. Pointwise Wilson intervals are descriptive;
    serially related winter times are not independent uncertainty estimates.

    Separate All / Vortex / Non-vortex / Unknown outputs use Notebook 35's completed
    screen status. Unchecked or failed screens never enter the Non-vortex stratum.
    Changed review labels update the strata without redownloading the forecasts.
    """),
    cell("code", """
    # @title 6 · Refresh event-conditioned reliability and group rank plots
    SNAPSHOT_THRESHOLD_1E5 = -2.0  # @param {type:"number"} # Exploratory; choose before analysis.
    RELIABILITY_THRESHOLDS = {region: SNAPSHOT_THRESHOLD_1E5 for region in POLYGONS}
    PROBABILITY_BINS = 10
    for group in ('All', 'Vortex', 'Non-vortex', 'Unknown'):
        folder = result_directory / 'event_conditioned' / group.lower().replace('-', '_')
        probabilities, bins = export_reliability(
            tables, folder, RELIABILITY_THRESHOLDS, group=group, n_bins=PROBABILITY_BINS)
        group_ranks = verification_subset(tables, group)
        n = 0 if group_ranks.empty else group_ranks.loc[group_ranks.in_common_sample, 'review_key'].nunique()
        print(f'{group}: {n} complete events; results in {folder}')
        for grouped, stem in ((False, 'rank_histograms_65ranks'), (True, 'rank_histograms')):
            if n:
                rank_histogram_counts(group_ranks, grouped=grouped).to_csv(folder / f'{stem}_counts.csv', index=False)
                figure = plot_rank_histograms(group_ranks, grouped=grouped, title_suffix=group)
                for extension in ('png', 'pdf'):
                    figure.savefig(folder / f'{stem}.{extension}', dpi=180)
                plt.close(figure)
            else:
                for extension in ('png', 'pdf', '_counts.csv'):
                    stale = folder / (f'{stem}{extension}' if extension.startswith('_') else f'{stem}.{extension}')
                    stale.unlink(missing_ok=True)
        if group == 'All' and not probabilities.empty:
            figure = plot_reliability_diagrams(probabilities, n_bins=PROBABILITY_BINS)
            display(figure)
            plt.close(figure)
    """),
    cell("markdown", """
    ## 7 · Scheduled winter reliability, including non-events

    This optional retrieval uses a regular UTC schedule fixed independently of observed
    events. Both threshold crossings and non-crossings are retained; missing reference or
    forecast inputs are marked failed/pending rather than coded as zero. Set the **exclusive**
    end to extend the winter sample. By default there is one target per day at 00 UTC in
    November–February; choose 6 or 12 hours for a denser schedule. Do not select only dates
    with catalog events or select dates according to forecast probabilities.

    This can require substantial WN2 reads: every target uses all members at three leads.
    The preview is free of field downloads; set RUN_SCHEDULED_RELIABILITY to True to retrieve.
    It shares compact wind-derived checkpoints with the catalog run, but saves its own
    probabilities, counts, settings, and status table. Its sample contains no event-derived
    vortex strata. Training overlap and missing-case selection still limit interpretation.

    To extend, rerun this cell with the same start and later end. Failed dates retry;
    completed WN2 targets/leads reuse their checkpoints. ERA5T targets refresh until final.
    A November 19 target requests initializations on November 18, 14, and 9 at the same UTC
    hour for the 1-, 5-, and 10-day leads. Exact WN2 availability is checked on each request;
    the workflow never substitutes a later run or another model.
    """),
    cell("code", """
    # @title 7 · Preview or run the outcome-independent winter schedule
    RUN_SCHEDULED_RELIABILITY = False  # @param {type:"boolean"}
    SCHEDULE_START_UTC = '2026-11-01'  # @param {type:"date"}
    SCHEDULE_END_UTC = '2027-03-01'  # @param {type:"date"} # Exclusive; future times stay pending.
    SCHEDULE_INTERVAL_HOURS = 24  # @param [6, 12, 24] {type:"raw"}
    scheduled = scheduled_verification_catalog(
        SCHEDULE_START_UTC, SCHEDULE_END_UTC, interval_hours=SCHEDULE_INTERVAL_HOURS)
    print(f'{len(scheduled)} winter targets; {len(scheduled)*len(LEAD_HOURS)} forecast requests.')
    display(scheduled[['event_number', 'target_valid_time']])
    if RUN_SCHEDULED_RELIABILITY:
        # Future schedule entries remain visible in the preview, but are not downloaded.
        due = scheduled.loc[scheduled.target_valid_time.le(pd.Timestamp.now(tz='UTC').tz_localize(None))]
        if due.empty:
            raise ValueError('No schedule targets are due yet.')
        open_verification_era5.cache_clear()
        archive.clear()
        scheduled_tables, scheduled_settings = run_ensemble_verification(
            due, archive, OUTPUT_DIR, allow_era5t=ALLOW_ERA5T)
        schedule_dir = OUTPUT_DIR / 'scheduled_winter'
        schedule_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in scheduled_tables.items():
            frame.to_csv(schedule_dir / f'{name}.csv', index=False)
        schedule_settings = {**scheduled_settings, 'start': SCHEDULE_START_UTC,
            'end_exclusive': SCHEDULE_END_UTC, 'interval_hours': SCHEDULE_INTERVAL_HOURS,
            'due_targets': len(due), 'future_targets': len(scheduled)-len(due), 'allow_era5t': ALLOW_ERA5T}
        atomic_json_file(schedule_dir / 'schedule_settings.json', schedule_settings)
        scheduled_probabilities, scheduled_bins = export_reliability(
            scheduled_tables, schedule_dir, RELIABILITY_THRESHOLDS,
            sample='scheduled_winter', n_bins=PROBABILITY_BINS)
        display(scheduled_tables['batch_status'])
        verified = scheduled_probabilities.review_key.nunique() if not scheduled_probabilities.empty else 0
        print(f'{verified}/{len(due)} due targets verified at all leads; {len(scheduled)-len(due)} future targets.')
        if not scheduled_probabilities.empty:
            figure = plot_reliability_diagrams(scheduled_probabilities, n_bins=PROBABILITY_BINS)
            display(figure)
            plt.close(figure)
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
