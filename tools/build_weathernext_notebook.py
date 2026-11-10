"""Regenerate standalone Colab Notebook 36 from its reusable source module."""
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/jpcz_catalog/weathernext_atlas.py"
TARGET = ROOT / "notebooks/36_weathernext2_jpcz_forecast_atlas.ipynb"


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
    return result


source = SOURCE.read_text()
core, ui = source.split("# --- Interactive browser ---", 1)
cells = [
    cell("markdown", """
    # Notebook 36 · WeatherNext 2 JPCZ forecast atlas

    **One selected event × one selected ensemble member × two variables × three leads = six forecast plots.**

    Run the numbered sections in order, then use the event browser. Start with one event and
    inspect the six plots before moving to another event or the optional batch export.
    The assignment defaults are **500-hPa geopotential height** and **925-hPa winds**, each
    at **1-, 5-, and 10-day leads**, using the same selected member throughout.

    The original UTC event start from Notebook 35 determines the target: nearest **00, 06,
    12, or 18 UTC on that same date**, ties earlier. The true onset and signed offset remain
    visible. For late starts, the same-date rule can give a target up to almost six hours
    before onset. All three forecasts verify at the **same target**; their initializations
    are target minus 24, 120, and 240 hours. No interpolation or silent nearest-run fallback.

    This standalone notebook contains its helper functions, so an unpublished GitHub change
    cannot prevent it from running. CPU runtime is sufficient. Google login happens only
    through Colab's normal authentication prompt; credentials are not saved to Drive.

    **Study selection:** 2023 onward is a provisional window, not a verified operational
    model training cutoff. The research model was trained through 2022, but the distributed
    operational version must be checked separately before claiming out-of-sample skill.
    The notebook counts the current catalog rather than assuming 34 events. It does not
    silently substitute an older historical catalog, another WN model, or ensemble means.

    Sources: [WeatherNext 2 model/data guide](https://developers.google.com/weathernext/guides/models-wn2),
    [official WN2 Zarr example](https://storage.googleapis.com/weathernext-public/colabs/WeatherNext_2_Starter_Guide_Zarr_on_Google_Cloud_Storage.ipynb),
    [dataset and attribution](https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_weathernext_2_0_0),
    [research training/evaluation](https://arxiv.org/html/2506.10772v1).
    """),
    cell("markdown", """
    ## 1 · Install dependencies

    Run once in a fresh Colab runtime. This installs the Zarr 3-capable stack needed for
    Google's forecast stores. The WN2 source array can advertise terabytes without loading
    them: reads remain lazy until one member, lead, level, and region are selected.
    Historical source chunks include multiple members, all pressure levels, and the full
    globe. They are decoded sequentially to limit memory; only selected regional results
    are retained and cached on Drive.
    """),
    cell("code", """
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
        'numpy>=1.26,<2.4', 'pandas>=2.2,<3', 'xarray>=2026.4,<2027',
        'zarr>=3.1,<4', 'gcsfs>=2025.3,<2027', 'dask[array]>=2025.1,<2027',
        'matplotlib>=3.9,<3.11', 'cartopy>=0.24,<0.26', 'ipywidgets>=8.1,<9',
        'netCDF4>=1.7,<2'], check=True)
    print('Dependencies installed. Continue to Google authentication.')
    """),
    cell("markdown", """
    ## 2 · Google login and Drive

    Authenticate with the **exact Google account approved for WeatherNext**. Drive mounting
    supplies your event catalog; Google Cloud authentication supplies the private forecasts.
    If those use different accounts, choose each intended account at its respective prompt.
    A successful Cloud Console folder listing alone does not authenticate this runtime.
    """),
    cell("code", """
    from google.colab import auth, drive, output
    drive.mount('/content/drive')
    auth.authenticate_user()
    output.enable_custom_widget_manager()
    import google.auth
    credentials, detected_project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform'])
    print('Google credentials acquired. The browser access check will verify WN2 permissions.')
    """),
    cell("markdown", """
    ## 3 · Paths and study window

    The catalog path is the same one used by Notebook 35. Its current catalog is read
    directly, while saved reviews/images are optional references. Catalogs and Notebook 35
    outputs are read only. This notebook writes to a separate `weathernext2_atlas` folder.

    **Vortex labels** come from Notebook 35's saved event-screen JSON records, which also
    drive its event selector. A completed positive scan is **Vortex**; a completed negative
    scan is **Non-vortex**. Unchecked, failed, or unknown results are labeled explicitly.
    Legacy manual `case_label` values do not classify these figures. If multiple screen
    settings folders exist without an active pointer, run **Refresh catalog** in updated
    Notebook 35 to identify its active settings. Event start/end/peak identities must match.

    Enter a Cloud **project ID** if you have one. Leaving it blank uses Google's detected
    project when available. If a project-related permission error appears, set your ID and
    rerun this section and the browser setup. No Cloud project or billing settings are changed.
    The root must remain the WN2 **ensemble** archive, not a path ending in `/success`.
    """),
    cell("code", """
    from pathlib import Path
    PROJECT_ID = ''  # @param {type:"string"}
    DRIVE_ROOT = Path('/content/drive/MyDrive/JPCZcatalog_outputs')
    CATALOG_PATH = DRIVE_ROOT / 'jpcz_catalog_ndjf_merged_12h.csv'
    NOTEBOOK35_DIR = DRIVE_ROOT / 'imerg_manual_case_review'
    OUTPUT_DIR = DRIVE_ROOT / 'weathernext2_atlas'
    STUDY_START_UTC = '2023-01-01'  # @param {type:"date"}
    STUDY_END_UTC = ''  # optional exclusive UTC end; blank includes all later catalog events
    # Only needed if discovery finds multiple stores for the SAME initialization.
    # Use the exact UTC initialization as key and full predictions.zarr path as value.
    RUN_OVERRIDES = {}
    """),
    cell("markdown", """
    ## 4 · Load forecast and plotting helpers

    Run this cell unchanged. Reusable source: `src/jpcz_catalog/weathernext_atlas.py`.
    It supports historical period stores and dated forecast-run folders, checking exact
    member labels, leads, initialization/valid times, levels, and units. Historical fields
    omit units; those known variables use Google's documented WN2 units, recorded in export
    metadata. Declared but unrecognized units remain an error. No forecast is substituted.
    """),
    cell("code", '# @title Load forecast and plotting helpers (run unchanged)\n' + core),
    cell("markdown", """
    ## 5 · Load the current event inventory

    The count and date range below come from your Drive catalog. The selection report
    includes excluded events and reasons. Forecast availability is checked per selected event.
    Signed peak D12 is a **12-hour divergence** measure, not precipitation intensity.
    """),
    cell("code", """
    catalog, selection_report = read_catalog(
        CATALOG_PATH, NOTEBOOK35_DIR, STUDY_START_UTC, STUDY_END_UTC or None)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selection_report.to_csv(OUTPUT_DIR / 'event_selection_report.csv', index=False)
    catalog.to_csv(OUTPUT_DIR / 'selected_event_inventory.csv', index=False)
    if catalog.empty:
        raise ValueError('No events in the configured study window. Check paths and dates above.')
    print(f'{len(catalog)} selected events from {catalog.event_start.min()} to {catalog.event_start.max()} UTC')
    print('Operational WN2 training cutoff: UNVERIFIED. Study start is provisional.')
    print('Current catalog:', CATALOG_PATH)
    print('New outputs:', OUTPUT_DIR)
    from IPython.display import display
    display(catalog[['event_number', 'event_start', 'event_end', 'event_peak',
                     'event_peak_D_1e5_s-1', 'target_valid_time', 'onset_offset_hours']])
    """),
    cell("markdown", """
    ## 6 · Open the event browser

    1. Choose an event or search its number/date/notes.
    2. Click **Check data & members**. This inspects metadata for all three runs,
       checks the assignment fields and units, and reads WN2's actual member labels.
    3. Choose **one member**. Assignment defaults are height at 500 hPa and winds at 925 hPa.
    4. Click **Load six plots**, inspect all labels/maps, then **Export six plots**.
    5. Use Previous/Next for another event. No forecast fields load just from changing a control.

    Field, level, map domain, and scale bounds can be changed without editing plotting code.
    Scale bounds stay fixed across leads and events until you change them. Values outside
    the scale are indicated by colorbar extensions. Use **Assignment defaults** to restore
    the requested fields. The member stays fixed until you explicitly select another.

    Figure headings and exports show **Observed event: Vortex / Non-vortex**, matching the
    Notebook 35 event screen. This is the reference event's class, not an automatic claim
    about a forecast member. After new scans in Notebook 35, click **Refresh vortex labels**
    here and then **Load six plots**; matching forecast arrays are reused from cache.
    Saved Notebook 35 PNGs contain baked-in titles: to replace an old title such as
    "Unclassified candidate", reopen that event in updated Notebook 35 and click **Load maps**
    to produce newly labeled reference exports. The reference browser flags stale titles.

    **Derived fields use these same dropdowns.** Select **horizontal divergence (derived)**
    or **Ertel PV (derived)**, choose its map pressure, and click **Load six plots**. Inputs
    are read for the same member, initialization, and lead; calculations happen automatically.
    Divergence defaults to **925 hPa**: **negative/blue = convergence, positive/red = divergence**,
    in 10⁻⁵ s⁻¹. It is an instantaneous wind diagnostic, not the catalog's 12-hour D12.

    Dry hydrostatic Ertel PV defaults to **1000 hPa**, in **PVU** (1 PVU = 10⁻⁶ K m² kg⁻¹ s⁻¹).
    It uses temperature and both wind components at three pressure levels, including the
    horizontal-temperature-gradient/vertical-wind-shear terms and absolute vorticity. At
    1000 hPa the 850/925/1000-hPa stencil gives a second-order **one-sided** pressure derivative;
    interior levels use a centered stencil. Potential temperature references 1000 hPa, which
    is a separate choice from the selected PV map level. Derivatives use Pa, radians, a
    spherical Earth radius of 6,371,000 m, and a two-cell horizontal halo where available.
    Both signed diagnostics use zero-centered scales; scale bounds must straddle zero.
    These are diagnostic calculations from WN2 fields, not additional native forecast variables.

    The implementations are checked against manufactured analytic fields and
    [MetPy's pressure-coordinate PV](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.potential_vorticity_baroclinic.html)
    and [horizontal divergence](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.divergence.html).

    Export is disabled after a selection change until its six plots load successfully.
    Missing runs/members are errors, not substitutions. Regional cache entries include run,
    lead, native member label, variables, level, conversion, and domain.

    **Interpretation:** pressure-level fields/diagnostics lack a below-surface mask.
    In particular, 1000 hPa can be below the surface over land and also over sea when
    surface pressure is less than 1000 hPa. This WN2 archive lacks the surface-pressure/
    terrain inputs needed to identify all such cells. Satellite references from Notebook 35 may be
    peak-centered and daily daytime context. The core six panels are forecast snapshots;
    timing-error and ensemble-spread validation require later workflows.
    """),
    cell("code", '# @title Load event-browser helpers (run unchanged)\n' + ui),
    cell("code", """
    archive = GoogleCloudArchive(credentials, PROJECT_ID or detected_project or '', overrides=RUN_OVERRIDES)
    browser = WeatherNextBrowser(catalog, archive, OUTPUT_DIR, NOTEBOOK35_DIR)
    browser.display()
    """),
    cell("markdown", """
    ## 7 · Optional: export the catalog after the pilot works

    Leave this off while checking the first event. To run later, set `RUN_BATCH = True`.
    This uses the **last successfully displayed pilot's one member, two fields, map domains,
    and scales** throughout. It does not use unapplied control changes. Each event has six
    plots plus an overview; PNG/PDF files, metadata, settings, and attribution are saved.
    Progress is checkpointed after every event. Reruns reuse regional cache entries and
    completed matching exports; missing or failed events remain visible in the progress CSV.
    """),
    cell("code", """
    RUN_BATCH = False
    BATCH_LIMIT = None  # use 2 for a small pilot, or None for all selected events
    if RUN_BATCH:
        display(export_catalog_single_member(browser, limit=BATCH_LIMIT))
    else:
        print('Batch is off. Use the browser to validate one event first.')
    """),
    cell("markdown", """
    ## Troubleshooting

    - **401/403:** rerun Section 2 with the approved account, then Section 6's browser setup.
      A project/quota error may also require your project ID in Section 3.
    - **Catalog missing:** verify the Drive location used in Notebook 35. No historical
      catalog is substituted. Refresh an incomplete merged catalog in Notebook 06.
    - **Run missing or ambiguous:** the message names the exact initialization. Check
      the archive partition; use `RUN_OVERRIDES` only for an explicit matching run.
    - **Units/schema mismatch:** the historical layout uses datetime `time` for model
      initialization and `prediction_timedelta` for lead. The notebook normalizes this
      after exact initialization selection. Absent units on known WN2 fields use Google's
      published schema, with provenance recorded; unrecognized declared units are rejected.
    - **Cartopy download error:** its coastlines/borders require a first-use Natural Earth
      download. Retry when connectivity is restored; no replacement basemap is fabricated.
    - **Runtime disconnected:** run the setup sections again; regional subsets and complete
      exports remain on Drive. Incomplete exports are rebuilt.

    Historical data attribution: © 2025 DeepMind Technologies Limited's machine learning
    models used to create the experimental data made available at
    https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_weathernext_2_0_0
    under CC BY 4.0 licence terms. This data is intended for experimental modelling only
    and is not intended, validated, or approved for real world use.
    """),
]

for index, item in enumerate(cells):
    item["id"] = f"wn2-{index:02d}"
    if item["cell_type"] == "code" and item["source"][0].startswith("# @title"):
        item["metadata"]["cellView"] = "form"
notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"colab": {"name": TARGET.name, "provenance": []},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "cells": cells,
}
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(TARGET)
