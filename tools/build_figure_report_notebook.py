"""Build a standalone, data-read-only Colab figure report notebook."""
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, name):
    source = textwrap.dedent(source).strip() + '\n'
    item = dict(cell_type=kind, id=name, metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        ast.parse(source)
        item.update(execution_count=None, outputs=[])
    return item


cells = [
    cell('markdown', '''
    # Notebook 39 · Saved ERA5 and WN2 figures by event

    Build a scrollable PDF **from figures already saved in Google Drive**.
    This notebook does not retrieve or calculate ERA5, WN2, IMERG, PV, or winds.
    Run the cells in order. CPU is sufficient; there is no Earthdata or weather-archive login.

    Each event contains one section per saved field and pressure level. Each section has:
    1. **ERA5: −12 h, peak, +12 h**, together on a landscape page.
    2. **WN2: 1-, 5-, 10-day leads**, together on the next page, for each saved member.

    WN2's actual valid time and initialization remain visible. Its target is based on
    event onset in Notebook 36; it is not automatically the ERA5 peak. The report does
    not pretend these are simultaneous observations. WN2 maps from Notebook 36 are
    labelled with their saved member; this does not turn them into ensemble means.

    **Rerunning:** export additional/updated figures, then rerun cells 4–5 here.
    Metadata discovery adds new field/level groups. Unchanged event PDFs are reused.
    Source images, metadata, and weather caches remain unchanged. No figure is
    inferred from a filename alone; event dates, field, level and timing must match.

    **Existing exports:** Notebook 35's six-panel phase PDFs are supported now:
    the known divergence panel is isolated with its original colorbar. Unrecognized
    layouts are shown intact and labelled. Notebook 36's standalone maps are supported.
    Notebook 37's aggregate statistical plots are not meteorological maps.

    **Future fields:** this notebook can organize PV, height, wind and onshore-flow
    figures after a source notebook actually exports them. It does not add those
    calculations to Notebook 35. Use the small `figure_manifest.csv` contract below
    for new exporters. Missing figures stay visible as gaps.
    ''', 'report-intro'),
    cell('markdown', '## 1 · Install the document tools', 'report-install-title'),
    cell('code', '''
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'pandas>=2.2,<4', 'reportlab>=4.4,<5',
                           'pypdf>=6.1,<7', 'pdfplumber>=0.11,<1', 'Pillow>=10,<13',
                           'pypdfium2>=4.30,<6'])
    ''', 'report-install'),
    cell('markdown', '## 2 · Mount Drive and choose the report', 'report-config-title'),
    cell('code', '''
    from pathlib import Path
    from google.colab import drive
    drive.mount('/content/drive')

    DRIVE_ROOT = Path('/content/drive/MyDrive/JPCZcatalog_outputs')
    CATALOG_PATH = DRIVE_ROOT / 'jpcz_catalog_ndjf_merged_12h.csv'
    ERA5_FIGURE_ROOT = DRIVE_ROOT / 'imerg_manual_case_review'
    WN2_FIGURE_ROOT = DRIVE_ROOT / 'weathernext2_atlas'
    REPORT_DIR = DRIVE_ROOT / 'event_figure_reports' / 'saved_map_comparison'

    START_UTC = None  # None = all catalog events; '2023-01-01' = current WN2 study window
    END_UTC = None  # Optional exclusive end date, e.g. '2027-01-01'
    EVENT_NUMBERS = None  # e.g. [276, 277]; None = all selected catalog events
    EVENT_LIMIT = None  # e.g. 2 for a quick layout check
    FIELDS = None  # Auto-add every discovered field/level; divergence only: [('divergence', 925)]
    MEMBERS = None  # All saved WN2 members; e.g. ['0'] selects exactly that saved member
    ERA5_BATCH_HOURS = 1  # Existing 35 batch: 1 = snapshot, 12 = averaged divergence
    EXTRA_MANIFESTS = []  # Optional Paths to figure_manifest.csv files outside these roots
    EVENTS_PER_VOLUME = 10
    COMBINE_ALL_EVENTS = True
    COMBINED_SIZE_LIMIT_MB = 350  # Larger reports remain available as bounded volumes
    print('Figures:', ERA5_FIGURE_ROOT, WN2_FIGURE_ROOT, sep='\\n')
    print('Report destination:', REPORT_DIR)
    ''', 'report-config'),
    cell('markdown', '## 3 · Load the embedded report helpers', 'report-helper-title'),
    cell('code', (ROOT / 'src/jpcz_catalog/figure_report.py').read_text(), 'report-helpers'),
    cell('markdown', '''
    ## 4 · Discover the saved figures and audit coverage

    No weather data are loaded. Every selected catalog event is retained, even if
    it has no WN2 maps. This can yield WN2 placeholders for pre-2023 cases; set
    `START_UTC = '2023-01-01'` above for the current verification study.
    Catalog events absent from the figure metadata are listed as missing.
    Orphan figure identities and unreadable metadata are saved for inspection.
    ''', 'report-inventory-title'),
    cell('code', '''
    catalog = pd.read_csv(CATALOG_PATH)
    assets, scan_issues = scan_saved_figures(ERA5_FIGURE_ROOT, WN2_FIGURE_ROOT, EXTRA_MANIFESTS)
    events, selected_figures = prepare_report_inventory(catalog, assets,
        start=START_UTC, end=END_UTC, fields=FIELDS, event_numbers=EVENT_NUMBERS,
        members=MEMBERS, era5_hours=ERA5_BATCH_HOURS)
    if EVENT_LIMIT is not None:
        events = events.head(EVENT_LIMIT).copy()
        selected_figures = selected_figures.loc[selected_figures.event_key.isin(events.event_key)].copy()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_csv(scan_issues, REPORT_DIR / 'scan_issues.csv')
    catalog_keys = set(catalog.apply(report_event_key, axis=1))
    _atomic_csv(assets.loc[~assets.event_key.isin(catalog_keys)], REPORT_DIR / 'unmatched_figures.csv')
    print(f'{len(events)} catalog events selected; {len(selected_figures)} saved-figure records.')
    print('Missing files:', int((~selected_figures.file_exists).sum()))
    print('Metadata issues:', len(scan_issues))
    display(events[['event_number', 'event_start', 'event_peak', 'event_end']])
    display(selected_figures.groupby(['source', 'field', 'level_hpa'], dropna=False).size().rename('figures').reset_index())
    if len(scan_issues):
        display(scan_issues)
    ''', 'report-inventory'),
    cell('markdown', '''
    ## 5 · Build or resume the PDF reports

    Produces one PDF per event, ten-event volumes, and `all_events.pdf` when within
    the size limit. PDFs contain event/field bookmarks. PDFs are assembled from the
    source vector pages when available; maps are not redrawn. Large full-catalog
    reports can take time to read and write on Drive, but do not repeat weather work.

    The build checkpoints after each event. If Colab disconnects, run this notebook
    again with the same settings. It reuses completed unchanged event PDFs and
    refreshes the combined files. Adding fields regenerates the affected events.
    ''', 'report-build-title'),
    cell('code', '''
    event_reports, coverage, report_summary = build_figure_report(events, selected_figures,
        REPORT_DIR, required_fields=FIELDS, events_per_volume=EVENTS_PER_VOLUME,
        combine=COMBINE_ALL_EVENTS, combined_limit_mb=COMBINED_SIZE_LIMIT_MB)
    display(event_reports[['event_number', 'event_peak', 'pages', 'status', 'missing_slots', 'pdf']])
    display(coverage.loc[coverage.availability.eq('missing')])
    print('Open Google Drive > My Drive > JPCZcatalog_outputs > event_figure_reports > saved_map_comparison')
    print('Current combined PDF:', report_summary['combined_pdf'] or report_summary['combined_note'])
    print('The JSON summary above is authoritative if an older all_events.pdf is still in this folder.')
    ''', 'report-build'),
    cell('markdown', '## 6 · Preview or download the finished report (optional)', 'report-preview-title'),
    cell('code', '''
    from IPython.display import display
    from google.colab import files
    import pypdfium2 as pdfium
    PREVIEW_EVENT_NUMBER = int(events.event_number.iloc[0])
    PREVIEW_PAGE = 1  # 1 = first ERA5 field page, 2 = its WN2 page
    preview_file = event_reports.loc[event_reports.event_number.eq(PREVIEW_EVENT_NUMBER), 'pdf'].iloc[0]
    print('Preview:', preview_file)
    with pdfium.PdfDocument(str(preview_file)) as document:
        page = document[max(0, min(PREVIEW_PAGE - 1, len(document) - 1))]
        bitmap = page.render(scale=1.2)
        display(bitmap.to_pil())
        bitmap.close()
        page.close()
    DOWNLOAD_REPORT = False  # Set True to download all_events.pdf, or the first volume if too large
    if DOWNLOAD_REPORT:
        files.download(report_summary['combined_pdf'] or report_summary['volumes'][0])
    ''', 'report-preview'),
    cell('markdown', '''
    ## Adding PV, height, winds or an onshore component later

    Notebook 36 already has multiple field/level exporters. Notebook 35 currently
    exports the fixed 925-hPa diagnostic overview; other fields need their own
    upstream figure exporter. This report does not calculate them from caches.

    A new exporter can place a **figure_manifest.csv** anywhere under the ERA5 or
    WN2 figure root (or add it to `EXTRA_MANIFESTS`). Write one row per standalone
    map, with these columns:

    | Column | Required content |
    |---|---|
    | event_start, event_end, event_peak | Exact current catalog UTC timestamps; event number alone is insufficient |
    | event_number | Display label; current catalog number takes precedence |
    | source | ERA5 or WN2 |
    | field, level_hpa | e.g. ertel_pv, 500; height, 500; winds, 925; onshore, 925 |
    | figure | Single-page PDF or PNG; absolute path or path relative to this CSV |
    | offset_hours | ERA5: -12, 0, or 12, relative to peak |
    | lead_hours, member | WN2: 24, 120, or 240; exact member/ensemble-summary label |
    | valid_time, init_time | Actual UTC times; init_time for WN2 |
    | units, domain, definition, status | Preserve scale/domain, averaging/derivation definition and missing-data status |

    Use the same field key for scientifically equivalent ERA5/WN2 quantities.
    `height` and `geopotential` are distinct (m versus m² s⁻²). For onshore flow,
    record the shoreline geometry, positive-onshore sign convention and averaging
    in `definition`; different definitions remain separate labelled pages.
    Extra pressure levels remain separate field sections. All saved WN2 members
    remain individually labelled; do not label a single-member plot as 64 members.

    Rerun cells 4–5 after exporting new fields. Review `figure_coverage.csv`,
    `missing_figures.csv`, `scan_issues.csv`, and `event_report_index.csv` beside
    the PDFs. A missing figure does not mean a zero meteorological value.
    ''', 'report-manifest'),
]
notebook = dict(cells=cells, metadata=dict(colab=dict(name='39_saved_event_figure_report.ipynb'),
    kernelspec=dict(display_name='Python 3', language='python', name='python3'),
    language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
target = ROOT / 'notebooks/39_saved_event_figure_report.ipynb'
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
print(target)
