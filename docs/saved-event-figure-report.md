# Notebook 39: read saved figures and build the event comparison PDF

Run `notebooks/39_saved_event_figure_report.ipynb` in a CPU Colab runtime, in order.
Mount the same Google Drive used by Notebooks 35 and 36. The defaults include all
current catalog events, with explicit missing-figure slots for any source not yet
exported. Set `START_UTC = '2023-01-01'` for the current WN2 study selection;
this is a study filter, not a claim about the operational model training cutoff.

Each event has one section per field and pressure level: an ERA5 page containing
the -12-hour, peak and +12-hour maps, followed by a WN2 page containing the saved
1-, 5- and 10-day forecast leads. Different WN2 members, map domains and diagnostic
definitions get separate pages. WN2 onset-based targets are labelled with their
actual valid and initialization times; they are not relabelled as peak times.

The notebook **does no weather calculation or retrieval**. It reads the shared
catalog, existing figure metadata and the saved PDF/PNG files. A selected member
map remains a selected member map; a saved 64-member verification table is not
silently converted into an ensemble-mean map.

## Inputs supported now

- Notebook 35 `batch_evolution/*/event_*/*_metadata.csv` with its phase PDFs.
  The report validates the six-panel layout and isolates the existing 925-hPa
  divergence panel, retaining its title, coordinates, units, sign description and
  colorbar. If the layout cannot be recognized, the original overview is retained
  intact and the page explains this. Satellite or precipitation gaps do not get
  interpreted as missing ERA5 divergence. Failed phases keep their status.
- Notebook 36 `weathernext2_atlas/figures/event_*/forecast_metadata.csv` and
  standalone `row*_FIELD_leadHHHh.pdf`/PNG panels. The exact field and level come
  from the metadata, not a filename guess. The newest existing export wins only
  within an identical event/field/level/domain/definition/member/timing selection.
- `figure_manifest.csv` from other/future field exporters. See the contract in
  Notebook 39. Explicit manifest entries take precedence over inferred legacy
  adapter entries for the same selection.

Notebook 35 does not currently have general PV/height/onshore exporters. Adding
those scientific calculations belongs upstream. Once those maps and metadata
exist, Notebook 39 will discover and group them without recalculating them.
Height versus geopotential, pressure levels, and different shoreline definitions
are not interchangeable. Record the shoreline geometry and positive-onshore
convention in the `definition` column when exporting an onshore component.

## Outputs and updates

Google Drive > My Drive > `JPCZcatalog_outputs/event_figure_reports/saved_map_comparison`:

- `all_events.pdf`: one scrollable document with event/field bookmarks, when
  below the configurable 350-MB source-size limit.
- `volume_001.pdf`, etc.: bounded volumes, ten events each by default.
- `events/event_NNNN_ID.pdf`: individual event reports and their checkpoints.
- `event_report_index.csv`: current event number, dates, PDF path, pages and gaps.
- `figure_coverage.csv` and `missing_figures.csv`: one row per report figure slot.
- `selected_figure_inventory.csv`, `scan_issues.csv`, `unmatched_figures.csv`:
  provenance and input issues. An unmatched old event boundary is not silently
  reassigned just because the event number is the same.
- `report_summary.json`: authoritative current report paths and counts.

To add another field or revised figure, export it upstream, then rerun cells 4-5.
`FIELDS = None` discovers all saved field/level pairs. To restrict the output, use
`FIELDS = [('divergence', 925)]` or a list of other pairs. Original figures and
weather caches are read-only. Each event is atomically checkpointed; unchanged
events are reused after a disconnect. A new field creates its section and missing
slots across the selected events. Small source-file or metadata changes rebuild
the affected event reports. Volumes and the combined report are then refreshed.

If the full report is too large, use the volumes. The current JSON summary says
whether a combined document was created; an older combined file may remain when
the current settings disable it. Reports from earlier broader selections may
remain on Drive but are not included in the current index/summary.

## Local validation

`tests/test_figure_report.py` uses synthetic saved PDFs, real metadata adapters,
current-catalog renumbering, missing cases, changed figures, extra field levels,
resume checkpoints, ambiguous WN2 targets and bounded merging. No weather service
is called. The notebook embeds the same helper source and can run without a
repository installation. Build it with `python tools/build_figure_report_notebook.py`.
