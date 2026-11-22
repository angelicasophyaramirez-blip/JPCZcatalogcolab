# Notebook 35: all-event three-time batch exports

Run the notebook's installation, helper, current catalog, and NASA login cells.
The installation cell now includes `pypdf`, used to combine one event's three
pages without holding the entire atlas in memory. The helper cell is collapsed
under “Load case-review helpers.”

In the final batch cell, set:

```python
RUN_BATCH = True
EVENTS_PER_RUN = None
BATCH_WINDOW_HOURS = 1
BATCH_SATELLITE_PLATFORM = 'Auto'
```

This loads the current shared Drive catalog again and processes every remaining
event at exactly peak −12 h, peak, and peak +12 h. `EVENTS_PER_RUN = 1` is an
optional pilot that processes all three phases for one event. A limit counts
attempted events, not individual plots. Partial or failed events are retried on
later runs; use `None` to visit all events despite gaps in earlier cases.

`BATCH_WINDOW_HOURS = 12` instead applies the trailing 12-hour divergence and
precipitation-rate window at each of those three target times. Circulation is
still a snapshot at the selected target. The running precipitation total begins
at the catalog onset and ends at the end of the selected hour; a phase before
onset is explicitly labeled “not begun.” `BATCH_ACCUMULATION_MODE = 'window'`
selects fixed-window totals instead. Satellite day offset is a separate setting.

Auto satellite selection calls the same `get_case` / `load_satellite_pair`
functions as the regular browser. It tries Terra and then Aqua on the same
requested Japanese calendar date if Terra is missing or blank. It exports one
selected platform per phase, not separate Terra and Aqua variants. Titles and
metadata record the platform actually used. These are daily daytime mosaics,
not satellite observations exactly 12 hours apart. A January 2000 event predates
MODIS imagery; unavailable archive imagery is retained as an explicit gap.

## Files in Drive

Everything is under:

`MyDrive/JPCZcatalog_outputs/imerg_manual_case_review/batch_evolution/`

A folder named for the chosen window, platform, and settings hash contains one
subfolder for each event. Settings such as the domain, polygons, precipitation
product, screen scheme, window, and accumulation mode are recorded and form
part of the checkpoint identity.

Each event folder contains:

- `all_three_times.pdf`: three pages ordered −12 h, peak, +12 h.
- `minus12h.png`, `peak.png`, `plus12h.png`: the usual six-panel case figures.
- Matching single-page PDFs and `_metadata.csv` files for each phase.
- `all_three_times_metadata.csv`: the three metadata records for that event.
- Phase JSON checkpoint records used to validate completed exports.

The batch folder contains `all_cases_metadata.csv`, `batch_manifest.csv`,
`batch_summary.json`, and `batch_settings.json`. The master metadata includes
all selected event/phase records and paths. The summary distinguishes a running
batch from a finished pass and records complete, partial, failed, and unattempted
phases. A finished pass can still contain missing data: inspect the statuses.

## Resume and memory behavior

Every phase saves its PNG, PDF, metadata, and checkpoint before moving on. A
matching completed export is reused only when all three files exist; a missing
PDF is rebuilt even if its PNG exists. Changed labels, event windows, or requested
settings invalidate the relevant completion checks. Existing map/data caches
are reused by the notebook readers. No catalog CSV or saved review is changed
by the batch exporter itself; the catalog refresh can update the review inventory.

Only one six-panel figure is rendered at a time and it is closed immediately.
Batch figures are saved rather than accumulating hundreds of large inline
outputs. The three-page PDF is assembled one event at a time. There is no single
PDF containing every event, avoiding an unnecessarily large file and merge.

If weather data cannot be loaded, the affected phase receives a labeled error
page and a failed metadata record; later events continue. Transient precipitation
or imagery gaps are partial and retried on the next pass. Satellite dates before
archive availability are marked `complete_archive_gap` when the other inputs
are complete, so an impossible old image does not cause endless retries.
Missing data are never converted to zero.

To resume after a disconnect, run setup again, use the same Drive folder and
batch settings, and rerun the batch cell. Old single-peak batch exports in the
parent review folder do not count as completed three-time exports.

## Maintenance and validation

The reusable exporter lives in `src/jpcz_catalog/case_review_batch.py` and is
embedded as ordinary multiline source by `tools/update_notebook35_batch.py`.
After rebuilding Notebook 35, apply its existing vortex-label updater and then
this batch updater. The updater is idempotent.

Tests cover real three-page PDF output, all-event iteration, exact offsets,
real browser Auto selection with synthetic Terra failure/Aqua success, bounded
pilot resume, deleted PDFs, failed phase placeholders, transient-data retries,
permanent pre-MODIS gaps, the embedded wrapper, and the actual six-panel renderer
without inline figure accumulation. Source fields are synthetic for offline
checks; private NASA downloads still require the user's authenticated Colab run.
