# Notebook 35: all-event three-time batch exports

Run the notebook's installation, helper, current catalog, and NASA login cells.
The installation cell now includes `pypdf`, used to combine one event's three
pages without holding the entire atlas in memory. The helper cell is collapsed
under “Load case-review helpers.”

The NASA authentication cell must run again in every new Colab runtime, even
when Drive still contains cached data. Wait for “NASA Earthdata authenticated.”
The batch checks the login before reading any weather data, and before each
new phase. Missing login stops with instructions rather than producing hundreds
of incomplete precipitation panels. Individual map requests also check login
when uncached IMERG intervals need downloading. Login failures propagate through
the case reader and stop the batch; completed phase files remain untouched.
Sign in and rerun with the same settings to reuse complete phases and retry
partial phases from an earlier unauthenticated run. Credentials are not saved.

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

Each batch phase now runs in a **fresh Python interpreter**. The child uses the
same case reader, satellite Auto fallback, renderer, numerical definitions,
and Drive caches as the browser. It writes the PNG/PDF, returns only metadata,
then exits. This releases all of its decoder, HTTP, native allocator, and plot
memory between phases, including buffers that garbage collection cannot reclaim
in a long-lived notebook. The notebook process assembles event PDFs and persists
checkpoints. The batch folder hash and completion identities are unchanged, so
exports saved before this fix are reused. The interactive browser remains in
the notebook process.

On Colab/Linux the child also limits its address space to 6 GiB and uses one
numerical-library thread. A child exit, timeout, or uncaught memory allocation
failure pauses the batch with `paused_worker`; it does not create a successful
checkpoint for that phase. Interrupting the notebook also terminates its child.
Normal weather-data gaps still receive the existing partial/failed metadata.
Worker PID and peak resident memory are recorded in new phase metadata for
diagnosis. The parent never receives weather arrays or figure objects.

The worker reuses the current NASA login token, handed through an in-memory
stdin pipe. It never saves credentials to Drive, notebook outputs, temporary
request files, or command arguments, and never prompts once per phase. The
versioned worker archive contains only the exact notebook helpers and entrypoint;
its SHA-256 is checked before execution. It is cached on the temporary runtime
disk. Every new runtime still requires running the NASA login cell once.

ERA5 source decoding is also bounded: maps and vortex checks now read one hour
and one variable at a time, with one Zarr read worker and one Blosc codec thread.
The selected 925-hPa regional arrays are copied into compact owned buffers so
they cannot retain the full decoded globe/all-pressure-level chunks. A 12-hour
map keeps only 12 small regional snapshots for the original mean calculation.
An uncached 25-hour vortex screen saves and releases each hour separately.
The screen criteria, divergence calculation, output settings, cache filenames,
and batch completion identities are unchanged by this I/O fix.

After each phase, references to the closed figure are released and garbage is
collected. On Linux/Colab the cleanup also asks the allocator to return free
pages to the operating system when `malloc_trim` is available. After a RAM
crash, reconnect the runtime, run setup and NASA login, and rerun the batch
using the same settings and Drive root. Successful phase exports and compact
vortex-hour caches remain on Drive and are reused.

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
this batch updater. It also invokes `tools/update_notebook35_memory.py` to embed
`src/jpcz_catalog/case_review_memory.py`, followed by
`tools/update_notebook35_isolation.py` to create the worker archive and embed
`src/jpcz_catalog/case_review_isolation.py`. These updaters are idempotent.

Tests cover real three-page PDF output, all-event iteration, exact offsets,
real browser Auto selection with synthetic Terra failure/Aqua success, bounded
pilot resume, deleted PDFs, failed phase placeholders, transient-data retries,
permanent pre-MODIS gaps, the embedded wrapper, and the actual six-panel renderer
without inline figure accumulation. Source fields are synthetic for offline
checks; private NASA downloads still require the user's authenticated Colab run.

Isolated-worker integration tests deliberately retain a 64-MiB buffer in each
phase, verify distinct PIDs and bounded per-process peak memory, and confirm
the parent performs no case reads or rendering. They check resume after an
abrupt worker exit, memory-error propagation, child cleanup on timeout/interrupt,
absence of the test token from files, and archive checksums. The bundled helpers
must match the notebook source exactly.

Memory regressions verify that each source load is a single time, level, and
variable; that retained arrays own only compact regional memory; that the map
divergence values and existing map caches are unchanged; and that a vortex
screen interrupted after six hours resumes only its 19 missing snapshots.
A cold read of the public ERA5 fields at 2001-12-21 06 UTC, where the reported
batch crash occurred, completed in a local test process with about 773 MiB peak
RSS and 55,956 bytes of retained regional data. This is a reader measurement,
not a peak-RAM guarantee for the full Colab batch.

A subsequent isolated-worker probe read public ERA5 at 2002-11-26 22 UTC and
2002-11-27 10 UTC (Event 34's peak and the +12-hour frame where Colab stopped),
using the real case reader and six-panel renderer. NASA login was mocked and
precipitation, satellite imagery, and vortex results were cached synthetic inputs.
Both PNG/PDF exports completed. Child peak RSS was about 1,116 and 1,075 MiB;
the parent stayed at 244.4 MiB after each child exited. This was a local macOS
probe, not a complete 280-event authenticated Colab run or a guarantee of every
future phase's peak memory.
