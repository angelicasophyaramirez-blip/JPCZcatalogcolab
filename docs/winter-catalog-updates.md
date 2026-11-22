# Updating the established catalog and ensemble verification

Start with **Notebook 38**, then refresh 35 and 36 and rerun 37. All four use the
same Drive file: `JPCZcatalog_outputs/jpcz_catalog_ndjf_merged_12h.csv`.
The event count is read from the user's Drive, never assumed to be 275.

## Which Shinoda detector?

Notebook **04** detects raw threshold fragments; Notebook **06** merges them.
The implementation computes hourly ERA5 925-hPa divergence using MetPy and
`lat_lon_grid_deltas`, averages the digitized JPCZ polygon with cosine-latitude
weights, and forms an end-labeled trailing 12-hour mean. Each calendar month's
criterion is strictly below its baseline mean minus two population standard
deviations. Notebook 06 merges fragments whose start-minus-previous-end is at
most 12 hours, retaining the strongest peak. This is the project's Shinoda-style
ERA5 adaptation, not a claim of reproducing unpublished numerical details.

The linked Notebook 37 revision is `196db5d`, on
`codex/notebook36-weathernext2`. Its Notebook 04 has calendar years 2000–2025;
repository `main` at `e6458fd` has an older 2000–2018 Notebook 04. Notebook **34**
is a separate full-record historical-baseline calculation and must not be used
to append to the established catalog. Notebook 38 loads a small versioned ZIP
of the detector helpers and checks its SHA-256 before importing them. Cell 3
contains a short loader, with progress and completion messages, rather than
the helper source as one very long escaped string. It does not clone the
repository, so an old Git clone cannot select the older detector version.
The actual original Drive baseline cannot be certified from a GitHub link;
Notebook 38 records the actual saved files and threshold values used.

## First update and later winters

Run Notebook 38's installation, Drive, helper-loading, and inspection cells.
Inspect the actual raw/merged counts, dates, and saved checkpoint coverage.
A last event date alone does not establish the last analyzed day. The notebook
suggests the next calendar month only when the last event's month has a complete
saved D12 checkpoint. Otherwise enter a verified first unscanned month explicitly.

The first scan freezes original raw and merged CSVs plus monthly thresholds in
`seasonal_updates/`. It uses `ndjf_monthly_thresholds.csv` from Notebook 04.
If this is missing, recovery requires all saved NDJF checkpoints for the years
represented in the baseline catalog. If those are unavailable or the original
baseline used different years, restore the original threshold CSV. Do not
estimate the baseline from only the new winter or silently use the separate
historical threshold table.

Keep the saved start date on every rerun; extend the inclusive end date or leave
it blank for the current UTC hour. Every NDJF month in that extension is scanned.
The established catalog is preserved; the extension is reconstructed from its
scalar checkpoints before merging. New raw fragments therefore cannot be
appended twice. Exact identities keep their display numbers; a changed window
or peak gets a fresh number and requires a fresh review. Original monsoon and
higher-vorticity classifications are not calculated for new cases by this
workflow. Notebook 35 supplies the separate vortex association review.

Each month initially discards 11 hours for the rolling-mean warmup, matching
Notebook 04. Monthly thresholds remain frozen, including in future winters.
This is an extension of an established detection climatology, not a refitted
climatology. Changing that baseline requires a separately versioned experiment.

`monthly_scan_status.csv`, `event_changes.csv`, and `scan_manifest.json` record
coverage, new/revised identities, provisional flags, and the exact detector.
The baseline copies and staged `candidate_raw.csv` / `candidate_merged.csv`
allow recovery if Drive disconnects while the two canonical files are being
replaced. A scientific/read failure during the scan leaves canonical files
untouched. Missing wind inputs are handled separately: a continuously verified
prefix can be promoted with the remaining dates explicitly pending. The two
file replacements are not a distributed transaction.

## Cloud reads, RAM, and recovery after a crash

Notebook 38 opens ARCO lazily (`chunks=None`), then selects the time range,
925-hPa winds, and regional domain before loading fields. It processes six
data hours per batch by default and limits Zarr read concurrency to one, with
two worker threads. The polygon means are retained; full monthly wind or
divergence maps are not. Source storage chunks still span the globe and all
levels, so actual peak RAM depends on decoding those chunks and has not been
profiled in the user's Colab runtime. `CHUNK_HOURS` can be lowered to one or two.

Hourly scalar progress is saved to mounted Drive after every 24 processed data
hours and at the end of the requested monthly slice. This means 24 data hours,
not 24 wall-clock hours. `CHECKPOINT_EVERY_HOURS` can be lowered, using a multiple
of `CHUNK_HOURS`. Individual NetCDF checkpoint writes use a temporary sibling
and file replacement. Restart the notebook and reconnect the same Drive to
resume: the hourly prefix must have matching times and finite scalar values.
Only work since the last completed checkpoint needs to be repeated for the
same source snapshot.

If a wind hour is missing, even inside the archive's advertised date range,
Notebook 38 immediately saves all finite hourly means before that hour. It
stops the continuous series there, records the exact time and nonfinite-cell
counts in `seasonal_updates/pending_data.csv`, and reports
`updated_with_pending_wind_data` when that prefix can update the catalog.
Dates after the gap are unverified, not non-events. A rolling mean never crosses
the gap. On retry, a compatible hourly checkpoint resumes at that missing hour.
Old checkpoints with a nonfinite tail are trimmed to their finite leading
prefix automatically; do not delete the Drive checkpoint folders.

If fewer than 12 good hourly values precede the gap, or the new source revision
would reduce existing catalog coverage, the catalog remains unchanged and
`pending_scan.json` records the attempted scan. A calculation failure with
finite wind inputs still raises an error. NaNs outside the polygon contribute
no weight; all divergence cells within the polygon, including their required
wind derivative stencils, must be finite.

Final ERA5 partials live in `seasonal_updates/hourly/`. Preliminary ERA5T
partials live in a revision-specific subdirectory there. Their revision key
includes the published `last_updated`, final stop, and preliminary stop metadata.
A restart using that same source revision resumes the saved prefix; a changed
revision uses a new prefix because earlier preliminary winds may have changed.
Finalization also recomputes preliminary months. If `last_updated` is absent,
progress is saved but reuse across runs is disabled because the source revision
cannot be verified. Old revision scalar files are small and remain available.

Completed D12 series live in `seasonal_updates/monthly/`. A completed slice with
unchanged source revision and end time reuses its scalar file without reading
winds again. Catalog CSVs, frozen thresholds, baseline backups, and scan status
also stay in Drive. No full historical ERA5 rebuild is needed for normal updates.

## A November 19 event

Search November 19–20 UTC in Notebook 38 after scanning through the available
reference time. A detected event appears in the shared CSV; an ongoing event or
ERA5T-based event is marked provisional. An event missing because reference
data have not arrived is not declared a non-event.

[ARCO's metadata and latency documentation](https://github.com/google-research/arco-era5#overview)
specifies inclusive `valid_time_stop` for final ERA5, `valid_time_stop_era5t` for
preliminary ERA5T, roughly one week of preliminary-data delay, and roughly three
months for final data. Read those metadata each run; allocated time coordinates
may extend beyond populated fields. Notebook 38 checks the archive revision on
each newly opened source and refreshes preliminary months when that revision
changes or the month becomes final. Notebook 37 refreshes preliminary verification
targets on each verification run. Completed WN2 checkpoints remain reusable.

For a target at November 19 00 UTC, request WN2 initializations November 18,
14, and 9 at 00 UTC for the 1-, 5-, and 10-day leads. The real event's target
retains Notebook 36's nearest-six-hour-time-on-the-original-UTC-date rule,
with ties earlier. Exact member IDs, lead, valid time, pressure and model
archive are required. Missing WN2 runs are reported and retried, with optional
explicit store overrides. The workflow does not generate unavailable WN2 data
or substitute WeatherNext 3. Check actual availability using the approved Google
account; no current-date WN2 availability is assumed from the model's name.

## Refreshing figures

- **35:** click Refresh catalog and screen new cases; exact event-window keys
  prevent old labels from attaching to a revised event. IMERG Final has its own
  publication delay and does not gate ERA5/WN2 verification.
- **36:** click the new Refresh catalog button. It reloads the Drive inventory
  and reviews and invalidates stale archive listings. Check members and load the
  new event; previously computed regional fields remain cached.
- **37:** run in Catalog mode, reload the inventory, and rerun sections 4–6.
  Existing WN2 targets/leads are reused; missing inputs retry. ERA5T values
  refresh. Regenerated tables and plots use the current catalog, rather than
  appending counts from a previous run. Only cases complete at every lead and
  both polygons enter the common sample. All / Vortex / Non-vortex / Unknown
  outputs have their own counts. Only completed valid screens enter the two
  named vortex strata; unchecked, ambiguous, and failed cases remain Unknown.

Updating a CSV does not execute other Colab runtimes or saved images. Rerun
the indicated stages to incorporate the event in the figures. Catalog ranks
remain conditioned on ERA5-detected events; a uniform-rank interpretation needs
that selection context. A 2023-onward study start is provisional until the
operational WN2 training cutoff is established.

## Reliability framework in Notebook 37

For a prespecified instantaneous polygon-divergence threshold T, the probability
is `count(64 WN2 member means < T) / 64`; the outcome is `ERA5 mean < T` at exactly
the same target. The editable −2 × 10⁻⁵ s⁻¹ setting is an exploratory snapshot
criterion, not a fitted threshold or the trailing-D12 JPCZ-event definition.
Each polygon can have its own threshold in `RELIABILITY_THRESHOLDS`.

Section 6 uses the event-conditioned sample shared with the rank histograms.
Section 7 constructs an outcome-independent regular NDJF schedule, including
crossings and non-crossings, for broader winter convergence reliability. Extend
its exclusive end date as winter progresses; future targets are previewed but
not downloaded, and failed due targets remain in the status table. A daily
00 UTC schedule is the default; 6- or 12-hour intervals are supported. Do not
filter that sample to observed events. Missing-case selection can still bias
results and must be reported from the status table.

Both paths save probability/outcome pairs, fixed bin edges, empty bins, counts,
observed frequencies, pointwise Wilson intervals, sample base rates, Brier
scores, forecast-count insets, settings, and six-panel PNG/PDF figures. Bins are
left-closed/right-open except the last bin includes probability one. Empty bins
are not interpreted as zero observed frequency. Wilson intervals ignore serial
dependence and are descriptive; they are not event-block bootstrap uncertainty.

These are snapshot-convergence probabilities. Probabilities of the full JPCZ
detector event need a separate temporal operator and evaluation of how a
six-hourly WN2 approximation matches hourly ERA5 trailing-D12 decisions. Vortex
strata describe reviewed observed events, not probabilistic vortex prediction.

## Validation

Offline synthetic tests cover extension without duplication, immutable baseline
and thresholds, pending ERA5, revised ongoing events, probability 0/1 binning,
strict threshold equality, incomplete cases, unchecked vortex labels, the exact
November 19 initialization dates, notebook compilation and checked helper imports, and the
existing 64-member verification/checkpoint/export workflow. Live private WN2
and Drive retrieval require the user's Colab Google authentication.

A simulated mid-month crash test uses the actual MetPy calculation and NetCDF
files. It confirms resumption after 24 saved hours, six-hour processing batches,
reuse of a completed unchanged ERA5T slice, and recalculation after an archive
revision or finalization. Its resumed hourly and D12 values match Notebook 04's
whole-window calculation on the same synthetic wind grid. This does not replace
an event-for-event comparison on real historical data.

Additional tests reproduce missing interior wind hours with later fields
present, retry and fill the gap, recover an older nonfinite checkpoint tail,
preserve a catalog when data are missing before the first D12 window or when
source revisions reduce coverage, distinguish numerical failures from missing
inputs, and ignore irrelevant nonfinite cells outside the polygon.

The short cell 3 loader is tested with the exact published helper ZIP: normal
imports, a repeat run using the verified local file, checksum rejection, and
an actionable connection error. The bundle is about 21 KB and uses only the
dependencies installed in cell 1. It is cached in the runtime's temporary
directory; scientific checkpoints and catalogs continue to persist in Drive.
Future builder runs must retain older content-addressed bundles in
`notebooks/support/` so previously shared notebook versions keep working.
