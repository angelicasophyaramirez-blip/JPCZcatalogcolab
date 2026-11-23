# Notebook 36/37 batches and saved-output reports

The current field is **derived 925-hPa horizontal divergence**. Its original sign is
preserved: negative is convergence. Both forecasts and ERA5 use instantaneous wind
snapshots at the same onset-aligned target, with the existing spherical derivative,
two-cell halo and polygon weighting. This is separate from the catalog's trailing-D12
event detector. IMERG availability is not required for this verification.

## Notebook 36: selected-member maps

Run the browser setup and inspect one successful event/member. The batch uses that
member's actual native label. Its default fields are:

```python
BATCH_FIELDS = [{'field':'divergence', 'level':925, 'domain':(125.,145.,33.,47.)}]
```

This produces three forecast panels at 1, 5 and 10 days, plus their overview, for each
selected event. Every figure is saved as PNG/PDF, with metadata, settings, attribution,
and a completion marker. Set `BATCH_FIELDS = None` to use the two displayed pilot rows,
or explicitly list multiple fields/levels/domains. The field choices do not silently
change the selected member. Geopotential/height at a non-default level requires explicit
plot limits; signed divergence/component scales must span zero.

Completed exports are validated **before** any archive request: exact event identity,
target/initializations, member, field/level, domains, scales, label metadata, algorithm
version, and every expected nonempty figure/metadata file must match. A changed source
override also invalidates reuse. Each unfinished event runs in a fresh process, with a
6-GiB Linux address-space limit and one numerical-library thread. It exits after writing
its files; no weather arrays or figures are returned to the notebook process. An
interruption terminates the child and leaves earlier complete exports reusable.

Outputs remain in `JPCZcatalog_outputs/weathernext2_atlas/figures/`. A settings-specific
`batch_<id>/batch_progress.csv` records the current inventory without duplicate append
rows. `figure_index.csv` lists each event's saved overview PNG/PDF for later report
assembly. The familiar root `batch_progress.csv` is a latest-run copy.

## Notebook 37: all-member verification

A Notebook 36 map represents one member; ranks and probabilities require **all 64
members**. Notebook 37 retrieves the missing ensemble inputs rather than treating a
single-member map as an ensemble. Both notebooks use
`JPCZcatalog_outputs/wn2_shared_inputs/`: compact, identity-checked regional native
variable arrays, grouped by four member labels, exact initialization/lead, level and
derivative-halo domain. Compatible u/v reads are reused across maps, components, speed
and divergence. Only finite raw inputs are persisted. No credentials are saved there.

Set:

```python
RUN_MODE = 'Catalog'
LOAD_SAVED_RESULTS_ONLY = False
VERIFY_FIELDS = [
    {'field':'divergence', 'level':925, 'threshold':-2.0, 'direction':'less'},
]
```

The example threshold retains the earlier **exploratory** snapshot criterion, not a
validated JPCZ occurrence definition. Choose the criterion before evaluating the sample.
Other supported verification fields are `u`, `v`, `winds` (pixel wind speed from u/v),
`geopotential` (native m²/s²), and `height` (geopotential divided by g, in m). Each field
has its own units, level, checkpoints, ranks, workbook, histograms and reliability
figures. Reliability requires a field-specific threshold and `less` or `greater`;
`threshold: None` exports values/ranks without inventing a probability criterion.
Different fields are never pooled into one histogram.

## Continuous scores and readable reliability plots

Notebook 37 section 5 now calculates **empirical ensemble CRPS** from the 64 saved
polygon-mean values and the matching ERA5 reference. It follows the forecast-CDF versus
observed-step definition in the proposal, using
`mean(abs(X-y)) - sum_ij(abs(X_i-X_j))/(2*M*M)`. Scores are nonnegative, lower is better,
and units match the selected field (10⁻⁵ s⁻¹ for the current signed divergence).
Replacing divergence D by convergence −D leaves CRPS unchanged. Values are not clipped
at zero, and a polygon score does not assess the spatial pattern within the polygon.

`crps_cases.csv` retains each scored event/polygon/lead; `crps_summary.csv` and
`crps_summary.png/pdf` compare the same complete events at every lead. The figure has
sample size, mean/median CRPS, interquartile range and ERA5T counts underneath.
The detailed workbook includes those score tables. The one-sheet collector joins the
saved score as `case_crps`, repeated for each member/reference row belonging to that
case. All/Vortex/Non-vortex/Unknown CRPS outputs also save under `event_conditioned/`.

**CRPSS is not calculated without an independent reference.** The proposal specifies
climatology outside the test sample. Test events are not recycled as their own baseline.
For now, raw CRPS is reported without a skill-against-climatology claim.

Reliability uses percent axes, a clear 45° reference line, and a statistics table below
each panel: usable cases, observed crossing count/frequency, mean forecast probability,
Brier score and occupied bins. Notebook display splits the two polygons into separate
figures, one polygon and one lead at a time, fitted responsively to the notebook width.
The six-panel overview and three-lead polygon figures remain saved for comparison.
`forecast_probability_counts.png/pdf` is a separate
companion figure. Pointwise Wilson 95% intervals remain in `reliability_bins.csv` and
can optionally be shown; empty bins are not presented as zero observed frequency.
The summary CSV also records dates, criterion, units and ensemble size.

`event_accounting.csv` lists all selected events, common-sample membership, number of
usable forecast leads, and full retrieval errors. The catalog preview prints disjoint
onset-year counts. Counts after 2024 are already within counts after 2022; they must not
be added together. In the Drive inventory audited on 2026-11-22, 14 events in 2023,
8 in 2024 and 12 in 2025 gave 34 older cases; five new 2026 cases brought selection to
39. Forty raw detector fragments become 39 merged events because one episode has two
fragments. These are audit results, not hard-coded limits.

The new-event wind failure was traced to `xarray.merge(combine_attrs='drop')`, which
also drops variable-level attributes. Shared cached u/v files retained correct units;
the merge now restores their variable attributes. Existing valid cached inputs and
the 34 completed forecast checkpoints remain reusable. This repair does not change
the divergence formula or invalidate the original results.

Each forecast lead is reduced in four-member source groups. Every group saves its two
polygon means immediately. The interpreter is recycled after 16 members, so a long
64-member verification does not retain all of its decoder buffers in one process.
If a lead is interrupted, its completed member groups remain reusable. Existing complete
925-hPa divergence target/lead checkpoints retain their original identities. ERA5T
references remain labeled preliminary and refresh under the existing retrieval policy.
Reference and forecast failures stay explicit; absent values are not zeros.

Each worker receives a current Google access token through an in-memory pipe. Refreshable
credentials remain in the Colab parent; tokens are not saved in Drive, request files,
or command arguments. Children use only the approved WN2 archive and explicit run
overrides. A content-verified worker archive packages the same implementation embedded
in the notebooks. Timeout/interruption cleans up its child process.
After closed checkpoint/result files and flushed logs, each single-job child exits
directly. This avoids gcsfs/aiohttp finalizers trying to close sessions on a different
event loop during interpreter teardown. Computation errors remain explicit in the
validated reply; completed member groups survive an interrupted worker.

## Reusing Drive outputs without weather recalculation

### CORP calibration view

Notebook 37 section 6 defaults to `RELIABILITY_METHOD = 'CORP'`. Each existing
`event_conditioned/<stratum>/` retains the fixed-bin figures; its `corp/` subfolder
contains six separate PNG/vector-PDF diagrams, exact probability-value counts,
`corp_points.csv`, `corp_cases.csv`, `corp_pools.csv`, `corp_summary.csv` and `corp_settings.json`.
The source is Dimitriadis, Gneiting and Jordan (2021), doi:10.1073/pnas.2016191118.

Only saved binary forecast probabilities and ERA5 crossing outcomes enter this view.
No ERA5/WN2 archive reader is called. Tied probability values are aggregated with their
case counts before weighted PAV; pool boundaries are automatic. Each distinct probability
is a plotted x-value, using the paper's discrete convention for 64-member probabilities.
Original p and outcomes remain in the case table alongside fitted diagnostic q.
The original Brier score, CRPS, rank positions and catalog membership are preserved.

Reported Brier components satisfy BS = MCB − DSC + UNC: MCB is raw BS minus fitted
in-sample BS; DSC is outcome variance minus fitted BS; UNC is outcome variance.
These descriptive in-sample components do not establish out-of-sample recalibration
skill. The base rate used here is not the independent climatology required for CRPSS.

Band defaults are `CORP_CONSISTENCY_BANDS = False` and
`ASSUME_EXCHANGEABLE_CASES = False`. Explicitly enabling both requests seeded 2,000-draw,
90% pointwise calibration-null consistency bands at fixed forecast values. These do not
model winter dependence and are not confidence bands. The curve itself remains available
without adopting that uncertainty assumption. The paper's serial-dependence discussion
points to Bröcker and Ben Bouallègue (2020), reference 41, for stratified rank histograms;
CORP does not replace the continuous-ensemble rank histograms.

Validation compares weighted PAV against SciPy's independent isotonic solver,
checks tied-value and case-order invariance, and reproduces all four published
Niamey Brier decompositions in Table 2 using the authors' public example data.

After retrieval, set `LOAD_SAVED_RESULTS_ONLY = True` to reopen the exported CSV tables.
Missing saved tables cause a clear error; this mode never silently starts a download.
New catalog cases without saved values are shown as not downloaded. The report cells
can redraw statistical summaries from the stored scalar values; they do not derive
divergence again or read ERA5/WN2 arrays.

Section 5 also collects the saved per-field `ordered_values.csv`, `member_values.csv`,
and `era5_reference.csv` rows into:

```text
JPCZcatalog_outputs/weathernext2_ensemble_verification/comparison/
  all_verification_values.xlsx
  all_verification_values.csv
```

The workbook has **one sheet, All values**, with event identity, variable, pressure,
units, polygon, lead, valid/initialization times, ERA5/WN2 source, native member label,
value, rank position where available, and saved-table provenance. ERA5 rows are
highlighted. Valid member values are retained even when a missing reference/lead
prevents a rank. Existing detailed per-field workbooks remain available separately.
Collecting rows does not retrieve or recalculate weather fields.

The future event-by-event comparison PDF is intentionally deferred. It should read
existing saved figure files and indexes, join by event identity, preserve the actual
timestamps/member labels, and place the ERA5 and WN2 pages together. It must not call
the archive readers or the field derivation functions. Notebook 35's peak±12-hour views
and Notebook 36's onset-aligned target are not automatically the same valid time.

## Validation and limits

Offline tests cover real PNG/PDF completion checks, preflight reuse with archive reads
forbidden, a three-panel divergence-only export, interrupted member-group recovery,
shared u/v cache reuse, native-component/height/geopotential units and ranks, separate
field identities, saved-only reporting, and the one-sheet Excel collector including
unranked member values. Worker tests execute fresh interpreters with deliberate retained
buffers, verify all 64 distinct native labels, reject mixed source/timing parts, check
token exclusion from files, and terminate children on timeout/interrupt.

These synthetic checks do not establish private WN2 access, eliminate network/archive
outages, or measure an end-to-end authenticated catalog run. Verify one authenticated
pilot before a large batch. Source chunk sizes, remote bandwidth, rendering, and Drive
latency still determine total runtime; reuse avoids repeating completed work.
