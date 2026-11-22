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

## Reusing Drive outputs without weather recalculation

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
