# Historical ERA5 JPCZ catalog: 1940/41--2025/26

## Purpose and catalog versioning

The established catalog remains the fixed 2000--2025 ERA5 product used for
the Shinoda-period comparison and the IMERG analysis.  The historical catalog
is a separate, versioned product for the question: *which NDJF episodes were
unusually convergent relative to the full available ERA5 historical record?*

No established CSV, NetCDF checkpoint, or threshold is overwritten by this
workflow.  Existing 2000--2025 monthly D12 checkpoints are read-only inputs
when their files are present and pass the inventory check.

## Complete-season sample

The historical record uses complete NDJF seasons:

\[
\text{NDJF 1940/41 through NDJF 2025/26}.
\]

This includes November--December 1940 through January--February 2026.  It
does not include January--February 1940 because the preceding
November--December 1939 months are outside ERA5's record.  It also does not
include November--December 2026, which belong to the next incomplete season.

## Detector and baseline

The historical detector is not a new physical method.  It uses the existing
Shinoda-style implementation:

1. hourly ERA5 925-hPa horizontal winds (`u`, `v`);
2. MetPy horizontal divergence on the ERA5 grid;
3. cosine-latitude-area-weighted mean divergence in the digitized Shinoda
   polygon;
4. trailing, end-labelled 12-hour mean, \(D_{12}\);
5. separate November, December, January, and February thresholds;
6. grouping consecutive threshold-hit hours into raw fragments; and
7. merging raw fragments separated by a quiet gap of no more than 12 hours.

The important difference is the anomaly baseline.  For each calendar month
\(m\), the threshold is recomputed from **all** eligible 12-hour values in
the complete 1940/41--2025/26 record:

\[
T_m = \overline{D_{12,m}} - 2\sigma_{D_{12,m}}.
\]

The existing 2000--2025 thresholds are not reused.  Reusing saved D12 values
does not make the historical mean partial: they are simply their portion of
the full-record calculation.

## Checkpoint and resume behavior

The historical workflow has its own Google Drive directory and writes:

- an inventory CSV after every completed calendar month;
- a NetCDF D12 checkpoint after every newly calculated month, using a
  temporary file followed by an atomic rename;
- raw threshold outputs only after the monthly inventory is complete;
- a separate checkpoint for the event-environment classification; and
- merged catalog and old-versus-new comparison tables.

On rerun, a completed historical checkpoint is preferred.  Otherwise, a
matching legacy 2000--2025 monthly checkpoint is reused read-only.  Only a
month missing from both locations is requested from ERA5/ARCO.

## Interpretation and comparison

Changing the baseline can change which 2000--2025 events qualify.  This is
expected and does not alter the established catalog.  The workflow compares
the two 12-hour-gap merged catalogs by temporal overlap:

- **retained:** an established episode overlaps a historical-baseline episode;
- **established only:** no historical episode overlaps it (not flagged by the
  historical baseline);
- **historical only:** a historical-baseline episode has no established
  overlap.

Temporal overlap, rather than exact peak-hour equality, is used because a
changed threshold can shorten or broaden a retained physical episode.

## Scope limitation

ERA5 supports the historical wind detector from 1940.  However, the observing
system is less complete before the satellite era, so pre-1970/1979 event
statistics should be interpreted cautiously.  IMERG precipitation does not
extend to this early period; precipitation association analyses remain
restricted to the IMERG era.
