# Validation Plan

## Purpose

Validate that the ERA5-based detector is scientifically aligned with the Shinoda December event-selection method and with the known February 2018 case before expanding to a larger NDJF catalog.

## Verification framework

The catalog should be checked in two complementary ways:

- catalog-level verification:
  - the December 2000-2018 detector should recover a similar number of major events to Shinoda, around 35 before subtype labels are applied
- case-study verification:
  - the known `2-7 February 2018` event should be present, with peak convergence on `3 February 2018`

These checks answer different questions:

- the December benchmark tests whether the detector behaves plausibly across many years
- the February 2018 case tests whether the detector can capture a known event in detail

## Validation stage 1: December benchmark

Target period:

- December 2000-2018

Reference behavior from Shinoda:

- 35 major JPCZ events over 19 Decembers
- about 1.84 events per month

Acceptance goal:

- reproduce the overall event frequency and seasonal behavior closely enough that the detector is credible for extension work
- recover the same or a similar number of major events, recognizing that ERA5 and WRF will not match perfectly
- make the event-count comparison before strong/weak monsoon and vorticity subtype labels are applied

What to compare:

- total event count
- event timing clusters within each December
- strength distribution of `D`
- spatial shape of the December climatological convergence band

Required QA plots:

- December climatological `925 hPa` divergence map with masks overlaid
- histogram of December `D` with threshold marker
- multi-year time series of `D`
- event-count summary by year

Expected verification artifact:

- `outputs/verification/december_benchmark_summary.md`

## Validation stage 2: February 2018 baseline case

Target period:

- `2-7 February 2018`

Known expectation from project guidance:

- a JPCZ event is present
- peak convergence occurs on `3 February 2018`

Acceptance goal:

- the detector identifies an event in this period
- the event peak time is consistent with the known baseline

Required QA plots:

- hourly or rolling-mean `D` time series for `2-7 February 2018`
- map sequence around the peak window
- divergence field with polygon overlaid

Expected verification artifact:

- `outputs/verification/feb2018_baseline_summary.md`

## Validation stage 3: classification sanity checks

Goals:

- verify that strong-monsoon cases show stronger northwest flow over the Sea of Japan
- verify that higher-vorticity cases show stronger low-level cyclonic curvature in the dashed rectangle

Checks:

- composites by event label
- event tables with SLP-difference and vorticity values
- spot-check a few events manually

## Adjustment rules

Allowed adjustments during validation:

- small changes to polygon vertices
- small changes to vorticity-box bounds
- choice of rolling-window label convention
- decision on whether to mask land points in the polygon

Rules for changes:

- change one thing at a time
- rerun the December benchmark after each change
- rerun the February 2018 baseline check after each change that affects event timing or peak strength
- document why the change was made

## Risks to monitor

- ERA5 and WRF climatologies are not identical
- the paper does not publish exact mask coordinates
- the paper text does not fully resolve every classification cutoff detail
- cloud-era data access patterns can still be slow if spatial subsetting is done too late

## Decision gate before NDJF production

Do not run the full production catalog until:

- the December benchmark is acceptable
- the February 2018 baseline behaves as expected
- the prototype polygon is judged stable
