# JPCZ Detector Spec

## Goal

Detect major JPCZ events in a way that is faithful to Shinoda et al. while adapting the workflow from WRF output to cloud-accessed ERA5.

Important reproduction principle:

- The Shinoda method is the conceptual detector framework.
- The exact finite-difference implementation on the ERA5 grid is ours.
- So this workflow should be described as a faithful ERA5 implementation of the Shinoda detector logic, not as a verbatim copy of every unpublished numerical implementation detail from the paper.

## Core Shinoda event metric

Primary field:

- `925 hPa horizontal divergence`
- units: `s^-1`
- display units for maps and time series: usually `10^-5 s^-1`
- sign convention:
  - negative = convergence
  - positive = divergence

Primary event metric:

- area-mean `925 hPa horizontal divergence` within the JPCZ detection polygon
- computed from hourly data
- converted to `12-hour mean` values, denoted here as `D`

Clean distinction:

What Shinoda did:

- detect major JPCZ events from polygon-mean `925 hPa` horizontal divergence
- use `12 h` mean values
- identify major events when the polygon-mean divergence becomes anomalously negative

What we do to reproduce that logic in ERA5:

- compute gridded divergence from ERA5 winds:
  - `div925 = du/dx + dv/dy`
- use finite differences on the ERA5 grid
- apply our digitized Shinoda polygon mask
- compute the cosine-latitude-weighted polygon mean:
  - `D_hourly(t) = sum(mask * cos(lat) * div925) / sum(mask * cos(lat))`
- compute the trailing `12 h` rolling mean:
  - `D_12h(t_k) = (1 / 12) * sum_{m=0}^{11} D_hourly(t_{k-m})`
- threshold:
  - `threshold = mean(D_12h) - 2 * std(D_12h)`
- group consecutive threshold hits into raw events
- merge nearby raw fragments if the quiet gap is `<= 12 h`

So the workflow is Shinoda-style because the detection logic is preserved, while the gridded numerical implementation is ours.

More explicitly:

- at each hourly time step, compute a full gridded `925 hPa` divergence field
  - `div925 = du/dx + dv/dy`
- on the ERA5 lat-lon grid, the finite-difference idea is:
  - `du/dx(i, j) ~= (u(i, j+1) - u(i, j-1)) / (x(i, j+1) - x(i, j-1))`
  - `dv/dy(i, j) ~= (v(i+1, j) - v(i-1, j)) / (y(i+1, j) - y(i-1, j))`
- the code uses MetPy's divergence operator together with ERA5 grid spacing from `lat_lon_grid_deltas`, so the implementation handles the exact grid-metric details on the native ERA5 mesh
- then apply the polygon mask and compute the hourly polygon-area-mean series:
  - `D_hourly(t) = sum_ij(mask(i, j) * cos(lat_i) * div925(i, j, t)) / sum_ij(mask(i, j) * cos(lat_i))`
- then compute the trailing `12 h` rolling mean:
  - `D_12h(t_k) = (1 / 12) * sum_{m=0}^{11} D_hourly(t_{k-m})`
- example:
  - `D_12h(2018-02-03 12:00 UTC)` is the mean of the hourly polygon-mean divergence values from `2018-02-03 01:00 UTC` through `2018-02-03 12:00 UTC`

Major-event threshold:

- flag a major JPCZ event when:
  - `D < mean(D) - 2 * std(D)`

Interpretation:

- the threshold is relative to the dataset climatology
- the detector is meant to isolate anomalously strong convergence, not average winter conditions

## Spatial definitions

### 1. Broad ERA5 working domain

Initial prototype domain for access and plotting:

- longitude: `120E to 150E`
- latitude: `30N to 50N`

Extended plots may optionally use:

- longitude: `120E to 160E`
- latitude: `30N to 50N`

### 2. JPCZ detection polygon

The paper defines this region graphically in Figure 2 but does not publish exact vertices. The prototype will use a first-pass digitized polygon that can be refined during benchmarking.

Initial prototype vertices, in `(lon, lat)`:

- `(129.5, 41.0)`
- `(136.0, 37.4)`
- `(134.5, 35.8)`
- `(128.8, 38.0)`

Status:

- these coordinates are an implementation estimate from the figure
- they are not printed in the paper
- they will be tuned only if the December benchmark clearly disagrees with Shinoda

Applied mask meaning:

- build a boolean `latitude x longitude` array on the ERA5 grid
- set `mask(i, j) = 1` if the grid-cell center `(lon_j, lat_i)` falls inside the polygon
- set `mask(i, j) = 0` otherwise
- only the `mask = 1` cells contribute to the polygon-area-mean detector metric

### 3. Relative-vorticity box

The paper also defines a dashed rectangle in Figure 2 for vorticity classification. Initial prototype rectangle:

- longitude: `127E to 140E`
- latitude: `37N to 45.5N`

Status:

- first-pass digitization from the figure
- subject to minor refinement during validation

## Temporal definitions

Input cadence:

- hourly ERA5 fields

Rolling aggregation:

- compute hourly divergence first
- then compute a `12-hour rolling mean` for the polygon-mean divergence series

Open implementation detail:

- the paper states that divergence was calculated every 1 hour and that 12-hour mean values were used, but it does not make the window-label convention fully explicit
- first implementation should use a trailing 12-hour rolling mean labeled by the window end time
- if event timing appears systematically shifted in validation, revisit this convention

Current implementation choice:

- use the trailing, end-labeled convention described above
- require all `12` hourly values to be present before a `D_12h` value is emitted

## Event grouping

If the major-event threshold is met at consecutive time steps:

- treat the consecutive windows as one event
- assign the event timestamp to the time of peak convergence
- peak convergence means the most negative `D` within the run

To avoid splitting one broader synoptic episode into many weakening-and-reforming fragments, the NDJF workflow then performs a second merge step:

- sort the raw threshold events by start time
- compute the quiet gap between one event end and the next event start
- if the gap is less than or equal to the chosen merge threshold, merge them into one event
- the current recommended merge threshold is `12 h`

Example:

- raw event A ends at `06:00 UTC`
- raw event B starts at `15:00 UTC`
- gap = `9 h`
- because `9 <= 12`, merge A and B into one broader episode

For the merged event:

- `event_start` = start of the first raw fragment
- `event_end` = end of the last raw fragment
- `event_peak` = time of the most negative `D_12h` across all merged fragments
- `threshold_hit_hours` = sum of hours actually below threshold
- `duration_hours` = total span of the merged episode, including internal above-threshold gaps

Recommended event table fields:

- `event_id`
- `event_time_peak_utc`
- `window_start_utc`
- `window_end_utc`
- `divergence_mean_peak_s-1`
- `divergence_mean_peak_1e5_s-1`
- `duration_hours_above_threshold`
- `month`
- `year`
- `notes`

## Classification fields

These do not trigger event detection. They are post-detection labels.

### Strong vs weak monsoon

Field:

- `sea level pressure difference`
- definition: `Seoul minus Sapporo`
- units: `hPa`

Planned labels:

- `Type 1 strong-monsoon`
- `Type 2 weak-monsoon`

Implementation note:

- the paper clearly defines the index, but the exact numeric split for "normal monsoon intensity" is not fully spelled out in the extracted text
- first implementation should use the period-mean index as the split
- document this as a reproduction assumption until we verify otherwise

### Lower vs higher vorticity

Field:

- area-mean `925 hPa relative vorticity`
- units: `s^-1`
- display units: usually `10^-5 s^-1`

Planned labels:

- `Type 1A lower-vorticity`
- `Type 1B higher-vorticity`

Threshold:

- split Type 1 events using:
  - `zeta > mean(zeta) + std(zeta)` for `Type 1B higher-vorticity`
  - otherwise `Type 1A lower-vorticity`

## Adaptation from WRF to ERA5

The paper's threshold is relative to their WRF climatology. We will carry over the method, not any WRF-specific numeric threshold.

ERA5 workflow:

1. compute ERA5 hourly divergence within the prototype polygon
2. compute ERA5 December `D`
3. compute ERA5 December `mean(D)` and `std(D)`
4. apply the same `mean - 2 sigma` rule

## Reproduction and production modes

### Mode A: December benchmark

Purpose:

- reproduce Shinoda-like December behavior for 2000-2018

Baseline rule:

- compute thresholds from December only

Target:

- event frequency should be broadly consistent with Shinoda's 35 major events over 19 Decembers
- the comparison should be made on the total detected-event catalog before strong/weak monsoon and vorticity subtype labels are applied
- a similar count is the goal, not an exact one-to-one replication, because ERA5 and WRF are not identical

### Mode B: NDJF production catalog

Purpose:

- expand the catalog to November, December, January, and February

Recommended thresholding strategy:

- compute thresholds separately by month
- example:
  - November windows compared against November climatology
  - December windows compared against December climatology
  - January windows compared against January climatology
  - February windows compared against February climatology

Reason:

- this keeps the anomaly threshold tied to each month's background state instead of pooling months with different seasonal means

## Open science items

- confirm whether land points inside the polygon should be retained or masked
- verify the best rolling-window convention
- refine polygon vertices if December benchmarking shows a clear mismatch
- confirm whether the strong/weak monsoon split should use the full-period mean or another baseline
- confirm whether the February 2018 baseline event suggests any timing adjustment to the rolling-window convention
