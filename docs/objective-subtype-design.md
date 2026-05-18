# Objective Subtype Design

## Purpose

This note captures the working experimental design for objective JPCZ subtype analysis after the merged `12 h` NDJF catalog has already been detected.

The goal is not to re-detect events. The goal is to determine whether the detected JPCZ events separate objectively into distinct forcing or modifier regimes using ERA5-derived fields.

## Core design principles

1. Keep the original Shinoda-style JPCZ detection polygon fixed for event detection.
2. Add new regions only for event characterization, not for event detection.
3. Do not train the clustering on subjective subtype labels.
4. Use the subjective labels only afterward as a physical interpretation check.
5. Separate cause or modifier clustering from magnitude clustering.
6. Prefer physically interpretable event-level metrics over raw image clustering in the first pass.

## Detection versus characterization

The project now has two separate layers.

### Detection layer

- Event detection remains tied to the existing JPCZ polygon.
- The core detector remains the `12 h` mean polygon-mean `925 hPa` divergence metric `D`.
- The merged NDJF catalog remains the working event list.

### Characterization layer

- Additional boxes or polygons may be defined to describe where the strongest convergence occurs and whether synoptic forcing is present.
- These added regions do not create new events and do not double count events.
- Each event remains one row in the catalog.
- New metrics simply describe how that one event behaves spatially and dynamically.

## Objective clustering target

The clustering should test whether the `201` detected JPCZ events separate naturally into different forcing or modifier structures.

The first objective clustering experiment should focus on:

- low-level convergence structure from `del dot u`
- passing-low or trough forcing from `850 hPa` geopotential-height anomaly
- frontal or baroclinic forcing from `850 hPa` temperature-gradient magnitude
- optionally low-level circulation from `del cross u`

## Feature naming rules

Feature names should be explicit enough that they are still understandable later without referring back to notebook code.

Use names that encode:

- field
- level
- region
- summary statistic
- time window when relevant

Avoid overly cryptic shorthand such as `C_core`, `Z850'`, or `V_soj` in saved tables.

## First-pass feature dictionary

The first clustering experiment should use the following event-level features.

### 1. JPCZ polygon convergence

Column names:

- `jpcz_polygon_mean_convergence_peak_925`
- `jpcz_polygon_max_convergence_peak_925`

Meaning:

- Convergence computed from `del dot u` at `925 hPa` within the original JPCZ polygon at the event peak time.

Calculation:

- Compute `925 hPa` divergence from hourly ERA5 `u` and `v`.
- Multiply by `-1` so positive values represent convergence magnitude.
- Summarize the convergence field only within the original JPCZ polygon.

Units:

- `s^-1`
- display units usually `1e-5 s^-1`

Purpose:

- Quantifies the strength of convergence in the canonical JPCZ detection region.

### 2. Coastal-Japan convergence

Column names:

- `coastal_japan_mean_convergence_peak_925`
- `coastal_japan_max_convergence_peak_925`
- `coastal_to_jpcz_convergence_ratio`

Meaning:

- Convergence computed from `del dot u` at `925 hPa` in a new coastal-Japan characterization region.

Calculation:

- Compute the same convergence field used above.
- Summarize it within a coastal-Japan box or polygon designed only for characterization.
- Compute the ratio:
  - `coastal_to_jpcz_convergence_ratio = coastal_japan_mean_convergence_peak_925 / jpcz_polygon_mean_convergence_peak_925`
  - or use the max-based ratio if later testing shows it is more stable.

Units:

- convergence fields in `s^-1`
- ratio is unitless

Purpose:

- Measures whether the strongest convergence is polygon-centered or shifted toward coastal Japan.

### 3. Pacific-side convergence

Column names:

- `pacific_east_of_japan_mean_convergence_peak_925`
- `pacific_east_of_japan_max_convergence_peak_925`
- `pacific_to_jpcz_convergence_ratio`

Meaning:

- Convergence computed from `del dot u` at `925 hPa` in a Pacific box east of Japan.

Purpose:

- Measures whether the event is coupled to stronger convergence east of Japan rather than being confined to the Sea of Japan side.

### 4. Hokkaido passing-low metric

Column names:

- `hokkaido_min_z850_anomaly_tminus12_to_tplus12`

Meaning:

- Minimum `850 hPa` geopotential-height anomaly in a Hokkaido or northern-Japan box over the event window.

Calculation:

- Convert ERA5 geopotential to geopotential height at `850 hPa`.
- Define:
  - `z850_anomaly = event-time z850 - background climatological z850`
- Evaluate the anomaly field at `t-12 h`, `t0`, and `t+12 h`.
- Save the most negative value in the Hokkaido box across those three times.

Units:

- `gpm`

Purpose:

- Quantifies the strength of passing-low or trough forcing near Hokkaido.

### 5. Sea of Japan synoptic-height metric

Column names:

- `sea_of_japan_min_z850_anomaly_tminus12_to_tplus12`

Meaning:

- Same geopotential-height anomaly logic as above, but in a Sea of Japan box.

Purpose:

- Quantifies whether the synoptic-height depression is centered more over the Sea of Japan than near Hokkaido.

### 6. Frontality metric

Column names:

- `front_box_max_temp_gradient_850_tminus12_to_tplus12`
- `pacific_box_max_temp_gradient_850_tminus12_to_tplus12`

Meaning:

- Maximum horizontal temperature-gradient magnitude at `850 hPa` in the chosen front-sensitive box over the `t-12`, `t0`, `t+12` window.

Calculation:

- Compute `|grad T850|` from ERA5 temperature.
- Use a regional maximum to capture the strongest frontal or baroclinic signature.

Units:

- raw units `K m^-1`
- display units may be scaled to `K (100 km)^-1`

Purpose:

- Quantifies whether a frontal or baroclinic zone is present.

### 7. Optional Sea of Japan circulation metric

Column names:

- `sea_of_japan_mean_vorticity_peak_925`
- `sea_of_japan_max_vorticity_peak_925`

Meaning:

- Relative vorticity from `del cross u` at `925 hPa` summarized in a Sea of Japan box.

Purpose:

- Quantifies circulation-centered events that may not be explained only by polygon or coastal convergence strength.

## Time-window logic

The event peak snapshot alone may miss a passing low or front. Therefore the first synoptic forcing experiment should use:

- `t-12 h`
- `t0`
- `t+12 h`

This window should be used for:

- `850 hPa` geopotential-height anomaly
- `850 hPa` temperature-gradient magnitude
- optionally Sea of Japan or Hokkaido vorticity

## First objective clustering experiment

The first forcing or modifier clustering should use a small, interpretable feature set.

Recommended first-pass feature set:

- `coastal_to_jpcz_convergence_ratio`
- `hokkaido_min_z850_anomaly_tminus12_to_tplus12`
- `front_box_max_temp_gradient_850_tminus12_to_tplus12`
- optionally `sea_of_japan_mean_vorticity_peak_925`

This first experiment is designed to test:

- polygon-centered versus coastal-enhanced convergence
- mesoscale versus synoptic forcing
- frontal influence versus non-frontal influence
- optional circulation-centered structure

## First diagnostic scatterplots

Before formal clustering, make event-level scatterplots such as:

1. `coastal_to_jpcz_convergence_ratio` versus `jpcz_polygon_max_convergence_peak_925`
2. `coastal_to_jpcz_convergence_ratio` versus `hokkaido_min_z850_anomaly_tminus12_to_tplus12`
3. `pacific_to_jpcz_convergence_ratio` versus `front_box_max_temp_gradient_850_tminus12_to_tplus12`
4. `sea_of_japan_mean_vorticity_peak_925` versus `hokkaido_min_z850_anomaly_tminus12_to_tplus12`

These plots should be inspected before clustering to see whether any natural separation exists.

## Clustering workflow

The clustering workflow should be:

1. build the event-level feature table
2. standardize the features
3. inspect 2D and 3D scatterplots
4. run PCA for visualization
5. test clustering algorithms such as:
   - hierarchical clustering
   - Gaussian mixture clustering
6. test several cluster counts such as `k = 3`, `4`, `5`, and `6`

## How success should be judged

The clustering should not be judged only by whether it reproduces subjective notes.

Success should be evaluated by:

- separation in feature space
- cluster stability under resampling
- physical interpretability of the cluster composites
- post hoc comparison with subjective satellite-based notes

## Role of subjective interpretation

Subjective notes remain valuable, but they are not the clustering target.

They should be used later to ask:

- do objectively derived clusters resemble any of the visually inferred event families?
- do they separate synoptic-scale forced cases from mesoscale or polygon-centered cases?

## Magnitude analysis is separate

Magnitude should not be mixed blindly into the first forcing-cluster experiment.

A later magnitude-focused experiment can use:

- `event_peak_D_1e5_s-1`
- duration
- `jpcz_polygon_max_convergence_peak_925`
- `coastal_japan_max_convergence_peak_925`
- integrated convergence metrics if added

## Not first priority

These are important but not first-priority inputs to the clustering design:

- cross sections
- station snowfall data
- vertical motion
- jet position and entrance-region analysis

Those should come later as diagnostic or interpretation tools after the first objective feature space is built.
