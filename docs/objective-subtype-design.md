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
- The core detector remains the `12 h` mean polygon-area-mean `925 hPa` divergence metric `D`.
- The merged NDJF catalog remains the working event list.

Detector-reproduction principle:

- The Shinoda method is the conceptual detector framework.
- The exact finite-difference implementation on the ERA5 grid is ours.
- So the current catalog should be described as a faithful ERA5 implementation of the Shinoda detector logic rather than a claim that every unpublished low-level numerical choice from the original workflow is known exactly.

What Shinoda did conceptually:

- Start with hourly `925 hPa` horizontal divergence.
- Average that divergence over one fixed JPCZ polygon.
- Convert the hourly polygon-mean series to `12 h` mean values.
- Define major events where the `12 h` polygon-mean divergence becomes anomalously negative relative to its own climatology.

What this project does in direct ERA5 implementation:

- Use hourly ERA5 `u` and `v` winds at `925 hPa`.
- Compute a full gridded divergence field at every hourly time step.
- Apply a fixed polygon mask corresponding to the digitized Shinoda JPCZ polygon.
- Compute an area-weighted polygon mean of that gridded divergence field.
- Apply a trailing `12 h` rolling mean to that hourly polygon-mean series.
- Threshold the smoothed series using `mean - 2 * std`.
- Group consecutive threshold hits into raw detector events.
- Merge short-gap fragments into broader NDJF episodes with the gap-merging workflow from `Notebook 06`.

Clean distinction:

What Shinoda did:

- detect major JPCZ events from polygon-mean `925 hPa` horizontal divergence
- use `12 h` mean values
- identify major events when the polygon-mean divergence becomes anomalously negative

What we do to reproduce that in ERA5:

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

So yes, the detector is Shinoda-style, but it is Shinoda-style because the major-event logic is preserved, not because the paper provided every exact low-level implementation detail.

Implemented detector math:

- For one hourly ERA5 `925 hPa` wind snapshot, horizontal divergence is computed at every grid cell as:
  - `div925 = du/dx + dv/dy`
- Here:
  - `du/dx` is the east-west derivative of the zonal wind
  - `dv/dy` is the north-south derivative of the meridional wind
- The derivatives are evaluated on the native lat-lon ERA5 grid using local grid spacing from `lat_lon_grid_deltas`.
- In the current implementation, the finite-difference derivatives are evaluated with MetPy's divergence operator, so the result is a full gridded divergence field with units of `s^-1`.
- A schematic centered-difference example for one grid cell `(i, j)` is:
  - `du/dx(i, j) ~= (u(i, j+1) - u(i, j-1)) / (x(i, j+1) - x(i, j-1))`
  - `dv/dy(i, j) ~= (v(i+1, j) - v(i-1, j)) / (y(i+1, j) - y(i-1, j))`
- MetPy handles the actual grid-metric details and edge treatment internally; the equations above are the reproducible finite-difference idea behind the implementation.
- A fixed polygon mask is then applied using the original Shinoda-style JPCZ polygon.
- In practice, "apply the polygon mask" means:
  - build a boolean array with the same `latitude x longitude` shape as the ERA5 slice
  - mark a cell `True` when that grid-cell center falls inside the polygon vertices
  - mark it `False` otherwise
- So the detector uses only the grid cells whose centers fall inside the polygon when computing the polygon mean.
- If the polygon vertices are `(lon_1, lat_1), ..., (lon_n, lat_n)` and the grid-cell center is `(lon_j, lat_i)`, then:
  - `mask(i, j) = 1` if `(lon_j, lat_i)` is inside the polygon
  - `mask(i, j) = 0` otherwise
- The current polygon vertices are the first-pass digitization of Shinoda Figure 2:
  - `(129.5, 41.0)`
  - `(136.0, 37.4)`
  - `(134.5, 35.8)`
  - `(128.8, 38.0)`
- Inside that polygon, the detector builds an hourly polygon-area-mean divergence series:
  - `D_hourly(t) = area_mean_polygon(div925(t))`
- The spatial mean is cosine-latitude weighted so that grid cells are area weighted rather than counted equally by row.
- More explicitly, if `w(i, j) = cos(lat_i)` and `mask(i, j)` is the polygon mask, then:
  - `D_hourly(t) = sum_ij(mask(i, j) * w(i, j) * div925(i, j, t)) / sum_ij(mask(i, j) * w(i, j))`
- The event detector then computes the Shinoda-style `12 h` rolling mean:
  - `D_12h(t) = rolling_mean_12h(D_hourly(t))`
- In the current implementation this is a trailing, end-labeled rolling mean:
  - `D_12h(t_k) = (1 / 12) * sum_{m=0}^{11} D_hourly(t_{k-m})`
- Example:
  - if the hourly polygon-mean series has values at `01:00, 02:00, ..., 12:00 UTC`
  - then `D_12h(12:00 UTC)` is the mean of those `12` hourly values
  - this is not a centered window and it is not a forward-looking window
- Because the detector is based on divergence, more negative `D_12h` values correspond to stronger polygon-mean convergence.
- The threshold is computed from the full valid `D_12h` series as:
  - `threshold = mean(D_12h) - 2 * std(D_12h)`
- An event is identified wherever the `12 h` polygon-mean divergence series drops below that threshold.
- Consecutive below-threshold hours are grouped into one event.
- For each grouped event:
  - `event_start` = first below-threshold time
  - `event_end` = last below-threshold time
  - `event_peak` = time of the minimum `D_12h` value within that grouped event
  - `event_peak_D` = minimum `D_12h` value within that grouped event
- The merged NDJF catalog later broadens nearby threshold fragments into larger synoptic episodes, but the core event membership still originates from this polygon-mean divergence-threshold process.

Why the merged NDJF catalog is not just a list of weakening and reforming fragments:

- The raw threshold detector only knows about consecutive below-threshold hours.
- If one threshold run ends, then a short above-threshold gap occurs, then another threshold run starts, the raw detector would initially create two separate events.
- `Notebook 06` resolves this by merging adjacent detector events separated by a short threshold-free gap.
- The current recommended merge gap is `12 h`.
- Two raw events are merged when:
  - `gap_hours = (next_event_start - previous_event_end) <= 12`
- Example:
  - raw event A ends at `2004-01-21 06:00 UTC`
  - raw event B starts at `2004-01-21 15:00 UTC`
  - `gap_hours = 9`
  - because `9 <= 12`, they are merged into one broader episode
- For the merged event:
  - `event_start` becomes the start of the first fragment
  - `event_end` becomes the end of the last fragment
  - `event_peak` remains the time of the most negative `D_12h` across the merged fragments
  - `threshold_hit_hours` stores the sum of threshold-hit hours across fragments
  - `duration_hours` becomes the total episode span, including the internal gap
- This means the working NDJF merged catalog is intended to represent broader synoptic episodes rather than every temporary threshold interruption as a new event.

Important distinction:

- The detection metric `D` is a polygon-mean divergence series, not a convergence-magnitude field.
- The subtype-characterization layer now uses the same signed divergence continuum:
  - `div925 = du/dx + dv/dy`
- So the detector and the subtype characterization both work from signed `del dot U`; convergence and divergence are allowed to cancel inside a predefined region when the regional mean is computed.

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

The first clustering notebook should test a small range of cluster counts such as `k = 2` through `k = 6` instead of assuming one correct answer from the start. It should report simple quality diagnostics such as mean silhouette score, smallest cluster size, and largest cluster size before choosing a first-pass solution for interpretation.

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

### 1. JPCZ polygon signed divergence

Column names:

- `jpcz_polygon_mean_divergence_peak_925`
- `jpcz_polygon_min_divergence_peak_925`

Meaning:

- Signed divergence computed from `del dot u` at `925 hPa` within the original JPCZ polygon at the event peak time.

Calculation:

- Compute `925 hPa` divergence from hourly ERA5 `u` and `v`.
- Keep the signed field:
  - `div925 = du/dx + dv/dy`
- Summarize the signed-divergence field only within the original JPCZ polygon.
- Use cosine-latitude area weighting for means.
- Save:
  - `jpcz_polygon_mean_divergence_peak_925 = mean_polygon(div925)`
  - `jpcz_polygon_min_divergence_peak_925 = min_polygon(div925)`

Units:

- `s^-1`
- display units usually `1e-5 s^-1`

Purpose:

- Quantifies the signed divergence in the canonical JPCZ detection region. More negative values mean stronger polygon-mean convergence; more positive values mean polygon-mean divergence.

Current zero and missing-data handling:

- Real zeros in `div925` are included in spatial means.
- Missing values inside a selected region are excluded from both the weighted numerator and weighted denominator.
- Region masks are still applied first so only in-region cells can contribute.
- This rule is applied consistently across polygon and box means and across all later composite averaging.
- A later sensitivity experiment should compare this current implementation against an alternate treatment to quantify whether zero-handling materially changes the subtype results.

### 2. Coastal-Japan signed divergence

Column names:

- `coastal_japan_mean_divergence_peak_925`
- `coastal_japan_min_divergence_peak_925`
- `coastal_to_jpcz_mean_divergence_ratio`

Meaning:

- Signed divergence computed from `del dot u` at `925 hPa` in a new coastal-Japan characterization region.

Calculation:

- Compute the same signed-divergence field used above.
- Summarize it within a coastal-Japan box or polygon designed only for characterization.
- Compute the ratio:
  - `coastal_to_jpcz_mean_divergence_ratio = coastal_japan_mean_divergence_peak_925 / jpcz_polygon_mean_divergence_peak_925`

Units:

- signed-divergence fields in `s^-1`
- ratio is unitless

Purpose:

- Measures how the coastal signed-divergence mean compares with the polygon mean. When both means are negative, values greater than `1` indicate the coastal box is more convergent than the polygon mean, values between `0` and `1` indicate it is less convergent, and negative values indicate opposite-signed regional means.

### 3. Pacific-side signed divergence

Column names:

- `pacific_east_of_japan_mean_divergence_peak_925`
- `pacific_east_of_japan_min_divergence_peak_925`
- `pacific_to_jpcz_mean_divergence_ratio`

Meaning:

- Signed divergence computed from `del dot u` at `925 hPa` in a Pacific box east of Japan.

Purpose:

- Measures whether the event is coupled to stronger Pacific-side convergence or divergence relative to the polygon mean.

### 4. Hokkaido passing-low metric

Column names:

- `hokkaido_min_z850_anomaly_tminus12_to_tplus12`

Meaning:

- Minimum `850 hPa` geopotential-height anomaly in a Hokkaido or northern-Japan box over the event window.

Calculation:

- Convert ERA5 geopotential to geopotential height at `850 hPa`.
- Define:
  - `z850_anomaly = event-time z850 - background climatological z850`
- The monthly climatology is computed month by month over the event years represented in the merged NDJF catalog.
- In the current workflow, if the merged working catalog contains event peaks spanning years `Y1 ... Yn`, then:
  - November climatology is the mean of all November `850 hPa` geopotential-height hourly fields across `Y1 ... Yn`
  - December climatology is the mean of all December hourly fields across `Y1 ... Yn`
  - January climatology is the mean of all January hourly fields across `Y1 ... Yn`
  - February climatology is the mean of all February hourly fields across `Y1 ... Yn`
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

- Quantifies circulation-centered events that may not be explained only by polygon or coastal signed-divergence structure.

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

- `coastal_to_jpcz_mean_divergence_ratio`
- `hokkaido_min_z850_anomaly_tminus12_to_tplus12`
- `front_box_max_temp_gradient_850_tminus12_to_tplus12`
- optionally `sea_of_japan_mean_vorticity_peak_925`

This first experiment is designed to test:

- polygon-centered versus coastal-enhanced signed divergence
- lower versus higher synoptic forcing
- frontal influence versus non-frontal influence
- optional circulation-centered structure

## First diagnostic scatterplots

Before formal clustering, make event-level scatterplots such as:

1. `coastal_to_jpcz_mean_divergence_ratio` versus `jpcz_polygon_min_divergence_peak_925`
2. `coastal_to_jpcz_mean_divergence_ratio` versus `hokkaido_min_z850_anomaly_tminus12_to_tplus12`
3. `pacific_to_jpcz_mean_divergence_ratio` versus `front_box_max_temp_gradient_850_tminus12_to_tplus12`
4. `sea_of_japan_mean_vorticity_peak_925` versus `hokkaido_min_z850_anomaly_tminus12_to_tplus12`

These plots should be inspected before clustering to see whether any natural separation exists.

The current first-pass scatterplot set uses the following pairings:

1. `coastal_to_jpcz_mean_divergence_ratio` versus `jpcz_polygon_min_divergence_peak_925`
2. `coastal_to_jpcz_mean_divergence_ratio` versus `hokkaido_min_z850_anomaly_tminus12_to_tplus12`
3. `pacific_to_jpcz_mean_divergence_ratio` versus `front_box_max_temp_gradient_850_tminus12_to_tplus12`
4. `sea_of_japan_mean_vorticity_peak_925` versus `hokkaido_min_z850_anomaly_tminus12_to_tplus12`

These are not the clustering themselves. They are raw event-level views of the original physical feature space:

- coastal-versus-polygon signed-divergence contrast versus polygon-core minimum divergence
- coastal-versus-polygon signed-divergence contrast versus synoptic-height forcing
- Pacific coupling versus frontality
- Sea of Japan circulation versus synoptic-height forcing

The scatterplots matter because they show whether physical structure is already visible before any dimensionality reduction or clustering is applied.

## Clustering workflow

The clustering workflow should be:

1. build the event-level feature table
2. standardize the features
3. inspect 2D and 3D scatterplots
4. run PCA for visualization
5. test clustering algorithms such as:
   - hierarchical clustering
   - Gaussian mixture clustering
6. test several cluster counts such as `k = 2`, `3`, `4`, `5`, and `6`

## Implemented first-pass math in Notebook 08

The current first-pass objective subtype workflow in `Notebook 08` uses one event-level feature vector per event and applies Ward hierarchical clustering to the standardized feature table.

### Step 1. Build one feature vector per event

For the current first-pass subtype experiment, each event is represented by four clustering features:

- `coastal_to_jpcz_mean_divergence_ratio`
- `hokkaido_min_z850_anomaly_tminus12_to_tplus12`
- `front_box_max_temp_gradient_850_tminus12_to_tplus12`
- `sea_of_japan_mean_vorticity_peak_925`

For event `i`, the feature vector is:

- `x_i = [x_i1, x_i2, x_i3, x_i4]`

Physical mapping of the four clustering axes:

- `coastal_to_jpcz_mean_divergence_ratio`
  - coastal-versus-polygon signed-divergence axis
  - Coastal Japan box versus JPCZ polygon
- `hokkaido_min_z850_anomaly_tminus12_to_tplus12`
  - synoptic-height-forcing axis
  - Hokkaido box
- `front_box_max_temp_gradient_850_tminus12_to_tplus12`
  - frontality axis
  - Hokkaido front box
- `sea_of_japan_mean_vorticity_peak_925`
  - Sea of Japan circulation axis
  - Sea of Japan box

These variables were selected because they represent different physical mechanisms and are less redundant than using many closely related convergence-only metrics at once.
These variables were selected because they represent different physical mechanisms and keep the low-level regional metric on the signed `del dot U` continuum rather than clipping divergence away before averaging.

### Step 2. Standardize each feature with a z score

The clustering does not operate on the raw units directly. Each feature is standardized column by column:

- `z_ij = (x_ij - mu_j) / sigma_j`

where:

- `x_ij` is feature `j` for event `i`
- `mu_j` is the feature mean over all events
- `sigma_j` is the feature standard deviation over all events

This prevents a feature with larger raw numerical range, such as geopotential-height anomaly, from dominating the distance calculation only because of its units.

### Step 3. Compute Euclidean distances in standardized feature space

After standardization, each event is a point in a four-dimensional feature space. The distance between events `a` and `b` is:

- `d(a, b) = sqrt(sum_j((z_aj - z_bj)^2))`

This Euclidean distance is the distance measure used in the clustering and silhouette calculations.

### Step 4. Apply Ward hierarchical clustering

The implemented clustering method is hierarchical clustering with Ward linkage.

Conceptually:

- begin with each event as its own cluster
- repeatedly merge the pair of clusters that causes the smallest increase in within-cluster variance
- continue until the tree can be cut at the requested cluster count `k`

For two candidate clusters `A` and `B`, Ward's merge cost is:

- `Delta(A, B) = (n_A * n_B / (n_A + n_B)) * ||mu_A - mu_B||^2`

where:

- `n_A`, `n_B` are cluster sizes
- `mu_A`, `mu_B` are cluster centroids in standardized feature space

### Step 5. Cut the hierarchy at a chosen cluster count

After the hierarchy is built, it is cut at a requested cluster count such as `k = 2`, `3`, or `4`. This assigns each event one cluster label for that solution.

The current comparison workflow in `Notebook 08` recomputes and saves the `k = 2`, `3`, and `4` solutions from the same fixed standardized feature table so that the outputs are directly comparable.

### Step 6. Evaluate each solution with silhouette score

For each event `i`, define:

- `a_i` = mean distance from event `i` to other events in its own cluster
- `b_i` = mean distance from event `i` to the nearest competing cluster

The silhouette value for event `i` is:

- `s_i = (b_i - a_i) / max(a_i, b_i)`

Interpretation:

- values near `1` indicate a well-separated event
- values near `0` indicate overlap between clusters
- values below `0` suggest the event may fit another cluster better

The notebook reports the mean silhouette score over all events for each tested `k`.

### Step 7. Summarize each cluster with medians in physical units

After labels are assigned, each cluster is summarized using medians of the raw event-level feature columns and additional diagnostic columns.

This is important because:

- clustering is performed on standardized variables
- physical interpretation should still be done in the original physical units

So the cluster median tables are the main science interpretation tables.

### Step 8. Use PCA only for visualization

Principal component analysis is used only to visualize the structure of the same standardized feature table in lower dimensions.

If the standardized feature matrix is `X`, singular value decomposition gives:

- `X = U * Sigma * V^T`

The PCA score matrix is:

- `T = X * V_k`

The explained variance ratio for principal component `j` is:

- `sigma_j^2 / sum_l(sigma_l^2)`

where `sigma_j` is the singular value for component `j`.

In this workflow:

- `PC1`, `PC2`, and `PC3` are used to make PCA scatterplots
- PCA is not the clustering method itself
- PCA is a visualization and dimensionality-compression step only

What the principal components represent:

- The variance here is the variance of the standardized event-by-feature matrix `X`, where rows are events and columns are the four standardized clustering features.
- `PC1` is the linear combination of the standardized clustering features that explains the largest fraction of total event-to-event variance.
- `PC2` is the next orthogonal linear combination that explains the next-largest fraction.
- `PC3` is the third orthogonal linear combination.
- Because the workflow uses four clustering features, PCA can in principle produce up to four components; the notebook currently visualizes the first three because they capture most of the standardized variance.

Current interpretation guidance:

- `PC1`, `PC2`, and `PC3` should be understood first as statistical axes of variation in the standardized feature table.
- They are not separate physical fields and they are not the clustering algorithm.
- Their physical meaning depends on the feature loadings on each component.
- In the current workflow, PCA is used to visualize the same structure already present in the raw scatterplots and clustering table, not to define the clusters.
- More explicitly, for event `i`:
  - `PC1_i = a_11 * z_i1 + a_12 * z_i2 + a_13 * z_i3 + a_14 * z_i4`
  - `PC2_i = a_21 * z_i1 + a_22 * z_i2 + a_23 * z_i3 + a_24 * z_i4`
  - `PC3_i = a_31 * z_i1 + a_32 * z_i2 + a_33 * z_i3 + a_34 * z_i4`
- Here:
  - `z_i1` through `z_i4` are the standardized clustering features for event `i`
  - the `a_mj` values are the PCA loadings
- The explained variance ratio for `PC1` is the fraction of total standardized event-to-event variance captured by the `PC1` direction; the same logic applies to `PC2` and `PC3`.

Relation between raw scatterplots and PCA:

- The first-pass scatterplots show the event cloud in the original physical variables.
- PCA rotates and compresses that same cloud into orthogonal axes of variance.
- The clustering is performed on the standardized four-feature table itself, not on subjective labels and not on manually drawn PCA groups.

Current practical interpretation of the first three components:

- `PC1` likely captures the dominant broad contrast across the four clustering variables, especially the low-synoptic versus higher-synoptic part of the event continuum.
- `PC2` and `PC3` capture additional orthogonal structure within that continuum, including how circulation, frontality, and coastal enhancement covary once the main broad contrast is removed.
- A more explicit physical interpretation of `PC1`, `PC2`, and `PC3` should eventually be based on saved PCA loading tables, not just the scatterplots.

### Step 9. Interpret the working clusters

The cluster labels themselves are algorithmic identifiers, not physical quantities.

For the current validated `k = 3` working solution:

- `Cluster 1`
  - least synoptic or weaker-background group
  - lower coastal enhancement, weaker Sea of Japan circulation, and weaker synoptic-height forcing
- `Cluster 2`
  - moderately synoptic or circulation-modified group
  - intermediate synoptic forcing and circulation strength
- `Cluster 3`
  - strongly synoptic, frontal, and coastal-enhanced group
  - stronger synoptic-height depression, stronger frontality, and stronger convergence/circulation signatures

These physical labels are post hoc interpretations of the objective cluster medians and validation tests. They are not used as clustering inputs.

## Composite workflow for Notebook 10

After `Notebook 09` validates `k = 3` as the working subtype framework, `Notebook 10` moves from event-level scalar summaries to full gridded composite maps.

### Composite principle

- The composite maps are not composites of the cluster variables themselves.
- The composite maps are composites of the full gridded physical fields used to calculate the cluster-defining event summaries.
- For the first primary composite field, the notebook now uses raw signed `del dot U = du/dx + dv/dy` so convergence and divergence remain on one continuum and can cancel in the mean.
- The primary composite stage uses only the validated `k = 3` framework:
  - `Cluster 1`
  - `Cluster 2`
  - `Cluster 3`
- The `k = 4` solution is not the main composite target.

### What is composited

There are nine primary composite maps:

- `3` variables times `3` clusters
- equivalently, `3` maps per cluster for `Cluster 1`, `Cluster 2`, and `Cluster 3`

The three physical fields are:

1. `925 hPa` signed divergence at the event peak time
2. `850 hPa` geopotential-height anomaly minimum over `t-12`, `t0`, `t+12`
3. `850 hPa` temperature-gradient magnitude maximum over `t-12`, `t0`, `t+12`

These are intended to be the full-domain gridded analogs of the physical ingredients used to define the cluster space.

### Composite-field formulas

For event `n`:

- peak-time signed-divergence field:
  - `div925_n(i, j) = (du/dx + dv/dy)_n(i, j, t0) * 1e5`
- time-window `z850` anomaly field:
  - `z850_anom_n(i, j, t) = z850_n(i, j, t) - z850_climatology(i, j, month(t))`
  - primary composite field:
    - `z850_anom_min_n(i, j) = min_{t in [-12, 0, +12]} z850_anom_n(i, j, t)`
- time-window temperature-gradient field:
  - `|grad T850|_n(i, j, t) = sqrt((dT/dx)^2 + (dT/dy)^2) * 1e5`
  - primary composite field:
    - `tempgrad_max_n(i, j) = max_{t in [-12, 0, +12]} |grad T850|_n(i, j, t)`

### Full-domain requirement

- These gridded fields are computed at every grid point across the full objective-subtype study domain.
- They are not restricted to the characterization boxes.
- They are not reduced to pre-aggregated scalar summaries before compositing.
- The characterization boxes are retained only for later comparison diagnostics.

### Pointwise composite mean

For cluster `c` and field `F_n(i, j)`:

- `N_c(i, j) = sum_n 1[finite(F_n(i, j))]`
- `Sum_c(i, j) = sum_n F_n(i, j) * 1[finite(F_n(i, j))]`
- `Mean_c(i, j) = Sum_c(i, j) / N_c(i, j)`

where the sums run only over events assigned to cluster `c`.

This is a full-domain pointwise mean at every grid cell, not a box-average-only product.

Equivalently, for each cluster:

- compute the full gridded field for every event assigned to that cluster
- then, at each grid cell, average the field value across the `N` events in that cluster

That pointwise averaging step is what produces the `9` primary full-grid maps.

### Pointwise sample counts and standard deviations

The notebook also saves:

- pointwise sample counts:
  - `N_c(i, j)`
- pointwise population-style standard deviations:
  - `Var_c(i, j) = sum_n(F_n(i, j)^2) / N_c(i, j) - Mean_c(i, j)^2`
  - `Std_c(i, j) = sqrt(max(Var_c(i, j), 0))`

These fields make it possible to:

- mask low-sample regions
- assess within-cluster spread
- diagnose whether a composite feature is spatially consistent or driven by a small subset of events

### Zero and missing-data rule in the composite notebook

The composite notebook uses the following rule consistently:

- include zeros if they are real physical values
- exclude only missing values from the numerator and denominator
- apply that same rule across every process in the composite workflow

So if a grid cell is valid and the signed divergence there is exactly `0`, that zero remains in the composite.
If a grid cell is missing, it does not contribute to `Sum_c(i, j)` or `N_c(i, j)`.

Because the first primary composite field now uses raw signed `del dot U`, convergence and divergence can cancel when the field is averaged over points or over events, which is the intended behavior for the composite maps.

This same rule should be used consistently for:

- the pointwise cluster means
- the pointwise count maps
- the pointwise standard-deviation maps
- the event-level box means
- the cluster-level box summaries
- any later difference-map calculations derived from those same composite fields

### Box-average outputs saved alongside the full maps

The notebook also computes weighted box averages of the same gridded composite fields over:

- Coastal Japan
- Pacific east of Japan
- Hokkaido
- Sea of Japan
- Hokkaido front
- Pacific front

For one composite field `F(i, j)` and one box:

- `box_mean(F) = sum_ij(F(i, j) * cos(lat_i) * 1[valid]) / sum_ij(cos(lat_i) * 1[valid])`

These are saved both:

- as event-level box means before cluster averaging
- and as box means of the final cluster composite maps

This allows direct comparison between:

- the full gridded composites
- the event-level box statistics
- the original scalar characterization logic from the clustering workflow

### Optional difference products

The notebook also saves difference maps such as:

- `Cluster 3 - Cluster 2`
- `Cluster 3 - Cluster 1`
- `Cluster 2 - Cluster 1`

These are not a new clustering step. They are simply pairwise differences between already-computed cluster-mean fields.

The same notebook may also save pointwise standard-deviation fields so the spread across events can be assessed directly alongside the mean composites.

## How success should be judged

The clustering should not be judged only by whether it reproduces subjective notes.

Success should be evaluated by:

- separation in feature space
- cluster stability under resampling
- physical interpretability of the cluster composites
- post hoc comparison with subjective satellite-based notes

## Validation after exploratory clustering

After the first exploratory clustering pass, the next step should be a dedicated validation workflow rather than continuing to overload the feature-construction notebook.

That validation workflow should test:

- whether the observed silhouette is stronger than a shuffled-null feature structure
- whether the clustering is stable under repeated event resampling
- whether the clusters differ on external variables that were not used directly in the clustering
- whether intermediate and strong synoptic subdivisions, such as cluster 2 versus cluster 3 in a `k = 3` solution, remain distinguishable on outside variables

In the current project layout, this validation step belongs in `Notebook 09` rather than in the feature-building and exploratory clustering notebook.

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
- `jpcz_polygon_min_divergence_peak_925`
- `coastal_japan_min_divergence_peak_925`
- integrated signed-divergence metrics if added

## Not first priority

These are important but not first-priority inputs to the clustering design:

- cross sections
- station snowfall data
- vertical motion
- jet position and entrance-region analysis

Those should come later as diagnostic or interpretation tools after the first objective feature space is built.
