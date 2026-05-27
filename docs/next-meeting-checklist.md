# Next Meeting Checklist

This file captures the prioritized questions, expected deliverables, and where they currently live in the workflow.

Primary notebooks:

- [08_objective_subtype_features.ipynb](/Users/angelica.ramirez/Documents/New%20project/notebooks/08_objective_subtype_features.ipynb)
- [09_cluster_validation_and_significance.ipynb](/Users/angelica.ramirez/Documents/New%20project/notebooks/09_cluster_validation_and_significance.ipynb)
- [10_cluster_composites_and_examples.ipynb](/Users/angelica.ramirez/Documents/New%20project/notebooks/10_cluster_composites_and_examples.ipynb)

Core methods note:

- [objective-subtype-design.md](/Users/angelica.ramirez/Documents/New%20project/docs/objective-subtype-design.md)

## 1. Convergence Calculation

Question:

- Exactly how was convergence/divergence computed?
- What formula was implemented?
- How were grid spacing, zeros, and missing values handled?

Deliverable:

- Short methods note stating:
  - implemented formula
  - grid-spacing handling
  - whether zeros were included
  - whether masked/missing values were excluded

Current source:

- [objective-subtype-design.md](/Users/angelica.ramirez/Documents/New%20project/docs/objective-subtype-design.md)
- [detect.py](/Users/angelica.ramirez/Documents/New%20project/src/jpcz_catalog/detect.py)
- [subtypes.py](/Users/angelica.ramirez/Documents/New%20project/src/jpcz_catalog/subtypes.py)

## 2. Composite Maps

Question:

- Do the nine primary composites exist as full-grid, everywhere-in-the-domain maps for:
  - signed divergence / convergence metric
  - `z850` anomaly
  - `|grad T850|`

Deliverable:

- PNGs for the `9` primary maps
- per-map sample-count grids
- underlying data arrays
- per-cluster precipitation/JPCC-strength proxy composite if available

Current source:

- [10_cluster_composites_and_examples.ipynb](/Users/angelica.ramirez/Documents/New%20project/notebooks/10_cluster_composites_and_examples.ipynb)

Expected outputs:

- composite means NetCDF
- composite counts NetCDF
- composite standard deviations NetCDF
- PNG maps in the notebook output/Drive export directory

## 3. Immediate Convergence Diagnostics

Question:

- How does the current divergence/convergence implementation compare with a centered finite-difference alternative?
- Would events change cluster assignment?

Deliverable:

- for each cluster:
  - implemented-field panel
  - finite-difference panel
  - difference panel
- small table of percent cluster switching

Status:

- not yet built

## 4. PCA Diagnostics and Interpretability

Question:

- What do `PC1`, `PC2`, and `PC3` represent numerically?
- How much variance does each explain?
- Which original clustering variables load most strongly on each PC?

Plain-language interpretation target:

- `PC1`, `PC2`, and `PC3` are not separate meteorological maps or separate clustering methods.
- Each PC is a weighted linear combination of the same four standardized clustering variables:
  - `coastal_to_jpcz_mean_divergence_ratio`
  - `hokkaido_min_z850_anomaly_tminus12_to_tplus12`
  - `front_box_max_temp_gradient_850_tminus12_to_tplus12`
  - `sea_of_japan_mean_vorticity_peak_925`
- The reported variance is the event-to-event variance of that standardized four-feature matrix, not the variance of only convergence, only `z850`, or the gridded composite maps.
- The loading table is what tells which original variables most strongly define `PC1`, `PC2`, and `PC3`.

Deliverable:

- variance-explained table
- cumulative variance table
- loadings table / heatmap
- uncertainty estimate if added later

Current source:

- Notebook 08 PCA outputs
- Notebook 10 PCA summary tables

Key outputs:

- `pca_variance_df`
- `pca_loadings_df`
- `pca_driver_df`

## 5. Cluster Definitions and Physics

Question:

- What are the physical signatures of each cluster?

Deliverable:

- one short plain-language paragraph per cluster
- explicitly tied to the composite maps

Current source:

- Notebook 08 feature medians
- Notebook 09 validation results
- Notebook 10 composites and grouped summaries

## 6. Statistical Significance and Robustness

Question:

- Are cluster differences statistically significant?
- How stable are the cluster assignments?

Deliverable:

- pairwise p-values/effect sizes for key variables
- cluster stability summary

Current source:

- [09_cluster_validation_and_significance.ipynb](/Users/angelica.ramirez/Documents/New%20project/notebooks/09_cluster_validation_and_significance.ipynb)

## 7. Sensitivity Tests

Question:

- How sensitive are the results to:
  - number of PCs
  - number of clusters
  - clustering method
  - zero/missing-data treatment

Deliverable:

- short summary table/report of changes in interpretation

Status:

- partly done for cluster count
- not yet complete for all requested sensitivities

## 8. Reproducibility and Methods Clarity

Question:

- Can the pipeline be described in reproducible terms from data extraction through composites and validation?

Deliverable:

- README / methods paragraph
- notebook paths
- exact variable definitions, boxes, and time offsets

Current source:

- [objective-subtype-design.md](/Users/angelica.ramirez/Documents/New%20project/docs/objective-subtype-design.md)
- notebooks `08`, `09`, and `10`

## 9. Link to Event Strength / Hypothesis

Question:

- Do the clusters differ in JPCC strength / precipitation proxy?
- What working hypothesis follows from that?

Deliverable:

- precipitation / moisture-ascent composite maps
- short hypothesis statement

Current source:

- Notebook 10 moisture-flux proxy composite

## 10. Meeting-Ready Materials

Question:

- What should be shown in the meeting?

Deliverable:

- one-page summary
- `3` to `5` slides
- thumbnail panel of the nine composites
- PCA numbers
- cluster sample sizes
- one-sentence cluster takeaways
- uncertainties and next steps

Status:

- not yet assembled into meeting-ready form
