# Planned Notebooks

This directory will hold the Colab notebooks for the JPCZ catalog workflow.

Planned order:

1. `01_data_access_and_masks.ipynb`
2. `02_december_benchmark_detector.ipynb`
3. `03_classification_and_qa.ipynb`
4. `04_ndjf_catalog.ipynb`
5. `05_manual_verification_and_position.ipynb`
6. `06_gap_merge_catalog.ipynb`
7. `07_event_diagnostic_atlas.ipynb`
8. `08_objective_subtype_features.ipynb`

Design rule:

- notebooks should orchestrate the workflow
- reusable science logic should live in Python modules instead of being duplicated across notebooks
- validation notebooks should save short human-readable summaries in `outputs/verification/`
- the setup notebook should be runnable directly from Colab after replacing the placeholder repository URL
- Notebook 02 and Notebook 03 now call reusable code in `src/jpcz_catalog/` instead of rebuilding the full workflow from scratch cell by cell
- Notebook 02 persists the December `D` series to `outputs/verification/december_D_timeseries.nc` so Notebook 03 can restart cleanly in a fresh Colab session
- Notebook 02 and Notebook 03 can also mirror their key outputs to Google Drive at `/content/drive/MyDrive/JPCZcatalog_outputs` so a Colab disconnect does not wipe out the saved benchmark artifacts
- Notebook 02 now writes one Drive checkpoint per December year in `JPCZcatalog_outputs/december_yearly/`, so interrupted benchmark runs can resume from the remaining years instead of starting from 2000 again
- Notebook 04 reuses completed December checkpoints, writes one checkpoint per NDJF month in `JPCZcatalog_outputs/ndjf_monthly/`, and produces the first-pass master catalog at `jpcz_catalog_ndjf.csv`
- Notebook 05 augments the NDJF catalog with peak-position diagnostics, candidate intensity metrics, a manual-review scaffold, and batch quicklook plots for visual event verification
- Notebook 06 merges nearby NDJF threshold fragments into broader synoptic episodes using configurable small-gap rules such as 6 h, 12 h, and 24 h
- Notebook 07 provides an event-by-event diagnostic atlas for convergence, satellite, OLR, and synoptic interpretation
- Notebook 08 builds the first objective subtype feature table and runs first-pass 2D, 3D, PCA, and hierarchical-clustering experiments
