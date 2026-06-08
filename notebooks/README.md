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
9. `09_cluster_validation_and_significance.ipynb`
10. `10_cluster_composites_and_examples.ipynb`
11. `11_temperature_gradient_sensitivity.ipynb`
12. `12_masked_t850_sensitivity.ipynb`
13. `13_geographic_t850_exclusion_sensitivity.ipynb`
14. `14_elevation_masked_t850_sensitivity.ipynb`
15. `15_cleaned_low_level_clustering_decision.ipynb`
16. `16_cleaned_cluster_composites_and_examples.ipynb`
17. `17_spatial_eof_analysis.ipynb`
18. `18_cleaned_cluster_quartile_analysis.ipynb`
19. `19_cleaned_cluster_date_pairing_and_sequence.ipynb`
20. `20_eastern_siberian_high_association.ipynb`
21. `21_*.ipynb` (reserved)
22. `22_objective_coastal_box_metrics_and_labels.ipynb`
23. `23_objective_regime_timing_and_impact.ipynb`
24. `24_objective_regime_manual_verification_atlas.ipynb`
25. `25_continuous_spell_evolution_and_onset_timing.ipynb`

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
- Notebook 09 validates the `k = 2, 3, 4` subtype solutions using permutation-null silhouettes, resampling stability, and external-variable significance tests
- Notebook 10 turns the validated `k = 3` working subtype framework into representative-event tables and peak-time physical composite maps
- Notebooks 11-15 progressively test terrain sensitivity, frontality sensitivity, and the cleaned low-level feature framework that led to the later `k = 2` interpretation
- Notebook 16 summarizes the cleaned-cluster composites and representative cases, including the supplementary broad upper-level context plots
- Notebook 17 stores the cleaned event-field stacks and exploratory EOF products
- Notebook 18 ranks the broad cleaned `Cluster 1` family by polygon-centered moisture-flux strength
- Notebook 19 tests whether the older cleaned-cluster labels behave like a before/after timing sequence
- Notebook 20 checks the eastern Siberian high / blocking-side context
- Notebook 22 resets the classification around a simpler objective coastal-wedge versus JPCZ-polygon framework using only moisture-flux and divergence means
- Notebook 23 keeps the peak-event timing baseline, transition counts, and the future merge point for a coastal snow / precipitation impact table
- Notebook 24 is the manual-review atlas for the new workflow, tying objective labels and spell timing to moisture, divergence, quicklook, OLR, and satellite context
- Notebook 25 extends the timing work to broader padded spell windows so offshore/coastal evolution can be tracked continuously instead of only at event peaks
