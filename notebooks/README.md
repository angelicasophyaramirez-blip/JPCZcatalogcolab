# Planned Notebooks

This directory will hold the Colab notebooks for the JPCZ catalog workflow.

Planned order:

1. `01_data_access_and_masks.ipynb`
2. `02_december_benchmark_detector.ipynb`
3. `03_classification_and_qa.ipynb`
4. `04_ndjf_catalog.ipynb`
5. `05_feb2018_baseline_case.ipynb`

Design rule:

- notebooks should orchestrate the workflow
- reusable science logic should live in Python modules instead of being duplicated across notebooks
- validation notebooks should save short human-readable summaries in `outputs/verification/`
- the setup notebook should be runnable directly from Colab after replacing the placeholder repository URL
