# JPCZ Event Catalog

This repository is for a cloud-native catalog of Japan Sea polar airmass convergence zone (JPCZ) events using ERA5 in Google Colab.

The project starts by reproducing the Shinoda et al. December event-detection method as closely as possible, but with ERA5 accessed directly in the cloud rather than bulk-downloaded to a local machine. The workflow is considered healthy when two verification checks are both reasonable: the December 2000-2018 run finds a similar number of major events to Shinoda, around 35 before any subtype labels are applied, and the known 2-7 February 2018 case shows a peak convergence signal on 3 February 2018. After those checks are stable, the same workflow will be extended to November, December, January, and February.

## Current scope

- Reproduce the Shinoda December detector with ERA5
- Classify events using clearer labels:
  - `Type 1 strong-monsoon`
  - `Type 2 weak-monsoon`
  - `Type 1A lower-vorticity`
  - `Type 1B higher-vorticity`
- Validate against:
  - December 2000-2018 benchmark behavior, including a total event count around Shinoda's 35 major December events over 19 years before subtyping
  - 2-7 February 2018 baseline event, with peak convergence on 3 February 2018

## Working principles

- Use cloud-hosted ERA5 access only
- Do not download the full dataset locally
- Keep units explicit in every derived field and plot
- Treat convergence as negative divergence
- Separate science logic in Python modules from notebook orchestration

## Repository map

- `docs/detector-spec.md`: event-definition and threshold rules
- `docs/colab-workflow.md`: Colab-first architecture and cloud data workflow
- `docs/validation-plan.md`: benchmark and validation plan
- `notebooks/README.md`: planned notebook sequence
- `outputs/verification/README.md`: expected verification artifacts from notebooks

## Immediate next build step

Implement the first notebook and helper modules for:

1. opening cloud ERA5 in Colab
2. building the JPCZ polygon and vorticity box masks
3. computing hourly 925 hPa divergence and 12-hour means
4. benchmarking December 2000-2018 event counts against Shinoda before subtype labels are assigned
5. verifying the 2-7 February 2018 baseline event with a peak on 3 February 2018
