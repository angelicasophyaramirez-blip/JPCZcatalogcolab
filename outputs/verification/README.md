# Verification Outputs

This directory is for compact verification artifacts produced by the notebooks.

Core artifacts:

- `december_events_first_pass.csv`
  - first-pass December event table from the Shinoda-style `mean - 2 std` detector
- `december_D_timeseries.nc`
  - persisted December `D` time series used for restart-safe QA and sensitivity checks
- `december_benchmark_summary.md`
  - total detected December events before subtype labels
  - comparison against Shinoda's approximately 35-event benchmark over 2000-2018
- `december_threshold_sensitivity.csv`
  - threshold sweep table for `mean - n std`
- `december_threshold_sensitivity.md`
  - human-readable interpretation of the threshold sweep
- `december_events_shinoda_style.csv`
  - first-pass classified catalog with strong/weak monsoon and Shinoda-style subtype labels
- `december_shinoda_style_summary.md`
  - summary of the first-pass subtype mix

Extended QA artifacts:

- `vorticity_box_sensitivity.csv`
  - simple dashed-box shift tests for the Type 1B sensitivity check
- `combo_sensitivity_table.csv`
  - combined threshold and vorticity-box sensitivity table
- `combo_sensitivity_summary.md`
  - short interpretation of the combined sensitivity results
- `final_first_pass_summary.md`
  - concise statement of the baseline validation, December benchmark, and first-pass class counts
- `next_steps.md`
  - current status and recommended follow-up work

These are verification artifacts, not the final science catalog.
