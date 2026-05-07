"""Helpers for writing compact verification summaries."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def write_text_summary(path: str | Path, text: str) -> Path:
    """Write a summary text file and return its path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return out_path


def render_december_benchmark_summary(
    *,
    total_events: int,
    D_mean: float,
    D_std: float,
    threshold: float,
    shinoda_benchmark: int = 35,
) -> str:
    """Render the December benchmark summary text."""
    return f"""# December Benchmark Summary

- Period: December 2000-2018
- Total detected major events: {total_events}
- Shinoda benchmark: about {shinoda_benchmark} major December events over 19 years
- First-pass ERA5 result: {total_events}
- Difference from Shinoda benchmark: {total_events - shinoda_benchmark}
- December D mean: {D_mean:.6e} s^-1
- December D std: {D_std:.6e} s^-1
- Threshold (mean - 2 std): {threshold:.6e} s^-1
- Threshold display units: {threshold * 1e5:.3f} 1e-5 s^-1

Interpretation:
This is a functioning first-pass detector. The event count is somewhat higher than Shinoda's benchmark, so spatial-mask refinement and land masking should be tested before finalizing the December reproduction.
"""


def render_threshold_sensitivity_summary(results: Mapping[str, int]) -> str:
    """Render the threshold sensitivity text summary."""
    lines = ["# December Threshold Sensitivity", ""]
    for label, count in results.items():
        lines.append(f"- {label}: {count} events")
    lines.extend(
        [
            "",
            "Interpretation:",
            "The first-pass ERA5 detector is sensitive near the event threshold. The difference between the ERA5 result and the Shinoda benchmark appears to be driven mainly by marginal near-threshold events rather than by land contamination or aggressive event splitting.",
        ]
    )
    return "\n".join(lines)


def render_classification_summary(
    classified_events: pd.DataFrame,
    *,
    title: str = "# December Shinoda-Style Classification Summary",
    slp_clim_mean: float | None = None,
    slp_clim_std: float | None = None,
) -> str:
    """Render a compact classification summary."""
    lines = [title, "", f"- Total first-pass December events: {len(classified_events)}", ""]
    if slp_clim_mean is not None and slp_clim_std is not None:
        lines.extend(
            [
                "## December climatological monsoon index",
                f"- Mean 12h Seoul-minus-Sapporo SLP index: {slp_clim_mean:.3f} hPa",
                f"- Std 12h Seoul-minus-Sapporo SLP index: {slp_clim_std:.3f} hPa",
                "",
            ]
        )

    if "monsoon_type" in classified_events:
        lines.extend(["## Monsoon types", classified_events["monsoon_type"].value_counts().to_string(), ""])
    if "shinoda_class" in classified_events:
        lines.extend(["## Final Shinoda-style classes", classified_events["shinoda_class"].value_counts().to_string(), ""])
    if "monsoon_type" in classified_events and "shinoda_class" in classified_events:
        lines.extend(["## Cross-tab", pd.crosstab(classified_events["monsoon_type"], classified_events["shinoda_class"]).to_string(), ""])

    lines.extend(
        [
            "Note:",
            "Type 1A and Type 1B are subdivisions of Type 1 strong-monsoon events only.",
        ]
    )
    return "\n".join(lines)


def render_first_pass_summary() -> str:
    """Return the first-pass ERA5 summary text from the successful Colab run."""
    return """
JPCZ catalog first-pass ERA5 summary

Baseline validation
- February 2-7, 2018 event detected
- Peak convergence occurs on February 3, 2018

December benchmark
- Shinoda benchmark: 35 events over 19 Decembers
- ERA5 first-pass result: 46 events
- Monthly rate from Shinoda: 1.84 events per December
- Monthly rate from ERA5 first-pass: 2.42 events per December

Threshold sensitivity
- mean - 2.0 std: 46 events
- mean - 2.2 std: 39 events
- mean - 2.3 std: 38 events

Final first-pass Shinoda-style classes
- Type 1 strong-monsoon: 23
- Type 2 weak-monsoon: 23
- Type 1A lower-vorticity: 21
- Type 1B higher-vorticity: 2

Interpretation
- The core ERA5 cloud-based detector is functioning.
- Event-count differences are influenced by near-threshold cases.
- The remaining subtype mismatch is concentrated in the higher-vorticity classification.
""".strip()


def render_combo_sensitivity_summary() -> str:
    """Return the combined threshold and vorticity-box sensitivity summary."""
    return """# Combined Threshold and Vorticity-Box Sensitivity

## Paper-faithful first pass
- Threshold: mean - 2.0 std
- Vorticity box: original
- Total events: 46
- Type 1 total: 23
- Type 1A: 21
- Type 1B: 2
- Type 2: 23

## Best simple compromise
- Threshold: mean - 2.3 std
- Vorticity box: west_1deg or south_1deg
- Total events: 38
- Type 1 total: 18
- Type 1A: 15
- Type 1B: 3
- Type 2: 20

## Interpretation
- Simple threshold tightening helps the total count.
- Simple box shifts help Type 1B modestly.
- No simple combination fully reproduces Shinoda.
- The remaining mismatch likely reflects ERA5 vs WRF differences and/or finer spatial-definition differences.
"""


def render_next_steps_summary() -> str:
    """Return the working next-steps note from the Colab session."""
    return """# Current Status and Next Steps

## Current status
- Cloud-based ERA5 access in Colab is working.
- February 2-7, 2018 baseline event was detected with peak convergence on February 3, 2018.
- First-pass December 2000-2018 detector produced 46 major events using the Shinoda-style mean - 2 std threshold.
- Threshold sensitivity shows that 2.2 to 2.3 std gives total event counts closer to Shinoda.
- Shinoda-style subtype reproduction is not fully matched in ERA5.
- The main remaining mismatch is the low number of Type 1B higher-vorticity events.

## Best current interpretations
- Paper-faithful first pass:
  - threshold = mean - 2.0 std
  - original vorticity box
  - 46 total events
  - 23 Type 1, 21 Type 1A, 2 Type 1B, 23 Type 2

- Best simple compromise:
  - threshold = mean - 2.3 std
  - west_1deg or south_1deg vorticity box
  - 38 total events
  - 18 Type 1, 15 Type 1A, 3 Type 1B, 20 Type 2

## Recommended next steps
1. Save and preserve the notebook and verification outputs.
2. Move the notebook logic into cleaner repo modules so Colab reruns are less fragile.
3. Create Notebook 02 and Notebook 03 in a more stable form from the cells that worked.
4. Test finer vorticity-box geometry changes, not just 1-degree shifts.
5. Decide whether the project goal is:
   - strict Shinoda-method reproduction
   - or an ERA5-adapted operational JPCZ catalog

## Important caution
- Files saved under /content/JPCZcatalog are in the Colab runtime and should be pushed/saved outside the runtime if they need to persist.
"""


def render_ndjf_catalog_summary(
    *,
    total_events: int,
    month_threshold_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    title: str = "# NDJF Catalog Summary",
) -> str:
    """Render a compact summary for the first-pass NDJF catalog."""
    lines = [title, "", f"- Total NDJF events: {total_events}", ""]
    lines.extend(["## Monthly thresholds", month_threshold_df.to_string(index=False), ""])

    if "month" in catalog_df:
        month_counts = (
            catalog_df["month"]
            .value_counts()
            .sort_index()
            .rename_axis("month")
            .to_string()
        )
        lines.extend(["## Event counts by month", month_counts, ""])

    if "monsoon_type" in catalog_df:
        lines.extend(["## Monsoon types", catalog_df["monsoon_type"].value_counts().to_string(), ""])

    if "shinoda_class" in catalog_df:
        lines.extend(["## Final classes", catalog_df["shinoda_class"].value_counts().to_string(), ""])

    lines.extend(
        [
            "Interpretation:",
            "This NDJF catalog extends the Shinoda-style December detector to November, December, January, and February using the same core divergence logic with month-specific climatological thresholds.",
        ]
    )
    return "\n".join(lines)


def render_manual_verification_summary(
    *,
    total_events: int,
    auto_catalog_name: str,
    scaffold_name: str,
    plot_dir_name: str,
    cloud_variable: str | None = None,
) -> str:
    """Render a short note describing the manual verification workflow outputs."""
    cloud_line = (
        f"- Optional cloud-band field requested: `{cloud_variable}`"
        if cloud_variable
        else "- Optional cloud-band field requested: none (wind + convergence quicklooks only)"
    )
    return f"""# NDJF Manual Verification Workflow

- Total NDJF events available for review: {total_events}
- Auto-augmented catalog: `{auto_catalog_name}`
- Manual verification scaffold: `{scaffold_name}`
- Quicklook plot directory: `{plot_dir_name}`
{cloud_line}

Manual review focus:
- `verified_event`: yes / no / uncertain
- `cloud_band_present`: yes / no / uncertain using a cloud proxy such as OLR if available
- `position_group_manual`: e.g. north-shifted / central / south-shifted
- `manual_peak_convergence_lat`, `manual_peak_convergence_lon`: corrected peak location if needed
- `satellite_checked`: yes / no / planned
- `satellite_cloud_band_match`: yes / no / uncertain
- `satellite_source`, `satellite_notes`: archive/source and freeform satellite check notes
- `upper_level_forcing_note`: short note about jet position / forcing pattern
- `verification_notes`: freeform event QA notes

Auto-added diagnostics:
- peak max convergence latitude / longitude
- convergence centroid latitude / longitude
- simple north/central/south and west/central/east automatic position groups
- candidate intensity metrics based on peak convergence, duration, and box-mean vorticity
"""


def render_gap_merge_summary(
    *,
    sensitivity_df: pd.DataFrame,
    recommended_gap_hours: float,
    merged_catalog_name: str,
) -> str:
    """Render a compact summary for gap-merged catalog candidates."""
    recommended_row = sensitivity_df.loc[sensitivity_df["gap_hours"] == recommended_gap_hours]
    recommended_count = (
        int(recommended_row["event_count"].iloc[0])
        if not recommended_row.empty
        else None
    )

    lines = [
        "# NDJF Gap-Merge Sensitivity",
        "",
        "The raw NDJF detector groups only strictly consecutive hourly threshold hits.",
        "This sensitivity table shows how many events remain after merging adjacent events separated by small threshold-free gaps.",
        "",
        sensitivity_df.to_string(index=False),
        "",
        f"Recommended candidate: merge gaps <= {recommended_gap_hours:.0f} hours",
    ]
    if recommended_count is not None:
        lines.append(f"- Resulting event count: {recommended_count}")
    lines.extend(
        [
            f"- Output catalog: `{merged_catalog_name}`",
            "",
            "Interpretation:",
            "A short-gap merge broadens the catalog from threshold-hit fragments toward synoptic episodes without rerunning the expensive ERA5 detection workflow.",
        ]
    )
    return "\n".join(lines)
