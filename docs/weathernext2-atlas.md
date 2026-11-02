# Notebook 36: WeatherNext 2 JPCZ forecast atlas

Open `notebooks/36_weathernext2_jpcz_forecast_atlas.ipynb` in Google Colab using
**File → Upload notebook**, or open it from GitHub after the change is published.
The notebook embeds its helper code and needs no repository clone or unpublished module download.
It supports historical period stores (`2023_to_2024/predictions.zarr`, containing 2023
initializations) and dated run folders used by the later archive. Historical `time` is an
initialization dimension; `prediction_timedelta` is the forecast lead. Selection is exact
before normalization to the plotting schema. Actual time coordinates determine coverage.

## First event

1. Run Sections 1–2: install dependencies, mount Drive, and authenticate with the Google
   account approved for WeatherNext. Drive and forecast authentication are separate.
2. In Section 3, verify the Drive paths. Supply your Google Cloud project ID if available;
   otherwise the notebook uses a detected project when there is one. No project, billing,
   or other Cloud settings are created or changed.
3. Run Sections 4–5. Review the actual selected catalog count, dates, and target-time offsets.
4. Run both code cells in Section 6 to open the browser. Choose an event, then click
   **Check data & members**. Review the three exact forecast-store paths and schema.
5. Keep one member selected. Initially use **500-hPa geopotential height** and
   **925-hPa winds**. Click **Load six plots**.
6. Inspect all six panels and their labels, then click **Export six plots**.
   Use Previous/Next to view another event. Section 7's batch export stays off by default.

## Forecast convention

For each catalog onset, choose the nearest 00, 06, 12, or 18 UTC on the **same UTC date**;
equal distances choose the earlier time. The original onset is never modified. For a
23:00 onset the target is 18:00 that day, not 00:00 the next day. Both onset and signed
target-minus-onset offset are recorded.

All six panels verify at that target. Forecast initializations are target minus 24, 120,
and 240 hours. Each row contains one variable at all three leads, and the same native WN2
`sample` label is used throughout. A member label is never interpreted as a positional index.
There is no nearest-run fallback or ensemble averaging.

## Catalog and references

- Source: `JPCZcatalog_outputs/jpcz_catalog_ndjf_merged_12h.csv`, as in Notebook 35.
- Optional reviews/references: `JPCZcatalog_outputs/imerg_manual_case_review`.
- Exact start/end/peak identities join reviews. Event numbers are preserved from the full
  catalog before filtering. If absent/invalid, numbers are assigned across the full inventory.
- The merged inventory is checked against saved raw detector fragments, when available.
- A 2023-onward filter is **provisional**. The operational archive's training cutoff has
  not been verified. The notebook does not describe these events as proven out of sample.
- The selection report identifies date exclusions. Forecast availability is checked on
  request; an included catalog event is not a guarantee that all three forecast runs exist.
- Peak D12 is signed 12-hour polygon-mean divergence, not precipitation intensity.
- Notebook 35 reference images are matched using metadata. They can be peak-centered;
  MODIS daily daytime imagery is not matched to the exact forecast hour.

## Maps and output

Each event/member export contains:

- Six individual forecast plots, each as PNG and PDF.
- One two-row, three-column overview, as PNG and PDF.
- `forecast_metadata.csv`: six records with event identity, member label, lead,
  initialization, valid time, units, domain, scale, source store, and cache identity.
- `settings.json`, historical-data attribution, and a completion marker.

The two row domains, scales, variables, and levels are adjustable through widgets.
All three leads in each row use identical scales and domains. The scale stays fixed across
events until changed. Winds show speed shading and vector arrows with a fixed 10 m/s key.
Height uses a constant contour interval for a given pressure-level range. Every panel
labels its member, level, initialization, valid time, and lead; color bars label units.
Pressure-level fields and derived diagnostics lack a below-surface mask and must be
interpreted accordingly, especially at 1000 hPa.

## Derived variables in the same dropdowns

**Horizontal divergence (derived)** uses eastward/northward winds at the selected
pressure level (default 925 hPa). Spherical divergence includes the meridional metric
term: `[du/dlambda + d(v*cos(phi))/dphi] / [R*cos(phi)]`. Values are displayed in
10⁻⁵ s⁻¹ with **negative values blue (convergence), positive values red (divergence)**,
and zero centered in the color scale. This is an instantaneous forecast diagnostic,
distinct from Notebook 35's 12-hour D12 peak-intensity measure.

**Ertel PV (derived)** is dry hydrostatic pressure-coordinate Ertel PV, defaulting to
a 1000-hPa map, in PVU (10⁻⁶ K m² kg⁻¹ s⁻¹). It uses temperature and both winds at
three pressure levels and includes both horizontal-theta-gradient/vertical-shear terms:

`PV = -g * [du/dp * dtheta/dy - dv/dp * dtheta/dx + (zeta + f) * dtheta/dp]`.

Potential temperature references 1000 hPa with kappa = 2/7; derivatives use pressure
in Pa. Vorticity and horizontal gradients include spherical latitude/longitude metrics,
with R = 6,371,000 m, g = 9.80665 m/s² and Earth rotation 7.292115×10⁻⁵ s⁻¹.
Centered, second-order three-point derivatives are used at interior pressure levels;
1000 hPa uses a one-sided 850/925/1000-hPa stencil, labeled on each PV panel.
Horizontal derivatives use a two-cell halo wherever the source grid permits, then crop
to the requested map domain. Stencil, formula, constants, units, and algorithm version
are included in cache identities and export metadata. The field dropdowns handle the
extra input retrieval and calculation automatically while preserving the selected member.

1000 hPa is a pressure surface, not the physical surface everywhere. It can lie below
land and below sea level in low-pressure systems. The archive lacks surface pressure
and terrain for a complete validity mask; no mask is inferred from mean sea-level pressure.

Both diagnostic color scales must straddle zero. **Assignment defaults** restores
500-hPa height and 925-hPa winds; adding diagnostics does not change the six-panel
layout or the fixed single-member rule.

New outputs live under `JPCZcatalog_outputs/weathernext2_atlas`. Existing catalogs and
Notebook 35 products are read only. Regional subsets are cached on Drive. Historical
source chunks span four members, all 13 levels, and the globe; reading one slice therefore
decodes unselected source values too. Chunks are computed sequentially to limit memory,
and only selected regional fields are retained. Known historical fields omit unit
attributes; defaults from Google's published WN2 schema and their provenance are recorded
in the export metadata. Declared units are never silently overridden. Exports include
their settings in their identities. Completed matching exports are reused after restart.

Changing a control disables export until a successful load; an older displayed figure
keeps its own labels. A failed load never creates a mixed or partially substituted six-panel set.

## Implementation and local validation

Reusable implementation: `src/jpcz_catalog/weathernext_atlas.py`.
After edits, regenerate the embedded notebook with:

```sh
python tools/build_weathernext_notebook.py
```

Regression checks: `tests/test_weathernext_atlas.py`. Tests use synthetic datasets with
the official WN2 coordinate structure, including a Zarr 3 round trip, and are not evidence
of authenticated access to Google's private forecast data. The live Colab access check and
first actual forecast render remain required before interpreting the research output.
The diagnostic tests additionally require `metpy>=1.7,<2` for an independent comparison
with MetPy's routines using explicit spherical map metrics; the notebook calculations
themselves use NumPy and do not add a runtime MetPy dependency.

## Sources

- [WeatherNext 2](https://developers.google.com/weathernext/guides/models-wn2)
- [Google's WN2 Zarr starter](https://storage.googleapis.com/weathernext-public/colabs/WeatherNext_2_Starter_Guide_Zarr_on_Google_Cloud_Storage.ipynb)
- [Dataset and attribution](https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_weathernext_2_0_0)
- [Research model's training and evaluation](https://arxiv.org/html/2506.10772v1)
- [MetPy pressure-coordinate PV](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.potential_vorticity_baroclinic.html)
- [MetPy divergence](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.divergence.html)
