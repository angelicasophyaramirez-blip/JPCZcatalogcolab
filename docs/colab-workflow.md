# Colab Workflow

## Objective

Build the JPCZ catalog in Google Colab with ERA5 accessed directly in the cloud, while keeping the science logic version-controlled in GitHub.

## Why Colab for this project

- ERA5 access stays in the cloud
- only regional subsets and small derived outputs are materialized
- notebooks are convenient for maps, threshold checks, and event review
- GitHub can store notebooks, modules, and small outputs without trying to version huge data files

## Data-access strategy

Principle:

- no full ERA5 downloads

Preferred workflow:

1. open the cloud-hosted ERA5 dataset from Colab
2. subset immediately by:
   - time
   - longitude
   - latitude
   - level
3. compute divergence and classification fields lazily
4. persist only small outputs:
   - event catalogs
   - QA figures
   - summary tables

## Expected variable set

Minimum fields for phase 1:

- `u` at `925 hPa`
- `v` at `925 hPa`
- `mean sea level pressure`

Likely phase 2 additions for forcing analysis:

- geopotential height
- temperature
- humidity
- vertical velocity
- upper-level wind fields
- any derived forcing terms required by the research design

## Suggested Python stack for Colab

- `xarray`
- `dask`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `zarr`
- `fsspec`
- `gcsfs`
- `shapely`
- `regionmask`

Optional:

- `cartopy` for publication-style maps

## Planned repo structure

- `docs/`
- `notebooks/`
- `src/jpcz_catalog/`

Planned Python modules:

- `src/jpcz_catalog/masks.py`
  - polygon and rectangle definitions
  - mask construction
- `src/jpcz_catalog/era5.py`
  - cloud dataset open helpers
  - variable selection helpers
- `src/jpcz_catalog/detect.py`
  - divergence computation
  - rolling means
  - threshold detection
  - event grouping
- `src/jpcz_catalog/classify.py`
  - monsoon and vorticity labels
- `src/jpcz_catalog/plotting.py`
  - QA time series
  - maps with units and sign conventions

## Planned notebook sequence

### Notebook 01: Data access and masks

- open ERA5 in Colab
- subset the working domain
- define and plot the JPCZ polygon
- define and plot the vorticity rectangle

### Notebook 02: December benchmark detector

- compute hourly `925 hPa` divergence
- compute polygon-mean `D`
- estimate `mean(D)` and `std(D)` for December 2000-2018
- detect major events
- compare count against Shinoda
- make the count comparison on the total event catalog before subtype labels are assigned
- write a compact verification summary for the December benchmark

### Notebook 03: Classification and QA

- compute `Seoul minus Sapporo` SLP index
- compute polygon and rectangle summary metrics
- assign strong/weak monsoon and lower/higher vorticity labels
- produce QA plots

### Notebook 04: NDJF production catalog

- run the production detector for November through February
- write the event catalog to a small CSV or Parquet file
- produce summary plots

### Notebook 05: February 2018 baseline review

- isolate `2-7 February 2018`
- compute hourly `925 hPa` divergence and rolling `D`
- verify a detected event with peak convergence on `3 February 2018`
- create a case-study plot set for the baseline event
- write a compact verification summary for the February 2018 case

## Colab usage rules

- subset before computing
- avoid calling `.load()` on a large multi-year domain
- keep outputs small and explicit
- restartable notebooks are required
- every notebook should include package install and import cells
- every plot should include field units
- time coordinates should stay in UTC unless a local-time view is explicitly needed

## GitHub workflow

Recommended pattern:

1. keep notebooks in GitHub
2. keep reusable science logic in `src/`
3. commit planning docs first
4. add notebook and module code in small steps
5. save only compact outputs that support validation
6. keep verification summaries in `outputs/verification/`

## First implementation milestone

The first coding milestone is successful when a notebook can:

- open ERA5 in Colab
- draw the prototype masks over the Sea of Japan
- compute a December divergence series with explicit units
- produce an initial set of detected December events for review
- summarize whether the total count is reasonably close to Shinoda's December benchmark
