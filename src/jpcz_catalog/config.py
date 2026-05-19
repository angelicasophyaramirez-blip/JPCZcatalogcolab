"""Shared project configuration for Colab notebooks and helper modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class BoundingBox:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class GeographicPoint:
    name: str
    latitude: float
    longitude: float


ARCO_ERA5_ZARR_STORE: Final[str] = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

WORKING_DOMAIN = BoundingBox(
    lon_min=120.0,
    lon_max=150.0,
    lat_min=30.0,
    lat_max=50.0,
)

EXTENDED_DOMAIN = BoundingBox(
    lon_min=120.0,
    lon_max=160.0,
    lat_min=30.0,
    lat_max=50.0,
)

# Tighter domain for objective subtype characterization and clustering.
# This still covers the JPCZ polygon plus the coastal, Pacific, Hokkaido,
# Sea of Japan, and frontal characterization regions, while reducing
# Colab memory pressure compared with the larger EXTENDED_DOMAIN.
OBJECTIVE_SUBTYPE_DOMAIN = BoundingBox(
    lon_min=129.0,
    lon_max=151.0,
    lat_min=33.0,
    lat_max=47.0,
)

VORTICITY_BOX = BoundingBox(
    lon_min=127.0,
    lon_max=140.0,
    lat_min=37.0,
    lat_max=45.5,
)

# First-pass characterization regions for objective subtype analysis.
# These do not affect event detection; they only describe where convergence
# and forcing are strongest after the Shinoda-style catalog has been built.
COASTAL_JAPAN_BOX = BoundingBox(
    lon_min=131.0,
    lon_max=140.0,
    lat_min=34.0,
    lat_max=40.5,
)

PACIFIC_EAST_OF_JAPAN_BOX = BoundingBox(
    lon_min=141.0,
    lon_max=150.0,
    lat_min=33.0,
    lat_max=42.0,
)

HOKKAIDO_BOX = BoundingBox(
    lon_min=139.0,
    lon_max=146.5,
    lat_min=41.0,
    lat_max=46.5,
)

SEA_OF_JAPAN_BOX = BoundingBox(
    lon_min=129.0,
    lon_max=140.0,
    lat_min=36.0,
    lat_max=45.0,
)

HOKKAIDO_FRONT_BOX = BoundingBox(
    lon_min=136.0,
    lon_max=147.0,
    lat_min=39.0,
    lat_max=47.0,
)

PACIFIC_FRONT_BOX = BoundingBox(
    lon_min=141.0,
    lon_max=151.0,
    lat_min=33.0,
    lat_max=42.0,
)

# First-pass digitization from Shinoda Figure 2. These values are intended
# to be refined only if validation indicates a clear mismatch.
JPCZ_POLYGON_VERTICES = (
    (129.5, 41.0),
    (136.0, 37.4),
    (134.5, 35.8),
    (128.8, 38.0),
)

SEOUL = GeographicPoint(name="Seoul", latitude=37.5665, longitude=126.9780)
SAPPORO = GeographicPoint(name="Sapporo", latitude=43.0618, longitude=141.3545)

DECEMBER_BENCHMARK_YEARS: Final[tuple[int, ...]] = tuple(range(2000, 2019))
BASELINE_START_UTC: Final[str] = "2018-02-02"
BASELINE_END_UTC: Final[str] = "2018-02-07T23:00:00"
BASELINE_PEAK_DATE_UTC: Final[str] = "2018-02-03"

EVENT_FIELD_UNITS = {
    "divergence_925hpa": "s^-1",
    "divergence_925hpa_display": "1e-5 s^-1",
    "relative_vorticity_925hpa": "s^-1",
    "relative_vorticity_925hpa_display": "1e-5 s^-1",
    "geopotential_height_anomaly_850hpa": "gpm",
    "temperature_gradient_850hpa_display": "K (100 km)^-1",
    "seoul_minus_sapporo_slp": "hPa",
}
