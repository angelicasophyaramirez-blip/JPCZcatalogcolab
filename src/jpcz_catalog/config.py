"""Shared project configuration for Colab notebooks and helper modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


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

VORTICITY_BOX = BoundingBox(
    lon_min=127.0,
    lon_max=140.0,
    lat_min=37.0,
    lat_max=45.5,
)

# First-pass digitization from Shinoda Figure 2. These values are intended
# to be refined only if validation indicates a clear mismatch.
JPCZ_POLYGON_VERTICES = (
    (129.5, 41.0),
    (136.0, 37.4),
    (134.5, 35.8),
    (128.8, 38.0),
)

EVENT_FIELD_UNITS = {
    "divergence_925hpa": "s^-1",
    "divergence_925hpa_display": "1e-5 s^-1",
    "relative_vorticity_925hpa": "s^-1",
    "relative_vorticity_925hpa_display": "1e-5 s^-1",
    "seoul_minus_sapporo_slp": "hPa",
}

