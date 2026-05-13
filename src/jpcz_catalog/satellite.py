"""Satellite image helpers for JPCZ event diagnostics."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from .config import BoundingBox

GIBS_WMS_BASE_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"

MODIS_LAYER_START_DATES = {
    "MODIS_Terra_CorrectedReflectance_TrueColor": pd.Timestamp("2000-02-24"),
    "MODIS_Aqua_CorrectedReflectance_TrueColor": pd.Timestamp("2002-07-04"),
}


def layer_available_for_date(layer_name: str, target_date: pd.Timestamp | str) -> bool:
    """Return whether the chosen MODIS layer should exist for the requested date."""
    first_available = MODIS_LAYER_START_DATES.get(layer_name)
    if first_available is None:
        return True
    return pd.Timestamp(target_date).normalize() >= first_available


def default_modis_layers_for_date(target_date: pd.Timestamp | str) -> list[str]:
    """Return the MODIS true-color layers that should be available for the date."""
    normalized = pd.Timestamp(target_date).normalize()
    layers: list[str] = []
    for layer_name, start_date in MODIS_LAYER_START_DATES.items():
        if normalized >= start_date:
            layers.append(layer_name)
    return layers


def build_gibs_wms_getmap_url(
    *,
    layer_name: str,
    target_date: pd.Timestamp | str,
    bbox: BoundingBox,
    width: int = 1600,
    height: int = 1200,
    image_format: str = "image/jpeg",
) -> str:
    """Build a direct NASA GIBS WMS GetMap URL for one daily MODIS snapshot."""
    params = {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.1.1",
        "layers": layer_name,
        "styles": "",
        "format": image_format,
        "transparent": "FALSE",
        "width": width,
        "height": height,
        "srs": "EPSG:4326",
        "bbox": f"{bbox.lon_min},{bbox.lat_min},{bbox.lon_max},{bbox.lat_max}",
        "time": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
    }
    return f"{GIBS_WMS_BASE_URL}?{urlencode(params)}"


def download_gibs_image(url: str, output_path: str | Path) -> Path:
    """Download one GIBS image to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response:
        image_bytes = response.read()

    output_path.write_bytes(image_bytes)
    return output_path
