"""Mask construction helpers for JPCZ polygons and vorticity boxes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr
from matplotlib.path import Path as MplPath

from .config import BoundingBox


def _coord_values(coord: xr.DataArray | np.ndarray | Sequence[float]) -> np.ndarray:
    values = getattr(coord, "values", coord)
    return np.asarray(values)


def build_polygon_mask(
    longitude: xr.DataArray | np.ndarray | Sequence[float],
    latitude: xr.DataArray | np.ndarray | Sequence[float],
    polygon_vertices: Sequence[tuple[float, float]],
) -> xr.DataArray:
    """Return a boolean mask for the polygon on a lat-lon grid."""
    lon_values = _coord_values(longitude)
    lat_values = _coord_values(latitude)
    lon2d, lat2d = np.meshgrid(lon_values, lat_values)
    polygon_path = MplPath(polygon_vertices)
    mask = polygon_path.contains_points(
        np.column_stack([lon2d.ravel(), lat2d.ravel()])
    ).reshape(lat2d.shape)
    return xr.DataArray(
        mask,
        coords={"latitude": lat_values, "longitude": lon_values},
        dims=("latitude", "longitude"),
        name="jpcz_polygon_mask",
    )


def build_coslat_weights(
    latitude: xr.DataArray | np.ndarray | Sequence[float],
    longitude: xr.DataArray | np.ndarray | Sequence[float],
    *,
    mask: xr.DataArray | np.ndarray | None = None,
) -> xr.DataArray:
    """Return cosine-latitude area weights, optionally masked."""
    lon_values = _coord_values(longitude)
    lat_values = _coord_values(latitude)
    lon2d, lat2d = np.meshgrid(lon_values, lat_values)
    weights = np.cos(np.deg2rad(lat2d))

    if mask is not None:
        mask_values = np.asarray(getattr(mask, "values", mask), dtype=float)
        weights = weights * mask_values

    return xr.DataArray(
        weights,
        coords={"latitude": lat_values, "longitude": lon_values},
        dims=("latitude", "longitude"),
        name="area_weights",
    )


def box_outline(box: BoundingBox) -> tuple[list[float], list[float]]:
    """Return longitude and latitude vertices for plotting a bounding box."""
    lons = [
        box.lon_min,
        box.lon_max,
        box.lon_max,
        box.lon_min,
        box.lon_min,
    ]
    lats = [
        box.lat_min,
        box.lat_min,
        box.lat_max,
        box.lat_max,
        box.lat_min,
    ]
    return lons, lats


def split_polygon_land_ocean_mask(
    longitude: xr.DataArray,
    latitude: xr.DataArray,
    polygon_vertices: Sequence[tuple[float, float]],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Split polygon cells into ocean and land subsets using Natural Earth."""
    import regionmask

    polygon_mask = build_polygon_mask(longitude, latitude, polygon_vertices)

    defined = regionmask.defined_regions
    if hasattr(defined, "natural_earth_v5_1_2"):
        ne = defined.natural_earth_v5_1_2
    elif hasattr(defined, "natural_earth_v5_0_0"):
        ne = defined.natural_earth_v5_0_0
    elif hasattr(defined, "natural_earth_v4_1_0"):
        ne = defined.natural_earth_v4_1_0
    else:
        raise RuntimeError("No supported Natural Earth region set found in regionmask.")

    land_regions = getattr(ne, "land_50", getattr(ne, "land_110"))
    land_mask = land_regions.mask(longitude, latitude)
    ocean_mask = np.isnan(land_mask.values)

    polygon_ocean_mask = xr.DataArray(
        polygon_mask.values & ocean_mask,
        coords=polygon_mask.coords,
        dims=polygon_mask.dims,
        name="polygon_ocean_mask",
    )
    polygon_land_mask = xr.DataArray(
        polygon_mask.values & (~ocean_mask),
        coords=polygon_mask.coords,
        dims=polygon_mask.dims,
        name="polygon_land_mask",
    )

    return polygon_mask, polygon_ocean_mask, polygon_land_mask

