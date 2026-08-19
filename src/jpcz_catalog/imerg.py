"""Small, reproducible helpers for GPM IMERG Final half-hourly data.

The IMERG HDF5 layout stores its V07 precipitation field as
``(time, longitude, latitude)``.  These helpers deliberately read only a
small requested spatial subset before transposing it to the conventional
``(latitude, longitude)`` analysis layout used elsewhere in this project.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

import numpy as np
import xarray as xr

from .config import BoundingBox
from .masks import build_coslat_weights, build_polygon_mask


# V07 documents the half-hour field as ``precipitation``.  Keep the older
# calibrated-field name as a fallback for legacy archive granules.
IMERG_PRECIPITATION_PATHS = (
    "Grid/precipitation",
    "Grid/precipitationCal",
)
IMERG_LATITUDE_PATH = "Grid/lat"
IMERG_LONGITUDE_PATH = "Grid/lon"


def parse_imerg_granule_start(path: str | Path) -> np.datetime64:
    """Return the UTC start time encoded in a standard IMERG granule name."""
    match = re.search(r"\.(\d{8})-S(\d{6})-E\d{6}", Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse an IMERG start time from {path!s}")
    date_part, time_part = match.groups()
    return np.datetime64(
        f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}T"
        f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
    )


def _contiguous_slice(values: np.ndarray, lower: float, upper: float) -> slice:
    indices = np.flatnonzero((values >= lower) & (values <= upper))
    if indices.size == 0:
        raise ValueError(f"No IMERG grid cells fall within {lower} to {upper}.")
    if not np.all(np.diff(indices) == 1):
        raise ValueError("Requested IMERG longitude range crosses a discontinuity.")
    return slice(int(indices[0]), int(indices[-1]) + 1)


def read_precipitation_cal_subset(
    path: str | Path,
    *,
    domain: BoundingBox,
) -> xr.DataArray:
    """Read one IMERG V07 precipitation field over a small lat/lon domain.

    The returned field is in mm h^-1 and has dimensions ``latitude`` then
    ``longitude``.  Negative fill values are converted to NaN.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        latitudes = np.asarray(handle[IMERG_LATITUDE_PATH][:], dtype=float)
        longitudes = np.asarray(handle[IMERG_LONGITUDE_PATH][:], dtype=float)
        lon_slice = _contiguous_slice(longitudes, domain.lon_min, domain.lon_max)
        lat_slice = _contiguous_slice(latitudes, domain.lat_min, domain.lat_max)

        dataset_path = next(
            (candidate for candidate in IMERG_PRECIPITATION_PATHS if candidate in handle),
            None,
        )
        if dataset_path is None:
            raise KeyError(
                "No supported IMERG precipitation field found. Expected one of "
                f"{IMERG_PRECIPITATION_PATHS}; available groups: {list(handle.keys())}."
            )
        dataset = handle[dataset_path]
        fill_value = float(dataset.attrs.get("_FillValue", -9999.9))
        # Native IMERG order is (time, longitude, latitude), with one time
        # element per granule.  Transpose to the project-wide lat/lon order.
        values = np.asarray(dataset[0, lon_slice, lat_slice], dtype=float).T

    values[np.isclose(values, fill_value) | (values < -100.0)] = np.nan
    return xr.DataArray(
        values,
        coords={
            "latitude": latitudes[lat_slice],
            "longitude": longitudes[lon_slice],
        },
        dims=("latitude", "longitude"),
        name="imerg_precipitation_rate",
        attrs={"units": "mm h^-1", "source_variable": dataset_path},
    )


def region_mean_rates(
    rate_field: xr.DataArray,
    regions: Mapping[str, Sequence[tuple[float, float]]],
) -> dict[str, float]:
    """Return cosine-latitude-area-weighted regional precipitation rates."""
    output: dict[str, float] = {}
    for region_name, vertices in regions.items():
        mask = build_polygon_mask(rate_field.longitude, rate_field.latitude, vertices)
        weights = build_coslat_weights(
            rate_field.latitude,
            rate_field.longitude,
            mask=mask,
        )
        valid_weights = weights.where(rate_field.notnull())
        denominator = float(valid_weights.sum().values)
        if denominator == 0.0:
            output[region_name] = float("nan")
        else:
            output[region_name] = float(
                ((rate_field * valid_weights).sum() / valid_weights.sum()).values
            )
    return output
