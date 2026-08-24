"""Resumable NDJF climatology and presentation-map helpers.

This module deliberately separates the seasonal background state from the
event-composite diagnostics used elsewhere in the project.  It reduces ERA5
hourly fields in small time chunks, writes one Drive-ready checkpoint per
calendar month, and then creates a large-scale pressure/wind map plus a
regional terrain/SST/temperature/wind map.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
import gc
import math
import shutil
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

from .config import BoundingBox
from .era5 import open_arco_era5


LARGE_SCALE_DOMAIN = BoundingBox(lon_min=80.0, lon_max=220.0, lat_min=20.0, lat_max=70.0)
REGIONAL_DOMAIN = BoundingBox(lon_min=110.0, lon_max=155.0, lat_min=30.0, lat_max=55.0)
SEA_OF_JAPAN_DISPLAY_BOX = BoundingBox(lon_min=127.0, lon_max=142.0, lat_min=35.0, lat_max=48.0)
SIBERIAN_HIGH_BOX = BoundingBox(lon_min=80.0, lon_max=140.0, lat_min=40.0, lat_max=70.0)
ALEUTIAN_LOW_BOX = BoundingBox(lon_min=160.0, lon_max=220.0, lat_min=30.0, lat_max=65.0)

SURFACE_VARIABLES = ("mean_sea_level_pressure", "2m_temperature", "sea_surface_temperature")
WIND_VARIABLES = ("u_component_of_wind", "v_component_of_wind")
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def _latitude_slice(field: xr.Dataset | xr.DataArray, domain: BoundingBox) -> slice:
    latitude = field["latitude"]
    if float(latitude[0]) >= float(latitude[-1]):
        return slice(float(domain.lat_max), float(domain.lat_min))
    return slice(float(domain.lat_min), float(domain.lat_max))


def subset_domain(field: xr.Dataset | xr.DataArray, domain: BoundingBox):
    """Return a coordinate-order-safe geographic subset."""
    return field.sel(
        longitude=slice(float(domain.lon_min), float(domain.lon_max)),
        latitude=_latitude_slice(field, domain),
    )


def ndjf_calendar_months(years: tuple[int, ...] | list[int], months: tuple[int, ...] | list[int]) -> list[pd.Timestamp]:
    """Return sorted calendar-month starts used for an NDJF climatology."""
    return [
        pd.Timestamp(year=int(year), month=int(month), day=1)
        for year in sorted({int(value) for value in years})
        for month in sorted({int(value) for value in months})
    ]


def _atomic_netcdf(dataset: xr.Dataset, destination: Path, local_work_dir: Path) -> None:
    """Write locally first, then copy a complete checkpoint to Drive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    local_work_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = local_work_dir / f"{destination.stem}.temporary.nc"
    dataset.to_netcdf(temporary_path)
    shutil.copy2(temporary_path, destination)
    temporary_path.unlink(missing_ok=True)


def _load_checkpoint(path: Path) -> xr.Dataset | None:
    if not path.exists():
        return None


def _checkpoint_is_readable(path: Path) -> bool:
    """Check a Drive checkpoint without retaining its gridded fields in RAM."""
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path) as checkpoint:
            required = {"mslp_hpa", "u925_ms", "v925_ms", "t2m_c", "sst_c"}
            return required.issubset(checkpoint.data_vars) and int(checkpoint.attrs.get("sample_count_hours", 0)) > 0
    except Exception as exc:  # A partial Drive file should be rebuilt, not trusted.
        print(f"Ignoring unreadable checkpoint {path.name}: {type(exc).__name__}: {exc}")
        return False
    try:
        return xr.open_dataset(path).load()
    except Exception as exc:  # A partial Drive file should be rebuilt, not trusted.
        print(f"Ignoring unreadable checkpoint {path.name}: {type(exc).__name__}: {exc}")
        return None


def _month_time_slices(month_start: pd.Timestamp, chunk_days: int):
    month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
    chunk_start = month_start
    while chunk_start <= month_end:
        chunk_end = min(chunk_start + pd.Timedelta(days=int(chunk_days)) - pd.Timedelta(hours=1), month_end)
        yield chunk_start, chunk_end
        chunk_start = chunk_end + pd.Timedelta(hours=1)


def _reduced_bundle_for_month(
    ds: xr.Dataset,
    month_start: pd.Timestamp,
    *,
    large_domain: BoundingBox,
    regional_domain: BoundingBox,
    chunk_days: int,
) -> xr.Dataset:
    """Safely reduce one calendar month without loading a full month at once."""
    total_large = None
    total_surface = None
    total_count = 0

    time_slices = list(_month_time_slices(month_start, chunk_days))
    for chunk_number, (chunk_start, chunk_end) in enumerate(time_slices, start=1):
        # Select the vertical level *before* spatial/time data are read.  The
        # prior bundled request could cause a Colab worker to materialize more
        # of the multi-level wind store than the 925-hPa map requires.
        def mean_field(variable: str, domain: BoundingBox, *, level: int | None = None) -> xr.DataArray:
            field = ds[variable]
            if level is not None:
                field = field.sel(level=level)
            field = field.sel(time=slice(chunk_start, chunk_end))
            field = subset_domain(field, domain)
            result = field.mean("time", skipna=True).astype(float).load()
            del field
            gc.collect()
            return result

        mslp_hpa = (mean_field("mean_sea_level_pressure", large_domain) / 100.0).rename("mslp_hpa")
        u925_ms = mean_field("u_component_of_wind", large_domain, level=925).rename("u925_ms")
        v925_ms = mean_field("v_component_of_wind", large_domain, level=925).rename("v925_ms")
        t2m_c = (mean_field("2m_temperature", regional_domain) - 273.15).rename("t2m_c")
        sst_c = (mean_field("sea_surface_temperature", regional_domain) - 273.15).rename("sst_c")

        chunk_count = int(mslp_hpa.attrs.get("time_count", 0))
        if chunk_count == 0:
            chunk_count = len(pd.date_range(chunk_start, chunk_end, freq="1h"))
        if chunk_count == 0:
            continue

        large_mean = xr.Dataset({"mslp_hpa": mslp_hpa, "u925_ms": u925_ms, "v925_ms": v925_ms})
        surface_mean = xr.Dataset({"t2m_c": t2m_c, "sst_c": sst_c})

        total_surface = surface_mean * chunk_count if total_surface is None else total_surface + surface_mean * chunk_count
        total_large = large_mean * chunk_count if total_large is None else total_large + large_mean * chunk_count
        total_count += chunk_count
        if chunk_number == 1 or chunk_number == len(time_slices) or chunk_number % 7 == 0:
            print(f"    {month_start:%Y-%m}: reduced {chunk_number}/{len(time_slices)} time chunks through {chunk_end:%d %b}")
        del mslp_hpa, u925_ms, v925_ms, t2m_c, sst_c, large_mean, surface_mean
        gc.collect()

    if total_count == 0 or total_large is None or total_surface is None:
        raise RuntimeError(f"No hourly ERA5 values available for {month_start:%Y-%m}.")

    result = xr.merge([total_large / total_count, total_surface / total_count])
    result.attrs.update(
        {
            "sample_count_hours": int(total_count),
            "source_month": month_start.strftime("%Y-%m"),
            "description": "Hourly ERA5 mean reduced in time chunks for NDJF climatology.",
        }
    )
    return result


def load_or_build_ndjf_concept_climatology(
    *,
    cache_dir: str | Path,
    local_work_dir: str | Path,
    years: tuple[int, ...] | list[int],
    months: tuple[int, ...] | list[int] = (11, 12, 1, 2),
    large_domain: BoundingBox = LARGE_SCALE_DOMAIN,
    regional_domain: BoundingBox = REGIONAL_DOMAIN,
    chunk_days: int = 1,
) -> tuple[xr.Dataset | None, pd.DataFrame]:
    """Load or build a full NDJF mean with month-by-month Drive checkpoints.

    Returns ``(None, inventory)`` until every calendar-month checkpoint exists.
    Rerun the same cell after a disconnect; completed months are never fetched
    again.
    """
    cache_dir = Path(cache_dir)
    local_work_dir = Path(local_work_dir)
    period_starts = ndjf_calendar_months(years, months)
    records: list[dict[str, object]] = []
    complete_periods: set[pd.Timestamp] = set()

    for period in period_starts:
        path = cache_dir / f"ndjf_concept_month_{period:%Y_%m}.nc"
        if _checkpoint_is_readable(path):
            complete_periods.add(period)
            records.append({"month": period.strftime("%Y-%m"), "status": "complete", "path": str(path)})
        else:
            records.append({"month": period.strftime("%Y-%m"), "status": "pending", "path": str(path)})

    inventory = pd.DataFrame(records)
    pending_periods = [period for period in period_starts if period not in complete_periods]
    print(f"NDJF concept climatology: {len(complete_periods)} complete; {len(pending_periods)} pending calendar months.")

    if pending_periods:
        ds = open_arco_era5(chunks={"time": 48})
        for number, period in enumerate(pending_periods, start=1):
            print(f"{period:%Y-%m}: computing {number}/{len(pending_periods)} with {chunk_days}-day reductions...")
            checkpoint = _reduced_bundle_for_month(
                ds,
                period,
                large_domain=large_domain,
                regional_domain=regional_domain,
                chunk_days=chunk_days,
            )
            path = cache_dir / f"ndjf_concept_month_{period:%Y_%m}.nc"
            _atomic_netcdf(checkpoint, path, local_work_dir)
            print(f"  saved Drive checkpoint: {path.name} ({checkpoint.attrs['sample_count_hours']} hourly samples)")

        inventory = pd.DataFrame(
            [
                {
                    "month": period.strftime("%Y-%m"),
                    "status": "complete",
                    "path": str(cache_dir / f"ndjf_concept_month_{period:%Y_%m}.nc"),
                }
                for period in period_starts
            ]
        )

    total = None
    total_hours = 0
    for number, period in enumerate(period_starts, start=1):
        part = _load_checkpoint(cache_dir / f"ndjf_concept_month_{period:%Y_%m}.nc")
        if part is None:
            print("Some checkpoints remain unavailable. Rerun this cell to resume.")
            return None, inventory
        count = int(part.attrs.get("sample_count_hours", 0))
        if count <= 0:
            raise RuntimeError("A completed concept-climatology checkpoint lacks sample_count_hours.")
        total = (part * count).load() if total is None else (total + part * count).load()
        total_hours += count
        part.close()
        if number % 20 == 0 or number == len(period_starts):
            print(f"Combining saved monthly means: {number}/{len(period_starts)}")
    mean = (total / total_hours).load()
    mean.attrs.update(
        {
            "period_label": f"NDJF {min(years)}-{max(years)}",
            "climatology_years": ",".join(str(int(year)) for year in years),
            "climatology_months": ",".join(str(int(month)) for month in months),
            "total_hourly_samples": int(total_hours),
        }
    )
    return mean, inventory


def find_pressure_center(mslp_hpa: xr.DataArray, box: BoundingBox, mode: str) -> dict[str, float]:
    """Find an absolute seasonal MSLP maximum or minimum inside a declared box."""
    subset = subset_domain(mslp_hpa, box)
    if mode == "high":
        index = int(np.nanargmax(subset.values))
    elif mode == "low":
        index = int(np.nanargmin(subset.values))
    else:
        raise ValueError("mode must be 'high' or 'low'.")
    lat_index, lon_index = np.unravel_index(index, subset.shape)
    return {
        "longitude_degE": float(subset.longitude.values[lon_index]),
        "latitude_degN": float(subset.latitude.values[lat_index]),
        "mslp_hpa": float(subset.values[lat_index, lon_index]),
    }


def _tile_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    scale = 256 * (2**zoom)
    x = (lon + 180.0) / 360.0 * scale
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale
    return x, y


def _terrain_colors(elevation: np.ndarray) -> np.ndarray:
    levels = np.array([-500.0, -1.0, 0.0, 300.0, 800.0, 1500.0, 2500.0, 4000.0])
    colors = np.array([
        [22, 56, 120], [41, 94, 166], [72, 123, 77], [115, 153, 83],
        [187, 175, 100], [175, 118, 70], [138, 88, 67], [242, 238, 226],
    ], dtype=float)
    rgb = np.empty((*elevation.shape, 3), dtype=float)
    for channel in range(3):
        rgb[..., channel] = np.interp(elevation, levels, colors[:, channel])
    return rgb


def _hillshade(rgb: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(elevation)
    slope = np.pi / 2.0 - np.arctan(np.hypot(grad_x, grad_y) * 1.4)
    aspect = np.arctan2(-grad_x, grad_y)
    illumination = (
        np.sin(math.radians(45.0)) * np.sin(slope)
        + np.cos(math.radians(45.0)) * np.cos(slope) * np.cos(math.radians(315.0) - aspect)
    )
    multiplier = np.clip(0.62 + 0.55 * illumination, 0.58, 1.14)
    land = elevation > 1.0
    rgb[land] *= multiplier[land, None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def build_terrain_relief(
    domain: BoundingBox,
    *,
    cache_dir: str | Path,
    zoom: int = 6,
    output_width: int = 1600,
    output_height: int = 900,
) -> np.ndarray:
    """Return an unlabelled equirectangular terrain-relief RGB image.

    The source is the public Mapzen/AWS Terrarium DEM.  Downloaded tiles are
    cached locally so regenerating the final figure does not re-download them.
    """
    from PIL import Image

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_left, y_top = _tile_pixel(domain.lon_min, domain.lat_max, zoom)
    x_right, y_bottom = _tile_pixel(domain.lon_max, domain.lat_min, zoom)
    tx0, ty0 = int(math.floor(x_left / 256)), int(math.floor(y_top / 256))
    tx1, ty1 = int(math.ceil(x_right / 256)) - 1, int(math.ceil(y_bottom / 256)) - 1
    tile_indices = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]

    def fetch(tile: tuple[int, int]):
        x, y = tile
        path = cache_dir / f"z{zoom}_{x}_{y}.png"
        if path.exists():
            return tile, path.read_bytes()
        request = urllib.request.Request(
            TERRARIUM_URL.format(z=zoom, x=x, y=y),
            headers={"User-Agent": "JPCZ academic conceptual figure"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read()
        path.write_bytes(data)
        return tile, data

    tile_data: dict[tuple[int, int], bytes] = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(fetch, tile) for tile in tile_indices]
        for future in as_completed(futures):
            tile, data = future.result()
            tile_data[tile] = data

    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for (x, y), data in tile_data.items():
        mosaic.paste(Image.open(BytesIO(data)).convert("RGB"), ((x - tx0) * 256, (y - ty0) * 256))

    global_x0, global_y0 = tx0 * 256, ty0 * 256
    encoded = np.asarray(mosaic, dtype=float)
    elevation = encoded[..., 0] * 256.0 + encoded[..., 1] + encoded[..., 2] / 256.0 - 32768.0
    terrain = _hillshade(_terrain_colors(elevation), elevation)

    lons = np.linspace(domain.lon_min, domain.lon_max, output_width)
    lats = np.linspace(domain.lat_max, domain.lat_min, output_height)
    x_values = np.array([_tile_pixel(lon, domain.lat_max, zoom)[0] for lon in lons])
    y_values = np.array([_tile_pixel(domain.lon_min, lat, zoom)[1] for lat in lats])
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    x_indices = np.clip(np.rint(x_grid - global_x0).astype(int), 0, terrain.shape[1] - 1)
    y_indices = np.clip(np.rint(y_grid - global_y0).astype(int), 0, terrain.shape[0] - 1)
    return terrain[y_indices, x_indices]
