"""Bound source-chunk decoding and detach compact ERA5 fields for Notebook 35."""
import gc
import sys

import numpy as np
import pandas as pd
import xarray as xr
import zarr


def release_review_memory():
    """Collect closed figures and return free allocator pages on Linux/Colab."""
    gc.collect()
    if sys.platform.startswith('linux'):
        try:
            import ctypes
            library = ctypes.CDLL(None)
            trim = getattr(library, 'malloc_trim', None)
            if trim is not None:
                trim(0)
        except (AttributeError, OSError):
            pass


def read_review_era5_hour(dataset, hour, variables, domain, level=925):
    """Read exactly one source time and one variable at a time, then copy the region.

    ARCO wind chunks contain every pressure level and the globe. A regional slice
    must not retain a view into that decoded buffer after the read is finished.
    """
    hour = pd.Timestamp(hour)
    west, east, south, north = domain
    latitude = slice(north, south) if dataset.latitude[0] > dataset.latitude[-1] else slice(south, north)
    longitude = slice(west, east) if dataset.longitude[0] < dataset.longitude[-1] else slice(east, west)
    selected = dataset[list(variables)].sel(time=hour, longitude=longitude, latitude=latitude)
    if 'level' in selected.dims or 'level' in selected.coords:
        selected = selected.sel(level=level)
    coordinates = {name: np.array(selected[name].values, copy=True) for name in ('latitude', 'longitude')}
    fields = {}
    with zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1}):
        from numcodecs import blosc
        blosc.set_nthreads(1)
        for name in variables:
            loaded = selected[name].transpose('latitude', 'longitude').load(scheduler='synchronous')
            compact = xr.DataArray(np.array(loaded.values, copy=True, order='C'),
                dims=('latitude', 'longitude'), coords=coordinates, attrs=dict(loaded.attrs), name=name)
            fields[name] = compact
            # Replace the loaded view before releasing it, in case its Variable is shared.
            selected[name] = compact
            del loaded, compact
            release_review_memory()
    del selected
    return xr.Dataset(fields, coords=coordinates)


def read_review_era5_window(dataset, times, variables, domain, level=925):
    """Retain only compact regional snapshots when a 12-hour mean is requested."""
    records = []
    for hour in pd.DatetimeIndex(times):
        record = read_review_era5_hour(dataset, hour, variables, domain, level=level)
        records.append(record.expand_dims(time=[hour]))
    result = xr.concat(records, dim='time')
    del records
    release_review_memory()
    return result
