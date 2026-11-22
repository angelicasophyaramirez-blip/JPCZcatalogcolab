"""Serial source reads, compact ownership, unchanged map fields, and per-hour vortex recovery."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
import metpy.calc as mpcalc
from metpy.units import units

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jpcz_catalog.case_review_memory import read_review_era5_hour, read_review_era5_window, release_review_memory


def winds(hours=25):
    times = pd.date_range('2001-12-20T18:00', periods=hours, freq='h')
    lat, lon = np.linspace(47, 33, 15), np.linspace(125, 145, 21)
    shape = (hours, 2, len(lat), len(lon))
    gradient = np.arange(hours)[:, None, None, None] * .1 + (lon-135)[None, None, None, :]
    return xr.Dataset({
        'u_component_of_wind': (('time','level','latitude','longitude'), np.broadcast_to(-gradient, shape).copy()),
        'v_component_of_wind': (('time','level','latitude','longitude'), np.broadcast_to(gradient*.1, shape).copy()),
        'mean_sea_level_pressure': (('time','latitude','longitude'), np.broadcast_to(101000+(lon-135)*10, (hours,len(lat),len(lon))).copy()),
        'surface_pressure': (('time','latitude','longitude'), np.full((hours,len(lat),len(lon)), 100000.)),
    }, coords={'time': times, 'level':[850,925], 'latitude':lat, 'longitude':lon})


def notebook_functions(names):
    nb=json.loads((ROOT / 'notebooks/35_imerg_manual_case_review.ipynb').read_text())
    source=next(''.join(c['source']) for c in nb['cells'] if 'def get_case(' in ''.join(c['source']))
    nodes=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in names]
    ns=dict(np=np,pd=pd,xr=xr,mpcalc=mpcalc,units=units,json=json,hashlib=hashlib,Path=Path,
            read_review_era5_window=read_review_era5_window, read_review_era5_hour=read_review_era5_hour,
            release_review_memory=release_review_memory)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'notebook35_memory','exec'),ns)
    return ns


@pytest.mark.parametrize('use_dask',[False,True])
def test_only_single_time_single_variable_is_loaded_and_arrays_own_compact_memory(monkeypatch,use_dask):
    source = winds(12)
    if use_dask:
        source = source.chunk({'time':3,'level':2})
    calls=[]
    load=xr.DataArray.load
    def inspected(array,**kwargs):
        calls.append((array.name,array.dims,dict(array.sizes)))
        assert 'time' not in array.dims and 'level' not in array.dims
        assert zarr.config.get('async.concurrency') == 1
        assert zarr.config.get('threading.max_workers') == 1
        assert kwargs['scheduler'] == 'synchronous'
        return load(array,**kwargs)
    monkeypatch.setattr(xr.DataArray,'load',inspected)
    variables=('u_component_of_wind','v_component_of_wind','mean_sea_level_pressure')
    result=read_review_era5_window(source,source.time.values,variables,(128,142,35,44))
    assert len(calls)==12*3
    assert [name for name,_,_ in calls]==list(variables)*12
    expected=source[list(variables)].sel(time=source.time.values,level=925,
        longitude=slice(128,142),latitude=slice(44,35))
    for name in variables:
        np.testing.assert_array_equal(result[name].values,expected[name].values)
    single=read_review_era5_hour(source,source.time.values[0],variables,(128,142,35,44))
    assert all(single[name].values.flags.owndata for name in variables)
    assert sum(single[name].nbytes for name in variables)<10000


def test_map_cache_and_original_numerical_definition_are_preserved(tmp_path):
    source=winds(12)
    ns=notebook_functions({'load_era5','window_bounds','cache_id'})
    ns.update(era5=source,CACHE_DIR=tmp_path,DOMAIN=(125,145,33,47),
              IMERG_PRODUCT='test',IMERG_VERSION='07',MIN_PIXEL_TEMPORAL_COVERAGE=.9)
    hour=pd.Timestamp(source.time.values[-1])
    result=ns['load_era5'](hour,12)
    old=source.sel(level=925)
    dx,dy=mpcalc.lat_lon_grid_deltas(old.longitude.values,old.latitude.values)
    expected=mpcalc.divergence(old.u_component_of_wind.mean('time').values*units('m/s'),
                             old.v_component_of_wind.mean('time').values*units('m/s'),dx=dx,dy=dy).magnitude*1e5
    np.testing.assert_array_equal(result.divergence,expected)
    np.testing.assert_array_equal(result.u,old.u_component_of_wind.isel(time=-1))
    np.testing.assert_array_equal(result.v,old.v_component_of_wind.isel(time=-1))
    # Already saved maps must not reread cloud fields after the memory fix.
    ns['read_review_era5_window']=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('cache miss'))
    cached=ns['load_era5'](hour,12)
    np.testing.assert_array_equal(cached.divergence,result.divergence)


def test_vortex_scan_checkpoints_each_hour_and_resumes_only_missing_hours(tmp_path):
    source=winds(25)
    ns=notebook_functions({'load_vortex_hourly','atomic_json'})
    ns.update(era5=source,DOMAIN=(125,145,33,47),vortex_cache_dir=lambda:tmp_path,
              detect_mslp_lows=lambda *a,**kw:[{'lon':132.,'lat':39.}],
              cyclone_circulation_evidence=lambda *a,**kw:{'passes_circulation_screen':True})
    calls=[]
    def crash(dataset,hour,variables,domain):
        calls.append(hour)
        if len(calls)==7:
            raise RuntimeError('simulated cloud interruption')
        return read_review_era5_hour(dataset,hour,variables,domain)
    ns['read_review_era5_hour']=crash
    start,end=pd.Timestamp(source.time.values[0]),pd.Timestamp(source.time.values[-1])
    with pytest.raises(RuntimeError,match='interruption'):
        ns['load_vortex_hourly'](start,end)
    assert len(list(tmp_path.glob('hour_*.json')))==6
    resumed=[]
    def read(dataset,hour,variables,domain):
        resumed.append(hour)
        return read_review_era5_hour(dataset,hour,variables,domain)
    ns['read_review_era5_hour']=read
    result=ns['load_vortex_hourly'](start,end)
    assert len(resumed)==19
    assert len(result)==25 and all(len(row['centers'])==1 for row in result)
    resumed.clear()
    ns['load_vortex_hourly'](start,end)
    assert not resumed
