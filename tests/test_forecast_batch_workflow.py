"""Bounded batch inputs, resumed member groups, configurable fields, and saved-only reports."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jpcz_catalog import weathernext_atlas as atlas
from jpcz_catalog import ensemble_verification as verification
from jpcz_catalog import verification_workflow as workflow
from test_weathernext_atlas import row as row_fixture,FakeArchive,make_panels


def synthetic_inputs():
    catalog=atlas.normalise_catalog(pd.DataFrame([dict(event_number=242,event_start='2023-01-23T19:00',
        event_end='2023-01-24T08:00',event_peak='2023-01-24T02:00')]))
    lat=np.arange(32.,49.,.5);lon=np.arange(124.,147.,.5)
    shape=(64,1,2,len(lat),len(lon));members=np.arange(64)[:,None,None,None,None]
    gradient=(lon-135)[None,None,None,None,:]
    coords=dict(sample=[str(i) for i in range(64)],time=pd.to_timedelta([24],unit='h'),
                level=[500,925],lat=lat,lon=lon)
    data=xr.Dataset({
        'u_component_of_wind':(('sample','time','level','lat','lon'),np.broadcast_to(members*gradient,shape).copy(),{'units':'m s-1'}),
        'v_component_of_wind':(('sample','time','level','lat','lon'),np.zeros(shape),{'units':'m s-1'}),
        'geopotential':(('sample','time','level','lat','lon'),np.broadcast_to((5000+10*members)*9.80665,shape).copy(),{'units':'m**2 s**-2'}),
    },coords=coords)
    reference=data.isel(sample=0,drop=True).rename({'lat':'latitude','lon':'longitude'}).copy(deep=True)
    reference=reference.assign_coords(time=[catalog.target_valid_time.iloc[0]])
    reference['u_component_of_wind'][:]=np.broadcast_to(31.5*(lon-135),(1,2,len(lat),len(lon)))
    reference['geopotential'][:]=(5000+315)*9.80665
    class Archive:
        root='synthetic';overrides={}
        def __init__(self):self.reads=0
        def run(self,init_time):
            self.reads+=1
            lead=int((catalog.target_valid_time.iloc[0]-atlas.utc_naive(init_time))/pd.Timedelta(hours=1))
            return 'synthetic/source',data.assign_coords(time=pd.to_timedelta([lead],unit='h'),
                init_time=atlas.utc_naive(init_time),datetime=('time',[catalog.target_valid_time.iloc[0]]))
    return catalog,Archive(),reference


def test_preflight_reuses_actual_figures_without_archive_access(tmp_path):
    row=row_fixture.__wrapped__();archive=FakeArchive()
    browser=atlas.WeatherNextBrowser(pd.DataFrame([row]),archive,tmp_path/'out',tmp_path/'refs')
    pilot=make_panels(row,browser.cache_dir,member=2)
    browser.loaded={'panels':pilot};browser.schema=archive.run(atlas.forecast_requests(row).init_time.iloc[0])[1]
    destination=atlas.export_event(pilot,row,browser.output_dir/'figures',features=False)
    times={p:p.stat().st_mtime_ns for p in destination.glob('*')}
    archive.run=lambda *a:(_ for _ in ()).throw(AssertionError('Already exported event must not access WN2'))
    report=atlas.export_catalog_single_member(browser)
    assert report.status.tolist()==['cached']
    assert all(path.stat().st_mtime_ns==value for path,value in times.items())
    assert len(pd.read_csv(next(browser.output_dir.glob('batch_*/figure_index.csv'))))==1
    specs=[pilot[i]['spec'] for i in (0,3)];domains=[pilot[i]['domain'] for i in (0,3)]
    assert atlas.find_completed_forecast_export(browser.output_dir/'figures',row,2,specs,domains)==destination
    (destination/'overview.png').unlink()
    assert atlas.find_completed_forecast_export(browser.output_dir/'figures',row,2,specs,domains) is None


def test_divergence_only_batch_has_three_leads_and_reuses_export(tmp_path):
    row=row_fixture.__wrapped__();archive=FakeArchive()
    spec=atlas.FIELD_PRESETS['divergence'];domain=(125.,145.,33.,47.)
    panels=atlas.event_panels(archive,tmp_path/'cache',row,2,[spec],[domain],progress=lambda _:None)
    assert len(panels)==3 and {p['spec'].key for p in panels}=={'divergence'}
    destination=atlas.export_event(panels,row,tmp_path/'figures',features=False)
    assert atlas.find_completed_forecast_export(tmp_path/'figures',row,2,[spec],[domain])==destination
    assert json.loads((destination/'complete.json').read_text())['forecast_panels']==3


def test_member_group_checkpoint_survives_mid_lead_failure(tmp_path,monkeypatch):
    catalog,archive,_=synthetic_inputs();request=atlas.forecast_requests(catalog.iloc[0]).iloc[0].to_dict()
    original=verification.polygon_divergence;calls=[]
    def fail(field):
        calls.append(field.sizes['sample'])
        if len(calls)==4:raise RuntimeError('synthetic interrupted lead')
        return original(field)
    monkeypatch.setattr(verification,'polygon_divergence',fail)
    with pytest.raises(RuntimeError,match='interrupted'):
        verification.forecast_polygon_values(archive,request,progress=lambda _:None,group_cache=tmp_path)
    assert len(list(tmp_path.glob('group_*.json')))==3
    calls.clear()
    def resume(field):calls.append(field.sizes['sample']);return original(field)
    monkeypatch.setattr(verification,'polygon_divergence',resume)
    result=verification.forecast_polygon_values(archive,request,progress=lambda _:None,group_cache=tmp_path)
    assert calls==[4]*13
    assert len(result['members'])==64
    assert all(len(result['values'][region])==64 for region in atlas.POLYGONS)


def test_shared_wind_input_cache_reuses_data_between_divergence_and_native_u(tmp_path,monkeypatch):
    catalog,archive,_=synthetic_inputs();archive.input_cache_dir=tmp_path
    request=atlas.forecast_requests(catalog.iloc[0]).iloc[0].to_dict();store,ds=archive.run(request['init_time'])
    domain=atlas.diagnostic_domain(ds,verification.VERIFICATION_DOMAIN)
    first=atlas.forecast_input_group(archive,store,ds,request,atlas.FIELD_PRESETS['divergence'],domain,0,4)
    monkeypatch.setattr(atlas,'compact_source_load',lambda *a:(_ for _ in ()).throw(AssertionError('Source reread')))
    cached=atlas.forecast_input_group(archive,store,ds,request,atlas.FIELD_PRESETS['u'],domain,0,4)
    np.testing.assert_array_equal(cached.u_component_of_wind,first.u_component_of_wind)
    assert len(list(tmp_path.glob('input_*.nc')))==2


@pytest.mark.parametrize('key,level,units',[('u',925,'m s⁻¹'),('v',925,'m s⁻¹'),('winds',925,'m s⁻¹'),
                                         ('height',500,'m'),('geopotential',500,'m² s⁻²')])
def test_configured_native_fields_keep_units_values_and_separate_checkpoints(tmp_path,key,level,units):
    catalog,archive,reference=synthetic_inputs();spec=replace(atlas.FIELD_PRESETS[key],level=level)
    tables,settings=verification.run_ensemble_verification(catalog,archive,tmp_path,
        era5_reader=lambda:reference,spec=spec,progress=lambda _:None)
    assert tables['case_ranks'].in_common_sample.all()
    assert tables['member_values'].units.eq(units).all()
    assert 'divergence_1e5_s-1' not in tables['member_values']
    assert tables['case_ranks'].field.eq(key).all()
    if key in ('height','geopotential'):
        expected=5315*(9.80665 if key=='geopotential' else 1)
        np.testing.assert_allclose(tables['era5_reference'].value,expected)
        assert tables['case_ranks'].era5_rank.eq(33).all()
    verification.export_verification(tables,settings,tmp_path/'report')
    ordered=pd.read_csv(tmp_path/'report/ordered_values.csv')
    assert len(ordered)==390 and ordered.units.eq(units).all()
    assert ordered['divergence_1e5_s-1'].isna().all()
    threshold=float(tables['era5_reference'].value.mean())
    probabilities=verification.reliability_probabilities(tables['member_values'],tables['case_ranks'],
        {region:threshold for region in atlas.POLYGONS},direction='greater')
    assert probabilities.units.eq(units).all() and probabilities.direction.eq('greater').all()


def test_saved_only_collection_keeps_all_values_in_one_sheet_and_never_calls_archive(tmp_path):
    catalog,archive,reference=synthetic_inputs()
    tables,settings=verification.run_ensemble_verification(catalog,archive,tmp_path,
        era5_reader=lambda:reference,progress=lambda _:None)
    destination=verification.export_verification(tables,settings,tmp_path/'catalog')
    archive.run=lambda *a:(_ for _ in ()).throw(AssertionError('Saved reports must not call archive'))
    configs=[{'field':'divergence','level':925,'threshold':-2}]
    runs=workflow.run_verification_fields(catalog,archive,tmp_path,destination,configs,saved_only=True)
    assert len(runs[0]['tables']['member_values'])==384
    combined,path=workflow.consolidate_saved_verification([destination],tmp_path/'comparison')
    assert len(combined)==390 and path.is_file()
    book=pd.ExcelFile(path);assert book.sheet_names==['All values']
    values=book.parse('All values',dtype={'member':str})
    assert len(values)==390 and values.source.eq('ERA5').sum()==6
    np.testing.assert_allclose(values.value,combined.value,rtol=1e-14,atol=1e-14)
    # One absent reference must not make its existing 64 member values disappear.
    tables['case_ranks']=tables['case_ranks'].loc[~((tables['case_ranks'].polygon=='JPCZ polygon')&tables['case_ranks'].lead_hours.eq(24))]
    verification.export_verification(tables,settings,destination)
    combined,_=workflow.consolidate_saved_verification([destination],tmp_path/'comparison')
    assert len(combined)==390


def test_notebook37_saved_native_wind_run_displays_and_keeps_base_directory(tmp_path):
    """Exercise the user-facing run cell for u, including a repeat run after settings edits."""
    import json
    from types import SimpleNamespace
    notebook=json.loads((ROOT/'notebooks/37_weathernext2_ensemble_verification.ipynb').read_text())
    source=next(''.join(cell['source']) for cell in notebook['cells']
                if ''.join(cell['source']).startswith('# @title 4 ·'))
    ranks=pd.DataFrame([dict(event_number=1,field='u',pressure_hpa=925,units='m s⁻¹',
        valid_time_utc='2023-01-01',polygon='JPCZ polygon',lead_days=1,member_count=64,
        era5_value=3.,era5_rank=20,in_common_sample=True,review_key='event1')])
    status=pd.DataFrame([dict(event_number=1,lead_days=1,era5_status='complete',
        wn2_status='complete',era5_error='',wn2_error='')])
    calls=[]
    def saved_run(*args,**kwargs):
        assert kwargs['saved_only'] is True
        calls.append(args[3])
        return [dict(tables={'case_ranks':ranks,'batch_status':status},settings={},
                     directory=tmp_path/'fields/u_925hpa')]
    env=dict(read_catalog=lambda:None,run_ensemble_verification=lambda:None,
        open_verification_era5=SimpleNamespace(cache_clear=lambda:None),archive=SimpleNamespace(clear=lambda:None),
        selected=ranks,OUTPUT_DIR=tmp_path,base_result_directory=tmp_path,VERIFY_FIELDS=[{'field':'u','level':925}],
        LOAD_SAVED_RESULTS_ONLY=True,ALLOW_ERA5T=True,run_verification_fields=saved_run,
        make_forecast_job_runner=lambda _:None,CATALOG_PATH=tmp_path/'catalog.csv',STUDY_START_UTC='2023-01-01',
        STUDY_END_UTC='',RUN_MODE='Catalog',LEAD_HOURS=atlas.LEAD_HOURS,display=lambda _:None)
    exec(compile(source,'notebook37_run_cell','exec'),env)
    exec(compile(source,'notebook37_run_cell','exec'),env)
    assert calls==[tmp_path,tmp_path]
