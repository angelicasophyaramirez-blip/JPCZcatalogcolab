"""Real child processes with synthetic readers: recycling, secrets, and failure cleanup."""
import ast
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jpcz_catalog import forecast_batch_runtime as runtime


def fake_bundle(tmp_path):
    atlas='''
from types import SimpleNamespace
class FieldSpec(SimpleNamespace): pass
class GoogleCloudArchive:
    def __init__(self,credentials,*args,**kwargs):
        assert credentials.token
def event_panels(*args,**kwargs): return []
def export_event(*args,**kwargs): raise AssertionError('No rendering in scalar worker tests')
'''
    verification='''
import os, resource, sys, time
from pathlib import Path
import numpy as np
HELD=[]
def open_verification_era5(): return None
def era5_polygon_values(time_text,source,**kwargs):
    if time_text=='sleep':time.sleep(30)
    return {'values':{'JPCZ polygon':-2.,'Coastal wedge':-3.}}
def forecast_polygon_values(archive,request,*,spec,group_cache,member_start,member_stop):
    assert member_stop-member_start==16
    HELD.append(np.ones(64*1024**2,dtype=np.uint8))
    root=Path(group_cache);root.mkdir(parents=True,exist_ok=True)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    (root/f'pid_{os.getpid()}.json').write_text(str(peak))
    return {'members':[str(i) for i in range(member_start,member_stop)],
        'values':{'JPCZ polygon':list(range(member_start,member_stop)),
                  'Coastal wedge':list(range(member_start,member_stop))},
        'init_time':str(request['init_time']),'valid_time':str(request['valid_time']),
        'lead_hours':request['lead_hours'],'source_store':'fake/source'+(str(member_start) if request.get('different_sources') else ''),
        'polygon_cells':{'JPCZ polygon':10,'Coastal wedge':10}}
'''
    payload=BytesIO()
    with zipfile.ZipFile(payload,'w') as bundle:
        for name,text in {'worker.py':(ROOT/'src/jpcz_catalog/forecast_batch_worker.py').read_text(),
            'jpcz_catalog/__init__.py':'','jpcz_catalog/weathernext_atlas.py':atlas,
            'jpcz_catalog/ensemble_verification.py':verification}.items():bundle.writestr(name,text)
    archive=tmp_path/'runtime.zip';archive.write_bytes(payload.getvalue())
    return archive,hashlib.sha256(payload.getvalue()).hexdigest()


def runner(tmp_path,**kwargs):
    path,digest=fake_bundle(tmp_path)
    archive=SimpleNamespace(root='weathernext/weathernext_2_0_0/zarr',overrides={},input_cache_dir=None,
        storage_options={'token':SimpleNamespace(valid=True,token='secret-for-pipe-only',expiry=None)})
    return runtime.make_forecast_job_runner(archive,archive_path=path,digest=digest,**kwargs)


def forecast_payload(tmp_path):
    return dict(spec={'key':'divergence','variables':['u','v'],'limits':[-10,10]},group_cache=str(tmp_path/'groups'),
                request=dict(init_time='2023-01-01T00:00:00Z',valid_time='2023-01-02T00:00:00Z',lead_hours=24))


def test_four_fresh_processes_return_all_64_members_and_keep_token_out_of_files(tmp_path):
    result=runner(tmp_path)('forecast',forecast_payload(tmp_path))
    assert result['members']==[str(i) for i in range(64)]
    assert result['values']['JPCZ polygon']==list(range(64))
    peaks=[float(path.read_text()) for path in (tmp_path/'groups').glob('pid_*.json')]
    assert len(peaks)==4 and max(peaks)-min(peaks)<96
    for path in tmp_path.rglob('*'):
        if path.is_file():assert b'secret-for-pipe-only' not in path.read_bytes()


def test_member_parts_from_different_sources_are_rejected(tmp_path):
    payload=forecast_payload(tmp_path);payload['request']['different_sources']=True
    with pytest.raises(ValueError,match='inconsistent forecast sources'):
        runner(tmp_path)('forecast',payload)


def test_timeout_and_interrupt_clean_up_child(tmp_path,monkeypatch):
    children=[];original=runtime.subprocess.Popen
    def capture(*args,**kwargs):
        process=original(*args,**kwargs);children.append(process);return process
    monkeypatch.setattr(runtime.subprocess,'Popen',capture)
    with pytest.raises(RuntimeError,match='timed out'):
        runner(tmp_path,timeout=.1)('reference',dict(valid_time='sleep',allow_era5t=False,
            spec={'key':'divergence','variables':['u','v'],'limits':[-10,10]}))
    assert all(p.poll() is not None for p in children)
    def interrupt(*args):raise KeyboardInterrupt
    monkeypatch.setattr(runtime.select,'select',interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner(tmp_path)('reference',dict(valid_time='sleep',allow_era5t=False,
            spec={'key':'divergence','variables':['u','v'],'limits':[-10,10]}))
    assert len(children)==2 and all(p.poll() is not None for p in children)


def test_generated_worker_bundle_is_identical_in_both_notebooks():
    digests=[]
    for name in ('36_weathernext2_jpcz_forecast_atlas.ipynb','37_weathernext2_ensemble_verification.ipynb'):
        nb=json.loads((ROOT/'notebooks'/name).read_text())
        source=next(''.join(c['source']) for c in nb['cells'] if 'def make_forecast_job_runner(' in ''.join(c['source']))
        assignments={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in ('FORECAST_WORKER_SHA256','FORECAST_WORKER_URL')}
        path=ROOT/'notebooks/support'/assignments['FORECAST_WORKER_URL'].rsplit('/',1)[-1]
        digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==assignments['FORECAST_WORKER_SHA256']
        with zipfile.ZipFile(path) as bundle:
            assert bundle.read('jpcz_catalog/weathernext_atlas.py').decode()==(ROOT/'src/jpcz_catalog/weathernext_atlas.py').read_text()
            assert bundle.read('jpcz_catalog/ensemble_verification.py').decode()==(ROOT/'src/jpcz_catalog/ensemble_verification.py').read_text()
            assert bundle.read('worker.py').decode()==(ROOT/'src/jpcz_catalog/forecast_batch_worker.py').read_text()
        digests.append(digest)
    assert len(set(digests))==1
