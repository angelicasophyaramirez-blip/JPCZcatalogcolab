"""Fresh-process forecast jobs, with credentials passed only over an in-memory pipe."""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
import zipfile

FORECAST_WORKER_SHA256 = ''
FORECAST_WORKER_URL = ''


def forecast_json(value):
    import numpy as np
    import pandas as pd
    if isinstance(value,dict):return {str(k):forecast_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [forecast_json(v) for v in value]
    if isinstance(value,np.generic):return forecast_json(value.item())
    if isinstance(value,(pd.Timestamp,Path)):return str(value)
    if value is None or value is pd.NA or (isinstance(value,float) and not np.isfinite(value)):return None
    return value


def prepare_forecast_worker(digest, url, archive_path=None):
    root=Path(tempfile.gettempdir())/'jpcz_forecast_workers'/digest
    root.mkdir(parents=True,exist_ok=True)
    archive=root/'runtime.zip'
    if archive_path is not None:payload=Path(archive_path).read_bytes()
    elif archive.exists():payload=archive.read_bytes()
    else:
        with urlopen(url,timeout=60) as response:payload=response.read()
    if hashlib.sha256(payload).hexdigest()!=digest:
        raise RuntimeError('Forecast worker checksum mismatch. Reload the updated notebook.')
    archive.write_bytes(payload)
    allowed={'worker.py','jpcz_catalog/__init__.py','jpcz_catalog/weathernext_atlas.py',
             'jpcz_catalog/ensemble_verification.py'}
    with zipfile.ZipFile(BytesIO(payload)) as bundle:
        if set(bundle.namelist())!=allowed:raise RuntimeError('Unexpected forecast worker files.')
        for name in sorted(allowed):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(bundle.read(name))
    return root/'worker.py'


def make_forecast_job_runner(archive, *, archive_path=None, digest=None, url=None, timeout=1200):
    """Jobs return compact metadata/scalars; weather arrays stay in their child process."""
    worker=None
    def run(kind,payload):
        nonlocal worker
        if kind=='forecast':
            # Recycle the interpreter after 16 members (four source groups), never all 64.
            parts=[run('forecast_part',{**payload,'member_start':start,'member_stop':start+16})
                   for start in range(0,64,16)]
            first=dict(parts[0])
            for part in parts[1:]:
                if any(part[key]!=first[key] for key in ('init_time','valid_time','lead_hours','source_store','polygon_cells')):
                    raise ValueError('Member workers returned inconsistent forecast sources or timing.')
            first['members']=[member for part in parts for member in part['members']]
            if len(first['members'])!=64 or len(set(first['members']))!=64:
                raise ValueError('Forecast workers did not return 64 distinct native members.')
            first['values']={region:[value for part in parts for value in part['values'][region]] for region in first['values']}
            return first
        if worker is None:
            worker=prepare_forecast_worker(digest or FORECAST_WORKER_SHA256,url or FORECAST_WORKER_URL,archive_path)
        credentials=archive.storage_options['token']
        expiry=getattr(credentials,'expiry',None)
        expiring=False
        if expiry is not None:
            import pandas as pd
            expiry=pd.Timestamp(expiry)
            expiry=expiry.tz_localize('UTC') if expiry.tzinfo is None else expiry.tz_convert('UTC')
            expiring=(expiry-pd.Timestamp.now(tz='UTC')).total_seconds()<timeout+120
        if not credentials.valid or expiring:
            from google.auth.transport.requests import Request
            credentials.refresh(Request())
        if not credentials.token:raise RuntimeError('Rerun Google authentication before retrieving forecasts.')
        request=forecast_json(dict(kind=kind,payload=payload,token=credentials.token,
            project=archive.storage_options.get('project',''),root=archive.root,overrides=archive.overrides,
            input_cache_dir=getattr(archive,'input_cache_dir',None)))
        env=os.environ.copy()
        env.update(MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='jpcz_forecast_reply_') as folder:
            reply=Path(folder)/'reply.json';request['reply_path']=str(reply)
            process=subprocess.Popen([sys.executable,'-u',str(worker)],stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
            try:
                process.stdin.write(json.dumps(request,allow_nan=False));process.stdin.close();del request
                deadline=time.monotonic()+timeout
                while process.poll() is None:
                    if time.monotonic()>deadline:raise RuntimeError('Forecast worker timed out; saved groups remain reusable.')
                    ready,_,_=select.select([process.stdout],[],[],1)
                    if ready:
                        line=process.stdout.readline()
                        if line:print(line,end='',flush=True)
                for line in process.stdout:print(line,end='',flush=True)
                if process.returncode or not reply.exists():
                    raise RuntimeError(f'Forecast worker exited {process.returncode}; completed checkpoints remain reusable.')
                result=json.loads(reply.read_text())
                if 'error' in result:raise RuntimeError(result['error'])
                return result['data']
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                for stream in (process.stdin,process.stdout):
                    if stream and not stream.closed:stream.close()
    return run
