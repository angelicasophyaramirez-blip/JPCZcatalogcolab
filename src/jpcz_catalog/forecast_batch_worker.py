"""Execute one bounded forecast/export job using the bundled, versioned implementation."""
import json
import os
from pathlib import Path
import resource
import sys


def main():
    request=json.load(sys.stdin)
    reply=Path(request.pop('reply_path'))
    try:
        if sys.platform.startswith('linux'):
            _,hard=resource.getrlimit(resource.RLIMIT_AS)
            limit=6*1024**3 if hard==resource.RLIM_INFINITY else min(6*1024**3,hard)
            resource.setrlimit(resource.RLIMIT_AS,(limit,hard))
        import pandas as pd
        from google.oauth2.credentials import Credentials
        from jpcz_catalog.weathernext_atlas import GoogleCloudArchive,FieldSpec,event_panels,export_event
        from jpcz_catalog.ensemble_verification import open_verification_era5,era5_polygon_values,forecast_polygon_values
        archive=GoogleCloudArchive(Credentials(request.pop('token')),request['project'],
            root=request['root'],overrides=request['overrides'],lazy_indexing=True,
            input_cache_dir=request.get('input_cache_dir'))
        payload=request['payload']
        def spec(data):
            data=dict(data);data['variables']=tuple(data['variables']);data['limits']=tuple(data['limits'])
            return FieldSpec(**data)
        if request['kind']=='reference':
            data=era5_polygon_values(payload['valid_time'],open_verification_era5(),
                allow_era5t=payload['allow_era5t'],spec=spec(payload['spec']))
        elif request['kind']=='forecast_part':
            for name in ('init_time','valid_time'):payload['request'][name]=pd.Timestamp(payload['request'][name])
            data=forecast_polygon_values(archive,payload['request'],spec=spec(payload['spec']),group_cache=payload['group_cache'],
                member_start=payload['member_start'],member_stop=payload['member_stop'])
        elif request['kind']=='atlas':
            row=pd.Series(payload['row'])
            for name in ('event_start','event_end','event_peak','target_valid_time'):row[name]=pd.Timestamp(row[name])
            panels=event_panels(archive,Path(payload['cache_dir']),row,payload['member'],
                [spec(s) for s in payload['specs']],payload['domains'])
            destination=export_event(panels,row,Path(payload['output_dir']),features=payload.get('features',True))
            data={'output_dir':str(destination)}
        else:raise ValueError('Unknown forecast worker job.')
        reply.write_text(json.dumps({'data':data},allow_nan=False))
    except Exception as error:
        reply.write_text(json.dumps({'error':f'{type(error).__name__}: {error}'}))


if __name__=='__main__':
    main()
    # main() closes all output/checkpoint files before returning. Zarr/gcsfs may
    # register aiohttp finalizers on a different event loop; interpreter teardown
    # can print tracebacks or wait on them after a successful result was written.
    # This single-job process owns no persistent state: flush logs and let the OS
    # release its sockets/memory. Genuine job errors remain in the checked reply.
    sys.stdout.flush();sys.stderr.flush()
    os._exit(0)
