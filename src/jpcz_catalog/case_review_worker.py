"""Entrypoint for a single phase; copied beside exact Notebook 35 helpers by the updater."""
import json
import os
from pathlib import Path
import resource
import runpy
import sys


def main():
    request = json.load(sys.stdin)
    reply_path = Path(request.pop('reply_path'))
    token = request.pop('earthdata_token')
    try:
        # A bad individual allocation must fail in the child, not exhaust the Colab kernel.
        if sys.platform.startswith('linux'):
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = 6 * 1024**3
            if hard != resource.RLIM_INFINITY:
                limit = min(limit, hard)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        settings = request['settings']
        for key in ('DRIVE_ROOT', 'OUTPUT_DIR', 'CACHE_DIR', 'REVIEW_PATH'):
            if key in settings:
                settings[key] = Path(settings[key])
        namespace = runpy.run_path(str(Path(__file__).with_name('helpers.py')), init_globals=settings)
        # Keep the active screen settings exactly equal to the parent notebook's settings.
        namespace.update(settings)
        try:
            os.environ['EARTHDATA_TOKEN'] = token
            namespace['auth'] = namespace['earthaccess'].login(strategy='environment', persist=False)
            namespace['require_earthdata_auth'](namespace['auth'])
        except Exception:
            reply_path.write_text(json.dumps(dict(authentication_required=True,
                error='NASA login could not be reused. Rerun the NASA authentication cell, then resume batch.')))
            return
        finally:
            os.environ.pop('EARTHDATA_TOKEN', None)
            del token
        # run_path's returned dict is distinct from its functions' global namespace.
        function_globals = namespace['get_case'].__globals__
        function_globals.update(settings, auth=namespace['auth'])
        pd = namespace['pd']
        row = pd.Series(request['row'])
        for name in namespace['DATE_COLUMNS']:
            row[name] = pd.Timestamp(row[name])
        files = {name: Path(path) for name, path in request['files'].items()}
        metadata, status = namespace['batch_export_phase'](row, files,
            load_case=namespace['get_case'],
            render_cases=lambda cases: namespace['plot_cases'](cases, show=False), **request['options'])
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata.update(worker_pid=os.getpid(), worker_peak_rss_mib=round(
            peak_rss / (1024**2 if sys.platform == 'darwin' else 1024), 1))
        reply_path.write_text(json.dumps(namespace['batch_json_value'](dict(metadata=metadata, status=status)),
                                         allow_nan=False))
    except Exception as error:
        # Do not print exception tracebacks containing the authenticated request's frame locals.
        message = 'Map worker ran out of memory.' if isinstance(error, MemoryError) else f'{type(error).__name__}: {error}'
        reply_path.write_text(json.dumps(dict(error=message + ' Saved phases remain reusable; batch is paused.')))


if __name__ == '__main__':
    main()
