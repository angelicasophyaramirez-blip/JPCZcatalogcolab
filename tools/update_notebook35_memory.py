"""Make Notebook 35's ERA5 map and vortex reads serial and bounded in memory."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'notebooks/35_imerg_manual_case_review.ipynb'
notebook = json.loads(TARGET.read_text())
module = (ROOT / 'src/jpcz_catalog/case_review_memory.py').read_text()
marker = '# --- Bounded ERA5 source reads ---'
vortex_reader = '''def load_vortex_hourly(start, end):
    """Read one hour at a time and persist each completed vortex snapshot to Drive."""
    expected = pd.date_range(start, end, freq='h')
    root = vortex_cache_dir()
    paths = {hour: root / f'hour_{hour:%Y%m%dT%H%M}.json' for hour in expected}
    missing = [hour for hour, path in paths.items() if not path.exists()]
    if missing:
        global era5
        if 'era5' not in globals():
            era5 = xr.open_zarr(ERA5_STORE, chunks=None, storage_options={'token': 'anon'})
        for index, hour in enumerate(missing, start=1):
            print(f'Vortex winds: {index}/{len(missing)} missing hours | {hour} UTC', flush=True)
            field = read_review_era5_hour(era5, hour,
                ('u_component_of_wind', 'v_component_of_wind', 'mean_sea_level_pressure', 'surface_pressure'), DOMAIN)
            dx, dy = mpcalc.lat_lon_grid_deltas(field.longitude.values, field.latitude.values)
            u, v = field.u_component_of_wind.values, field.v_component_of_wind.values
            zeta = mpcalc.vorticity(u*units('m/s'), v*units('m/s'), dx=dx, dy=dy).magnitude*1e5
            pressure = field.mean_sea_level_pressure.values/100
            surface = field.surface_pressure.values
            valid = np.isfinite(surface) & (surface > 92500)
            centers, rejected = [], []
            for center in detect_mslp_lows(pressure, field.longitude.values, field.latitude.values, max_centers=100):
                evidence = cyclone_circulation_evidence(center, field.longitude.values, field.latitude.values,
                                                       u, v, zeta, valid_mask=valid)
                if evidence['passes_circulation_screen']:
                    centers.append(dict(center, **evidence))
                else:
                    rejected.append(dict(center, **evidence))
            atomic_json(dict(hour_utc=hour.isoformat(), centers=centers, rejected=rejected), paths[hour])
            del field, u, v, zeta, pressure, surface, valid, centers, rejected
            release_review_memory()
    return [json.loads(paths[hour].read_text()) for hour in expected]
'''
for cell in notebook['cells']:
    source = ''.join(cell.get('source', []))
    if cell['cell_type'] == 'code' and "'pip', 'install'" in source:
        source = source.replace("'zarr'", "'zarr>=3.1,<4'")
        cell['source'] = source.splitlines(keepends=True)
    if cell['cell_type'] == 'code' and 'def get_case(' in source:
        source = source.split(marker, 1)[0].rstrip() + '\n'
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_era5')
        original = ast.get_source_segment(source, node)
        start = original.index('    start, end = window_bounds(hour, hours)')
        end = original.index('    dx, dy = mpcalc.lat_lon_grid_deltas(')
        bounded = '''    start, end = window_bounds(hour, hours)
    expected = pd.date_range(start, end, freq='h', inclusive='left')
    print(f'Loading {len(expected)} ERA5 hour(s), one variable at a time...', flush=True)
    snap = read_review_era5_window(era5, expected,
        ('u_component_of_wind', 'v_component_of_wind', 'mean_sea_level_pressure'), DOMAIN)
'''
        replacement = original[:start] + bounded + original[end:]
        source = source.replace(original, replacement, 1)
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_vortex_hourly')
        source = source.replace(ast.get_source_segment(source, node), vortex_reader.strip(), 1)
        if 'load_case=get_case, release_memory=' not in source:
            source = source.replace('frame, OUTPUT_DIR, load_case=get_case,',
                                    'frame, OUTPUT_DIR, load_case=get_case, release_memory=release_review_memory,')
        source += '\n\n' + marker + '\n' + module
        ast.parse(source)
        cell['source'] = source.splitlines(keepends=True)
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
print(TARGET)
