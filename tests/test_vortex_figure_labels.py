"""Event identity, current Notebook 35 screen results, and figure-label regressions."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jpcz_catalog import weathernext_atlas as wn
from test_weathernext_atlas import row as row_fixture, FakeArchive, make_panels


@pytest.fixture
def row():
    return row_fixture.__wrapped__()


@pytest.mark.parametrize("status,detected,expected", [
    ("complete",True,"Vortex"),("complete",False,"Non-vortex"),
    ("complete","True","Vortex"),("complete","False","Non-vortex"),
    ("complete","0","Non-vortex"),("complete",np.nan,"Vortex status unknown"),
    ("complete",None,"Vortex status unknown"),("not_scanned",False,"Vortex not checked"),
    ("failed",False,"Vortex status unavailable"),("ambiguous",False,"Vortex status ambiguous"),
])
def test_only_completed_known_negative_results_are_non_vortex(status,detected,expected):
    record = {"vortex_scan_status": status, "vortex_detected": detected,
              "case_label": "Unclassified candidate"}
    assert wn.vortex_classification(record) == expected
    expected_title = "Vortex-associated" if expected == "Vortex" else "Non-vortex-associated"
    assert wn.vortex_plot_label(record) == expected_title


def save_result(review_dir,row,signature,detected,**changes):
    folder=review_dir/'vortex_screen'/signature
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/('event_'+hashlib.sha256(row.review_key.encode()).hexdigest()[:20]+'.json')
    path.write_text(json.dumps(dict(status='complete',review_key=row.review_key,signature=signature,
                                    vortex_detected=detected,**changes)))
    return path


def test_per_event_cache_overrides_stale_inventory_without_scanning(row,tmp_path):
    signature='a'*16
    path=save_result(tmp_path,row,signature,True)
    frame=pd.DataFrame([{**row.to_dict(),'vortex_scan_status':'not_scanned','vortex_detected':False}])
    result=wn.refresh_vortex_labels(frame,tmp_path).iloc[0]
    assert result.vortex_classification == 'Vortex'
    assert result.vortex_label_source == str(path)
    path.write_text(path.read_text().replace('"vortex_detected": true','"vortex_detected": false'))
    result=wn.refresh_vortex_labels(frame,tmp_path).iloc[0]
    assert result.vortex_classification == 'Non-vortex'
    path.unlink()
    # Having no record is not a negative result, even if an inventory once said complete.
    frame['vortex_scan_status']='complete'
    result=wn.refresh_vortex_labels(frame,tmp_path).iloc[0]
    assert result.vortex_classification == 'Vortex not checked'


def test_active_screen_selects_matching_settings_and_rejects_wrong_event(row,tmp_path):
    save_result(tmp_path,row,'a'*16,True)
    path=save_result(tmp_path,row,'b'*16,False)
    frame=pd.DataFrame([row])
    assert wn.refresh_vortex_labels(frame,tmp_path).iloc[0].vortex_classification == 'Vortex status ambiguous'
    (tmp_path/'vortex_screen/active_screen.json').write_text(json.dumps({'signature':'b'*16}))
    assert wn.refresh_vortex_labels(frame,tmp_path).iloc[0].vortex_classification == 'Non-vortex'
    saved=json.loads(path.read_text());saved['review_key']='different event'
    path.write_text(json.dumps(saved))
    assert wn.refresh_vortex_labels(frame,tmp_path).iloc[0].vortex_classification == 'Vortex status unavailable'


def test_forecast_titles_and_exports_follow_reference_class_without_new_forecast_reads(row,tmp_path):
    import matplotlib.pyplot as plt
    positive=row.copy();positive['vortex_scan_status']='complete';positive['vortex_detected']=True
    positive['case_label']='ordinary candidate'
    panels=make_panels(positive,tmp_path/'cache')
    fig=wn.render_event(panels,positive,features=False)
    assert 'Observed event: Vortex-associated' in fig._suptitle.get_text()
    assert all(ax.get_title().startswith('Observed event: Vortex-associated\n') for ax in fig.axes[:6])
    assert all(p['metadata']['vortex_classification']=='Vortex' for p in panels)
    first=wn.export_event(panels,positive,tmp_path/'figures',fig,features=False)
    negative=positive.copy();negative['vortex_detected']=False
    changed=make_panels(negative,tmp_path/'cache')
    assert all(p['metadata']['cache_hit'] for p in changed)
    assert all(p['metadata']['vortex_classification']=='Non-vortex' for p in changed)
    fig2=wn.render_event(changed,negative,features=False)
    assert 'Observed event: Non-vortex-associated' in fig2._suptitle.get_text()
    assert all(ax.get_title().startswith('Observed event: Non-vortex-associated\n') for ax in fig2.axes[:6])
    second=wn.export_event(changed,negative,tmp_path/'figures',fig2,features=False)
    assert first != second  # a completed export with an old class cannot hide the new label
    metadata = pd.read_csv(second/'forecast_metadata.csv')
    assert metadata.vortex_classification.eq('Non-vortex').all()
    assert metadata.vortex_plot_label.eq('Non-vortex-associated').all()
    plt.close(fig);plt.close(fig2)


def test_browser_refresh_updates_dropdown_and_blocks_export_until_new_labels_applied(row,tmp_path):
    folder=tmp_path/'reviews'
    save_result(folder,row,'a'*16,True)
    browser=wn.WeatherNextBrowser(pd.DataFrame([row]),FakeArchive(),tmp_path/'out',folder)
    browser._refresh_labels()
    assert browser.event.options[0][0].endswith(' | Vortex-associated')
    assert browser.export.disabled
    assert browser.selected_row().vortex_classification == 'Vortex'


def notebook35_functions():
    notebook=json.loads((ROOT/'notebooks/35_imerg_manual_case_review.ipynb').read_text())
    source=next(''.join(c['source']) for c in notebook['cells'] if 'def get_case(' in ''.join(c.get('source',[])))
    tree=ast.parse(source)
    names={'vortex_classification','vortex_plot_label','get_case','event_option_label','plot_cases'}
    functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    namespace={'np':np,'pd':pd,'re':__import__('re'),'DATE_COLUMNS':list(wn.DATE_COLUMNS),'VORTEX_LABEL_SCHEME':wn.VORTEX_LABEL_SCHEME}
    exec(compile(functions,'notebook35_label_functions','exec'),namespace)
    return source,namespace


@pytest.mark.parametrize('status,detected,expected',[
    ('complete', True, 'Vortex-associated'),
    ('complete', False, 'Non-vortex-associated'),
    ('not_scanned', None, 'Non-vortex-associated'),
])
def test_notebook35_titles_use_scan_result_even_when_manual_label_disagrees(row,status,detected,expected):
    import cartopy.crs as ccrs
    from matplotlib import colors, pyplot as plt

    source,ns=notebook35_functions()
    row['vortex_scan_status'] = status
    row['vortex_detected'] = detected
    coords = {'latitude': [36., 38., 40.], 'longitude': [130., 132., 134.]}
    wind = xr.DataArray(np.ones((3, 3)), coords=coords, dims=tuple(coords))
    dynamics = xr.Dataset(dict(divergence=-wind, vorticity=wind,
                               mslp=wind * np.arange(1008, 1020, 4), u=wind, v=wind))
    def no_imerg(*args):raise OSError('synthetic missing precipitation')
    ns.update(accumulation_bounds=lambda *args:(row.event_start,row.event_start),
              load_era5=lambda *args:dynamics,load_imerg=no_imerg,
              window_bounds=lambda hour,hours:(hour,hour+pd.Timedelta(hours=1)),
              vortex_case_metadata=lambda *args:dict(vortex_scan_status=status,vortex_detected=detected),
              load_satellite_pair=lambda *args,**kwargs:(None,{},None,{}),POLYGONS={},
              plt=plt,ccrs=ccrs,colors=colors,textwrap=textwrap,
              decorate=lambda ax: ax.set_extent((125,145,33,47),ccrs.PlateCarree()),
              annotate_mslp_lows=lambda *args:None,PRECIP_LEVELS=[0,1,2],
              DIV_LEVELS=[-2,0,2],VORT_LEVELS=[-2,0,2],ACCUM_LEVELS=[0,1,2])
    case=ns['get_case'](row,offset=12,label='Case A: ordinary candidate')
    assert case['metadata']['label']==expected
    assert case['metadata']['label_scheme']==wn.VORTEX_LABEL_SCHEME
    assert case['metadata']['vortex_scan_status']==status
    assert case['metadata']['comparison_slot']=='A'
    assert ns['event_option_label'](row).endswith(' | ' + expected)
    fig = ns['plot_cases']([case])
    assert all(expected in ax.get_title() for ax in fig.axes[:6])
    assert 'offset +12 h from peak' in fig.axes[1].get_title()
    assert '925-hPa divergence' in fig.axes[1].get_title()
    plt.close(fig)
    assert 'Unclassified candidate' not in source
    assert "'active_screen.json'" in source
    shared = {'vortex_classification', 'vortex_plot_label'}
    def label_functions(text):
        return {n.name: ast.dump(n) for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name in shared}
    assert label_functions(source) == label_functions((ROOT/'src/jpcz_catalog/weathernext_atlas.py').read_text())


def test_saved_reference_images_are_flagged_until_regenerated(row,tmp_path):
    positive=row.copy();positive['vortex_scan_status']='complete';positive['vortex_detected']=True
    metadata={c:row[c] for c in wn.DATE_COLUMNS}
    pd.DataFrame([{**metadata,'label':'Unclassified candidate'}]).to_csv(tmp_path/'old_metadata.csv',index=False)
    pd.DataFrame([{**metadata,'label':'Vortex-associated','label_scheme':wn.VORTEX_LABEL_SCHEME}]).to_csv(tmp_path/'new_metadata.csv',index=False)
    refs=wn.find_notebook35_references(tmp_path,positive)
    assert refs.reference_label_current.tolist()==[True,False]
    assert refs.current_vortex_classification.eq('Vortex-associated').all()
