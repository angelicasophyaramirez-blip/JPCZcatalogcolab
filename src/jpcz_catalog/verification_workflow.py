"""Configurable verification runs and Drive-only report collection."""
from dataclasses import replace
import json
from pathlib import Path
import pandas as pd

from .weathernext_atlas import FIELD_PRESETS, POLYGONS, LEAD_HOURS
from .ensemble_verification import (
    run_ensemble_verification, atomic_json_file, export_verification, plot_rank_histograms,
    rank_histogram_counts, export_reliability, verification_subset,
    checkpoint_identity,
)

SUPPORTED_VERIFICATION_FIELDS=('divergence','u','v','winds','height','geopotential')


def verification_field_configs(configs):
    if not configs:raise ValueError('Select at least one verification field.')
    records=[]
    for config in configs:
        key=config['field']
        if key not in SUPPORTED_VERIFICATION_FIELDS:
            raise ValueError(f'Choose a supported verification field: {SUPPORTED_VERIFICATION_FIELDS}.')
        spec=replace(FIELD_PRESETS[key],level=int(config.get('level',FIELD_PRESETS[key].level)))
        if key=='winds':spec=replace(spec,label='wind speed (derived from u/v)')
        if spec.level not in (50,100,150,200,250,300,400,500,600,700,850,925,1000):
            raise ValueError('Select an available pressure level.')
        identifier=f'{spec.key}_{spec.level}hpa'
        if any(item['id']==identifier for item in records):raise ValueError('Duplicate verification field and level.')
        threshold=config.get('threshold')
        if threshold is not None:
            import math
            if not math.isfinite(float(threshold)):raise ValueError('A reliability threshold must be finite.')
        direction=config.get('direction','less')
        if direction not in ('less','greater'):raise ValueError('Threshold direction must be less or greater.')
        records.append(dict(id=identifier,spec=spec,config={**config,'level':spec.level,'threshold':threshold,'direction':direction}))
    return records


def load_saved_verification(destination, *, catalog=None, spec=None):
    """Read exported CSVs only. This function has no archive/reader callback."""
    destination=Path(destination)
    names=('case_ranks','member_values','era5_reference','batch_status')
    paths=[destination/f'{name}.csv' for name in names]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f'Saved verification tables are incomplete in {destination}. Retrieve missing inputs first.')
    settings=json.loads((destination/'calculation_settings.json').read_text())
    if spec is not None:
        if settings.get('field','divergence')!=spec.key or settings.get('pressure_hpa',925)!=spec.level:
            raise ValueError('Saved tables use a different verification field/level.')
    tables={}
    for name,path in zip(names,paths):
        try:frame=pd.read_csv(path,dtype={'member':'string','review_key':'string'},float_precision='round_trip')
        except pd.errors.EmptyDataError:frame=pd.DataFrame()
        if catalog is not None and 'review_key' in frame:
            frame=frame.loc[frame.review_key.isin(catalog.review_key)].copy()
            if 'valid_time_utc' in frame and 'target_valid_time' in catalog:
                targets=pd.to_datetime(catalog.set_index('review_key').target_valid_time,utc=True)
                frame=frame.loc[pd.to_datetime(frame.valid_time_utc,utc=True,errors='coerce').eq(frame.review_key.map(targets))].copy()
            labels=[c for c in ('vortex_label','vortex_scan_status','vortex_classification') if c in frame]
            for key in labels:
                column='vortex_plot_label' if key=='vortex_label' else key
                if column in catalog:frame[key]=frame.review_key.map(catalog.set_index('review_key')[column])
        tables[name]=frame
    if catalog is not None:
        present=set(zip(tables['batch_status'].get('review_key',[]),tables['batch_status'].get('lead_hours',[])))
        missing=[]
        for _,row in catalog.iterrows():
            for lead in LEAD_HOURS:
                if (row.review_key,lead) not in present:
                    missing.append(dict(event_number=int(row.event_number),review_key=row.review_key,
                        lead_hours=lead,lead_days=lead//24,era5_status='not_downloaded',wn2_status='not_downloaded',
                        era5_error='Saved-only mode: target/lead not yet retrieved.',wn2_error='Saved-only mode: target/lead not yet retrieved.'))
        if missing:tables['batch_status']=pd.concat([tables['batch_status'],pd.DataFrame(missing)],ignore_index=True)
    return tables,settings


def run_verification_fields(catalog, archive, output_dir, result_directory, configs, *,
                            saved_only=False,job_runner=None,allow_era5t=False,progress=print):
    """Keep each variable/level's physical units, checkpoints, and plots independent."""
    results=[]
    for item in verification_field_configs(configs):
        spec=item['spec']
        destination=Path(result_directory) if item['id']=='divergence_925hpa' else Path(result_directory)/'fields'/item['id']
        progress(f"Field {spec.title} | {'saved tables only' if saved_only else 'resume missing inputs'}")
        if saved_only:
            tables,settings=load_saved_verification(destination,catalog=catalog,spec=spec)
            expected=json.loads(json.dumps(checkpoint_identity(archive,spec),sort_keys=True))
            if any(settings.get(key)!=value for key,value in expected.items()):
                raise ValueError('Saved verification settings differ from this source/field configuration. '
                                 'Use the original settings or retrieve the new configuration explicitly.')
        else:
            tables,settings=run_ensemble_verification(catalog,archive,output_dir,spec=spec,
                job_runner=job_runner,allow_era5t=allow_era5t,progress=progress)
            destination.mkdir(parents=True,exist_ok=True)
            for name,frame in tables.items():
                temporary=destination/f'.{name}.tmp.csv';frame.to_csv(temporary,index=False);temporary.replace(destination/f'{name}.csv')
            atomic_json_file(destination/'calculation_settings.json',settings)
        results.append({**item,'tables':tables,'settings':settings,'directory':destination})
    return results


def consolidate_saved_verification(sources, destination):
    """Combine existing ordered-value rows into ONE Excel sheet, without any field calculation."""
    frames=[]
    seen=set()
    for source in sources:
        path=Path(source)/'ordered_values.csv'
        if str(path.resolve()) in seen:continue
        seen.add(str(path.resolve()))
        if not path.is_file():raise FileNotFoundError(f'Run the report export first: {path}')
        try:frame=pd.read_csv(path,dtype={'member':'string','review_key':'string'},float_precision='round_trip')
        except pd.errors.EmptyDataError:frame=pd.DataFrame()
        if 'value' not in frame and 'divergence_1e5_s-1' in frame:frame['value']=frame['divergence_1e5_s-1']
        if 'field' not in frame:frame['field']='divergence'
        if 'units' not in frame:frame['units']='10⁻⁵ s⁻¹'
        frame['saved_table']=str(path)
        # Keep native values even when a missing reference/lead prevents a rank from being assigned.
        supplements=[]
        for table,source_name in (('member_values','WN2'),('era5_reference','ERA5')):
            raw_path=Path(source)/f'{table}.csv'
            if not raw_path.is_file():continue
            try:raw=pd.read_csv(raw_path,dtype={'member':'string','review_key':'string'},float_precision='round_trip')
            except pd.errors.EmptyDataError:continue
            if raw.empty:continue
            raw['source']=source_name
            raw['saved_table']=str(raw_path)
            if 'value' not in raw:raw['value']=raw['divergence_1e5_s-1']
            if 'field' not in raw:raw['field']='divergence'
            if 'units' not in raw:raw['units']='10⁻⁵ s⁻¹'
            if source_name=='ERA5':
                raw['member']=''
                copies=[]
                for lead in LEAD_HOURS:
                    copy=raw.copy();copy['lead_hours']=lead;copy['lead_days']=lead//24
                    copy['init_time_utc']=(pd.to_datetime(copy.valid_time_utc,utc=True)-pd.Timedelta(hours=lead)).astype(str)
                    copies.append(copy)
                raw=pd.concat(copies,ignore_index=True)
            supplements.append(raw)
        if supplements:
            frame=pd.concat([frame,*supplements],ignore_index=True,sort=False)
            frame['member']=frame['member'].fillna('').astype(str)
            keys=['review_key','field','pressure_hpa','polygon','lead_hours','source','member']
            for col in ('pressure_hpa','lead_hours'):
                frame[col]=pd.to_numeric(frame[col],errors='raise').astype('Int64')
            frame=frame.drop_duplicates(keys,keep='first')
        frames.append(frame)
    if not frames:raise ValueError('Select saved result folders to combine.')
    values=pd.concat(frames,ignore_index=True,sort=False)
    preferred=['event_number','review_key','field','pressure_hpa','units','vortex_classification','polygon','lead_days',
               'valid_time_utc','init_time_utc','source','member','value','position','bin_number','in_common_sample','saved_table']
    values=values[[c for c in preferred if c in values]+[c for c in values if c not in preferred]]
    values=values.sort_values([c for c in ['event_number','field','pressure_hpa','polygon','lead_days','position'] if c in values])
    if len(values)>1048575:raise ValueError('The combined values exceed one Excel sheet; select fewer saved folders.')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    csv=destination/'all_verification_values.csv';tmp=csv.with_suffix('.tmp.csv')
    values.to_csv(tmp,index=False);tmp.replace(csv)
    workbook=destination/'all_verification_values.xlsx';tmp=workbook.with_suffix('.tmp.xlsx')
    with pd.ExcelWriter(tmp,engine='xlsxwriter',engine_kwargs={'options':{'strings_to_formulas':False,'strings_to_urls':False}}) as writer:
        values.to_excel(writer,sheet_name='All values',index=False)
        sheet=writer.sheets['All values'];sheet.freeze_panes(1,0);sheet.hide_gridlines(2)
        sheet.autofilter(0,0,len(values),max(0,len(values.columns)-1));sheet.set_column(0,len(values.columns)-1,18)
        if 'source' in values:
            from xlsxwriter.utility import xl_col_to_name
            col=xl_col_to_name(values.columns.get_loc('source'))
            sheet.conditional_format(1,0,len(values),len(values.columns)-1,{'type':'formula',
                'criteria':f'=${col}2="ERA5"','format':writer.book.add_format({'bg_color':'#DDF0DE','bold':True})})
    tmp.replace(workbook)
    return values,workbook


def saved_verification_sources(result_directory):
    """Find saved field tables within one catalog/pilot run; do not mix other study folders."""
    root=Path(result_directory)
    sources=([root] if (root/'ordered_values.csv').is_file() else [])
    sources.extend(path.parent for path in sorted((root/'fields').glob('*/ordered_values.csv')))
    return sources


def export_field_reports(runs, selection_report=None):
    """Build report artifacts from already retrieved tables; never fetch weather fields."""
    import matplotlib.pyplot as plt
    for run in runs:
        tables,settings,destination=run['tables'],run['settings'],Path(run['directory'])
        if selection_report is not None:tables['event_selection']=selection_report
        export_verification(tables,settings,destination)
        ranks=tables['case_ranks']
        for grouped,stem in ((False,'rank_histograms_65ranks'),(True,'rank_histograms')):
            if not ranks.empty and ranks.in_common_sample.any():
                fig=plot_rank_histograms(ranks,grouped=grouped)
                for extension in ('png','pdf'):fig.savefig(destination/f'{stem}.{extension}',dpi=180)
                plt.close(fig)
            else:
                for extension in ('png','pdf'):(destination/f'{stem}.{extension}').unlink(missing_ok=True)


def export_field_reliability(runs, *, n_bins=10):
    """Rebuild calibration figures from saved scalar tables with an explicit per-field criterion."""
    import matplotlib.pyplot as plt
    folders=[]
    for run in runs:
        threshold=run['config'].get('threshold')
        if threshold is None:
            print(f"{run.get('id','Field')}: no reliability threshold selected; values/ranks remain saved.")
            continue
        for group in ('All','Vortex','Non-vortex','Unknown'):
            folder=Path(run['directory'])/'event_conditioned'/group.lower().replace('-','_')
            p,_=export_reliability(run['tables'],folder,{region:float(threshold) for region in POLYGONS},
                group=group,n_bins=n_bins,direction=run['config'].get('direction','less'))
            ranks=verification_subset(run['tables'],group)
            n=0 if ranks.empty else ranks.loc[ranks.in_common_sample,'review_key'].nunique()
            print(f"{run.get('id','Field')} / {group}: {n} complete events; {folder}")
            for grouped,stem in ((False,'rank_histograms_65ranks'),(True,'rank_histograms')):
                if n:
                    rank_histogram_counts(ranks,grouped=grouped).to_csv(folder/f'{stem}_counts.csv',index=False)
                    fig=plot_rank_histograms(ranks,grouped=grouped,title_suffix=group)
                    for extension in ('png','pdf'):fig.savefig(folder/f'{stem}.{extension}',dpi=180)
                    plt.close(fig)
                else:
                    for suffix in ('.png','.pdf','_counts.csv'):(folder/(stem+suffix)).unlink(missing_ok=True)
            folders.append(folder)
    return folders
