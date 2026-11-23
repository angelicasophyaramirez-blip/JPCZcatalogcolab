"""CORP/PAV calibration diagnostics from saved binary probabilities, without archive access.

Dimitriadis, Gneiting & Jordan (2021), doi:10.1073/pnas.2016191118.
The fit is descriptive and in-sample. It does not replace the original forecasts.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ensemble_verification import POLYGONS, LEAD_HOURS, atomic_json_file, reliability_bin_counts

CORP_REFERENCE = 'https://www.pnas.org/doi/10.1073/pnas.2016191118'
CORP_POINT_COLUMNS = ['forecast_probability','case_count','observed_count','empirical_frequency',
    'corp_probability','pool_id','pool_count','pool_observed_count','pool_min_probability','pool_max_probability']


def pav_binary_probabilities(probabilities, outcomes):
    """Weighted PAV on tied forecast values; return unique-x points and fitted case values.

    Tied probabilities are aggregated before pooling, so ordering within ties cannot
    change the result. Integer cross-products determine pool merges without tolerances.
    Adjacent equal levels are merged into maximal constant pools for reproducible counts.
    """
    p=np.asarray(probabilities,dtype=float);y=np.asarray(outcomes,dtype=float)
    if p.ndim!=1 or y.shape!=p.shape or not len(p):raise ValueError('Supply paired, nonempty probability/outcome vectors.')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isin(y,[0,1]).all():
        raise ValueError('PAV needs finite probabilities in [0,1] and binary outcomes.')
    unique,inverse,counts=np.unique(p,return_inverse=True,return_counts=True)
    hits=np.bincount(inverse,weights=y,minlength=len(unique)).astype(int)
    pools=[]
    for index,(count,hit) in enumerate(zip(counts,hits)):
        pools.append([index,index+1,int(count),int(hit)])
        while len(pools)>1 and pools[-2][3]*pools[-1][2]>=pools[-1][3]*pools[-2][2]:
            right=pools.pop();left=pools.pop()
            pools.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
    fitted=np.empty(len(unique));rows=[]
    for number,(start,stop,count,hit) in enumerate(pools,1):
        fitted[start:stop]=hit/count
        for index in range(start,stop):
            rows.append(dict(forecast_probability=float(unique[index]),case_count=int(counts[index]),
                observed_count=int(hits[index]),empirical_frequency=float(hits[index]/counts[index]),
                corp_probability=hit/count,pool_id=number,pool_count=count,pool_observed_count=hit,
                pool_min_probability=float(unique[start]),pool_max_probability=float(unique[stop-1])))
    return pd.DataFrame(rows,columns=CORP_POINT_COLUMNS),fitted[inverse]


def corp_calibration_tables(probabilities, *, consistency_bands=False, assume_exchangeable=False,
                            n_resamples=2000, seed=37):
    """Fit each polygon/lead separately and compute the exact CORP Brier decomposition.

    Optional bands simulate Y*~Bernoulli(original p) at fixed forecast values, refit
    PAV, and take 5/95 percentiles. These are pointwise 90% CONSISTENCY bands under
    calibration and exchangeable/independent cases, not confidence or simultaneous bands.
    No uncertainty band is made by default for the small clustered winter event sample.
    """
    metadata=['polygon','lead_days','field','field_title','pressure_hpa','units','threshold','direction','sample']
    summary_columns=metadata+['cases','distinct_probabilities','pav_pools','observed_crossings','base_rate',
        'brier_score','pav_brier_score','miscalibration_mcb','discrimination_dsc','uncertainty_unc',
        'decomposition_residual','valid_time_start','valid_time_end','ensemble_members','plot_type',
        'band_type','n_resamples','seed']
    if consistency_bands and not assume_exchangeable:
        raise ValueError('Consistency bands require an explicit independent/exchangeable-case assumption.')
    if consistency_bands and (not isinstance(n_resamples,int) or n_resamples<100):
        raise ValueError('Use at least 100 consistency resamples.')
    if probabilities.empty:
        return (pd.DataFrame(columns=metadata+CORP_POINT_COLUMNS+['consistency_lower','consistency_upper']),
                probabilities.assign(corp_probability=pd.Series(dtype=float)),pd.DataFrame(columns=summary_columns))
    keys=['review_key','polygon','lead_hours']
    if probabilities.duplicated(keys).any():raise ValueError('Duplicate cases would overweight CORP.')
    rows=[];cases=[];summaries=[]
    for (region,lead),frame in probabilities.groupby(['polygon','lead_days'],sort=False):
        frame=frame.copy()
        for name in ('field','pressure_hpa','units','threshold','direction','sample','member_count'):
            if name in frame and frame[name].nunique(dropna=False)!=1:
                raise ValueError(f'Plot separate CORP diagnostics for different {name} settings.')
        first=frame.iloc[0]
        info={name:first[name] for name in metadata}
        p=frame.forecast_probability.to_numpy(dtype=float);y=frame.observed.to_numpy(dtype=float)
        points,q=pav_binary_probabilities(p,y)
        raw=float(np.mean((p-y)**2));cal=float(np.mean((q-y)**2));base=float(y.mean())
        unc=base*(1-base);mcb=raw-cal;dsc=unc-cal
        if min(mcb,dsc)<-1e-12:raise ArithmeticError('CORP score decomposition is not nonnegative.')
        mcb=max(0.,mcb);dsc=max(0.,dsc)
        points['consistency_lower']=np.nan;points['consistency_upper']=np.nan
        if consistency_bands:
            identity=json.dumps(info,sort_keys=True,default=lambda value:value.item() if isinstance(value,np.generic) else str(value))
            panel_seed=(int(seed)+int(hashlib.sha256(identity.encode()).hexdigest()[:8],16))%(2**32)
            rng=np.random.default_rng(panel_seed)
            fits=np.empty((n_resamples,len(points)))
            for index in range(n_resamples):
                simulated,_=pav_binary_probabilities(p,rng.binomial(1,p))
                fits[index]=simulated.corp_probability
            points['consistency_lower'],points['consistency_upper']=np.quantile(fits,[.05,.95],axis=0)
        rows.append(points.assign(**info))
        frame['corp_probability']=q
        frame['original_brier_score']=(p-y)**2;frame['pav_in_sample_brier_score']=(q-y)**2
        cases.append(frame)
        times=pd.to_datetime(frame.valid_time_utc,utc=True)
        differences=np.diff(points.forecast_probability)
        plot_type='discrete' if len(differences)==0 or float(differences.min())>=.01-1e-12 else 'continuous'
        summaries.append(dict(**info,cases=len(frame),distinct_probabilities=len(points),pav_pools=points.pool_id.nunique(),
            observed_crossings=int(y.sum()),base_rate=base,brier_score=raw,pav_brier_score=cal,
            miscalibration_mcb=mcb,discrimination_dsc=dsc,uncertainty_unc=unc,
            decomposition_residual=raw-(mcb-dsc+unc),valid_time_start=str(times.min()),valid_time_end=str(times.max()),
            ensemble_members=int(first.member_count),plot_type=plot_type,
            band_type='90% pointwise consistency; independent/exchangeable cases assumed' if consistency_bands else 'none',
            n_resamples=n_resamples if consistency_bands else 0,seed=int(seed)))
    return pd.concat(rows,ignore_index=True),pd.concat(cases,ignore_index=True),pd.DataFrame(summaries,columns=summary_columns)


def plot_corp_reliability(points, summary, *, region, lead_hours, title_suffix='All'):
    """A publication-style discrete CORP graph, with a separate statistics panel underneath."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    selected=points.loc[points.polygon.eq(region)&points.lead_days.eq(lead_hours//24)].sort_values('forecast_probability')
    stats=summary.loc[summary.polygon.eq(region)&summary.lead_days.eq(lead_hours//24)]
    if selected.empty or len(stats)!=1:raise ValueError('No unique CORP panel for the selected polygon/lead.')
    stat=stats.iloc[0]
    fig=plt.figure(figsize=(7.2,9.3),facecolor='white')
    ax=fig.add_axes([.14,.385,.78,.47]);table_ax=fig.add_axes([.09,.075,.85,.235])
    ax.plot([0,100],[0,100],'--',color='#323a42',linewidth=1.1,label='Perfect reliability',zorder=1)
    if selected.consistency_lower.notna().any():
        ax.fill_between(100*selected.forecast_probability,100*selected.consistency_lower,100*selected.consistency_upper,
            color='#bec9d1',alpha=.5,linewidth=0,label='90% consistency band',zorder=2)
    ax.plot(100*selected.forecast_probability,100*selected.corp_probability,'o-',color='#087f8c',
            linewidth=2,markersize=4.5,label='CORP / PAV estimate',zorder=3)
    ax.set(xlim=(-2,102),ylim=(-2,102),xlabel='Original WN2 forecast probability',
           ylabel='Estimated ERA5 crossing frequency',xticks=np.arange(0,101,20),yticks=np.arange(0,101,20))
    ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    ax.set_aspect('equal',adjustable='box');ax.grid(color='#e0e6eb',linewidth=.6);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=11)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='center',bbox_to_anchor=(.53,.885),ncol=2,
               fontsize=10,frameon=False)
    operator='<' if stat.direction=='less' else '>'
    criterion=f'{stat.threshold:g} × {stat.units}' if stat.field=='divergence' else f'{stat.threshold:g} {stat.units}'
    fig.suptitle(f'CORP reliability · {title_suffix}',fontsize=16,y=.985)
    fig.text(.5,.947,f'{region} · {lead_hours//24}-day lead',ha='center',fontsize=14)
    fig.text(.5,.917,f'{stat.field_title}: polygon mean {operator} {criterion}',ha='center',fontsize=10.5)
    rows=[['Events / members',f'{int(stat.cases)} / {int(stat.ensemble_members)}'],
          ['Distinct forecasts / PAV pools',f'{int(stat.distinct_probabilities)} / {int(stat.pav_pools)}'],
          ['ERA5 crossings / observed frequency',f'{int(stat.observed_crossings)} / {stat.base_rate:.1%}'],
          ['Original Brier score',f'{stat.brier_score:.4f}'],
          ['Miscalibration (MCB)',f'{stat.miscalibration_mcb:.4f}'],
          ['Discrimination (DSC)',f'{stat.discrimination_dsc:.4f}'],
          ['Uncertainty (UNC)',f'{stat.uncertainty_unc:.4f}']]
    table_ax.axis('off');table=table_ax.table(cellText=rows,colLabels=['Statistic','Value'],
        colWidths=[.7,.3],cellLoc='left',bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(10)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#d7e0e7');cell.set_linewidth(.55);cell.PAD=.07
        if r==0:cell.set_facecolor('#e8eef3');cell.set_text_props(weight='bold')
        elif r==5:cell.set_facecolor('#e2f0ef');cell.set_text_props(weight='bold',color='#075e65')
        elif r%2==0:cell.set_facecolor('#f6f8fa')
    start=pd.Timestamp(stat.valid_time_start);end=pd.Timestamp(stat.valid_time_end)
    band_note='90% pointwise consistency band; exchangeability assumed.' if stat.band_type!='none' else 'Uncertainty bands not shown; winter cases may be dependent.'
    fig.text(.09,.018,f'{start:%Y-%m-%d} to {end:%Y-%m-%d} · ERA5-selected events · BS = MCB − DSC + UNC\n'
             f'{band_note}\nCORP/PAV: Dimitriadis et al. (2021). Diagnostic fit; original forecasts retained.',fontsize=8.5,color='#435567')
    return fig


def plot_corp_forecast_counts(points, *, title_suffix='All'):
    """Exact native probability-value counts, saved separately from the CORP graph."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter,MaxNLocator
    fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained')
    for r,region in enumerate(POLYGONS):
        for c,lead in enumerate(LEAD_HOURS):
            frame=points.loc[points.polygon.eq(region)&points.lead_days.eq(lead//24)]
            ax=axes[r,c]
            ax.bar(100*frame.forecast_probability,frame.case_count,width=1.2,color='#087f8c')
            ax.set(xlim=(-2,102),ylim=(0,None),xlabel='Original forecast probability',ylabel='Number of events',
                   title=f'{region} · {lead//24} days · N={int(frame.case_count.sum())}')
            ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'Forecast counts at native probability values · {title_suffix}\n'
                 '64-member probabilities occur in increments of 1/64; one case per event and panel.',fontsize=13)
    return fig


def export_corp_reliability(probabilities, destination, *, title_suffix='All', consistency_bands=False,
                             assume_exchangeable=False, n_resamples=2000, seed=37):
    """Export only saved probability diagnostics; this API accepts no archive or weather arrays."""
    import matplotlib.pyplot as plt
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    points,cases,summary=corp_calibration_tables(probabilities,consistency_bands=consistency_bands,
        assume_exchangeable=assume_exchangeable,n_resamples=n_resamples,seed=seed)
    for name,frame in (('corp_points',points),('corp_cases',cases),('corp_summary',summary)):
        temporary=destination/f'.{name}.tmp.csv';frame.to_csv(temporary,index=False);temporary.replace(destination/f'{name}.csv')
    pool_rows=[]
    for _,frame in points.groupby(['polygon','lead_days','pool_id'],sort=False):
        first=frame.iloc[0]
        pool_rows.append(dict(polygon=first.polygon,lead_days=first.lead_days,pool_id=first.pool_id,
            cases=int(frame.case_count.sum()),observed_crossings=int(frame.observed_count.sum()),
            distinct_probabilities=len(frame),forecast_min=first.pool_min_probability,forecast_max=first.pool_max_probability,
            mean_forecast_probability=float(np.average(frame.forecast_probability,weights=frame.case_count)),
            corp_probability=first.corp_probability))
    pd.DataFrame(pool_rows,columns=['polygon','lead_days','pool_id','cases','observed_crossings',
        'distinct_probabilities','forecast_min','forecast_max','mean_forecast_probability','corp_probability']).to_csv(
            destination/'corp_pools.csv',index=False)
    for polygon,tag in (('JPCZ polygon','jpcz_polygon'),('Coastal wedge','coastal_wedge')):
        for lead in LEAD_HOURS:
            stem=f'corp_{tag}_lead{lead//24:02d}d'
            present=not summary.loc[summary.polygon.eq(polygon)&summary.lead_days.eq(lead//24)].empty
            if present:
                fig=plot_corp_reliability(points,summary,region=polygon,lead_hours=lead,title_suffix=title_suffix)
                with plt.rc_context({'pdf.fonttype':42,'ps.fonttype':42}):
                    for extension in ('png','pdf'):fig.savefig(destination/f'{stem}.{extension}',dpi=300)
                plt.close(fig)
            else:
                for extension in ('png','pdf'):(destination/f'{stem}.{extension}').unlink(missing_ok=True)
    if not points.empty:
        fig=plot_corp_forecast_counts(points,title_suffix=title_suffix)
        with plt.rc_context({'pdf.fonttype':42,'ps.fonttype':42}):
            for extension in ('png','pdf'):fig.savefig(destination/f'corp_forecast_counts.{extension}',dpi=300)
        plt.close(fig)
    else:
        for extension in ('png','pdf'):(destination/f'corp_forecast_counts.{extension}').unlink(missing_ok=True)
    atomic_json_file(destination/'corp_settings.json',dict(method='CORP; weighted isotonic regression / PAV',
        reference=CORP_REFERENCE,original_forecasts_changed=False,in_sample_diagnostic=True,
        panels=json.loads(summary[['polygon','lead_days','field','pressure_hpa','units','threshold','direction','cases']].to_json(orient='records')),
        png_dpi=300,pdf_fonts='embedded TrueType',
        use='threshold crossing calibration within the ERA5-selected sample',
        monotonicity='conditional crossing frequency constrained nondecreasing in forecast probability',
        band_type='90% pointwise consistency' if consistency_bands else 'none',
        exchangeability_assumed=assume_exchangeable,n_resamples=n_resamples if consistency_bands else 0,seed=int(seed),
        brier_identity='original BS = MCB - DSC + UNC',
        reference_for_unc='within-sample base rate; descriptive decomposition, not independent CRPSS climatology'))
    return points,cases,summary
