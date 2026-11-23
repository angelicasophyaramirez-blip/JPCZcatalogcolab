"""Per-event CRPS figures and matched CRPSS, using saved scalar values only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ensemble_verification import LEAD_HOURS, POLYGONS, atomic_json_file, empirical_ensemble_crps

SCORE_CASE_KEYS = ['review_key', 'polygon', 'lead_hours']


def validate_saved_case_scores(scores):
    required = SCORE_CASE_KEYS + ['event_number', 'valid_time_utc', 'init_time_utc', 'lead_days',
        'field', 'field_title', 'pressure_hpa', 'units', 'era5_value', 'crps']
    missing = set(required) - set(scores)
    if missing: raise ValueError(f'Missing saved CRPS columns: {sorted(missing)}. Run section 5 first.')
    if scores.empty: raise ValueError('No complete saved CRPS cases. Run section 5 first.')
    if scores.duplicated(SCORE_CASE_KEYS).any(): raise ValueError('Duplicate saved CRPS case identities.')
    if not np.isfinite(scores[['crps', 'era5_value']].to_numpy(dtype=float)).all() or scores.crps.lt(0).any():
        raise ValueError('Saved scores and ERA5 values must be finite; CRPS must be nonnegative.')
    for name in ('field', 'pressure_hpa', 'units'):
        if scores[name].nunique(dropna=False) != 1: raise ValueError(f'Keep each {name} in a separate report.')
    expected = {(region, lead) for region in POLYGONS for lead in LEAD_HOURS}
    for _, frame in scores.groupby('review_key', sort=False):
        if set(zip(frame.polygon, frame.lead_hours)) != expected:
            raise ValueError('Each event must have both polygons and all three leads in the common sample.')
        if frame.event_number.nunique() != 1 or pd.to_datetime(frame.valid_time_utc, utc=True).nunique() != 1:
            raise ValueError('An event identity has inconsistent labels or valid times.')
    if not scores.lead_days.eq(scores.lead_hours / 24).all(): raise ValueError('Inconsistent lead units.')
    valid = pd.to_datetime(scores.valid_time_utc, utc=True)
    initialized = pd.to_datetime(scores.init_time_utc, utc=True)
    if valid.isna().any() or initialized.isna().any() or not (initialized+pd.to_timedelta(scores.lead_hours, unit='h')).eq(valid).all():
        raise ValueError('Saved initialization, lead and valid time do not agree.')
    if 'in_common_sample' in scores and not scores.in_common_sample.astype(str).str.lower().eq('true').all():
        raise ValueError('Use only the complete common sample for these comparisons.')
    return scores.sort_values(['valid_time_utc', 'event_number', 'polygon', 'lead_hours']).copy()


def matched_climatology_skill(scores, reference, settings, calculation_settings):
    """Score a frozen empirical climatology on exactly the saved model cases.

    CRPSS is a ratio of mean scores. Individual cases retain model/reference CRPS
    and their difference, not an averageable individual-case skill percentage.
    """
    scores = validate_saved_case_scores(scores)
    if not settings.get('complete'): raise ValueError('Reference incomplete: finish its saved schedule before CRPSS.')
    if settings.get('weighting') != 'one target, one equal weight': raise ValueError('Unsupported reference weighting.')
    for name in ('era5_store', 'domain', 'polygons', 'pressure_hpa', 'diagnostic'):
        actual = json.loads(json.dumps(calculation_settings.get(name), sort_keys=True))
        if settings.get('calculation', {}).get(name) != actual:
            raise ValueError(f'Reference and forecast {name} definitions differ.')
    required = ['reference_key', 'valid_time_utc', 'polygon', 'value', 'field', 'pressure_hpa', 'units', 'reference_product']
    if set(required) - set(reference): raise ValueError('Incomplete reference-value schema.')
    if reference.duplicated(['reference_key', 'polygon']).any(): raise ValueError('Duplicate climatology targets.')
    if not reference.reference_product.eq('ERA5').all(): raise ValueError('Use final historical ERA5 for climatology.')
    if not np.isfinite(reference.value.to_numpy(dtype=float)).all(): raise ValueError('Nonfinite reference values.')
    for name in ('field', 'pressure_hpa', 'units'):
        if not reference[name].eq(scores[name].iloc[0]).all(): raise ValueError(f'Reference {name} mismatch.')
    times = pd.to_datetime(reference.valid_time_utc, utc=True)
    start = pd.Timestamp(settings['start_utc']); end = pd.Timestamp(settings['end_utc_exclusive'])
    if not ((times >= start) & (times < end)).all(): raise ValueError('Reference dates exceed its frozen period.')
    earliest_init = pd.to_datetime(scores.init_time_utc, utc=True).min()
    if end > earliest_init or (times >= earliest_init).any():
        raise ValueError('Climatology must end before every evaluated forecast initialization.')
    expected = int(settings['expected_targets'])
    if expected < 2: raise ValueError('Climatology requires at least two historical targets.')
    if set(reference.polygon) != set(POLYGONS): raise ValueError('Climatology must cover both polygons.')
    target_sets = []
    distributions = {}
    for region, frame in reference.groupby('polygon', sort=False):
        if len(frame) != expected: raise ValueError('Reference inventory differs from the complete frozen schedule.')
        target_sets.append(set(zip(frame.reference_key, frame.valid_time_utc)))
        distributions[region] = frame.value.to_numpy(dtype=float)
    if target_sets[0] != target_sets[1]: raise ValueError('Climatology polygons have different targets.')
    frozen_targets = {(row['reference_key'], row['valid_time_utc']) for row in settings['schedule']}
    if target_sets[0] != frozen_targets: raise ValueError('Reference values differ from the frozen target list.')
    cases = scores.copy()
    cases['reference_crps'] = [empirical_ensemble_crps(distributions[row.polygon], row.era5_value)
                               for row in cases.itertuples()]
    cases['crps_improvement'] = cases.reference_crps - cases.crps
    cases['reference_targets'] = expected
    cases['reference_kind'] = settings['kind']
    rows = []
    for (region, lead), frame in cases.groupby(['polygon', 'lead_days'], sort=False):
        model = float(frame.crps.mean()); baseline = float(frame.reference_crps.mean())
        rows.append(dict(polygon=region, lead_days=int(lead), cases=len(frame),
            mean_model_crps=model, mean_reference_crps=baseline, mean_improvement=baseline-model,
            crpss_percent=100*(1-model/baseline) if baseline > 0 else np.nan,
            status='complete' if baseline > 0 else 'undefined_zero_reference_crps',
            reference_targets=expected, reference_kind=settings['kind'],
            field=frame.field.iloc[0], pressure_hpa=int(frame.pressure_hpa.iloc[0]), units=frame.units.iloc[0]))
    return cases, pd.DataFrame(rows)


def plot_event_crps(scores, review_key):
    """One case, six saved scores. No boxplots: there is one score per lead/polygon."""
    import matplotlib.pyplot as plt
    frame = scores.loc[scores.review_key.eq(review_key)]
    if frame.empty: raise ValueError('Event not present in saved scores.')
    first = frame.iloc[0]; time = pd.Timestamp(first.valid_time_utc)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    fig.subplots_adjust(left=.08, right=.98, top=.75, bottom=.34, wspace=.29)
    fig.suptitle(f'Event {int(first.event_number)} · CRPS at 1, 5 and 10 days', y=.97, fontsize=16)
    fig.text(.5, .89, f'{first.field_title} · verifying {time:%Y-%m-%d %H:%M} UTC', ha='center', fontsize=11)
    fig.text(.5, .84, 'One score per polygon and lead · all 64 WN2 members · lower is better', ha='center', fontsize=10)
    for ax, region in zip(axes, POLYGONS):
        panel = frame.loc[frame.polygon.eq(region)].set_index('lead_days').reindex([1, 5, 10])
        ax.plot([1, 5, 10], panel.crps, 'o-', color='#087f8c', label='WN2 CRPS')
        rows = [['WN2 CRPS', *[f'{v:.3f}' for v in panel.crps]]]
        if 'reference_crps' in panel:
            ax.plot([1, 5, 10], panel.reference_crps, 's--', color='#b56a19', label='Climatology CRPS')
            rows.extend([['Reference CRPS', *[f'{v:.3f}' for v in panel.reference_crps]],
                         ['Improvement', *[f'{v:+.3f}' for v in panel.crps_improvement]]])
        ax.set(title=region, xlabel='Forecast lead (days)', ylabel=f'CRPS ({first.units})',
               xticks=[1, 5, 10], ylim=(0, None))
        ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=9)
        table = ax.table(cellText=rows, colLabels=['Saved score', '1 day', '5 days', '10 days'],
            colWidths=[.43, .19, .19, .19], cellLoc='center', bbox=[0, -.49, 1, .27])
        table.auto_set_font_size(False); table.set_fontsize(9)
        for (r, _), cell in table.get_celld().items():
            cell.set_edgecolor('#d8e1e8')
            if r == 0: cell.set_facecolor('#e8eef3'); cell.set_text_props(weight='bold')
    fig.text(.08, .035, 'Scores describe polygon means, not spatial placement. '
        'CRPSS is computed from mean paired scores across cases.', fontsize=9, color='#435567')
    return fig


def plot_all_event_crps(scores):
    """Every case is labelled; color shows each case's score rather than a pooled boxplot."""
    import matplotlib.pyplot as plt
    events = scores[['review_key', 'event_number', 'valid_time_utc']].drop_duplicates('review_key')
    n = len(events); fig, axes = plt.subplots(1, 2, figsize=(11, max(5, .25*n+2.4)), layout='constrained')
    maximum = float(scores.crps.max()) or 1.
    labels = [f'#{int(r.event_number)} · {pd.Timestamp(r.valid_time_utc):%Y-%m-%d}' for r in events.itertuples()]
    for ax, region in zip(axes, POLYGONS):
        values = scores.loc[scores.polygon.eq(region)].pivot(index='review_key', columns='lead_days', values='crps')
        values = values.reindex(index=events.review_key, columns=[1, 5, 10])
        artist = ax.imshow(values, aspect='auto', vmin=0, vmax=maximum, cmap='YlGnBu', interpolation='nearest')
        ax.set(xticks=[0, 1, 2], xticklabels=['1 day', '5 days', '10 days'], yticks=np.arange(n),
               yticklabels=labels, title=region, xlabel='Forecast lead')
        ax.tick_params(axis='y', labelsize=8)
        for row in range(n):
            for col in range(3):
                value = values.iloc[row, col]
                ax.text(col, row, f'{value:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if value > .55*maximum else '#182d3b')
    fig.colorbar(artist, ax=list(axes), shrink=.6, label=f'CRPS ({scores.units.iloc[0]}); lower is better')
    fig.suptitle(f'Every event · {scores.field_title.iloc[0]}\nN = {n}; each row is one case, not one member', fontsize=14)
    return fig


def export_event_crps_reports(scores, destination):
    """Write each case and one combined PDF; keep only one event figure in RAM at once."""
    import matplotlib.pyplot as plt
    from pypdf import PdfReader, PdfWriter
    scores = validate_saved_case_scores(scores)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    scores.to_csv(destination/'all_event_scores.csv', index=False)
    events = scores[['review_key', 'event_number', 'valid_time_utc']].drop_duplicates('review_key')
    wide = scores.pivot(index=['review_key', 'event_number', 'valid_time_utc'],
                        columns=['polygon', 'lead_days'], values='crps')
    wide.columns = [f'{region} | {int(lead)} day CRPS' for region, lead in wide.columns]
    wide = wide.reset_index().sort_values(['valid_time_utc', 'event_number'])
    for value, label in (('reference_crps', 'reference CRPS'), ('crps_improvement', 'CRPS improvement')):
        if value in scores:
            extra = scores.pivot(index=['review_key', 'event_number', 'valid_time_utc'],
                                 columns=['polygon', 'lead_days'], values=value)
            extra.columns = [f'{region} | {int(lead)} day {label}' for region, lead in extra.columns]
            wide = wide.merge(extra.reset_index(), on=['review_key', 'event_number', 'valid_time_utc'], validate='one_to_one')
    wide.to_csv(destination/'one_row_per_event.csv', index=False)
    with pd.ExcelWriter(destination/'all_event_scores.xlsx', engine='xlsxwriter',
            engine_kwargs={'options': {'strings_to_formulas': False, 'strings_to_urls': False}}) as writer:
        wide.to_excel(writer, sheet_name='All events', index=False)
        sheet = writer.sheets['All events']; sheet.freeze_panes(1, 3); sheet.set_column(0, len(wide.columns)-1, 24)
        sheet.autofilter(0, 0, len(wide), len(wide.columns)-1)
    index = []
    with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        temporary_pdf = destination/'.all_events_crps.tmp.pdf'
        with PdfWriter() as combined:
            for row in events.itertuples():
                stamp = pd.Timestamp(row.valid_time_utc).strftime('%Y%m%dT%H%MZ')
                stem = f'event_{int(row.event_number):04d}_{stamp}_crps'
                case = scores.loc[scores.review_key.eq(row.review_key)]
                signature = hashlib.sha256(('event_crps_v1|'+case.to_json(orient='split', double_precision=15)).encode()).hexdigest()
                marker = destination/f'{stem}.json'
                cached = False
                if marker.is_file() and all((destination/f'{stem}.{ext}').is_file() for ext in ('png', 'pdf', 'csv')):
                    try: cached = json.loads(marker.read_text()).get('signature') == signature
                    except (ValueError, OSError): pass
                if not cached:
                    fig = plot_event_crps(scores, row.review_key)
                    try:
                        fig.savefig(destination/f'{stem}.png', dpi=220)
                        fig.savefig(destination/f'{stem}.pdf')
                    finally: plt.close(fig)
                    case.to_csv(destination/f'{stem}.csv', index=False)
                    atomic_json_file(marker, dict(signature=signature))
                combined.append(PdfReader(destination/f'{stem}.pdf'))
                index.append(dict(event_number=int(row.event_number), review_key=row.review_key,
                    event_png=str(destination/f'{stem}.png'), event_pdf=str(destination/f'{stem}.pdf'),
                    status='cached' if cached else 'exported'))
            combined.write(temporary_pdf)
        temporary_pdf.replace(destination/'all_events_crps.pdf')
        fig = plot_all_event_crps(scores)
        try:
            fig.savefig(destination/'all_events_crps_overview.png', dpi=220)
            fig.savefig(destination/'all_events_crps_overview.pdf')
        finally: plt.close(fig)
    index = pd.DataFrame(index); index.to_csv(destination/'event_figure_index.csv', index=False)
    atomic_json_file(destination/'case_score_settings.json', dict(events=len(events), scores=len(scores),
        source='saved crps_cases.csv', weather_fields_recalculated=False, score_unit=scores.units.iloc[0],
        combined_pdf='all_events_crps.pdf', sample='same complete cases at all leads'))
    return index, wide


def export_crpss_report(scores, reference_directory, destination, calculation_settings):
    """No archive access. Missing/partial climatology yields an explicit pending status."""
    import matplotlib.pyplot as plt
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    reference_directory = Path(reference_directory)
    paths = [reference_directory/'reference_values.csv', reference_directory/'reference_settings.json']
    if not all(path.is_file() for path in paths):
        status = dict(status='reference_not_prepared', message='Build the independent ERA5 climatology in section 5c first.')
        cases = summary = None
    else:
        settings = json.loads(paths[1].read_text())
        if not settings.get('complete'):
            status = dict(status='reference_incomplete', completed_targets=settings.get('completed_targets', 0),
                          expected_targets=settings.get('expected_targets', 0), message='Resume section 5c; no partial-reference skill is reported.')
            cases = summary = None
        else:
            reference = pd.read_csv(paths[0], float_precision='round_trip')
            cases, summary = matched_climatology_skill(scores, reference, settings, calculation_settings)
            status = dict(status='complete', reference_directory=str(reference_directory),
                definition='100 * (1 - mean(model CRPS) / mean(reference CRPS))',
                sample='identical cases and equal event weights', reference_kind=settings['kind'])
    atomic_json_file(destination/'crpss_status.json', status)
    if cases is None:
        for name in ('crpss_cases.csv', 'crpss_summary.csv', 'crpss_summary.png', 'crpss_summary.pdf'):
            (destination/name).unlink(missing_ok=True)
        return status, cases, summary
    cases.to_csv(destination/'crpss_cases.csv', index=False); summary.to_csv(destination/'crpss_summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7))
    fig.subplots_adjust(left=.08, right=.98, top=.77, bottom=.39, wspace=.3)
    for ax, region in zip(axes, POLYGONS):
        panel = summary.loc[summary.polygon.eq(region)].sort_values('lead_days')
        ax.axhline(0, color='#45535e', linestyle='--', linewidth=1)
        ax.plot(panel.lead_days, panel.crpss_percent, 'o-', color='#087f8c')
        ax.set(title=region, xticks=[1, 5, 10], xlabel='Forecast lead (days)', ylabel='CRPSS (%)')
        ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.2)
        panel = panel.set_index('lead_days').reindex([1, 5, 10])
        rows = [['Paired events', *[str(int(x)) for x in panel.cases]],
                ['Reference targets', *[str(int(x)) for x in panel.reference_targets]],
                ['Mean WN2 CRPS', *[f'{x:.3f}' for x in panel.mean_model_crps]],
                ['Mean reference CRPS', *[f'{x:.3f}' for x in panel.mean_reference_crps]],
                ['CRPSS (%)', *[f'{x:+.1f}' if np.isfinite(x) else 'undefined' for x in panel.crpss_percent]]]
        table = ax.table(cellText=rows, colLabels=['Statistic', '1 day', '5 days', '10 days'],
            colWidths=[.46, .18, .18, .18], cellLoc='center', bbox=[0, -.76, 1, .51])
        table.auto_set_font_size(False); table.set_fontsize(9)
        for (r, _), cell in table.get_celld().items():
            cell.set_edgecolor('#d8e1e8')
            if r == 0: cell.set_facecolor('#e8eef3'); cell.set_text_props(weight='bold')
    first = scores.iloc[0]
    fig.suptitle(f'CRPSS · {first.field_title}\nSkill relative to a frozen historical ERA5 distribution', y=.97, fontsize=14)
    fig.text(.5, .84, f"{settings['kind']} · {settings['start_utc'][:10]} to {settings['end_utc_exclusive'][:10]} (end exclusive)",
             ha='center', fontsize=10)
    fig.text(.08, .035, f'Positive: better than reference. CRPS units: {first.units}. '
        'CRPSS is a ratio of mean paired scores.\nReference dates precede evaluated forecasts; '
        'this does not verify the WN2 checkpoint training cutoff.', fontsize=9, color='#435567')
    with plt.rc_context({'pdf.fonttype': 42}):
        fig.savefig(destination/'crpss_summary.png', dpi=220); fig.savefig(destination/'crpss_summary.pdf')
    plt.close(fig)
    return status, cases, summary
