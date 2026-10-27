"""Single customer planning flow using explicit historical strategy statistics."""
from __future__ import annotations

from io import BytesIO, StringIO
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape

from .business_config import control_name, default_config, fingerprint
from .business_data import canonical, read_table
from .business_plan import build_plan
from .business_store import code_fingerprint
from .portfolio import load_experiments, upsert_experiment

COLUMNS = ['Partner', 'FICO group', 'PD score group', 'Strategy', 'Metric', 'Sample mean', 'Sample variance', 'Sample number']


def demo_data():
    rows = []
    for partner, shift in [('Holland', 0), ('PriceLine', 120), ('Princess', 220), ('Carnival', 300)]:
        for i, fico in enumerate(['FICO 650–699', 'FICO 700–749', 'FICO 750+']):
            for k, pd_group in enumerate(['PD low', 'PD medium', 'PD high']):
                for j, strategy in enumerate(['Credit line 1000', 'Credit line 2000', 'Credit line 3000']):
                    for metric, mean, sd in [
                        ('Revolving Balance', 1100+shift+100*i-80*k+100*j, 300+20*i+10*k+20*j),
                        ('RAR', 120+shift/20+10*i-8*k+5*j, 45+3*i+2*k+3*j),
                        ('NCL', 35+shift/50-4*i+10*k+5*j, 20+2*i+3*k+2*j),
                    ]:
                        rows.append([partner, fico, pd_group, strategy, metric, mean, sd**2, 200+25*i+20*k+15*j])
    return pd.DataFrame(rows, columns=COLUMNS)


def _canonical_input(frame, default_metric='Primary metric'):
    d = canonical(frame)
    d = d.rename(columns={'strategy / option': 'strategy', 'option':'strategy', 'credit line':'strategy', 'group by':'group_by', 'groupby':'group_by'})
    if d.columns.duplicated().any():
        raise ValueError('Column names overlap after mapping. Rename duplicate fields.')
    if 'metric' not in d:
        d['metric'] = default_metric.strip()
    return d


def grouping_columns(frame):
    """Prefer common business dimensions while allowing other categorical columns."""
    d = _canonical_input(frame)
    reserved = {'partner', 'strategy', 'metric', 'sample_mean', 'sample_variance', 'sample_number', 'sample_sd',
                'group_by', 'ym', 'month', 'year_month', 'date', 'snapshot_date', 'historical_start', 'historical_end', 'successes'}
    candidates = []
    for original, col in zip(frame.columns, d.columns):
        if col in reserved or col.endswith('_id') or col == 'id':
            continue
        series = d[col].dropna()
        # Do not treat continuous scores or customer IDs as arbitrary categories.
        if len(series) == 0 or (pd.to_numeric(series, errors='coerce').notna().all() and series.nunique() > 20):
            continue
        preferred = any(x in col for x in ('segment', 'fico', 'risk', 'group', 'band', 'score'))
        candidates.append((0 if preferred else 1, str(original), col))
    return [(label, col) for _, label, col in sorted(candidates, key=lambda x: x[0])]


def pool_statistics(frame, keys):
    """Combine disjoint sample summaries using the exact sample variance formula."""
    result = []
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        n = group.sample_number.to_numpy(dtype=float)
        means = group.sample_mean.to_numpy(dtype=float)
        variances = group.sample_variance.to_numpy(dtype=float)
        total = float(n.sum())
        mean = float(np.dot(n, means) / total)
        variance = float(np.sum((n-1)*variances + n*(means-mean)**2) / (total-1))
        result.append({**dict(zip(keys, key)), 'sample_mean':mean, 'sample_variance':variance, 'sample_number':int(total)})
    return pd.DataFrame(result)


def normalize_table(frame: pd.DataFrame, default_metric='Primary metric', grouping=None, allow_pool=False) -> pd.DataFrame:
    d = _canonical_input(frame, default_metric)
    fields = ['partner', 'strategy', 'metric', 'sample_mean', 'sample_variance', 'sample_number']
    missing = [x for x in fields if x not in d]
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    legacy = 'group_by' in d and 'segment' in d
    candidates = grouping_columns(frame)
    if legacy:
        d['group_by'] = d['group_by'].astype(str).str.strip()
        grouping = grouping or d['group_by'].iloc[0]
        d = d[d['group_by'].eq(grouping)].copy()
        dimensions = ['segment']
    else:
        if not candidates:
            raise ValueError('Add a grouping column such as Segment, FICO group or PD score group. Continuous scores must be grouped first.')
        grouping = grouping or candidates[0][0]
        mapping = dict(candidates)
        if grouping not in mapping:
            raise ValueError('Choose an available grouping column.')
        dimensions = [col for _, col in candidates]
        d = d.rename(columns={mapping[grouping]: 'segment'})
        dimensions = ['segment' if c == mapping[grouping] else c for c in dimensions]
        d['group_by'] = grouping
    for col in list(dict.fromkeys(['partner', 'group_by', 'strategy', 'metric'] + dimensions)):
        if d[col].isna().any() or d[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'{col}: fill in every row.')
        d[col] = d[col].astype(str).str.strip()
    for col in ['sample_mean', 'sample_variance', 'sample_number']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
        if not np.isfinite(d[col]).all():
            raise ValueError(f'{col}: every row must contain a valid number.')
    if (d.sample_variance < 0).any() or (d.sample_number < 2).any() or (d.sample_number % 1 != 0).any():
        raise ValueError('Sample variance must be non-negative and sample number must be a whole number of at least 2.')
    full_keys = list(dict.fromkeys(['partner', 'group_by', 'strategy', 'metric'] + dimensions))
    if d.duplicated(full_keys).any():
        raise ValueError('Duplicate historical combinations. Do not add monthly rows when the same customers can appear more than once; provide non-overlapping summaries.')
    keys = ['partner', 'group_by', 'segment', 'strategy', 'metric']
    if d.duplicated(keys).any():
        if not allow_pool:
            raise ValueError('This grouping combines rows across other dimensions. Confirm that those rows contain non-overlapping customers before combining them.')
        return pool_statistics(d, keys)
    return d[keys + ['sample_mean', 'sample_variance', 'sample_number']].reset_index(drop=True)

def prepare_history(data, partners, segments, control, tests, metrics):
    selected = data[data.partner.isin(partners) & data.segment.isin(segments)]
    # All strata represented by selected partners must have all selected arms and metrics.
    cells = pd.DataFrame([(partner, segment) for partner in partners for segment in segments], columns=['partner', 'segment'])
    missing = []
    for cell in cells.itertuples(index=False):
        for arm in [control] + list(tests):
            for metric in metrics:
                if selected[(selected.partner == cell.partner) & (selected.segment == cell.segment) & (selected.strategy == arm) & (selected.metric == metric)].empty:
                    missing.append(f'{cell.partner} / {cell.segment} / {arm} / {metric}')
    if cells.empty:
        raise ValueError('Select at least one available Partner and segment.')
    if missing:
        raise ValueError('Missing historical statistics for: ' + '; '.join(missing[:15]) + (f' (and {len(missing)-15} more)' if len(missing)>15 else ''))
    return selected[selected.strategy.isin([control]+list(tests)) & selected.metric.isin(metrics)].rename(columns={'strategy':'arm'}).drop(columns='group_by')


def calculate_plan(data, partners, segments, control, tests, primary, guardrail, effect, effect_scale, alpha, power, multiplicity, guardrail_margin=0):
    metrics = [primary] + ([guardrail] if guardrail else [])
    history = prepare_history(data, partners, segments, control, tests, metrics)
    config = default_config()
    config.update(name='Customer experiment', owner='Not provided', question='Compare selected options', alpha=alpha, power=power, multiplicity=multiplicity)
    config['arms'] = [{'name': arm, 'role': 'Control' if arm == control else 'Test', 'definition': arm} for arm in [control]+list(tests)]
    base = config['metrics'][0]
    base.update(name=primary, unit='Not provided', definition='Historical customer-level metric; observation window not provided', effect=effect, effect_scale=effect_scale)
    if guardrail:
        config['metrics'].append({**base, 'name':guardrail, 'role':'Guardrail', 'direction':'Lower', 'margin':guardrail_margin or 1., 'include_in_plan':bool(guardrail_margin)})
    plan = build_plan(config, history, code_version=code_fingerprint())
    if plan['variance_assumptions']:
        raise ValueError('Some selected options lack historical variance; complete the missing data before calculating.')
    return plan


def pdf_summary(plan):
    output = BytesIO()
    styles = getSampleStyleSheet()
    font = 'CustomerPlanSans'
    if font not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font, str(Path(__file__).parent / 'assets' / 'fonts' / 'BusinessSansSC-Regular.ttf')))
    for style in styles.byName.values():
        style.fontName = font
        style.wordWrap = 'CJK'
    c = plan['config']
    control = control_name(c)
    roles = {arm['name']: arm['role'] for arm in c['arms']}
    story = [Paragraph('Customer experiment sample size plan', styles['Title']), Spacer(1, 12),
             Paragraph(escape(c['name']), styles['Heading2']),
             Paragraph('Historical summaries are estimates, not evidence of a causal effect. Historical control counts determine the assumed population mix. Maintain this mix across arms.', styles['BodyText']), Spacer(1,12)]
    if c.get('synthetic'):
        story.insert(1, Paragraph('SYNTHETIC DEMO - inputs and results are illustrative.', styles['BodyText']))
    rows = [['Role', 'Option', 'Analyzable N', 'Required customers']] + [[roles.get(r['arm'], 'Test'), str(r['arm']), str(r['analyzable_n']), str(r['recruitment_n'])] for r in plan['arm_totals']]
    rows.append(['Total', '', str(sum(r['analyzable_n'] for r in plan['arm_totals'])), str(sum(r['recruitment_n'] for r in plan['arm_totals']))])
    table = Table(rows, repeatRows=1, hAlign='LEFT', colWidths=[60,170,100,110])
    table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),font),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f8e7e6')),('GRID',(0,0),(-1,-1),.4,colors.lightgrey),('PADDING',(0,0),(-1,-1),7)]))
    story += [table, Spacer(1,12), Paragraph('Statistical settings', styles['Heading2'])]
    primary = next(metric for metric in c['metrics'] if metric['role'] == 'Primary')
    details = [f"Control: {control}. Primary metric: {primary['name']}. MDE: {primary['effect']:g} {primary['effect_scale']}.",
               f"Data source: {plan.get('input_source', 'Not recorded')}; Group by: {plan.get('input_grouping', 'Not recorded')}; calculated: {plan.get('saved_at', 'Not recorded')}.",
               f"Significance level (alpha): {c['alpha']:g}; target power: {c['power']:.1%} per comparison; expected missing-outcome fraction: {c['attrition']:.1%}.",
               f"Multiple-comparison protection: {'on' if c['multiplicity'] else 'off'}. Comparison family: {plan['family_size']}; planning alpha per comparison: {plan['planning_alpha']:.8g}.",
               f"Allocation: {c['allocation']} (pairwise Neyman with shared Control). Decision scope: {c['decision_scope']}. Unit: {c['unit']}."]
    for metric in c['metrics']:
        if metric['role'] == 'Guardrail':
            status = (f"included in sample planning; maximum tolerated deterioration {metric['margin']:g} metric units" if metric['include_in_plan'] else 'monitoring only; not a sample-size constraint')
            details.append(f"Guardrail: {metric['name']}; {status}.")
    details.append('Fixed-weight stratified Welch planning uses historical variances, assumes the same Partner / Segment mix across groups, and does not treat historical option differences as causal effects. The target is per-comparison power, not simultaneous power. Historical parameter uncertainty is not fully propagated.')
    story.extend(Paragraph(escape(text), styles['BodyText']) for text in details)
    story.append(Paragraph('Calculation formulas', styles['Heading2']))
    formulas = [
        'Historical sample pooling (only for non-overlapping customers): N = sum(n_i); mean = sum(n_i * mean_i) / N; variance = [sum((n_i - 1) * variance_i + n_i * (mean_i - mean)^2)] / (N - 1).',
        'Population weight for stratum h: w_h = historical Control N_h / sum(historical Control N). The same weights are used for Control and each Test.',
        'Control baseline = sum(w_h * Control mean_h). For a relative MDE, delta = abs(Control baseline) * MDE / 100; for an absolute MDE, delta = MDE.',
        'Variance of each weighted Test - Control difference: V = sum_h[w_h^2 * (s_Control,h^2 / n_Control,h + s_Test,h^2 / n_Test,h)].',
        'Welch-Satterthwaite degrees of freedom: df = V^2 / sum_j[v_j^2 / (n_j - 1)], where each v_j is a weighted variance component in V.',
        'Primary metric power uses the noncentral t distribution with noncentrality delta / sqrt(V) and critical value t_(1 - alpha_comparison/2, df). The engine searches integer sample sizes and verifies the resulting power.',
        'Pairwise Neyman starting ratio for Test k: n_Test,k / n_Control = sqrt(sum_h(w_h * s_Test,k,h^2)) / sqrt(sum_h(w_h * s_Control,h^2)). Each Test is first sized against Control; the largest pairwise Control N is retained, then each Test N is recalculated and checked with that shared Control.',
        'When protection is enabled, alpha_comparison = alpha / number of comparisons in the family; otherwise alpha_comparison = alpha. Recruitment target per stratum and arm = ceil(analyzable N / (1 - missing-outcome fraction)).',
    ]
    story.extend(KeepTogether([Paragraph(escape(text), styles['BodyText'])]) for text in formulas)
    if any(metric['role'] == 'Guardrail' and metric['include_in_plan'] for metric in c['metrics']):
        story.append(Paragraph('For a guardrail included in planning, the engine uses its absolute deterioration margin as delta and checks one-sided non-inferiority power under no true difference, using the same conservative t critical value.', styles['BodyText']))
    story.append(Paragraph('Pairwise calculations and final verification', styles['Heading2']))
    for pair in plan['pairwise']:
        story.append(Paragraph(escape(f"{pair['test']} vs {control}: initial pairwise Control N = {pair['pair_control_n']}, Test N = {pair['pair_test_n']}, Neyman ratio = {pair['neyman_ratio']:.5g}."), styles['BodyText']))
    for comparison in plan['comparisons']:
        story.append(Paragraph(escape(f"{comparison['metric']} / {comparison['test']}: final Control N = {comparison['control_n']}, Test N = {comparison['test_n']}, baseline = {comparison['baseline']:.6g}, detectable change or margin = {comparison['detectable_change_or_margin']:.6g}, variance = {comparison['variance_of_difference']:.6g}, df = {comparison['degrees_of_freedom']:.6g}, verified power = {comparison['power']:.2%}."), styles['BodyText']))
    story.append(Paragraph('Population weights', styles['Heading2']))
    for row in plan['weights']:
        story.append(Paragraph(escape(f"{row['partner']} / {row['segment']}: {row['weight']:.1%} of population"),styles['BodyText']))
    story += [Spacer(1, 12), Paragraph('Historical inputs', styles['Heading2']), Paragraph('One row per Partner, Segment, option and metric. Variance is the sample variance supplied by the data owner.', styles['BodyText'])]
    history_rows = [['Partner / Segment', 'Option', 'Metric', 'Mean', 'Variance', 'N']]
    for r in plan['history']:
        history_rows.append([Paragraph(escape(f"{r['partner']} / {r['segment']}"), styles['BodyText']), Paragraph(escape(str(r['arm'])), styles['BodyText']), Paragraph(escape(str(r['metric'])), styles['BodyText']), f"{r['sample_mean']:.4g}", f"{r['sample_variance']:.4g}", str(int(r['sample_number']))])
    history_table = Table(history_rows, repeatRows=1, colWidths=[112,100,70,50,55,50], hAlign='LEFT')
    history_table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),font), ('FONTSIZE',(0,0),(-1,-1),8), ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f8e7e6')), ('GRID',(0,0),(-1,-1),.4,colors.lightgrey), ('VALIGN',(0,0),(-1,-1),'TOP')]))
    story.append(history_table)
    SimpleDocTemplate(output, pagesize=A4).build(story)
    return output.getvalue()


def _run_plan(data, partners, segments, control, tests, primary, guardrail, effect, scale, alpha, power, multiplicity, margin, source, raw, grouping):
    if not tests:
        st.session_state.customer_plan_error = 'Choose at least one Test option.'
        return
    try:
        plan = calculate_plan(data, partners, segments, control, tests, primary, guardrail, effect, scale, alpha, power, multiplicity, margin)
        plan['config']['synthetic'] = source == 'Synthetic demo'
        plan['input_source'] = source
        plan['input_grouping'] = grouping
        st.session_state.customer_plan_draft = dict(source=source, raw=raw.copy(), partners=partners, segments=segments, control=control, tests=tests, primary=primary, guardrail=guardrail, effect=effect, scale=scale, alpha=alpha, power=power, multiplicity=multiplicity, margin=margin, grouping=grouping)
        st.session_state.customer_plan_result = plan
        st.session_state.customer_plan_result_view = True
        st.session_state.customer_plan_error = ''
    except (ValueError, OverflowError) as exc:
        st.session_state.customer_plan_error = str(exc)


def render_customer_plan():
    st.markdown('### Plan an experiment')
    st.caption('Choose historical statistics, your audience and test options. We will calculate the customers needed for each group.')
    if st.session_state.get('customer_plan_result_view'):
        plan = st.session_state['customer_plan_result']
        st.markdown('#### Your sample size plan')
        if plan['config'].get('synthetic'):
            st.warning('Synthetic demo: these numbers are illustrative.')
        control = control_name(plan['config'])
        roles = {arm['name']: arm['role'] for arm in plan['config']['arms']}
        totals = pd.DataFrame(plan['arm_totals'])[['arm','recruitment_n']].rename(columns={'arm':'Option','recruitment_n':'Required customers'})
        totals.insert(0, 'Role', totals['Option'].map(roles))
        total = int(totals['Required customers'].sum())
        totals['Allocation'] = totals['Required customers'].map(lambda n: f'{n/total:.1%}')
        control_total = int(totals.loc[totals['Role'].eq('Control'), 'Required customers'].iloc[0])
        control_metric, total_metric = st.columns(2)
        control_metric.metric('Control customers needed', f'{control_total:,}', help=f'Control option: {control}.')
        total_metric.metric('Total customers needed', f'{total:,}')
        st.caption(f'Control option: {control}. The total includes the Control and all Test options.')
        st.dataframe(totals, hide_index=True, use_container_width=True)
        m=plan['config']['metrics'][0]
        st.info(f"Primary metric: {m['name']} · MDE: {m['effect']:g} {m['effect_scale']} · Power: {plan['config']['power']:.0%} · Confidence level: {1-plan['config']['alpha']:.0%}")
        st.caption('Historical Control counts define the assumed Partner / Segment mix. Keep this same mix across all groups. Historical option differences are not assumed to be causal. Power is per comparison; estimates depend on historical variances.')
        with st.expander('Population mix and sample detail'):
            st.dataframe(pd.DataFrame(plan['weights']), hide_index=True, use_container_width=True)
            st.dataframe(pd.DataFrame(plan['targets'])[['partner','segment','arm','recruitment_n']], hide_index=True, use_container_width=True)
        a,b,c=st.columns(3)
        a.download_button('Download PDF summary', pdf_summary(plan), 'customer-plan.pdf', 'application/pdf', use_container_width=True)
        b.download_button('Download sample sizes', totals.to_csv(index=False), 'sample-size-plan.csv', 'text/csv', use_container_width=True)
        def back_to_settings():
            st.session_state.customer_plan_result_view = False
        c.button('Back to settings', use_container_width=True, on_click=back_to_settings)
        st.markdown('#### Save experiment')
        name=st.text_input('Experiment name', value=st.session_state.get('customer_saved_name','Customer experiment'), key='customer_plan_name')
        if st.button('Save to Your experiments', type='primary'):
            if not name.strip():
                st.error('Enter an experiment name.')
            else:
                p=plan.copy(); p['config']={**plan['config'],'name':name.strip()}
                try:
                    display_name = ('Demo · ' if p['config'].get('synthetic') and not name.strip().startswith('Demo · ') else '') + name.strip()
                    previous = next((item for item in load_experiments() if item.get('id') == p['config']['experiment_id']), {})
                    collected = int(previous.get('sample_collected', 0))
                    upsert_experiment({**previous, 'id':p['config']['experiment_id'], 'name':display_name, 'type':'Customer', 'status':previous.get('status', 'Planning'), 'sample_target':total, 'sample_collected':collected, 'progress':min(int(100*collected/total),100), 'action':'Update sample progress', 'primary_outcome':m['name'], 'config_payload':{'customer_plan':p}, 'description':'Synthetic demo' if p['config'].get('synthetic') else 'Customer sample size plan'})
                    st.success('Saved to Your experiments. Progress starts at zero and needs an update from experiment data.')
                except (ValueError, OSError) as exc:
                    st.error(f'Could not save experiment: {exc}')
        return

    draft = st.session_state.get('customer_plan_draft', {})
    sources = ['Upload file', 'Databricks', 'Synthetic demo']
    source = st.radio('Data source', sources, index=sources.index(draft.get('source', sources[0])), horizontal=True, key='customer_plan_source')
    if source == 'Databricks':
        st.info('Databricks is not connected yet. Choose Upload file or Synthetic demo to continue.')
        return
    metric_default='Primary metric'
    if source == 'Synthetic demo':
        st.info('Synthetic example: credit line strategies, FICO groups and PD score groups. Figures are illustrative.')
        raw=demo_data()
    else:
        st.download_button('Download CSV template', demo_data().head(6).to_csv(index=False), 'customer-history-template.csv', 'text/csv')
        st.caption('Each row describes a Partner, strategy, metric and combination of grouping columns. Use a consistent observation window and non-overlapping customers.')
        input_method=st.radio('How would you like to add data?', ['Upload CSV / Excel', 'Enter or paste data'], horizontal=True)
        if input_method == 'Upload CSV / Excel':
            upload=st.file_uploader('Upload historical statistics', type=['csv','xlsx','xlsm'])
            if upload is None and draft.get('source') == 'Upload file' and isinstance(draft.get('raw'), pd.DataFrame):
                raw = draft['raw'].copy()
                st.caption('Using the data from your previous calculation. Upload a new file to replace it.')
            elif upload is None:
                st.caption('Each row is one Partner × Segment × Strategy × Metric. Excel: .xlsx or .xlsm.')
                return
            else:
                try:
                    raw=pd.read_excel(BytesIO(upload.getvalue()), dtype=str, keep_default_na=False, engine='openpyxl') if upload.name.lower().endswith(('.xlsx','.xlsm')) else read_table(upload.getvalue())
                except (ValueError, UnicodeError, ImportError, OSError) as exc:
                    st.error(f'Could not read file: {exc}')
                    return
        else:
            input_style=st.radio('Input method', ['Editable table', 'Paste CSV / tab-separated'], horizontal=True)
            if input_style == 'Editable table':
                raw=st.data_editor(pd.DataFrame([['','','','','',None,None,None]],columns=COLUMNS), num_rows='dynamic', hide_index=True, use_container_width=True, key='customer_plan_editor')
            else:
                pasted=st.text_area('Paste table with a header row', height=140)
                if not pasted.strip():
                    return
                try: raw=read_table(pasted)
                except (ValueError, UnicodeError) as exc:
                    st.error(f'Could not read pasted data: {exc}'); return
        if not any(str(c).strip().lower() == 'metric' for c in raw.columns):
            metric_default=st.text_input('Name of the metric in this file', value='Primary metric')
    try:
        input_data = _canonical_input(raw, metric_default)
        if 'partner' not in input_data:
            raise ValueError('Missing column: partner')
        if input_data['partner'].isna().any():
            raise ValueError('Fill in Partner for each row.')
        is_legacy = 'group_by' in input_data and 'segment' in input_data
        candidates = grouping_columns(raw)
        groups = list(input_data['group_by'].dropna().astype(str).str.strip().unique()) if is_legacy else [label for label, _ in candidates]
        if not groups:
            raise ValueError('Add a grouping column such as Segment, FICO group or PD score group. Continuous scores must be grouped first.')
    except ValueError as exc:
        st.warning(str(exc)); return
    left,right=st.columns(2)
    partner_options = list(input_data['partner'].astype(str).str.strip().unique())
    partners=left.multiselect('Partners', partner_options, default=[p for p in draft.get('partners', partner_options) if p in partner_options])
    grouping=right.selectbox('Group by', groups, index=groups.index(draft.get('grouping')) if draft.get('grouping') in groups else 0)
    allow_pool = source == 'Synthetic demo'
    if not is_legacy and len(candidates) > 1 and source != 'Synthetic demo':
        allow_pool = st.checkbox('The rows being combined contain different, non-overlapping customers', value=False,
                                 key='customer_nonoverlap_' + fingerprint({'data': raw.to_csv(index=False), 'grouping': grouping})[:16],
                                 help='Required when the selected grouping combines multiple rows across other columns. Do not add monthly snapshots of the same customers.')
    try:
        data=normalize_table(raw,metric_default,grouping=grouping,allow_pool=allow_pool)
    except ValueError as exc:
        st.warning(str(exc)); return
    available=list(data[data.partner.isin(partners)].segment.unique())
    segments=st.multiselect('Segments to include', available, default=[s for s in draft.get('segments', available) if s in available])
    view=data[data.partner.isin(partners)&data.segment.isin(segments)]
    if view.empty:
        st.info('Select at least one Partner and segment.'); return
    st.caption('Each historical count belongs to a Strategy and Metric. Counts across options, metrics or time periods may overlap and are not necessarily unique customers.')
    st.markdown('**Historical data overview**')
    st.dataframe(view.groupby(['partner','segment','strategy','metric'],as_index=False).sample_number.sum().rename(columns={'sample_number':'historical sample number'}), hide_index=True, use_container_width=True)
    with st.expander('Uploaded data preview'):
        st.dataframe(raw, hide_index=True, use_container_width=True)
        month_col = next((c for c in raw.columns if str(c).strip().lower() in ('ym', 'month', 'year_month')), None)
        if month_col is not None:
            st.write('Months represented:', ', '.join(sorted(raw[month_col].dropna().astype(str).unique())))
            st.caption('Month is informational. Multiple monthly rows for the same combination must be combined outside the app after checking for overlapping customers.')
    st.markdown('### Metrics')
    names=list(view.metric.unique())
    x,y=st.columns(2)
    primary=x.selectbox('Primary metric', names, index=names.index(draft.get('primary')) if draft.get('primary') in names else 0)
    guardrail_options = ['None']+[n for n in names if n != primary]
    guardrail=y.selectbox('Guardrail metric (optional)', guardrail_options, index=guardrail_options.index(draft.get('guardrail')) if draft.get('guardrail') in guardrail_options else 0)
    guardrail=None if guardrail=='None' else guardrail
    st.markdown('### Control and test options')
    options=list(view.strategy.unique())
    if len(options)<2:
        st.warning('Provide historical statistics for at least two Strategy / Option values.'); return
    a,b=st.columns(2)
    control=a.selectbox('Control', options, index=options.index(draft.get('control')) if draft.get('control') in options else 0)
    choices=[x for x in options if x != control]
    tests=b.multiselect('Test options', choices, default=[x for x in draft.get('tests', choices[:1]) if x in choices])
    mde_col, scale_col=st.columns(2)
    scales=['Relative %','Absolute']
    scale=scale_col.selectbox('MDE type', scales, index=scales.index(draft.get('scale')) if draft.get('scale') in scales else 0)
    effect=mde_col.number_input('Minimum detectable effect (MDE)', min_value=0.001, value=float(draft.get('effect',5.0)), help='Smallest change worth detecting, not a prediction of the outcome.')
    with st.expander('Advanced settings'):
        a,b=st.columns(2)
        alpha=a.number_input('False positive rate (alpha)', min_value=.001,max_value=.2,value=float(draft.get('alpha',.05)),step=.01,format='%.3f')
        power=b.number_input('Detection chance (power)', min_value=.5,max_value=.99,value=float(draft.get('power',.8)),step=.05)
        multiplicity=st.checkbox('Protect against multiple comparisons', value=draft.get('multiplicity',True))
        st.caption('Default: pairwise Neyman allocation with a shared Control; historical Control customer counts determine the population mix across arms.')
        margin=st.number_input('Guardrail maximum tolerated deterioration (absolute units; 0 = monitoring only)', min_value=0.,value=float(draft.get('margin',0.))) if guardrail else 0.
    st.markdown('### Historical data visualizations')
    st.caption('Historical differences describe the supplied data; they do not establish an effect caused by the strategy. Sample counts are shown separately by Partner and option.')
    chart_data = view[view.metric.eq(primary) & view.strategy.isin([control] + tests)].copy()
    if not chart_data.empty:
        chart_data['historical_sd'] = np.sqrt(chart_data.sample_variance)
        for title, value, label in [('Historical sample by group', 'sample_number', 'Historical sample number'),
                                    ('Historical primary metric by strategy', 'sample_mean', f'{primary} historical mean'),
                                    ('Historical variability by strategy', 'historical_sd', f'{primary} historical standard deviation')]:
            st.markdown(f'**{title}**')
            fig=px.bar(chart_data, x='segment', y=value, color='strategy', barmode='group', facet_col='partner', facet_col_wrap=2,
                       labels={'segment': grouping, 'strategy':'Strategy', value:label},
                       color_discrete_sequence=['#e85959', '#16856b', '#d69a1f', '#6384c5'])
            fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split('=', 1)[-1]))
            fig.update_layout(height=360 if len(chart_data.partner.unique()) <= 2 else 590, margin=dict(t=40,b=45,l=30,r=20), legend_title_text='Strategy')
            st.plotly_chart(fig, use_container_width=True, key=f'customer_history_chart_{value}')
    else:
        st.info('Select at least one Test option to compare historical data.')
    if st.session_state.get('customer_plan_error'):
        st.error(st.session_state.customer_plan_error)
    st.button('Get plan', type='primary', use_container_width=True, on_click=_run_plan,
              args=(data, partners, segments, control, tests, primary, guardrail, effect, scale, alpha, power, multiplicity, margin, source, raw, grouping))
