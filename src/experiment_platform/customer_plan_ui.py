"""Single customer planning flow using explicit historical strategy statistics."""
from __future__ import annotations

from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from .business_config import control_name, default_config, fingerprint
from .business_data import canonical, read_table
from .business_plan import build_plan
from .business_store import code_fingerprint
from .customer_demo import historical_demo
from .customer_palette import STRATEGY_COLORS, strategy_styles
from .portfolio import load_experiments, upsert_experiment

COLUMNS = ['Partner', 'FICO group', 'PD score group', 'Strategy', 'Metric', 'Sample mean', 'Sample variance', 'Sample number']
def demo_data():
    return historical_demo()


def historical_metric_chart(data, grouping, primary, options):
    """Compare ordered options within each Partner and segment, keeping gaps visible."""
    partners = list(data.partner.unique())
    segments = list(data.segment.unique())
    index = pd.MultiIndex.from_product([partners, segments, options], names=['partner', 'segment', 'strategy'])
    rows = data.set_index(['partner', 'segment', 'strategy']).reindex(index).reset_index()
    fig = px.line(rows, x='strategy', y='sample_mean', color='segment', line_group='segment',
                  facet_col='partner', facet_col_wrap=2, markers=True,
                  category_orders={'strategy': options, 'segment': segments, 'partner': partners},
                  labels={'strategy': 'Control and Test options', 'segment': grouping,
                          'sample_mean': f'{primary} historical mean', 'partner': 'Partner'},
                  color_discrete_sequence=STRATEGY_COLORS)
    fig.update_traces(connectgaps=False)
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split('=', 1)[-1]))
    fig.update_layout(height=360 if len(partners) <= 2 else 590, margin=dict(t=40, b=70, l=30, r=20),
                      legend_title_text=grouping)
    return fig


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
        # Keep numeric scores available; the UI requests bins for continuous values.
        if len(series) == 0:
            continue
        preferred = any(x in col for x in ('segment', 'fico', 'risk', 'group', 'band', 'score'))
        candidates.append((0 if preferred else 1, str(original), col))
    return [(label, col) for _, label, col in sorted(candidates, key=lambda x: x[0])]


def bin_group_column(frame, column, boundaries):
    """Apply user-supplied upper boundaries; never infer score bins."""
    values = pd.to_numeric(frame[column], errors='coerce')
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError('All values in the selected score column must be numeric and non-missing.')
    try:
        cuts = [float(part.strip()) for part in boundaries.split(',') if part.strip()]
    except ValueError:
        raise ValueError('Enter numeric group boundaries separated by commas.') from None
    if not cuts or not np.isfinite(cuts).all() or any(a >= b for a, b in zip(cuts, cuts[1:])):
        raise ValueError('Enter one or more strictly increasing numeric boundaries.')
    labels = [f'<= {cuts[0]:g}'] + [f'{a:g} to {b:g}' for a, b in zip(cuts, cuts[1:])] + [f'> {cuts[-1]:g}']
    result = frame.copy()
    original_key = f'__original_{column}'
    if original_key in result:
        raise ValueError('The file contains a reserved original-score column name.')
    result[original_key] = result[column]
    result[column] = pd.cut(values, [-np.inf, *cuts, np.inf], labels=labels, right=True).astype(str)
    return result


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


def demo_overall_summary(frame, by_partner=False):
    """Summarize disjoint demo cells separately for each strategy and metric."""
    data = _canonical_input(frame)
    keys = (['partner'] if by_partner else []) + ['strategy', 'metric']
    result = pool_statistics(data, keys)
    result['standard_deviation'] = np.sqrt(result['sample_variance'])
    return result.drop(columns='sample_variance').rename(columns={
        'partner': 'Partner', 'strategy': 'Strategy', 'metric': 'Metric',
        'sample_mean': 'Historical mean', 'sample_number': 'Historical sample number',
        'standard_deviation': 'Historical standard deviation',
    })


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
        dimensions = ['segment'] + [c for c in dimensions if c != mapping[grouping] and d[c].notna().all()]
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
    from .customer_report import build_design_package
    return build_design_package(plan)


def _run_plan(data, partners, segments, control, tests, primary, guardrail, effect, scale, alpha, power, multiplicity, margin, source, raw, grouping, boundaries):
    if not tests:
        st.session_state.customer_plan_error = 'Choose at least one Test option.'
        return
    try:
        plan = calculate_plan(data, partners, segments, control, tests, primary, guardrail, effect, scale, alpha, power, multiplicity, margin)
        plan['config']['synthetic'] = source == 'Synthetic demo'
        plan['input_source'] = source
        plan['input_grouping'] = grouping
        plan['group_boundaries'] = boundaries
        st.session_state.customer_plan_draft = dict(source=source, raw=raw.copy(), partners=partners, segments=segments, control=control, tests=tests, primary=primary, guardrail=guardrail, effect=effect, scale=scale, alpha=alpha, power=power, multiplicity=multiplicity, margin=margin, grouping=grouping, boundaries=boundaries)
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
        name=st.text_input('Experiment name', value=plan['config'].get('name', 'Customer experiment'), key='customer_plan_name',
                           help='This name appears in the PDF and in your saved experiments.')
        report_plan = {**plan, 'config': {**plan['config'], 'name': name.strip() or plan['config']['name']}}
        a,b,c=st.columns(3)
        a.download_button('Download design package (PDF)', pdf_summary(report_plan), 'experiment-design-package.pdf', 'application/pdf', use_container_width=True)
        b.download_button('Download sample sizes', totals.to_csv(index=False), 'sample-size-plan.csv', 'text/csv', use_container_width=True)
        def back_to_settings():
            st.session_state.customer_plan_result_view = False
        c.button('Back to settings', use_container_width=True, on_click=back_to_settings)
        st.markdown('#### Save experiment')
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
    source = st.radio('Data source', sources, index=sources.index(draft.get('source', 'Synthetic demo')), horizontal=True, key='customer_plan_source')
    if source == 'Databricks':
        st.info('Databricks is not connected yet. Choose Upload file or Synthetic demo to continue.')
        return
    metric_default='Primary metric'
    if source == 'Synthetic demo':
        st.info('Synthetic example: credit line strategies, FICO groups and PD score groups. Metrics are simplified illustrations, not calibrated portfolio estimates.')
        raw=demo_data()
        st.markdown('**Data preview**')
        st.dataframe(raw.iloc[np.linspace(0, len(raw) - 1, 12, dtype=int)], hide_index=True, use_container_width=True)
        st.caption('Demo rows within each Strategy and Metric represent non-overlapping customers.')
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
        if raw.empty:
            st.info('Add historical statistics to continue.')
            return
        st.markdown('**Data preview**')
        st.dataframe(raw.head(15), hide_index=True, use_container_width=True)
        try:
            detected = _canonical_input(raw)
        except ValueError as exc:
            st.warning(str(exc)); return
        original_columns = list(raw.columns)
        normalized_columns = list(detected.columns)[:len(original_columns)]
        fields = [('Partner', 'partner'), ('Strategy / Option', 'strategy'), ('Metric', 'metric'), ('Sample mean', 'sample_mean'), ('Sample variance', 'sample_variance'), ('Sample number', 'sample_number')]
        column_choices = ['Not provided'] + original_columns
        mapping = {}
        with st.expander('Review or change column mapping', expanded=False):
            st.caption('Select the matching column for each field. Metric may be omitted when the file contains one metric.')
            for label, field in fields:
                guess = original_columns[normalized_columns.index(field)] if field in normalized_columns else 'Not provided'
                chosen = st.selectbox(label, column_choices, index=column_choices.index(guess), key='customer_map_' + field + '_' + fingerprint({'columns': [str(c) for c in original_columns]})[:10])
                if chosen != 'Not provided':
                    if chosen in mapping:
                        st.error('Choose a different column for each field.'); return
                    mapping[chosen] = field
        raw = raw.rename(columns=mapping)
        if 'metric' not in raw:
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
    st.markdown('**Choose your grouping column**')
    st.caption('Select one column from the data above. The available group values and historical statistics update when you change it.')
    if not is_legacy:
        st.caption('Suggested columns: ' + ', '.join(groups[:6]))
    left,right=st.columns(2)
    partner_options = list(input_data['partner'].astype(str).str.strip().unique())
    partners=left.multiselect('Partners', partner_options, default=[p for p in draft.get('partners', partner_options) if p in partner_options])
    default_group = draft.get('grouping') if draft.get('grouping') in groups else ('FICO group' if source == 'Synthetic demo' and 'FICO group' in groups else None)
    grouping=right.selectbox('Group by', groups, index=groups.index(default_group) if default_group else None,
                             placeholder='Choose a column, such as FICO group or PD score group')
    if grouping is None:
        st.info('Choose a Group by column to see its segments and historical statistics.')
        return
    raw_before_binning = raw.copy()
    boundaries = ''
    if not is_legacy:
        selected_values = raw[grouping].dropna()
        numeric_values = pd.to_numeric(selected_values, errors='coerce')
        if len(selected_values) and numeric_values.notna().all() and selected_values.nunique() > 20:
            st.info(f'{grouping} contains many numeric values. Enter boundaries to group this continuous score; each boundary is included in the lower group.')
            boundaries = st.text_input('Group boundaries', value=draft.get('boundaries', '') if grouping == draft.get('grouping') else '', placeholder='For example: 0.02, 0.05, 0.10', key='customer_score_bins_' + fingerprint({'column':grouping})[:10])
            if not boundaries.strip():
                return
            try:
                raw = bin_group_column(raw, grouping, boundaries)
            except ValueError as exc:
                st.warning(str(exc)); return
        st.caption(f'Grouping by the column {grouping}. Available values: ' + ', '.join(str(value) for value in raw[grouping].dropna().unique()[:12]))
    allow_pool = source == 'Synthetic demo'
    if not is_legacy and source != 'Synthetic demo':
        normalized = _canonical_input(raw, metric_default)
        group_field = dict(grouping_columns(raw)).get(grouping)
        pool_keys = ['partner', group_field, 'strategy', 'metric']
        if all(key in normalized.columns for key in pool_keys):
            combining = normalized.duplicated(pool_keys, keep=False)
            if combining.any():
                st.caption('These rows share the selected Partner, group, strategy and metric and would be combined. Previewing up to 30 rows.')
                st.dataframe(raw.loc[combining].head(30), hide_index=True, use_container_width=True)
                allow_pool = st.checkbox('The rows being combined contain different, non-overlapping customers', value=False,
                                         key='customer_nonoverlap_' + fingerprint({'data': raw.to_csv(index=False), 'grouping': grouping})[:16],
                                         help='Do not combine monthly snapshots of the same customers.')
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
    month_col = next((c for c in raw.columns if str(c).strip().lower() in ('ym', 'month', 'year_month')), None)
    if month_col is not None:
        st.caption('Month is informational. Check for overlapping customers before combining monthly rows.')
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
        chart_options = [control] + list(tests)
        colors, patterns = strategy_styles([control] + sorted((option for option in options if option != control), key=str.casefold))
        if len(chart_options) > len(STRATEGY_COLORS):
            st.caption('For more than 24 options, patterns supplement the colors. Use the legend to focus on individual options.')
        chart_data['historical_sd'] = np.sqrt(chart_data.sample_variance)
        for title, value, label in [('Historical sample by group', 'sample_number', 'Historical sample number'),
                                    ('Historical primary metric by strategy', 'sample_mean', f'{primary} historical mean'),
                                    ('Historical variability by strategy', 'historical_sd', f'{primary} historical standard deviation')]:
            st.markdown(f'**{title}**')
            if value == 'sample_mean':
                st.caption('Each line is a historical segment within a Partner. Lines connect selected options for comparison; they do not imply a causal effect or values between options.')
                fig = historical_metric_chart(chart_data, grouping, primary, chart_options)
            else:
                fig=px.bar(chart_data, x='segment', y=value, color='strategy', pattern_shape='strategy', barmode='group', facet_col='partner', facet_col_wrap=2,
                           labels={'segment': grouping, 'strategy':'Strategy', value:label},
                           category_orders={'strategy': chart_options}, color_discrete_map=colors, pattern_shape_map=patterns)
                fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split('=', 1)[-1]))
                fig.update_layout(height=360 if len(chart_data.partner.unique()) <= 2 else 590, margin=dict(t=40,b=45,l=30,r=20), legend_title_text='Strategy')
            st.plotly_chart(fig, use_container_width=True, key=f'customer_history_chart_{value}')
    else:
        st.info('Select at least one Test option to compare historical data.')
    if st.session_state.get('customer_plan_error'):
        st.error(st.session_state.customer_plan_error)
    st.button('Get plan', type='primary', use_container_width=True, on_click=_run_plan,
              args=(data, partners, segments, control, tests, primary, guardrail, effect, scale, alpha, power, multiplicity, margin, source, raw_before_binning, grouping, boundaries))
