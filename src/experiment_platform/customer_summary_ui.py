"""Business-facing analysis of post-experiment aggregate statistics."""
from __future__ import annotations
from io import BytesIO
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from .business_data import read_table
from .customer_demo import customer_demo_results
from .customer_decision import decision_context, rule_based_decision
from .customer_palette import strategy_styles
from .customer_summary_results import normalize_summary, calculate_summary
from .portfolio import configuration_fingerprint
from .customer_results_report import build_results_report


@st.cache_data(show_spinner=False)
def summary_demo():
    """Summarize the population-specific credit-line response demo."""
    # All rows summarize distinct experimental units within a metric; the
    # underlying simulated records are not retained in the resulting dataset.
    frame = customer_demo_results(n=72000)
    keys = ['Partner','FICO group','PD score group','Assigned option']
    rows = []
    for metric in ['Revolving Balance','RAR','NCL']:
        result = frame.groupby(keys,sort=False)[metric].agg(**{'Sample mean':'mean','Sample variance':'var','Sample number':'size'}).reset_index()
        result['Metric'] = metric
        rows.append(result)
    output = pd.concat(rows,ignore_index=True).rename(columns={'Assigned option':'Strategy'})
    output[['Sample mean','Sample variance']] = output[['Sample mean','Sample variance']].round(6)
    return output[['Partner','FICO group','PD score group','Strategy','Metric','Sample mean','Sample variance','Sample number']]


def render_summary_analysis(source):
    from .customer_ui import average_chart, _result_rows, render_decision_card
    demo = summary_demo()
    if source == 'Synthetic demo':
        frame = demo
    else:
        st.download_button('Download summary CSV template', demo.head(12).to_csv(index=False), 'customer-results-summary-template.csv', 'text/csv', key='summary_template')
        st.caption('Provide strategy, metric, sample mean, sample variance and sample number. Partner and grouping columns are optional.')
        mode = st.radio('How would you like to add data?', ['Upload CSV / Excel','Enter or paste data'], horizontal=True, key='summary_input_method')
        if mode == 'Upload CSV / Excel':
            uploaded = st.file_uploader('Upload experiment summary statistics', type=['csv','xlsx','xlsm'], key='summary_upload')
            if uploaded is None:
                st.info('Upload one row for each strategy, metric and population combination.'); return
            try:
                frame = pd.read_excel(BytesIO(uploaded.getvalue()), dtype=str, keep_default_na=False, engine='openpyxl') if uploaded.name.lower().endswith(('.xlsx','.xlsm')) else read_table(uploaded.getvalue())
            except (ValueError,UnicodeError,pd.errors.ParserError,ImportError,OSError) as exc:
                st.error(f'Could not read the file: {exc}'); return
        else:
            how = st.radio('Input method', ['Editable table','Paste CSV / tab-separated'], horizontal=True, key='summary_input_style')
            if how == 'Editable table':
                frame = st.data_editor(pd.DataFrame([{'Partner':'','Segment':'','Strategy':'','Metric':'',
                    'Sample mean':None,'Sample variance':None,'Sample number':None}]), num_rows='dynamic',
                    hide_index=True, use_container_width=True, key='summary_editor')
            else:
                pasted = st.text_area('Paste summary data with a header row',height=140,key='summary_paste')
                if not pasted.strip(): return
                try: frame = read_table(pasted)
                except (ValueError,UnicodeError,pd.errors.ParserError) as exc:
                    st.error(f'Could not read pasted data: {exc}'); return
    st.markdown('**Data preview**')
    visible_columns = [col for col in frame.columns if frame[col].notna().any() and frame[col].fillna('').astype(str).str.strip().ne('').any()]
    st.dataframe(frame.loc[:, visible_columns].head(12), hide_index=True, use_container_width=True)
    try: data, dimensions = normalize_summary(frame)
    except ValueError as exc:
        st.warning(str(exc)); return
    partners = next((col for col in dimensions if col.lower() == 'partner'), None)
    if partners:
        available = data[partners].drop_duplicates().tolist()
        chosen = st.multiselect('Partners',available,default=available,key='summary_partners')
        data = data[data[partners].isin(chosen)]
        if data.empty: st.info('Select at least one Partner.'); return
    groups = ['No segment breakdown'] + dimensions
    default_group = next((name for name in dimensions if name.lower() == 'fico group'), None)
    left, right = st.columns(2)
    metrics = data.metric.drop_duplicates().tolist()
    preferred = 'Revolving Balance' if 'Revolving Balance' in metrics else None
    saved_primary = st.session_state.get('summary_primary_saved')
    default_primary = saved_primary if saved_primary in metrics else preferred
    def remember_primary():
        st.session_state.summary_primary_saved = st.session_state.summary_primary
    primary = left.selectbox('Primary metric',metrics,index=metrics.index(default_primary) if default_primary else None,
                             placeholder='Choose a metric',key='summary_primary',on_change=remember_primary)
    if primary is not None:
        st.session_state.summary_primary_saved = primary
    group = right.selectbox('Group by (optional)',groups,index=groups.index(default_group) if source == 'Synthetic demo' and default_group else 0,key='summary_group')
    if primary is None: st.info('Choose the primary metric.'); return
    choices = [m for m in metrics if m != primary]
    slots = st.session_state.setdefault('summary_extra_slots',[])
    def add():
        counter = st.session_state.get('summary_next_slot',0)
        st.session_state.summary_extra_slots.append(counter)
        st.session_state.summary_next_slot = counter + 1
    def remove(slot): st.session_state.summary_extra_slots.remove(slot)
    st.button('＋ Add metric',on_click=add,disabled=len(slots)>=len(choices),key='summary_add_metric')
    # Reset selections that disappeared from the uploaded data or became the
    # primary metric; keep surviving selections when the dataset changes.
    used = set()
    for slot in slots:
        key = f'summary_extra_{slot}'
        value = st.session_state.get(key)
        if value is not None and (value not in choices or value in used):
            st.session_state[key] = None
        elif value is not None:
            used.add(value)
    # Initialize new slots before rendering any selectboxes, so every dropdown
    # excludes selections made by later slots on this same run.
    for slot in slots:
        key = f'summary_extra_{slot}'
        if key not in st.session_state:
            suggested = next((x for x in ['RAR','NCL'] if x in choices and x not in used),None)
            st.session_state[key] = suggested
            if suggested is not None:
                used.add(suggested)
    extra = []
    for position, slot in enumerate(slots, 1):
        a,b = st.columns([5,1])
        key = f'summary_extra_{slot}'
        old = st.session_state.get(key)
        other_selected = {st.session_state.get(f'summary_extra_{other}') for other in slots if other != slot}
        available_metrics = [metric for metric in choices if metric not in other_selected]
        suggested = old if old in available_metrics else next((x for x in ['RAR','NCL'] if x in available_metrics),None)
        selected = a.selectbox(f'Additional metric {position}',available_metrics,
                               index=available_metrics.index(suggested) if suggested else None,
                               key=key,placeholder='Choose a metric')
        b.button('Remove',key=f'summary_remove_{slot}',on_click=remove,args=(slot,))
        if selected is not None: extra.append(selected)
    if len(extra)!=len(slots) or len(extra)!=len(set(extra)):
        st.info('Choose a different metric for each additional selection.'); return
    if extra: st.caption('Additional metrics are exploratory. Within each metric, multiple Tests follow the selected protection; comparisons are not adjusted across metrics.')
    selected_metrics = [primary]+extra
    options = data[data.metric.eq(primary)].strategy.drop_duplicates().tolist()
    if len(options)<2: st.warning('Provide at least two strategies for the primary metric.'); return
    preferred_control = next((x for x in options if x.lower() in {'control','bau','current','credit line 1000'}),None)
    control = st.selectbox('Control strategy',options,index=options.index(preferred_control) if preferred_control else None,
                           placeholder='Choose the Control strategy',key='summary_control')
    if control is None: st.info('Choose which strategy is Control.'); return
    with st.expander('Advanced settings'):
        a,b = st.columns(2)
        alpha = a.number_input('Significance level',min_value=.001,max_value=.2,value=.05,step=.01,format='%.3f',key='summary_alpha')
        protection = b.selectbox('Multiple Test protection',['Holm','None'],key='summary_protection')
        directions = {}
        for metric in selected_metrics:
            lower = any(x in metric.lower() for x in ('ncl','loss','default','cost'))
            directions[metric] = st.selectbox(f'Preferred direction: {metric}',['Higher','Lower'],index=1 if lower else 0,
                key='summary_dir_'+configuration_fingerprint({'metric':metric})[:12])
    if source != 'Synthetic demo':
        confirmed = st.checkbox('I confirm the summary rows represent non-overlapping customers within each metric, and all strategies use a consistent outcome window.',key='summary_disjoint_confirmation')
        if not confirmed:
            st.info('Confirm the rows do not overlap before combining them. Repeated monthly snapshots can count the same customer more than once.'); return
    st.caption('Overall statistics pool the observed customers in each arm. Summary data cannot verify unique customer IDs, assignment integrity, missing outcomes, or the frozen planning mix.')
    config = {'data':data.to_json(orient='split'),'metrics':selected_metrics,'control':control,'group':group,'alpha':alpha,'protection':protection}
    fingerprint = configuration_fingerprint(config)
    if st.button('Calculate results',type='primary',key='summary_calculate'):
        try:
            analyses = {}
            for metric in selected_metrics:
                subset_options = set(data[data.metric.eq(metric)].strategy)
                if subset_options != set(options):
                    raise ValueError(f'{metric}: provide the same Control and Test strategies as the primary metric.')
                analyses[metric] = calculate_summary(data,metric,control,None if group == groups[0] else group,alpha,protection)
            st.session_state.summary_analyses = analyses
            st.session_state.summary_result_fingerprint = fingerprint
            st.session_state.summary_result_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        except (ValueError,OverflowError,ZeroDivisionError) as exc:
            st.error(str(exc)); return
    if st.session_state.get('summary_result_fingerprint') != fingerprint:
        if st.session_state.get('summary_result_fingerprint'): st.warning('Inputs changed. Calculate results again.')
        return
    colors, patterns = strategy_styles([control] + sorted((option for option in options if option != control),key=str.casefold))
    prepared = []
    for metric in selected_metrics:
        overall, subgroups, audit, issues = st.session_state.summary_analyses[metric]
        binary = audit['metric_type'] == 'Binary'
        key = configuration_fingerprint({'metric':metric})[:12]
        overall_rows = _result_rows(overall,'Overall',audit['metric_type'],data,'strategy',metric,control,None,alpha)
        segment_rows = [(str(label), result, _result_rows(result,label,audit['metric_type'],data,'strategy',metric,control,group,alpha))
                        for label,result in subgroups.items()]
        group_details = pd.concat([rows for _,_,rows in segment_rows],ignore_index=True) if segment_rows else overall_rows.iloc[0:0]
        context = decision_context(overall_rows,group_details,metric,control,directions[metric],alpha,protection,source=='Synthetic demo',binary=binary)
        prepared.append({'metric':metric,'overall':overall,'overall_rows':overall_rows,'segments':segment_rows,
                         'audit':audit,'issues':issues,'binary':binary,'key':key,'insight':rule_based_decision(context)})

    st.markdown('### Overall results')
    st.caption('Compare each strategy with Control. Whiskers show the confidence interval for the Test–Control difference, shifted to the average scale.')
    for item in prepared:
        metric = item['metric']
        st.markdown(f'#### {metric}' + (' · Primary metric' if metric == primary else ''))
        st.plotly_chart(average_chart(item['overall'],item['overall_rows'],item['binary'],metric,control,colors,patterns),
                        use_container_width=True,key='summary_chart_'+item['key'])

    if group != groups[0]:
        st.markdown(f'### Results by {group}')
        st.caption('Each segment is compared with its own Control. Charts for the same metric share a vertical scale. Segment comparisons are exploratory and unadjusted.')
        for item in prepared:
            metric = item['metric']
            st.markdown(f'#### {metric}')
            segment_charts = [(label, average_chart(result,rows,item['binary'],metric,control,colors,patterns))
                              for label,result,rows in item['segments']]
            # Independent axis scaling can make different groups look alike.
            # Use the full range required by every comparison within a metric.
            if segment_charts:
                ranges = [chart.layout.yaxis.range for _,chart in segment_charts]
                common_range = [min(r[0] for r in ranges), max(r[1] for r in ranges)]
                for label, chart in segment_charts:
                    chart.update_yaxes(range=common_range)
                    st.markdown(f'**{label}**')
                    st.plotly_chart(chart,use_container_width=True,
                                    key='summary_segment_'+item['key']+'_'+configuration_fingerprint({'group':label})[:12])
            for issue in item['issues']: st.warning(f'Not calculated — {issue}')

    st.markdown('### Decision insights')
    report_metrics = []
    for item in prepared:
        metric = item['metric']
        st.markdown(f'#### {metric}')
        if metric != primary: st.caption('Exploratory metric: no adjustment across metrics and no automatic Guardrail decision.')
        render_decision_card(item['insight'])
        exports = [item['overall_rows']] + [rows for _,_,rows in item['segments']]
        details = pd.concat(exports,ignore_index=True)
        details.insert(0,'Metric',metric)
        with st.expander(f'Detailed statistics · {metric}'):
            st.dataframe(details,hide_index=True,use_container_width=True)
            st.caption('Overall Holm p-values and Bonferroni intervals are adjusted across Tests within the metric when protection is enabled. Segments and extra metrics are exploratory.')
            st.download_button('Download analysis results (CSV)',details.to_csv(index=False),'customer-summary-analysis.csv','text/csv',key='summary_export_'+item['key'])
        report_metrics.append({'name':str(metric), 'role':'Primary metric' if metric == primary else 'Additional metric',
            'type':item['audit']['metric_type'], 'direction':directions[metric], 'overall':item['overall_rows'].to_dict('records'),
            'groups':[(label,rows.to_dict('records')) for label,_,rows in item['segments']], 'issues':item['issues'], 'insight':item['insight']})
    st.caption('Confirm randomization, SRM, sample completion, balance, missing outcomes and guardrails before deciding. Test-to-Test differences are descriptive.')

    st.markdown('### Final report')
    st.caption('Download a shareable report with all selected metrics and segment comparisons.')
    report_name = st.text_input('Experiment name for the report', value='Customer experiment', key='summary_report_name')
    report = {'name':report_name, 'source':source, 'input_format':'Summary statistics', 'synthetic':source == 'Synthetic demo',
              'control':control, 'partners':chosen if partners else [], 'group_by':None if group == groups[0] else group, 'alpha':alpha, 'protection':protection,
              'calculated_at':st.session_state.get('summary_result_timestamp'), 'fingerprint':fingerprint, 'metrics':report_metrics}
    st.download_button('Download final report (PDF)', build_results_report(report), 'customer-experiment-results-report.pdf',
                       'application/pdf', key='summary_pdf_download')
