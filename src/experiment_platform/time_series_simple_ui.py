"""Business-facing analysis of a completed time-series campaign."""
from datetime import datetime
from io import BytesIO
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from .business_data import read_table
from .core import load_timeseries_demo, normalize_uploaded_dataset, safe_datetime_series, safe_numeric_series
from .charts import time_series_line, pre_post_bar, bsts_counterfactual_chart, timeseries_chart_context
from .portfolio import configuration_fingerprint
from .customer_demo import PARTNERS, FICO_GROUPS
from .timeseries import (infer_timeseries_columns, prepare_timeseries_data, predictor_candidates,
                         run_pre_post_analysis, fit_bsts_model, data_fingerprint, method_config_fingerprint,
                         humanize_column_name)

SEGMENTS = FICO_GROUPS


@st.cache_data(show_spinner=False)
def analysis_demo():
    """Illustrative weekly Partner x Segment data aligned with customer demo labels."""
    base = load_timeseries_demo()
    rng = np.random.default_rng(4381)
    records = []
    for partner, pw, peffect in zip(PARTNERS, [.32,.27,.22,.19], [.01,.025,-.025,-.005]):
        for segment, sw, seffect in zip(SEGMENTS, [.30,.42,.28], [-.045,.015,.055]):
            flag = base['campaign_flag'].to_numpy()
            expected = np.maximum(base['applications'].to_numpy() * pw * sw * (1 + flag * (peffect + seffect)), 0)
            records.append(pd.DataFrame({'date':base['date'], 'Partner':partner, 'Segment':segment,
                'applications':rng.poisson(expected), 'campaign_flag':flag}))
    return pd.concat(records,ignore_index=True).sort_values(['date','Partner','Segment'],kind='stable').reset_index(drop=True)


def _safe_pct(value):
    try: return f'{float(value):+.1%}' if np.isfinite(float(value)) else 'Unavailable'
    except (ValueError, TypeError): return 'Unavailable'


def _grouping_columns(frame, date_col, metric, flag):
    """Suggest stable, low-cardinality business dimensions, not numeric metrics."""
    out = []
    for col in frame.columns:
        name = str(col).strip().lower()
        if col in {date_col,metric,flag} or any(x in name for x in ('campaign','date','month','week','id','name')):
            continue
        values = frame[col].fillna('').astype(str).str.strip()
        distinct = values[values.ne('')].nunique()
        likely = any(x in name for x in ('partner','segment','group','fico','band','channel','region','score'))
        numeric_share = safe_numeric_series(frame,col).notna().mean()
        if 1 <= distinct <= 30 and (likely or numeric_share < .2):
            out.append(col)
    return sorted(out, key=lambda x:(not any(t in str(x).lower() for t in ('segment','fico','group')),str(x)))


def _make_series(frame, date_col, metric, aggregation, predictors, missing):
    """Pool selected rows by date, retaining shared unaffected signals unchanged."""
    data = frame.copy()
    dates = safe_datetime_series(data,date_col)
    outcome = safe_numeric_series(data,metric)
    if dates.isna().any() or not np.isfinite(outcome).all():
        raise ValueError('Some rows have an invalid date or missing/non-numeric metric. Correct those rows before analysis.')
    data['_source_date'] = dates
    data['_source_value'] = outcome
    duplicates = dates.duplicated().any()
    if duplicates and aggregation not in {'Sum','Simple average'}:
        raise ValueError('Choose how to combine rows with the same date before calculating.')
    for name in predictors:
        values = safe_numeric_series(data,name)
        if not np.isfinite(values).all():
            raise ValueError(f'{name}: comparison signals must have a numeric value for every row.')
        data[name] = values
        # A shared external signal may repeat across Partner/Segment rows. Do
        # not sum or average divergent signals without knowing their meaning.
        if duplicates and (data.groupby('_source_date')[name].nunique(dropna=False) > 1).any():
            raise ValueError(f'{name}: provide one identical unaffected signal per date across the selected rows.')
    grouped = data.groupby('_source_date',as_index=False,sort=True).agg(
        **{'_source_value':('_source_value','mean' if aggregation == 'Simple average' else 'sum'),
           **{name:(name,'first') for name in predictors}})
    return prepare_timeseries_data(grouped,'_source_date','_source_value','Fail validation',missing,'Auto')


def _decision_insight(metric, prepost, model, model_error, segment=None):
    """Describe evidence conservatively using only results from the current run."""
    scope = f'For {segment}, ' if segment else ''
    before, after = prepost['pre_value'], prepost['post_value']
    relative = (after-before)/before if before > 0 else np.nan
    observed = f"{scope}the average {metric} changed from {before:,.2f} to {after:,.2f} per period"
    if np.isfinite(relative): observed += f' ({relative:+.1%})'
    observed += '. This before-and-after change is descriptive.'
    if model is None:
        return [observed, 'A model estimate is unavailable for this selection' + (f': {model_error}' if model_error else '.'),
                'Review the data or collect more history before using a model to support a decision.']
    low, high = model['prediction_interval']
    difference = model['cumulative_impact']
    reliability = model.get('reliability','Review')
    interval = f"{model['interval_level']:.0%} model interval {low:+,.2f} to {high:+,.2f}"
    if not all(np.isfinite(x) for x in (low,high,difference)):
        model_text = 'The model uncertainty is unavailable; do not rely on the estimated direction.'
    elif low <= 0 <= high:
        model_text = f'The estimated total difference is {difference:+,.2f} ({interval}); the interval includes zero, so its direction is uncertain.'
    else:
        model_text = f"The model estimates {abs(difference):,.2f} {'more' if difference > 0 else 'fewer'} total {metric} than its no-campaign baseline ({interval})."
    if model.get('expected_total',0)>0 and np.isfinite(model.get('relative_impact',np.nan)):
        model_text += f" Estimated relative change: {model['relative_impact']:+.1%}."
    if reliability == 'Poor':
        next_step = 'Improve the baseline or add an unaffected comparison signal before relying on the estimate.'
    elif np.isfinite(low) and np.isfinite(high) and low <= 0 <= high:
        next_step = 'Review the model and consider collecting more outcome data before deciding.'
    else:
        next_step = 'Compare the estimated change with business targets, costs and risk before deciding.'
    caution = f'Model validation: {reliability}. {next_step} Review other events and outcome maturity; the estimate depends on the model assumptions.'
    return [observed,model_text,caution]


def _insight_card(lines):
    body = ''.join('<p style="margin:.35rem 0;line-height:1.5">'+escape(line)+'</p>' for line in lines)
    st.markdown('<div style="border:1px solid #d1e4ed;border-radius:9px;background:#f7fbfd;padding:1rem 1.2rem;color:#153449">'
                '<div style="font-weight:750;color:#006b9b;font-size:.82rem">✦ &nbsp; RESULT INSIGHTS</div>'+body+'</div>',unsafe_allow_html=True)


def render_time_series_analysis():
    cfg = st.session_state.ts_config
    st.markdown('### Analyze results')
    source = st.radio('Data source',['Upload file','Databricks','Synthetic demo'],index=2,horizontal=True,key='ts_simple_source')
    st.session_state.ts_prepared_df = pd.DataFrame()
    if source == 'Databricks':
        st.info('Databricks is not connected yet. Choose Upload file or Synthetic demo to continue.'); return
    demo = analysis_demo()
    if source == 'Synthetic demo':
        frame = demo.copy()
        cfg.dataset_name, cfg.data_source = 'Synthetic Campaign Time Series', 'Synthetic Demo'
    else:
        cfg.data_source = 'Upload File'
        st.caption('Include a date and numeric metric. Partner and Segment columns are optional.')
        template = demo[['date','Partner','Segment','applications']].head(24)
        st.download_button('Download CSV template',template.to_csv(index=False),'time-series-template.csv','text/csv',key='ts_simple_template')
        mode = st.radio('How would you like to add data?',['Upload CSV / Excel','Enter or paste data'],horizontal=True,key='ts_simple_input_mode')
        if mode == 'Upload CSV / Excel':
            uploaded = st.file_uploader('Upload time-series data',type=['csv','xlsx','xlsm'],key='ts_simple_upload')
            if uploaded is None: st.info('Upload a file with a date and numeric metric.'); return
            try:
                frame = pd.read_excel(BytesIO(uploaded.getvalue()),dtype=str,keep_default_na=False,engine='openpyxl') if uploaded.name.lower().endswith(('.xlsx','.xlsm')) else read_table(uploaded.getvalue())
                cfg.dataset_name = uploaded.name
            except (ValueError,UnicodeError,pd.errors.ParserError,ImportError,OSError) as exc:
                st.error(f'Could not read the file: {exc}'); return
        else:
            how = st.radio('Input method',['Editable table','Paste CSV / tab-separated'],horizontal=True,key='ts_simple_input_style')
            if how == 'Editable table':
                frame = st.data_editor(pd.DataFrame([{'Date':'','Partner':'','Segment':'','Metric':None}]),num_rows='dynamic',hide_index=True,use_container_width=True,key='ts_simple_editor')
            else:
                pasted = st.text_area('Paste data with a header row',height=140,key='ts_simple_paste')
                if not pasted.strip(): return
                try: frame = read_table(pasted)
                except (ValueError,UnicodeError,pd.errors.ParserError) as exc:
                    st.error(f'Could not read pasted data: {exc}'); return
            cfg.dataset_name = 'Entered time series'
    frame = normalize_uploaded_dataset(frame)
    st.session_state.ts_raw_df = frame.copy()
    if frame.empty or len(frame.columns)<2:
        st.warning('Provide at least a date and a numeric metric.'); return
    st.markdown('**Data preview**')
    preview_columns = [col for col in frame.columns if str(col).strip().lower().replace('_',' ') != 'campaign name']
    if source == 'Synthetic demo':
        preview = frame.iloc[np.linspace(0,len(frame)-1,min(12,len(frame)),dtype=int)].loc[:,preview_columns]
        st.caption('Sample rows across dates, Partners and segments.')
    else:
        preview = frame.loc[:,preview_columns].head(12)
    st.dataframe(preview,hide_index=True,use_container_width=True,height=245)
    signature = data_fingerprint(frame)
    if st.session_state.get('ts_simple_data_signature') != signature:
        for key in ('ts_simple_date','ts_simple_metric','ts_simple_launch','ts_simple_end','ts_simple_predictors','ts_simple_partners','ts_simple_group','ts_simple_aggregation'):
            st.session_state.pop(key,None)
        st.session_state.ts_simple_data_signature = signature
    columns = list(frame.columns)
    inferred = infer_timeseries_columns(frame)
    date_candidates = [col for col in columns if not pd.api.types.is_numeric_dtype(frame[col]) and safe_datetime_series(frame,col).notna().mean() >= .5]
    numeric = [col for col in columns if safe_numeric_series(frame,col).notna().mean() >= .5]
    if not date_candidates or not numeric:
        st.warning('The file needs a recognizable date column and a numeric metric column.'); return
    default_date = inferred['date'] if inferred['date'] in date_candidates else date_candidates[0]
    default_metric = inferred['outcome'] if inferred['outcome'] in numeric else numeric[0]
    a,b = st.columns(2)
    date_col = a.selectbox('Date column',date_candidates,index=date_candidates.index(default_date),key='ts_simple_date')
    metrics = [c for c in numeric if c != date_col and not any(x in str(c).lower() for x in ('flag','true_effect','true_impact'))]
    if not metrics: st.warning('Choose a numeric outcome column.'); return
    metric = b.selectbox('Metric',metrics,index=metrics.index(default_metric) if default_metric in metrics else 0,key='ts_simple_metric')
    flag = inferred['campaign_flag'] if inferred['campaign_flag'] in columns else 'None'
    dims = _grouping_columns(frame,date_col,metric,flag)
    partner = next((c for c in dims if str(c).lower()=='partner'),None)
    selected = frame.copy()
    if partner:
        options = [x for x in frame[partner].dropna().astype(str).str.strip().unique() if x]
        chosen_partners = st.multiselect('Partners',options,default=options,key='ts_simple_partners')
        selected = selected[selected[partner].astype(str).str.strip().isin(chosen_partners)]
        if selected.empty: st.info('Select at least one Partner.'); return
    else: chosen_partners = []
    group_options = ['No segment breakdown'] + dims
    group = st.selectbox('Group by (optional)',group_options,index=0,key='ts_simple_group')
    stable_dimensions = [col for col in dims if any(token in str(col).lower() for token in ('partner','segment','fico','group','band','region','channel'))]
    dimension_cols = list(dict.fromkeys(stable_dimensions + ([group] if group != group_options[0] else [])))
    for col in dimension_cols:
        if selected[col].isna().any() or selected[col].astype(str).str.strip().eq('').any():
            st.error(f'{col}: fill in every row before analysis.'); return
        selected[col] = selected[col].astype(str).str.strip()
    dates = safe_datetime_series(selected,date_col)
    values = safe_numeric_series(selected,metric)
    invalid = dates.isna() | ~np.isfinite(values)
    if invalid.any():
        st.error(f'{int(invalid.sum()):,} rows have an invalid date or missing/non-numeric metric. Correct those rows before analysis.'); return
    # A stable mix is essential: otherwise shifts in the population can look
    # like a campaign change. We cannot infer omitted dimensions or overlaps.
    if dimension_cols:
        coverage = selected.assign(_coverage_date=dates).groupby('_coverage_date')[dimension_cols].apply(lambda x:len(x.drop_duplicates()))
        total_cells = len(selected[dimension_cols].drop_duplicates())
        if (coverage != total_cells).any():
            st.error('Some Partner/Segment combinations are missing on certain dates. Use a consistent population across the full period, or select a complete subset.'); return
    multiple = dates.duplicated().any()
    if multiple:
        choices = ['Sum','Simple average']
        aggregation = st.selectbox('Combine rows for the same date',choices,index=0 if source=='Synthetic demo' else None,
            placeholder='Choose how to combine',key='ts_simple_aggregation',
            help='Sum is appropriate for additive counts and totals. Simple average weights each row equally; rates and averages often need customer-count weights.')
        st.caption('Sum counts or totals only for non-overlapping rows. Simple average gives every row equal weight; do not use it for rates or averages unless that weighting is appropriate.')
        if aggregation is None: st.info('Choose how to combine the selected rows.'); return
        if source != 'Synthetic demo':
            confirmation_scope = configuration_fingerprint({'data':signature,'partners':chosen_partners,'group':group,'metric':metric,'aggregation':aggregation})
            if confirmation_scope != st.session_state.get('ts_simple_aggregation_confirmation_scope'):
                st.session_state.pop('ts_simple_disjoint',None)
                st.session_state.ts_simple_aggregation_confirmation_scope = confirmation_scope
            confirmed = st.checkbox('I confirm rows are non-overlapping and the chosen aggregation is appropriate for this metric.',key='ts_simple_disjoint')
            if not confirmed: st.info('Confirm how the rows are combined before calculating.'); return
    else: aggregation = None
    min_date,max_date = dates.min().date(),dates.max().date()
    default_launch = None
    if source=='Synthetic demo' and flag!='None':
        launch_dates = dates[pd.to_numeric(selected[flag],errors='coerce').fillna(0)>0]
        if not launch_dates.empty: default_launch = launch_dates.min().date()
    prior_launch = st.session_state.get('ts_simple_launch')
    if prior_launch is not None and not min_date <= prior_launch <= max_date: st.session_state.pop('ts_simple_launch',None)
    launch = st.date_input('Campaign launch date',value=default_launch,min_value=min_date,max_value=max_date,key='ts_simple_launch')
    if launch is None: st.info('Choose the campaign launch date.'); return
    st.caption('Both a before-and-after comparison and a modeled estimate will be calculated. The model needs sufficient history.')
    with st.expander('Advanced settings'):
        missing = st.selectbox('Missing periods',['Fail validation','Keep','Interpolate','Fill missing counts with zero'],key='ts_simple_missing')
        end_enabled = st.checkbox('Set a campaign end date',key='ts_simple_end_enabled')
        old_end = st.session_state.get('ts_simple_end')
        if old_end is not None and not launch <= old_end <= max_date: st.session_state.pop('ts_simple_end',None)
        end = st.date_input('Campaign end date',value=max_date,min_value=launch,max_value=max_date,key='ts_simple_end') if end_enabled else None
        candidates = predictor_candidates(selected,metric,date_col,flag)
        candidates = [x for x in candidates if x not in dims]
        predictors = st.multiselect('Unaffected comparison signals (optional)',candidates,key='ts_simple_predictors',help='Choose only signals the campaign could not affect.')
        signal_scope = configuration_fingerprint({'data':signature,'partners':chosen_partners,'group':group,'signals':predictors})
        if signal_scope != st.session_state.get('ts_simple_confirmed_signal_selection'):
            st.session_state.pop('ts_simple_predictor_confirm',None)
            st.session_state.ts_simple_confirmed_signal_selection = signal_scope
        affirmed = not predictors or st.checkbox('I confirm these comparison signals were not affected by the campaign.',key='ts_simple_predictor_confirm')
        seasonality = st.selectbox('Seasonality',['Auto','None','Weekly','Monthly','Annual'],key='ts_simple_seasonality')
        interval = st.selectbox('Confidence level',[.95,.90,.80],format_func=lambda x:f'{x:.0%}',key='ts_simple_interval')
        trend = st.checkbox('Use local trend',value=True,key='ts_simple_trend')
    if not affirmed: st.info('Confirm the selected comparison signals are unaffected before calculating.'); return
    if group != group_options[0] and selected[group].nunique()>12:
        st.warning('Choose a grouping column with at most 12 segments.'); return
    try:
        scopes = [('Overall',selected)]
        if group != group_options[0]: scopes += [(str(name),part) for name,part in selected.groupby(group,sort=True)]
        prepared_scopes = [(name,_make_series(part,date_col,metric,aggregation,predictors,missing)) for name,part in scopes]
        if any(not np.isfinite(pd.to_numeric(data['_outcome'],errors='coerce')).all() for _,data in prepared_scopes):
            raise ValueError('Some periods have no outcome. Select a missing-period option in Advanced settings.')
    except ValueError as exc:
        st.error(f'Data validation: {exc}'); return
    prepared = prepared_scopes[0][1]
    pre = prepared[prepared['_date']<pd.Timestamp(launch)]
    post = prepared[prepared['_date']>=pd.Timestamp(launch)]
    if end: post = post[post['_date']<=pd.Timestamp(end)]
    if pre.empty or post.empty: st.warning('Choose a launch date with observations both before and after it.'); return
    st.caption(f"{len(prepared):,} {prepared.attrs['frequency'].lower()} periods · {len(pre):,} before launch · {len(post):,} in the selected outcome window")
    cfg.date_column,cfg.outcome_column,cfg.campaign_flag_column = str(date_col),str(metric),str(flag)
    cfg.intervention_date,cfg.campaign_end_date = launch.isoformat(),end.isoformat() if end else ''
    cfg.duplicate_policy,cfg.missing_policy = 'Fail validation',missing
    cfg.selected_covariates,cfg.seasonality,cfg.prediction_interval,cfg.local_trend = predictors,seasonality,interval,trend
    cfg.frequency = prepared.attrs['frequency']
    cfg.comparison_metric,cfg.comparison_window,cfg.symmetric_periods = 'Average per period','Symmetric',min(len(pre),len(post))
    st.session_state.ts_prepared_df = prepared
    fingerprint = configuration_fingerprint({'data':signature,'scope':chosen_partners,'group':group,'aggregation':aggregation,
        'date_column':date_col,'metric':metric,'launch':cfg.intervention_date,'end':cfg.campaign_end_date,'missing':missing,'predictors':predictors,
        'seasonality':seasonality,'interval':interval,'trend':trend,'seed':cfg.simulation_seed,'holdout':cfg.holdout_length})
    if group != group_options[0]:
        st.caption('A separate model is fitted for each segment; calculation may take longer.')
    if st.button('Calculate results',type='primary',key='ts_simple_calculate'):
        bundle = []
        model_cache = st.session_state.setdefault('ts_simple_model_cache',{})
        progress = st.progress(0,text='Preparing analysis...')
        with st.spinner('Calculating both analyses...'):
            for position,(name,data) in enumerate(prepared_scopes):
                progress.progress(position / len(prepared_scopes),text=f'Analyzing {name} ({position+1} of {len(prepared_scopes)})')
                scope_pre = data[data['_date']<pd.Timestamp(launch)]
                scope_post = data[data['_date']>=pd.Timestamp(launch)]
                if end: scope_post = scope_post[scope_post['_date']<=pd.Timestamp(end)]
                try:
                    pp = run_pre_post_analysis(data,cfg.intervention_date,cfg.campaign_end_date,comparison_metric='Average per period',comparison_window='Symmetric',symmetric_periods=min(len(scope_pre),len(scope_post)))
                except ValueError as exc:
                    bundle.append({'name':name,'data':data,'prepost':None,'model':None,'error':str(exc)}); continue
                try:
                    model_key = configuration_fingerprint({'data':data_fingerprint(data),'launch':cfg.intervention_date,
                        'end':cfg.campaign_end_date,'frequency':data.attrs['frequency'],'predictors':predictors,'interval':interval,
                        'seasonality':seasonality,'trend':trend,'seed':cfg.simulation_seed,'holdout':cfg.holdout_length})
                    if model_key not in model_cache:
                        model_cache[model_key] = fit_bsts_model(data,cfg.intervention_date,data.attrs['frequency'],predictors,interval,
                            campaign_end_date=cfg.campaign_end_date,seasonality=seasonality,local_trend=trend,
                            seed=int(cfg.simulation_seed),holdout_length=int(cfg.holdout_length))
                        if len(model_cache)>16: model_cache.pop(next(iter(model_cache)))
                    model = model_cache[model_key]
                    problem = None
                except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:
                    model,problem = None,str(exc)
                bundle.append({'name':name,'data':data,'prepost':pp,'model':model,'error':problem})
        progress.empty()
        st.session_state.ts_simple_bundle = bundle
        st.session_state.ts_simple_bundle_fingerprint = fingerprint
        overall = bundle[0]
        st.session_state.ts_prepost_result = overall['prepost']
        st.session_state.ts_prepost_fingerprint = f"{method_config_fingerprint(cfg,'Pre–Post Analysis')}:{data_fingerprint(prepared)}" if overall['prepost'] else ''
        st.session_state.ts_prepost_timestamp = datetime.now().isoformat(sep=' ',timespec='seconds')
        st.session_state.ts_bsts_result = overall['model']
        st.session_state.ts_structural_fingerprint = f"{method_config_fingerprint(cfg,'Structural Time Series Counterfactual')}:{data_fingerprint(prepared)}" if overall['model'] else ''
        st.session_state.ts_structural_timestamp = st.session_state.ts_prepost_timestamp if overall['model'] else ''
        st.session_state.ts_structural_status = 'completed' if overall['model'] else 'failed'
        cfg.analysis_method = 'Structural Time Series Counterfactual' if overall['model'] else 'Pre–Post Analysis'
    if st.session_state.get('ts_simple_bundle_fingerprint') != fingerprint:
        if st.session_state.get('ts_simple_bundle_fingerprint'): st.warning('Inputs changed. Calculate results again.')
        return
    bundle = st.session_state.ts_simple_bundle
    # Keep the legacy save/monitor snapshot aligned with the available overall result.
    cfg.analysis_method = 'Structural Time Series Counterfactual' if bundle[0]['model'] else 'Pre–Post Analysis'
    label = humanize_column_name(metric)
    st.markdown('### Overall results')
    st.caption('The model uses the selected post-launch window. The before-and-after chart uses matching-length windows, which can be shorter.')
    _render_scope(bundle[0],label,cfg,group,is_overall=True)
    if len(bundle)>1:
        st.markdown(f'### Results by {group}')
        st.caption('Segment results are exploratory; differences between segments are not formally tested.')
        for item in bundle[1:]: _render_scope(item,label,cfg,group,is_overall=False)
    st.markdown('### Decision insights')
    for index,item in enumerate(bundle):
        st.markdown(f"#### {item['name']}")
        if item['prepost'] is None:
            st.warning('Not calculated: '+str(item['error'])); continue
        _insight_card(_decision_insight(label,item['prepost'],item['model'],item['error'],None if index == 0 else item['name']))
    st.caption('A before-and-after change is descriptive. The model is conditional on pre-launch patterns and unaffected signals; neither result alone proves causality or approves a decision.')


def _render_scope(item,label,cfg,group,is_overall):
    name,data,pp,model = item['name'],item['data'],item['prepost'],item['model']
    if not is_overall: st.markdown(f'#### {name}')
    if pp is None:
        st.warning('Not calculated: '+str(item['error'])); return
    a,b,c = st.columns(3)
    a.metric('Average before',f"{pp['pre_value']:,.2f}")
    b.metric('Average after',f"{pp['post_value']:,.2f}")
    c.metric('Observed change',_safe_pct(pp['percent_change']) if pp['pre_value']>0 else 'Unavailable')
    st.caption(f"Matching before/after windows: {pp['pre_observations']} / {pp['post_observations']} periods; {pp['pre_start'].date()}–{pp['pre_end'].date()} and {pp['post_start'].date()}–{pp['post_end'].date()}.")
    chart_key = configuration_fingerprint({'scope':name,'group':group,'overall':is_overall})[:14]
    if model is not None:
        low,high = model['prediction_interval']
        relative = _safe_pct(model['relative_impact']) if model['expected_total']>0 else 'Unavailable'
        st.caption(f"Model estimate: {model['cumulative_impact']:+,.2f} ({relative}) · {model['interval_level']:.0%} model interval: {low:+,.2f} to {high:+,.2f} · Validation: {model.get('reliability','Review')}")
        st.markdown('**Actual vs estimated without campaign**')
        context = timeseries_chart_context(data,cfg.intervention_date,cfg.campaign_end_date)
        st.plotly_chart(bsts_counterfactual_chart(data,model['result'],cfg.intervention_date,label,context),use_container_width=True,key='ts_simple_model_'+chart_key)
        for warning in model.get('warnings',[]): st.warning(warning)
        with st.expander(f'Model diagnostics · {name}'):
            st.write('Validation:',model.get('reliability','Review'))
            st.json(model.get('backtest',{}))
            st.write('Comparison signals used:',', '.join(model.get('covariates',[])) or 'None')
    else:
        st.warning('Model estimate unavailable: '+str(item['error']))
        st.plotly_chart(time_series_line(data,cfg.intervention_date,label),use_container_width=True,key='ts_simple_history_'+chart_key)
    st.markdown('**Before-and-after comparison**')
    st.plotly_chart(pre_post_bar(pp['pre_value'],pp['post_value'],label),use_container_width=True,key='ts_simple_prepost_'+chart_key)
    with st.expander(f'Detailed statistics · {name}'):
        st.json({k:str(v) if isinstance(v,pd.Timestamp) else v for k,v in pp.items()})
        if model is not None: st.dataframe(model['result'],hide_index=True,use_container_width=True)
