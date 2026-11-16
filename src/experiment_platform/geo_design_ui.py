"""Business-facing setup and review for multivariate matched DMA designs."""
import pandas as pd
import plotly.express as px
import streamlit as st
from .business_config import fingerprint
from .business_data import frame_fingerprint
from .geo_matching import build_matched_design, matching_candidates


@st.cache_data(show_spinner=False)
def _design(history, covariates, candidates, required, excluded, cluster, seed, limit, holdout, version='multivariate-paired-v1'):
    return build_matched_design(history,covariates,candidates,required,excluded,cluster,seed,limit,holdout)


def render_matching_setup(panel, cfg):
    history = panel.loc[panel._date < pd.Timestamp(cfg.planned_launch_date)].copy()
    if history.empty:
        st.info('Choose a launch date after the history you want to use for matching.')
        return None
    names = sorted(history._dma.unique())
    candidates = st.multiselect('Candidate DMAs',names,default=names,key='geo_match_candidates')
    available = matching_candidates(history.loc[history._dma.isin(candidates)], [cfg.outcome_column,cfg.date_column,cfg.dma_column,cfg.group_column,cfg.pair_column])
    preferred = [c for c in available if any(term in c.lower() for term in ('spend','customer','eligible','population','fico','pd','share'))]
    covariates = st.multiselect('Match using historical business data',available,default=preferred[:4],key='geo_match_covariates',
        help='The primary outcome path, level, trend and volatility are always included. Select additional pre-launch business measures with consistent definitions.')
    st.caption('Only data before the launch date is used. The latest 25% is reserved for validation by default; it does not determine the pairs.')
    st.caption('The 12-period minimum is a software guard, not evidence of adequate history. Use longer history for seasonality, stability and validation of realistic campaign lengths.')
    with st.expander('Matching constraints and advanced settings',expanded=False):
        required = st.multiselect('Markets that must be included',candidates,key='geo_match_required')
        excluded = st.multiselect('Markets to exclude',candidates,key='geo_match_excluded')
        cluster_options = ['None'] + [c for c in history if not c.startswith('_') and c not in {cfg.date_column,cfg.dma_column,cfg.outcome_column,cfg.group_column,cfg.pair_column}]
        cluster = st.selectbox('Spillover cluster column',cluster_options,key='geo_match_cluster',
            help='Markets with the same non-empty cluster label must receive the same assignment. Use business knowledge of shared media or travel; proximity alone is not inferred.')
        limit = st.number_input('Maximum matching distance',min_value=.1,value=3.,step=.1,key='geo_match_distance')
        holdout = st.slider('History reserved for validation',min_value=10,max_value=50,value=25,step=5,format='%d%%',key='geo_match_holdout')
        seed = st.number_input('Random assignment seed',min_value=0,max_value=2**32-1,
                               value=int(cfg.assignment_seed or 91),step=1,key='geo_match_seed')
        st.caption('Distance is a normalized dissimilarity, not a probability of a good match. Defaults weight the outcome path 50%, outcome summaries 25%, and auxiliary features 25%. With no varying auxiliary features, the outcome blocks receive 2/3 and 1/3. Fix settings before reviewing validation; repeatedly trying seeds or weights can overfit the held-out history.')
    cluster = None if cluster == 'None' else cluster
    fields = list(dict.fromkeys(['_date','_dma','_outcome',*covariates,*([cluster] if cluster else [])]))
    history = history[fields].sort_values(['_dma','_date']).reset_index(drop=True)
    signature = fingerprint({'version':1,'history':frame_fingerprint(history),'variables':covariates,'candidates':candidates,
                             'required':required,'excluded':excluded,'cluster':cluster,'limit':limit,'holdout':holdout,'seed':seed})
    if st.button('Find matched markets',type='primary',key='geo_find_matches'):
        try:
            with st.spinner('Matching candidates and checking held-out history…'):
                result = _design(history,covariates,candidates,required,excluded,cluster,int(seed),float(limit),holdout/100)
            st.session_state.geo_matched_design = result
            st.session_state.geo_matching_signature = signature
            cfg.assignment_seed = int(seed)
        except (ValueError, RuntimeError) as exc:
            st.session_state.geo_matched_design = None
            st.error(str(exc))
    if st.session_state.get('geo_matching_signature') != signature or not st.session_state.get('geo_matched_design'):
        st.info('Choose candidates and matching variables, then select Find matched markets to review a design.')
        return None
    result = st.session_state.geo_matched_design
    st.markdown('### Recommended market pairs')
    st.caption(f"Matched on {result['training_periods']} periods ({result['training_start']} to {result['training_end']}). "
               f"Validated on {result['validation_periods']} later periods ({result['validation_start']} to {result['validation_end']}).")
    st.dataframe(result['pairs'],hide_index=True,use_container_width=True,
        column_config={'Matching distance':st.column_config.NumberColumn(format='%.2f'),
                       'Training correlation':st.column_config.NumberColumn(format='%.2f'),
                       'Mean gap (%)':st.column_config.NumberColumn(format='%.1f')})
    st.caption(f"{result['independent_units']} independent randomization units. Each pair has one Test and one Control; spillover constraints can link several pairs into one randomization unit. Review and freeze the assignment before launch.")
    if result['independent_units'] <= 10:
        st.caption(f"The smallest attainable exact two-sided p-value is {2 / 2**result['independent_units']:.4f}. Too few independent units can make the requested significance level unattainable, regardless of customer volume.")
    validation = result['validation']
    st.markdown('### Check on unused historical data')
    a,b,c = st.columns(3)
    a.metric('Held-out prediction error (MAE)',f"{validation['mae']:,.2f}")
    b.metric('Historical A/A effect',f"{validation['aa_effect']:+,.2f}")
    c.metric('A/A randomization p-value',f"{validation['aa_pvalue']:.3f}")
    if validation['aa_flagged']:
        st.warning('The untouched historical window shows an A/A difference. Review the design and events before launching; do not keep rerunning seeds to remove the flag.')
    else:
        st.caption(validation['note'])
    st.caption('The A/A diagnostic uses a 5% threshold. It is separate from the detection settings chosen below.')
    predictions = validation['predictions'].copy()
    predictions['Date'] = pd.to_datetime(predictions['Date'])
    pair = st.selectbox('Review pair history',result['pairs']['Pair'].tolist(),key='geo_match_review_pair')
    row = result['pairs'].loc[result['pairs'].Pair.eq(pair)].iloc[0]
    display = history.loc[history._dma.isin([row['Test DMA'],row['Control DMA']])]
    fig = px.line(display,x='_date',y='_outcome',color='_dma',labels={'_date':'Date','_outcome':cfg.outcome_column,'_dma':'DMA'},
                  color_discrete_sequence=['#007eb8','#d86b24'])
    fig.add_vline(x=pd.Timestamp(result['validation_start']).timestamp()*1000,line_dash='dash',line_color='#75899a')
    fig.update_layout(height=330,margin=dict(t=20,b=30,l=20,r=20))
    st.plotly_chart(fig,use_container_width=True,key='geo_matching_pair_history')
    st.caption('The dashed line separates matching history from untouched validation history.')
    if not result['reserves'].empty:
        st.markdown('**Reserve and excluded markets**')
        st.dataframe(result['reserves'],hide_index=True,use_container_width=True)
    with st.expander('Matching details'):
        st.markdown('**Baseline balance across assigned DMAs**')
        st.dataframe(result['balance'],hide_index=True,use_container_width=True)
        st.caption('SMD compares training features across DMAs, not individual customers. It is descriptive with few markets; it does not prove equivalence. The seed is not rerun to optimize this table.')
        st.dataframe(result['features'],hide_index=True,use_container_width=True)
        st.json(result['distance_settings'])
        st.dataframe(predictions,hide_index=True,use_container_width=True)
    from .geo_design_report import build_geo_design_html
    st.download_button('Download matching report (HTML)',build_geo_design_html(result,history,cfg),
                       'geo-matching-report.html','text/html',key='geo_matching_report')
    return result
