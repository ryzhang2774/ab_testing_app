from itertools import combinations
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from src.experiment_platform.geo_matching import (
    _optimal_pairs, build_matched_design, matching_candidates, paired_power_table, plan_matched_experiment,
)
from src.experiment_platform.geo_design_report import build_geo_design_html


def panel(count=12, periods=40):
    dates = pd.date_range('2025-01-05',periods=periods,freq='W')
    t = np.arange(periods)
    rows = []
    for i in range(count):
        group = i//2
        y = 200+group*60 + group*.6*t + 12*np.sin(t/3+group) + (i%2)*1.5
        y += np.random.default_rng(i).normal(0,2,periods)
        for k,date in enumerate(dates):
            rows.append({'_date':date,'_dma':f'DMA {i:02}','_outcome':y[k],
                         'customers':10000+group*3000+20*k,'spend':y[k]*3,'cluster':''})
    return pd.DataFrame(rows)


def test_global_pair_optimization_beats_greedy_and_enforces_constraints():
    names = ['A','B','C','D']
    cost = {(0,1):1.,(0,2):2.,(0,3):100.,(1,2):100.,(1,3):2.,(2,3):100.}
    pairs,_ = _optimal_pairs(names,cost,set(),{},200.)
    assert set(pairs)=={(0,2),(1,3)}
    constrained,_ = _optimal_pairs(names,cost,{'A','B','C','D'},{'A':'same','C':'same'},200.)
    assert (0,2) not in constrained
    assert len(constrained)==2


def test_matching_is_stable_and_validation_outcomes_do_not_select_pairs():
    data = panel()
    result = build_matched_design(data,['customers','spend'],seed=21)
    assert len(result['pairs'])==6
    assert result['assignment'].groupby('Pair').Group.nunique().eq(2).all()
    shuffled = build_matched_design(data.sample(frac=1,random_state=4),['customers','spend'],seed=21)
    pd.testing.assert_frame_equal(shuffled['assignment'],result['assignment'])
    changed = data.copy()
    post = changed._date.gt(pd.Timestamp(result['training_end']))
    tests = result['assignment'].loc[result['assignment'].Group.eq('Test'),'DMA']
    changed.loc[post & changed._dma.isin(tests),'_outcome'] += 500
    changed.loc[post,'customers'] *= 2
    other = build_matched_design(changed,['customers','spend'],seed=21)
    pd.testing.assert_frame_equal(other['assignment'],result['assignment'])
    pd.testing.assert_frame_equal(other['pairs'],result['pairs'])
    pd.testing.assert_frame_equal(other['features'],result['features'])
    assert other['validation']['aa_effect']-result['validation']['aa_effect']==pytest.approx(500)
    assert other['validation']['aa_flagged']


def test_reserves_do_not_receive_singleton_assignment_and_required_markets_fail_clearly():
    data = panel(7)
    result = build_matched_design(data,['customers'])
    assert len(result['assignment'])==6 and len(result['reserves'])==1
    reserve = result['reserves'].iloc[0].DMA
    assert reserve not in result['assignment'].DMA.tolist()
    missing = data.drop(data.index[0])
    r = build_matched_design(missing,['customers'])
    assert 'Incomplete' in r['reserves'].set_index('DMA').at['DMA 00','Reason']
    with pytest.raises(ValueError,match='Mandatory market'):
        build_matched_design(missing,['customers'],required=['DMA 00'])
    with pytest.raises(ValueError,match='cannot also be excluded'):
        build_matched_design(data,required=['DMA 00'],excluded=['DMA 00'])
    excluded = build_matched_design(panel(),excluded=['DMA 00','DMA 01'])
    assert not {'DMA 00','DMA 01'} & set(excluded['assignment'].DMA)


def test_spillover_clusters_stay_in_one_arm_and_reduce_independent_units():
    data = panel()
    data.loc[data._dma.isin(['DMA 00','DMA 02']),'cluster']='shared media'
    design = build_matched_design(data,['customers'],cluster_column='cluster')
    assignment = design['assignment'].set_index('DMA')
    assert assignment.at['DMA 00','Group']==assignment.at['DMA 02','Group']
    assert assignment.at['DMA 00','Randomization unit']==assignment.at['DMA 02','Randomization unit']
    assert assignment.at['DMA 00','Pair']!=assignment.at['DMA 02','Pair']
    assert design['independent_units']<len(design['pairs'])
    assert assignment.groupby('Pair').Group.nunique().eq(2).all()


def test_auxiliary_values_matter_and_missing_covariates_are_not_imputed():
    data = panel(6)
    data['_outcome']=data._date.map({d:100+np.sin(i) for i,d in enumerate(sorted(data._date.unique()))})
    design = build_matched_design(data,['customers'])
    assert set(map(frozenset,design['pairs'][['Test DMA','Control DMA']].to_numpy())) == {
        frozenset(['DMA 00','DMA 01']),frozenset(['DMA 02','DMA 03']),frozenset(['DMA 04','DMA 05'])}
    data.loc[data._dma.eq('DMA 00'),'customers']=np.nan
    missing = build_matched_design(data,['customers'])
    assert missing['reserves'].set_index('DMA').at['DMA 00','Reason'].startswith('Missing')
    candidates = matching_candidates(panel().assign(campaign_spend=3,customer_id=range(len(panel())),
        ID=range(len(panel())),customerID=range(len(panel()))))
    assert 'campaign_spend' not in candidates
    assert 'customer_id' not in candidates
    assert 'ID' not in candidates and 'customerID' not in candidates


def test_exact_power_uses_pair_randomization_resolution_and_large_effect():
    data = panel()
    design = build_matched_design(data,['customers'])
    rows = paired_power_table(data,design,[4,100],100.,alpha=.05)
    assert len(rows)==1 and rows[0]['duration']==4
    assert rows[0]['power'] > .9
    assert rows[0]['null_rejection_rate'] <= .05
    small = build_matched_design(panel(4),['customers'])
    assert paired_power_table(panel(4),small,[4],10000.,alpha=.05)[0]['power']==0


def test_plan_is_limited_to_holdout_and_report_records_reproducible_inputs():
    data = panel()
    design = build_matched_design(data,['customers'])
    plan = plan_matched_experiment(data,design,'2026-01-01','Weekly','Relative lift',10.,max_duration=26)
    assert plan['table']['Campaign Duration'].max() <= design['validation_periods']
    assert plan['table']['Required Total DMAs'].eq(len(design['assignment'])).all()
    cfg = SimpleNamespace(dataset_name='<script>alert(1)</script>',outcome_column='Applications',planned_launch_date='2026-01-01')
    html = build_geo_design_html(design,data,cfg,plan).decode()
    assert '<script>' not in html and '&lt;script&gt;' in html
    for phrase in ['Historical input values','Untouched historical validation','How matching is calculated',
                   'Random seed','Maximum requested duration','Randomization unit','conditional historical estimate']:
        assert phrase in html


def test_short_history_is_rejected_instead_of_reusing_training_for_validation():
    with pytest.raises(ValueError,match='12 pre-launch periods'):
        build_matched_design(panel(periods=11))


def test_matching_ui_requires_current_design_and_downloads_complete_plan():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file('app.py',default_timeout=90).run()
    app.button(key='home_choose_geo').click().run()
    app.button(key='geo_demo_plan').click().run()
    assert not app.exception
    assert app.radio(key='geo_market_design_mode').value=='Automatically match candidates'
    assert app.button(key='geo_find_matches')
    app.button(key='geo_find_matches').click().run()
    assert not app.exception and not app.error
    design = app.session_state.geo_matched_design
    assert len(design['assignment'])==12
    assert design['validation_periods']>0
    assert app.button(key='geo_get_plan')
    app.button(key='geo_get_plan').click().run()
    assert not app.exception and not app.error
    assert app.session_state.geo_plan['matched_design']['seed']==design['seed']
    assert any('Download complete design report' in item.label for item in app.get('download_button'))
    app.button(key='geo_edit_plan').click().run()
    app.multiselect(key='geo_match_candidates').set_value(design['assignment'].DMA.tolist()[:-1]).run()
    assert not app.exception
    assert not any(item.key=='geo_get_plan' for item in app.button)


def test_global_calendar_gaps_are_not_treated_as_consecutive_weeks():
    data = panel()
    missing_week = data._date.unique()[12]
    with pytest.raises(ValueError,match='reporting calendar'):
        build_matched_design(data.loc[data._date.ne(missing_week)])


def test_uploaded_history_supports_matching_without_recorded_groups(monkeypatch):
    from io import BytesIO
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    data = panel().rename(columns={'_date':'week','_dma':'dma','_outcome':'applications'}).drop(columns='cluster')
    upload = BytesIO(data.to_csv(index=False).encode())
    upload.name = 'business-history.csv'
    def uploader(*args,**kwargs):
        return upload if kwargs.get('key')=='geo_upload' else None
    monkeypatch.setattr(st,'file_uploader',uploader)
    app = AppTest.from_file('app.py',default_timeout=90).run()
    app.button(key='home_choose_geo').click().run()
    app.button(key='geo_demo_plan').click().run()
    app.radio(key='geo_source').set_value('Upload File').run()
    assert not app.exception and not app.error
    assert set(app.multiselect(key='geo_match_covariates').options)=={'customers','spend'}
    app.button(key='geo_find_matches').click().run()
    assert not app.exception and not app.error
    design = app.session_state.geo_matched_design
    assert len(design['pairs'])==6
    assert all(row['Basis'].startswith('Mean gap') for row in design['pairs'].to_dict('records'))
    app.button(key='geo_get_plan').click().run()
    assert not app.exception and not app.error
    assert app.session_state.geo_plan['matched_design']['input_fingerprint']==design['input_fingerprint']
