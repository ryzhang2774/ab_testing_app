import json
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.experiment_platform.customer_results import demo_results, calculate_results
from src.experiment_platform.customer_summary_results import normalize_summary, calculate_summary
from src.experiment_platform.customer_summary_ui import summary_demo


def _summaries(frame, metric):
    out = frame.groupby(['Partner','FICO group','Assigned option'],sort=False)[metric].agg(
        **{'Sample mean':'mean','Sample variance':'var','Sample number':'size'}).reset_index()
    out['Metric'] = metric
    return out


def test_continuous_pooled_statistics_match_customer_records():
    raw = demo_results()
    summary, dims = normalize_summary(_summaries(raw, 'Revolving Balance'))
    assert dims == ['Partner','FICO group']
    overall, groups, audit, issues = calculate_summary(summary,'Revolving Balance','Credit line 1000','FICO group')
    raw_overall, _, _, _ = calculate_results(raw,'Customer ID','Assigned option','Revolving Balance','Credit line 1000')
    raw_groups, _ = __import__('src.experiment_platform.customer',fromlist=['analyze_customer_records']).analyze_customer_records(
        raw,'Customer ID','Assigned option','Revolving Balance','Credit line 1000',partner_column='FICO group',joint_decision=False,multiplicity_method='None')
    assert not issues and audit['accounts'] == len(raw)
    for field in ['Mean / Rate','Effect vs Control','Raw p-value','Adjusted p-value','CI Lower','CI Upper']:
        left = overall.set_index('Credit Line')[field].sort_index()
        right = raw_overall.set_index('Credit Line')[field].sort_index()
        np.testing.assert_allclose(left,right,rtol=1e-9,atol=1e-9,equal_nan=True)
        for label in groups:
            np.testing.assert_allclose(groups[label].set_index('Credit Line')[field].sort_index(),
                raw_groups[label].set_index('Credit Line')[field].sort_index(),rtol=1e-9,atol=1e-9,equal_nan=True)


def test_binary_summaries_match_raw_including_sparse_fisher():
    raw = pd.DataFrame({'id':[f'x{i}' for i in range(24)], 'arm':['Control']*12+['Test']*12,
                        'partner':['A']*6+['B']*6+['A']*6+['B']*6,
                        'outcome':[0]*12+[1]+[0]*11})
    summary = raw.groupby(['partner','arm'])['outcome'].agg(**{'Success count':'sum','Sample number':'size'}).reset_index()
    summary['Metric'] = 'Conversion'
    summary['Metric type'] = 'Binary'
    data,_ = normalize_summary(summary)
    result, _, _, _ = calculate_summary(data,'Conversion','Control')
    raw_result, _, _, _ = calculate_results(raw,'id','arm','outcome','Control')
    a = result.set_index('Credit Line').loc['Test']
    b = raw_result.set_index('Credit Line').loc['Test']
    assert a['Method'] == 'Fisher exact test'
    for field in ['Mean / Rate','Effect vs Control','Raw p-value','CI Lower','CI Upper']:
        assert a[field] == pytest.approx(b[field])


def test_validation_blocks_bad_variance_duplicates_and_monthly_snapshots():
    row = {'Partner':'A','Strategy':'Control','Metric':'RAR','Sample mean':10,'Sample variance':5,'Sample number':20}
    with pytest.raises(ValueError,match='Duplicate'):
        normalize_summary(pd.DataFrame([row,row]))
    with pytest.raises(ValueError,match='variance'):
        normalize_summary(pd.DataFrame([{**row,'Sample variance':-1}]))
    with pytest.raises(ValueError,match='monthly'):
        normalize_summary(pd.DataFrame([{**row,'month':'2026-01'}]))
    with pytest.raises(ValueError,match='success count'):
        normalize_summary(pd.DataFrame([{**row,'Metric type':'Binary','Success count':21}]))
    sd_row = {k:v for k,v in row.items() if k != 'Sample variance'}
    sd_data, _ = normalize_summary(pd.DataFrame([{**sd_row,'Sample SD':3}]))
    assert sd_data.iloc[0].sample_variance == 9
    with pytest.raises(ValueError,match='do not agree'):
        normalize_summary(pd.DataFrame([{**row,'Sample SD':3}]))


def test_summary_demo_and_ui_support_multiple_metrics():
    demo = summary_demo()
    data, _ = normalize_summary(demo)
    assert data.sample_number.min() >= 2
    assert set(data.metric) == {'Revolving Balance','RAR','NCL'}
    _, fico_groups, _, _ = calculate_summary(data,'Revolving Balance','Credit line 1000','FICO group')
    _, pd_groups, _, _ = calculate_summary(data,'Revolving Balance','Credit line 1000','PD score group')
    def mean(groups, label, arm):
        return float(groups[label].set_index('Credit Line').loc[arm, 'Mean / Rate'])
    assert mean(fico_groups,'FICO 750+','Credit line 4000') > mean(fico_groups,'FICO 650–699','Credit line 4000') + 250
    assert mean(pd_groups,'PD low','Credit line 4000') > mean(pd_groups,'PD high','Credit line 4000') + 200
    assert mean(fico_groups,'FICO 650–699','Credit line 5000') - mean(fico_groups,'FICO 650–699','Credit line 4000') < 50

    app = AppTest.from_file('app.py',default_timeout=90).run()
    app.button(key='home_choose_customer').click().run()
    app.button(key='customer_demo_analyze').click().run(timeout=90)
    assert not app.exception
    assert not any(item.label == 'Data format' for item in app.radio)
    assert set(demo.columns) == {'Partner','FICO group','PD score group','Strategy','Metric','Sample mean','Sample variance','Sample number'}
    assert not any('One row per non-overlapping' in x.value for x in app.caption)
    assert not any('Synthetic experiment summaries' in x.value for x in app.info)
    assert app.selectbox(key='summary_group').value == 'FICO group'
    app.button(key='summary_add_metric').click().run(timeout=90)
    assert app.selectbox(key='summary_extra_0').value == 'RAR'
    app.button(key='summary_add_metric').click().run(timeout=90)
    assert app.selectbox(key='summary_extra_1').value == 'NCL'
    assert 'RAR' not in app.selectbox(key='summary_extra_1').options
    assert 'NCL' not in app.selectbox(key='summary_extra_0').options
    assert 'Revolving Balance' not in app.selectbox(key='summary_extra_0').options
    app.button(key='summary_calculate').click().run(timeout=90)
    assert not app.exception
    assert set(app.session_state.summary_analyses) == {'Revolving Balance','RAR','NCL'}
    assert app.text_input(key='summary_report_name').value == 'Customer experiment'
    assert any(x.proto.label == 'Download final report (PDF)' for x in app.get('download_button'))
    assert len(app.get('plotly_chart')) == 12
    charts = [json.loads(chart.proto.spec) for chart in app.get('plotly_chart')]
    assert charts[3]['layout']['yaxis']['range'] == charts[4]['layout']['yaxis']['range'] == charts[5]['layout']['yaxis']['range']
    assert charts[3]['data'][0]['y'] != charts[4]['data'][0]['y']

    labels = [str(x.value) for x in app.markdown]
    overall = labels.index('### Overall results')
    segments = labels.index('### Results by FICO group')
    decisions = labels.index('### Decision insights')
    assert overall < segments < decisions
    assert not any('Additional metric (exploratory)' in label for label in labels)
    for metric in ['Revolving Balance', 'RAR', 'NCL']:
        assert any(metric in label for label in labels[overall:segments])
    app.selectbox(key='summary_group').set_value('PD score group').run(timeout=90)
    app.button(key='summary_calculate').click().run(timeout=90)
    assert not app.exception
    assert any('Results by PD score group' in x.value for x in app.markdown)
    labels = [x.value for x in app.markdown]
    assert all(f'**PD {level}**' in labels for level in ['low','medium','high'])



def test_pasted_summary_requires_nonoverlap_confirmation():
    app=AppTest.from_file('app.py',default_timeout=90).run()
    app.button(key='home_choose_customer').click().run()
    app.button(key='customer_demo_analyze').click().run(timeout=90)
    app.radio(key='simple_source').set_value('Upload file').run()
    app.radio(key='summary_input_method').set_value('Enter or paste data').run()
    app.radio(key='summary_input_style').set_value('Paste CSV / tab-separated').run()
    app.text_area(key='summary_paste').set_value('Segment,Strategy,Metric,Sample mean,Sample variance,Sample number\nA,Control,RAR,10,4,100\nA,Test,RAR,12,5,120\n').run()
    app.selectbox(key='summary_primary').set_value('RAR').run()
    assert not any(b.key == 'summary_calculate' for b in app.button)
    app.checkbox(key='summary_disjoint_confirmation').check().run()
    app.button(key='summary_calculate').click().run(timeout=90)
    assert not app.exception and 'RAR' in app.session_state.summary_analyses


def test_metric_options_update_with_pasted_dataset_and_removal():
    app = AppTest.from_file('app.py',default_timeout=90).run()
    app.button(key='home_choose_customer').click().run()
    app.button(key='customer_demo_analyze').click().run()
    app.radio(key='simple_source').set_value('Upload file').run()
    app.radio(key='summary_input_method').set_value('Enter or paste data').run()
    app.radio(key='summary_input_style').set_value('Paste CSV / tab-separated').run()
    header = 'Strategy,Metric,Sample mean,Sample variance,Sample number\n'
    def rows(metrics):
        return header + ''.join(f'{strategy},{metric},10,4,100\n' for metric in metrics for strategy in ['Control','Test'])
    app.text_area(key='summary_paste').set_value(rows(['Balance','Revenue','Loss','Retention'])).run()
    app.selectbox(key='summary_primary').set_value('Balance').run()
    app.button(key='summary_add_metric').click().run()
    app.selectbox(key='summary_extra_0').set_value('Revenue').run()
    app.button(key='summary_add_metric').click().run()
    assert 'Revenue' not in app.selectbox(key='summary_extra_1').options
    app.selectbox(key='summary_extra_1').set_value('Loss').run()
    assert 'Loss' not in app.selectbox(key='summary_extra_0').options
    app.button(key='summary_remove_0').click().run()
    assert 'Revenue' in app.selectbox(key='summary_extra_1').options
    app.text_area(key='summary_paste').set_value(rows(['Balance','Loss','New metric'])).run()
    assert not app.exception
    assert 'New metric' in app.selectbox(key='summary_extra_1').options
    assert 'Revenue' not in app.selectbox(key='summary_extra_1').options
    assert app.selectbox(key='summary_extra_1').value == 'Loss'
    app.text_area(key='summary_paste').set_value(rows(['Balance','New metric'])).run()
    assert not app.exception
    assert app.selectbox(key='summary_extra_1').value is None
    assert app.selectbox(key='summary_extra_1').options == ['New metric']
