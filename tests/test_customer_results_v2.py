import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from streamlit.testing.v1 import AppTest
from src.experiment_platform.customer_results import demo_results, calculate_results
from src.experiment_platform.customer_ui import average_chart
from src.experiment_platform.customer_decision import decision_context, rule_based_decision, llm_settings, generate_llm_insight


def test_demo_matches_design_and_results_have_overall_and_fico_segments():
    frame = demo_results()
    assert 'Active account (0/1)' not in frame.columns
    assert set(frame.Partner) == {'Holland', 'PriceLine', 'Princess', 'Carnival'}
    assert set(frame['Assigned option']) == {f'Credit line {n}' for n in [1000, 2000, 3000, 4000, 5000]}
    limits = frame['Assigned option'].str.extract(r'(\d+)$')[0].astype(float)
    assert (frame['Revolving Balance'] <= limits).all()
    assert (frame['NCL'] == 0).mean() > .8
    averages = frame.groupby('Assigned option')['Revolving Balance'].mean()
    first_step = averages['Credit line 2000'] - averages['Credit line 1000']
    last_step = averages['Credit line 5000'] - averages['Credit line 4000']
    assert 0 <= last_step < first_step * .25
    overall, segments, audit, issues = calculate_results(frame, 'Customer ID', 'Assigned option', 'Revolving Balance',
        'Credit line 1000', group_column='FICO group')
    assert not issues and len(segments) == 3 and audit['accounts'] == len(frame)
    result = overall.set_index('Credit Line').loc['Credit line 2000']
    test = frame[frame['Assigned option'] == 'Credit line 2000']['Revolving Balance']
    control = frame[frame['Assigned option'] == 'Credit line 1000']['Revolving Balance']
    assert result['Raw p-value'] == pytest.approx(stats.ttest_ind(test, control, equal_var=False).pvalue)
    assert result['Effect vs Control'] == pytest.approx(test.mean()-control.mean())
    assert result['CI Lower'] < result['Effect vs Control'] < result['CI Upper']
    assert audit['health'][0]['Allocation check'].startswith('Not checked')


def test_binary_and_missing_subgroup_and_real_input_validation():
    frame = demo_results().head(300).copy()
    frame['Binary outcome'] = (frame['Baseline revolving balance'] > 0).astype(int)
    frame.loc[frame.index[:2], 'FICO group'] = 'Small segment'
    overall, segments, audit, issues = calculate_results(frame, 'Customer ID', 'Assigned option', 'Binary outcome',
        'Credit line 1000', group_column='FICO group')
    assert audit['metric_type'] == 'Binary' and issues
    assert np.isfinite(overall.loc[~overall['Is Control'], 'Raw p-value']).all()
    control = frame[frame['Assigned option'] == 'Credit line 1000']['Binary outcome']
    test = frame[frame['Assigned option'] == 'Credit line 2000']['Binary outcome']
    _, p_value = proportions_ztest([test.sum(), control.sum()], [len(test), len(control)])
    assert overall.set_index('Credit Line').loc['Credit line 2000', 'Raw p-value'] == pytest.approx(p_value)
    with pytest.raises(ValueError, match='unique'):
        calculate_results(frame.assign(**{'Customer ID':'same'}), 'Customer ID', 'Assigned option', 'Revolving Balance', 'Credit line 1000')
    with pytest.raises(ValueError, match='finite'):
        calculate_results(frame.assign(**{'Revolving Balance':None}), 'Customer ID', 'Assigned option', 'Revolving Balance', 'Credit line 1000')




def test_charts_sort_means_and_show_reported_p_values_and_intervals():
    rows = pd.DataFrame([{'Test option':'A', 'Difference':.04, 'CI lower':-.01, 'CI upper':.09,
                          'Adjusted p-value':.04, 'Significant':True}])
    result = pd.DataFrame([{'Credit Line':'A','Mean / Rate':.14,'N':10,'Is Control':False},
                           {'Credit Line':'Control','Mean / Rate':.10,'N':12,'Is Control':True}])
    chart = average_chart(result, rows, True, 'RAR', 'Control', {'A':'#008fc5','Control':'#005b88'}, {'A':'','Control':''})
    assert list(chart.data[0].x) == ['Control','A']
    assert list(chart.data[0].y) == pytest.approx([10, 14])
    assert list(chart.data[1].y) == pytest.approx([9, 19])  # Control mean + difference interval
    assert 'Lift +40.0%' in chart.data[2].text[1]
    assert 'p 0.040' in chart.data[2].text[1] and 'Significant' in chart.data[2].text[1]
    result.loc[result['Is Control'], 'Mean / Rate'] = 0
    chart = average_chart(result, rows, True, 'RAR', 'Control', {'A':'#008fc5','Control':'#005b88'}, {'A':'','Control':''})
    assert 'Lift unavailable' in chart.data[2].text[1]



def test_decision_notes_use_results_and_do_not_claim_pairwise_proof():
    overall = pd.DataFrame([
        {'Scope':'Overall','Test option':f'Credit line {n}', 'Control N':100,'Test N':100,
         'Control mean / rate':100, 'Test mean / rate':mean, 'Difference':mean-100,
         'CI lower':1, 'CI upper':100, 'Adjusted p-value':.01,'Significant':True}
        for n, mean in [(2000,200),(3000,225),(4000,235),(5000,238)]
    ])
    empty = overall.iloc[0:0]
    context = decision_context(overall, empty, 'Balance', 'Credit line 1000', 'Higher', .05, 'Holm', True)
    text = rule_based_decision(context)
    assert 'Credit line 5000' in text['primary'] and 'not been directly compared' in text['primary']
    assert any('Additional growth tapers' in item and 'descriptive' in item for item in text['observations'])
    assert len(text['comparisons']) == 4
    assert all(item['status'] == 'Favorable' for item in text['comparisons'])
    context['preferred_direction'] = 'Lower'
    assert 'No Test' in rule_based_decision(context)['primary']
    assert all(item['status'] == 'Unfavorable' for item in rule_based_decision(context)['comparisons'])
    context['overall'][0]['significant'] = False
    assert rule_based_decision(context)['comparisons'][0]['status'] == 'Inconclusive'
    context['overall'][0]['difference'] = -12
    assert '-12.00' in rule_based_decision(context)['comparisons'][0]['detail']


def test_databricks_model_sends_only_aggregated_data_and_blocks_redirects(monkeypatch):
    import json
    from src.experiment_platform import customer_decision
    overall = pd.DataFrame([{'Scope':'Overall','Test option':'A','Control N':100,'Test N':100,
        'Control mean / rate':100, 'Test mean / rate':105, 'Difference':5, 'CI lower':1,'CI upper':9,
        'Adjusted p-value':.02,'Significant':True}])
    small = overall.assign(Scope='small segment', **{'Control N':5})
    context = decision_context(overall, small, 'Balance', 'Control', 'Higher', .05, 'Holm', False)
    assert context['segments_exploratory'] == [] and context['small_segment_comparisons_withheld'] == 1
    config = llm_settings({'DATABRICKS_HOST':'https://workspace.example', 'DATABRICKS_TOKEN':'test-token', 'EXPERIMENT_LLM_ENDPOINT':'model-name'})
    assert config and llm_settings({'DATABRICKS_HOST':'http://example', 'DATABRICKS_TOKEN':'x', 'EXPERIMENT_LLM_ENDPOINT':'model'}) is None
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return b'{"choices":[{"message":{"content":"Review the evidence."}}]}'
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == 'https://workspace.example/serving-endpoints/model-name/invocations'
            payload = json.loads(request.data)
            assert payload['stream'] is False and 'Customer ID' not in request.data.decode()
            assert 'small segment' not in request.data.decode()
            return Response()
    monkeypatch.setattr(customer_decision, 'build_opener', lambda handler: Opener())
    assert generate_llm_insight(context, config) == 'Review the evidence.'
