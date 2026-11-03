from types import SimpleNamespace
import pandas as pd
from src.experiment_platform.time_series_html_report import build_analysis_html, build_plan_html


def test_analysis_report_contains_actual_counterfactual_and_escapes_input():
    dates = pd.date_range('2025-01-01', periods=4)
    data = pd.DataFrame({'_date': dates, '_outcome': [10, 12, 13, 14]})
    pp = {'pre_value':11., 'post_value':13.5, 'absolute_change':2.5, 'percent_change':2.5/11,
          'pre_start':dates[0], 'pre_end':dates[1], 'post_start':dates[2], 'post_end':dates[3],
          'pre_observations':2, 'post_observations':2, 'significance_test':{'available':False, 'reason':'short'}}
    result = pd.DataFrame({'_date':dates[2:], '_outcome':[13.,14.], 'counterfactual':[10.,11.],
                           'lower':[8.,9.], 'upper':[12.,13.], 'impact':[3.,3.]})
    model = {'observed_total':27.,'expected_total':21.,'cumulative_impact':6.,'relative_impact':6/21,
             'prediction_interval':(1.,11.),'interval_level':.95,'reliability':'Review','seasonal_period':None,
             'model_specification':{'level':'local level'},'result':result,'covariates':[]}
    cfg = SimpleNamespace(intervention_date='2025-01-03',campaign_end_date='')
    report = build_analysis_html([{'name':'<script>','data':data,'prepost':pp,'model':model,'error':None}],cfg,'Upload','Segment','Revenue').decode()
    assert '&lt;script&gt;' in report and '<script>' not in report
    assert 'Counterfactual by period' in report and '27.00' in report and '21.00' in report
    assert 'Newey–West' in report and 'Significant' not in report


def test_planning_report_marks_assumption_based_and_inputs():
    data = pd.DataFrame({'_date':pd.date_range('2025-01-01',periods=2), '_outcome':[10.,20.]})
    cfg = SimpleNamespace(outcome_column='Revenue', frequency='Daily', planned_launch_date='2025-01-03')
    plan = {'target_power':.8,'alpha':.05,'effect':5,'effect_type':'Relative Lift (%)','ramp':1,
            'validation_mode':'Assumption-based','recommended_duration':7,'predictors':[],
            'power':[{'duration':7,'power':.82,'power_ci':(float('nan'),float('nan')),
                      'minimum_detectable_effect':.04,'calibration_status':'assumption_based','status':'Recommended'}]}
    report = build_plan_html(plan,data,cfg).decode()
    assert 'Assumption-based' in report and 'does not empirically validate' in report
    assert '2025-01-01' in report and '<td>10</td>' in report and '4.0000' in report
