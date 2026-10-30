import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.experiment_platform.time_series_simple_ui import analysis_demo, _make_series, _decision_insight


def enter_analysis():
    app = AppTest.from_file('app.py',default_timeout=90).run()
    app.button(key='home_choose_timeseries').click().run(timeout=90)
    app.button(key='ts_demo_analyze').click().run(timeout=90)
    return app


def test_synthetic_demo_and_both_analyses_are_shown():
    demo = analysis_demo()
    assert {'Partner','Segment','date','applications'}.issubset(demo.columns)
    assert 'campaign_name' not in demo.columns
    assert set(demo.Partner) == {'Holland','PriceLine','Princess','Carnival'}
    app = enter_analysis()
    assert not app.exception and app.radio(key='ts_simple_source').value == 'Synthetic demo'
    assert not any(radio.label == 'Analysis method' for radio in app.radio)
    assert app.selectbox(key='ts_simple_metric').value == 'applications'
    assert app.selectbox(key='ts_simple_group').value == 'No segment breakdown'
    assert app.date_input(key='ts_simple_launch').value.isoformat() == '2025-07-06'
    preview = app.dataframe[0].value
    assert {'Partner','Segment'}.issubset(preview.columns) and 'campaign_name' not in preview.columns
    app.button(key='ts_simple_calculate').click().run(timeout=120)
    assert not app.exception and not app.error
    assert app.session_state.ts_prepost_result['causal'] is False
    assert app.session_state.ts_bsts_result['pre_observations'] >= 20
    assert len(app.get('plotly_chart')) == 2
    texts = ' '.join(x.value for x in app.markdown)
    assert 'Before-and-after comparison' in texts and 'RESULT INSIGHTS' in texts
    assert 'no-campaign baseline' in texts and 'descriptive' in texts
    app.multiselect(key='ts_simple_partners').set_value(['Holland']).run(timeout=90)
    assert any('Calculate results again' in x.value for x in app.warning)
    assert not app.get('plotly_chart')


def test_real_pasted_data_requires_launch_and_reports_model_limit():
    app = enter_analysis()
    app.radio(key='ts_simple_source').set_value('Upload file').run()
    app.radio(key='ts_simple_input_mode').set_value('Enter or paste data').run()
    app.radio(key='ts_simple_input_style').set_value('Paste CSV / tab-separated').run()
    dates = pd.date_range('2025-01-06',periods=12,freq='W-MON')
    text = 'Date,Revenue,Costs,campaign_name\n' + ''.join(f'{date.date()},{100+i},{40+i},example\n' for i,date in enumerate(dates))
    app.text_area(key='ts_simple_paste').set_value(text).run()
    assert not app.exception and app.date_input(key='ts_simple_launch').value is None
    assert 'campaign_name' not in app.dataframe[0].value.columns
    app.date_input(key='ts_simple_launch').set_value(dates[6].date()).run()
    app.button(key='ts_simple_calculate').click().run(timeout=90)
    assert not app.exception and app.session_state.ts_bsts_result is None
    assert app.session_state.ts_prepost_result['pre_observations'] == 6
    assert any('Model estimate unavailable' in w.value and '20 pre' in w.value for w in app.warning)
    app.selectbox(key='ts_simple_metric').set_value('Costs').run()
    assert any('Calculate results again' in w.value for w in app.warning)
    assert not app.get('plotly_chart')
    app.selectbox(key='ts_simple_metric').set_value('Revenue').run()
    invalid = text.replace('2025-01-20,102,42,example','2025-01-20,,42,example')
    app.text_area(key='ts_simple_paste').set_value(invalid).run()
    assert not app.exception and any('invalid date or missing' in e.value for e in app.error)
    assert not app.get('plotly_chart')


def test_grouped_results_and_explicit_aggregation():
    app = enter_analysis()
    app.radio(key='ts_simple_source').set_value('Upload file').run()
    app.radio(key='ts_simple_input_mode').set_value('Enter or paste data').run()
    app.radio(key='ts_simple_input_style').set_value('Paste CSV / tab-separated').run()
    dates = pd.date_range('2025-01-06',periods=12,freq='W-MON')
    text = 'Date,Partner,Segment,Revenue\n'+''.join(f'{d.date()},{p},{seg},{100+i+(20 if seg=="High" else 0)}\n' for i,d in enumerate(dates) for p in ['A','B'] for seg in ['Low','High'])
    app.text_area(key='ts_simple_paste').set_value(text).run()
    assert app.selectbox(key='ts_simple_aggregation').value is None
    app.selectbox(key='ts_simple_group').set_value('Segment').run()
    app.selectbox(key='ts_simple_aggregation').set_value('Sum').run()
    app.checkbox(key='ts_simple_disjoint').check().run()
    app.date_input(key='ts_simple_launch').set_value(dates[6].date()).run()
    app.button(key='ts_simple_calculate').click().run(timeout=90)
    assert not app.exception
    bundle = app.session_state.ts_simple_bundle
    assert [item['name'] for item in bundle] == ['Overall','High','Low']
    assert bundle[0]['prepost']['pre_value'] == pytest.approx(450)
    assert bundle[1]['prepost']['pre_value'] > bundle[2]['prepost']['pre_value']
    assert any('Results by Segment' in x.value for x in app.markdown)
    assert sum('RESULT INSIGHTS' in x.value for x in app.markdown) == 3
    app.multiselect(key='ts_simple_partners').set_value(['A']).run()
    assert not app.checkbox(key='ts_simple_disjoint').value
    assert not app.get('plotly_chart')
    incomplete = text.replace('2025-01-20,A,High,122\n','')
    app.text_area(key='ts_simple_paste').set_value(incomplete).run()
    assert any('combinations are missing' in item.value for item in app.error)
    assert not any(button.key == 'ts_simple_calculate' for button in app.button)


def test_aggregation_preserves_shared_signals_and_rejects_divergent():
    dates = pd.date_range('2025-01-06',periods=8,freq='W-MON')
    frame = pd.DataFrame([{'date':d,'value':n,'signal':n*10} for n,d in enumerate(dates) for _ in range(2)])
    summed = _make_series(frame,'date','value','Sum',['signal'],'Fail validation')
    averaged = _make_series(frame,'date','value','Simple average',['signal'],'Fail validation')
    assert summed['_outcome'].tolist() == [2*n for n in range(8)]
    assert averaged['_outcome'].tolist() == list(range(8))
    assert summed['signal'].tolist() == [n*10 for n in range(8)]
    frame.loc[0,'signal'] = -1
    with pytest.raises(ValueError,match='one identical unaffected signal'):
        _make_series(frame,'date','value','Sum',['signal'],'Fail validation')


def test_insight_tracks_data_and_does_not_invent_a_model():
    pp = {'pre_value':100,'post_value':120}
    no_model = ' '.join(_decision_insight('Revenue',pp,None,'not enough history'))
    assert '+20.0%' in no_model and 'unavailable' in no_model and 'not enough history' in no_model
    model = {'prediction_interval':(-5,20),'cumulative_impact':8,'interval_level':.95,'reliability':'Poor'}
    text = ' '.join(_decision_insight('Revenue',pp,model,None))
    assert 'includes zero' in text and 'Poor' in text and 'not enough' not in text


def test_comparison_signal_confirmation_is_refreshed():
    app = enter_analysis()
    app.radio(key='ts_simple_source').set_value('Upload file').run()
    app.radio(key='ts_simple_input_mode').set_value('Enter or paste data').run()
    app.radio(key='ts_simple_input_style').set_value('Paste CSV / tab-separated').run()
    dates = pd.date_range('2025-01-06',periods=12,freq='W-MON')
    text = 'Date,Revenue,External signal\n' + ''.join(f'{d.date()},{100+i},{500+i}\n' for i,d in enumerate(dates))
    app.text_area(key='ts_simple_paste').set_value(text).run()
    app.date_input(key='ts_simple_launch').set_value(dates[6].date()).run()
    app.multiselect(key='ts_simple_predictors').set_value(['External signal']).run()
    assert not app.checkbox(key='ts_simple_predictor_confirm').value
    app.checkbox(key='ts_simple_predictor_confirm').check().run()
    assert app.checkbox(key='ts_simple_predictor_confirm').value
    app.multiselect(key='ts_simple_predictors').set_value([]).run()
    app.multiselect(key='ts_simple_predictors').set_value(['External signal']).run()
    assert not app.checkbox(key='ts_simple_predictor_confirm').value
    assert not any(button.key == 'ts_simple_calculate' for button in app.button)
