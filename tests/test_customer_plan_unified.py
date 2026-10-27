import io
import pandas as pd
import pytest
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest
from src.experiment_platform.customer_plan_ui import demo_data, normalize_table, grouping_columns, calculate_plan, pdf_summary


def _calculate(data):
    return calculate_plan(data, ['Holland','PriceLine'], list(data.segment.unique()), 'Credit line 1000', ['Credit line 2000','Credit line 3000'], 'Revolving Balance', 'NCL', 5, 'Relative %', .05, .8, True)


def test_strategies_are_required_and_shared_control_is_calculated():
    data=normalize_table(demo_data(), grouping='FICO group', allow_pool=True)
    plan=_calculate(data)
    assert len(plan['arm_totals']) == 3
    assert all(r['power'] >= .8 for r in plan['comparisons'])
    assert not plan['variance_assumptions']
    incomplete=data[~((data.partner=='Holland') & (data.segment=='FICO 700–749') & (data.strategy=='Credit line 2000') & (data.metric=='Revolving Balance'))]
    with pytest.raises(ValueError, match='Missing historical statistics.*Holland'):
        _calculate(incomplete)
    pdf = pdf_summary(plan)
    assert pdf.startswith(b'%PDF')
    contents = '\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    for text in ['Statistical settings', 'Calculation formulas', 'Credit line 1000', 'Pairwise calculations and final verification', 'Welch-Satterthwaite', 'Neyman']:
        assert text in contents
    assert str(next(r['recruitment_n'] for r in plan['arm_totals'] if r['arm'] == 'Credit line 1000')) in contents


def test_excel_template_matches_csv():
    original=demo_data()
    b=io.BytesIO()
    original.to_excel(b,index=False,engine='openpyxl')
    excel=normalize_table(pd.read_excel(io.BytesIO(b.getvalue()), dtype=str, keep_default_na=False, engine='openpyxl'), allow_pool=True)
    csv=normalize_table(pd.read_csv(io.StringIO(original.to_csv(index=False)), dtype=str, keep_default_na=False), allow_pool=True)
    pd.testing.assert_frame_equal(excel,csv)


def test_grouping_columns_and_pooled_variance():
    raw = pd.DataFrame([
        {'Partner':'Holland','FICO group':'700+', 'PD score group':'low','Strategy':'1000','Metric':'RAR','Sample mean':10,'Sample variance':4,'Sample number':3},
        {'Partner':'Holland','FICO group':'700+', 'PD score group':'high','Strategy':'1000','Metric':'RAR','Sample mean':20,'Sample variance':9,'Sample number':2},
    ])
    assert [name for name, _ in grouping_columns(raw)] == ['FICO group', 'PD score group']
    with pytest.raises(ValueError, match='non-overlapping'):
        normalize_table(raw, grouping='FICO group')
    result = normalize_table(raw, grouping='FICO group', allow_pool=True).iloc[0]
    assert result.sample_number == 5
    assert result.sample_mean == pytest.approx(14)
    assert result.sample_variance == pytest.approx((2*4 + 1*9 + 3*16 + 2*36)/4)
    assert len(normalize_table(raw, grouping='PD score group')) == 2
    with pytest.raises(ValueError, match='Duplicate'):
        normalize_table(pd.concat([raw,raw.iloc[[0]]]), grouping='FICO group', allow_pool=True)
    monthly = pd.concat([raw.iloc[[0]].assign(YM='2026-01'), raw.iloc[[0]].assign(YM='2026-02')])
    with pytest.raises(ValueError, match='monthly'):
        normalize_table(monthly, grouping='FICO group', allow_pool=True)


def test_legacy_group_by_column_remains_supported():
    raw = pd.DataFrame([{'Partner':'Holland','Group by':'FICO group','Segment':'700+',
                         'Strategy':'1000','Metric':'RAR','Sample mean':10,'Sample variance':4,'Sample number':3}])
    out = normalize_table(raw, grouping='FICO group')
    assert out.iloc[0]['segment'] == '700+'


def test_customer_plan_demo_flow(tmp_path,monkeypatch):
    from src.experiment_platform import customer_plan_ui
    saved = []
    monkeypatch.setattr(customer_plan_ui, 'upsert_experiment', lambda record: saved.append(record))
    monkeypatch.setattr(customer_plan_ui, 'load_experiments', lambda: [])
    app=AppTest.from_file('app.py',default_timeout=60).run()
    app.button(key='home_choose_customer').click().run()
    assert not app.exception
    assert not any('workspace' in str(x.label).lower() for x in app.selectbox)
    app.button(key='customer_demo_plan').click().run()
    assert not app.exception
    app.radio(key='customer_plan_source').set_value('Synthetic demo').run()
    assert not app.exception
    assert app.radio(key='customer_plan_source').options == ['Upload file', 'Databricks', 'Synthetic demo']
    assert next(s for s in app.selectbox if s.label == 'Group by').options == ['FICO group', 'PD score group']
    assert len(app.get('plotly_chart')) == 3
    assert next(s for s in app.selectbox if s.label == 'Primary metric').options == ['Revolving Balance', 'RAR', 'NCL']
    next(s for s in app.selectbox if s.label == 'Group by').set_value('PD score group').run()
    assert not app.exception
    next(b for b in app.button if b.label == 'Get plan').click().run(timeout=60)
    assert not app.exception
    assert any('Total customers needed' == m.label for m in app.metric)
    assert any('Control customers needed' == m.label for m in app.metric)
    assert not any('Duration' in m.label for m in app.metric)
    next(b for b in app.button if b.label == 'Save to Your experiments').click().run()
    assert not app.exception
    assert saved[0]['sample_target'] > 0 and saved[0]['sample_collected'] == 0
    assert saved[0]['config_payload']['customer_plan']['config']['synthetic']
    next(b for b in app.button if b.label == 'Back to settings').click().run()
    assert not app.exception
    assert app.radio(key='customer_plan_source').value == 'Synthetic demo'
    assert next(s for s in app.selectbox if s.label == 'Control').value == 'Credit line 1000'
    assert next(s for s in app.selectbox if s.label == 'Group by').value == 'PD score group'
