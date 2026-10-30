import io
import pandas as pd
import pytest
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest
from src.experiment_platform.customer_plan_ui import demo_data, demo_overall_summary, normalize_table, grouping_columns, calculate_plan, pdf_summary, strategy_styles, historical_metric_chart


def _calculate(data):
    return calculate_plan(data, ['Holland','PriceLine'], list(data.segment.unique()), 'Credit line 1000', ['Credit line 2000','Credit line 3000'], 'Revolving Balance', 'NCL', 5, 'Relative %', .05, .8, True)


def test_many_credit_lines_have_distinct_styles_and_plausible_demo():
    names = [f'Credit line {1000 * i}' for i in range(1, 31)]
    colors, patterns = strategy_styles(names)
    assert len(set(colors[name] for name in names[:24])) == 24
    assert colors[names[0]] == colors[names[24]] and patterns[names[0]] != patterns[names[24]]
    demo = demo_data()
    balance = demo[demo.Metric.eq('Revolving Balance')]
    limits = balance.Strategy.str.extract(r'(\d+)$')[0].astype(float)
    assert (balance['Sample mean'].reset_index(drop=True) <= limits.reset_index(drop=True)).all()
    assert balance['Sample number'].nunique() > 10


def test_historical_line_chart_orders_options_and_keeps_missing_cells_separate():
    data = pd.DataFrame([
        {'partner':'Holland', 'segment':'High', 'strategy':'Test B', 'sample_mean':30},
        {'partner':'Holland', 'segment':'High', 'strategy':'Control', 'sample_mean':10},
        {'partner':'Holland', 'segment':'Low', 'strategy':'Control', 'sample_mean':20},
        {'partner':'Holland', 'segment':'Low', 'strategy':'Test A', 'sample_mean':25},
    ])
    fig = historical_metric_chart(data, 'FICO group', 'Revolving Balance', ['Control', 'Test A', 'Test B'])
    high = next(trace for trace in fig.data if trace.name == 'High')
    assert list(high.x) == ['Control', 'Test A', 'Test B']
    assert high.y[0] == 10 and pd.isna(high.y[1]) and high.y[2] == 30
    assert high.connectgaps is False
    assert fig.layout.xaxis.title.text == 'Control and Test options'
    assert fig.layout.yaxis.title.text == 'Revolving Balance historical mean'


def test_strategies_are_required_and_shared_control_is_calculated():
    data=normalize_table(demo_data(), grouping='FICO group', allow_pool=True)
    plan=_calculate(data)
    assert len(plan['arm_totals']) == 3
    assert all(r['power'] >= .8 for r in plan['comparisons'])
    assert not plan['variance_assumptions']
    incomplete=data[~((data.partner=='Holland') & (data.segment=='FICO 700–749') & (data.strategy=='Credit line 2000') & (data.metric=='Revolving Balance'))]
    with pytest.raises(ValueError, match='Missing historical statistics.*Holland'):
        _calculate(incomplete)
    plan['config']['synthetic'] = True
    pdf = pdf_summary(plan)
    assert pdf.startswith(b'%PDF')
    contents = '\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    for text in ['Experiment Design Package', 'Sample size recommendation', 'Statistical settings', 'Calculation formulas', 'Credit line 1000', 'Calculation verification', 'Welch-Satterthwaite', 'Neyman', 'Historical inputs', 'Reproducibility']:
        assert text in contents
    assert all(symbol in contents for symbol in ['Σ', '√', '−', '×'])
    assert 'SYNTHETIC DEMO' in contents
    assert f"{next(r['recruitment_n'] for r in plan['arm_totals'] if r['arm'] == 'Credit line 1000'):,}" in contents


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


def test_demo_overall_summary_pools_each_strategy_and_metric_separately():
    raw = pd.DataFrame([
        {'Partner':'Holland','Strategy':'1000','Metric':'RAR','Sample mean':10,'Sample variance':4,'Sample number':3},
        {'Partner':'PriceLine','Strategy':'1000','Metric':'RAR','Sample mean':20,'Sample variance':9,'Sample number':2},
        {'Partner':'Holland','Strategy':'2000','Metric':'RAR','Sample mean':30,'Sample variance':16,'Sample number':4},
    ])
    summary = demo_overall_summary(raw)
    row = summary[summary.Strategy.eq('1000')].iloc[0]
    assert row['Historical sample number'] == 5
    assert row['Historical mean'] == pytest.approx(14)
    assert row['Historical standard deviation'] == pytest.approx(((2*4 + 9 + 3*16 + 2*36)/4)**.5)
    assert summary[summary.Strategy.eq('2000')].iloc[0]['Historical sample number'] == 4
    assert len(demo_overall_summary(raw, by_partner=True)) == 3


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
    assert app.radio(key='customer_plan_source').value == 'Synthetic demo'
    assert not app.exception
    assert any('Choose your grouping column' in item.value for item in app.markdown)
    assert any('Data preview' in item.value for item in app.markdown)
    assert not any('Overall historical summary' in item.value or 'Historical data overview' in item.value for item in app.markdown)
    assert app.radio(key='customer_plan_source').options == ['Upload file', 'Databricks', 'Synthetic demo']
    group_select = next(s for s in app.selectbox if s.label == 'Group by')
    assert group_select.options == ['FICO group', 'PD score group']
    assert group_select.value == 'FICO group'
    assert not app.exception
    assert next(m for m in app.multiselect if m.label == 'Segments to include').options == ['FICO 650–699', 'FICO 700–749', 'FICO 750+']
    assert len(app.get('plotly_chart')) == 3
    assert next(s for s in app.selectbox if s.label == 'Primary metric').options == ['Revolving Balance', 'RAR', 'NCL']
    next(s for s in app.selectbox if s.label == 'Group by').set_value('PD score group').run()
    assert not app.exception
    next(b for b in app.button if b.label == 'Get plan').click().run(timeout=60)
    assert not app.exception
    assert any('Total customers needed' == m.label for m in app.metric)
    assert any('Control customers needed' == m.label for m in app.metric)
    assert not any('Duration' in m.label for m in app.metric)
    rendered = []
    real_pdf_summary = customer_plan_ui.pdf_summary
    def capture_pdf(plan):
        rendered.append(plan['config']['name'])
        return real_pdf_summary(plan)
    monkeypatch.setattr(customer_plan_ui, 'pdf_summary', capture_pdf)
    app.text_input(key='customer_plan_name').set_value('Credit line test').run()
    assert not app.exception
    assert rendered[-1] == 'Credit line test'
    next(b for b in app.button if b.label == 'Save to Your experiments').click().run()
    assert not app.exception
    assert saved[0]['sample_target'] > 0 and saved[0]['sample_collected'] == 0
    assert saved[0]['config_payload']['customer_plan']['config']['name'] == 'Credit line test'
    assert saved[0]['config_payload']['customer_plan']['config']['synthetic']
    next(b for b in app.button if b.label == 'Back to settings').click().run()
    assert not app.exception
    assert app.radio(key='customer_plan_source').value == 'Synthetic demo'
    assert next(s for s in app.selectbox if s.label == 'Control').value == 'Credit line 1000'
    assert next(s for s in app.selectbox if s.label == 'Group by').value == 'PD score group'


def test_numeric_score_bins_are_explicit_and_preserve_cells():
    from src.experiment_platform.customer_plan_ui import bin_group_column
    rows = pd.DataFrame([{'Partner':'Holland','PD score':score,'Strategy':'1000','Metric':'RAR', 'Sample mean':10+score, 'Sample variance':2, 'Sample number':3} for score in range(30)])
    assert 'PD score' in [name for name, _ in grouping_columns(rows)]
    with pytest.raises(ValueError, match='increasing'):
        bin_group_column(rows, 'PD score', '15, 10')
    grouped = bin_group_column(rows, 'PD score', '10, 20')
    result = normalize_table(grouped, grouping='PD score', allow_pool=True)
    assert result.sample_number.sum() == 90
    assert list(result.segment) == ['<= 10', '10 to 20', '> 20']
    assert list(result.sample_number) == [33, 30, 27]


def test_pasted_data_uses_actual_grouping_columns():
    app=AppTest.from_file('app.py',default_timeout=60).run()
    app.button(key='home_choose_customer').click().run()
    app.button(key='customer_demo_plan').click().run()
    app.radio(key='customer_plan_source').set_value('Upload file').run()
    next(r for r in app.radio if r.label == 'How would you like to add data?').set_value('Enter or paste data').run()
    next(r for r in app.radio if r.label == 'Input method').set_value('Paste CSV / tab-separated').run()
    app.text_area[0].set_value('Partner,FICO group,PD band,Strategy,Metric,Sample mean,Sample variance,Sample number\nHolland,700+,Low,1000,RAR,10,4,100\nHolland,700+,Low,2000,RAR,12,5,100\n').run()
    assert not app.exception
    grouping = next(s for s in app.selectbox if s.label == 'Group by')
    assert grouping.options == ['FICO group', 'PD band']
    grouping.set_value('PD band').run()
    assert not app.exception
    assert next(m for m in app.multiselect if m.label == 'Segments to include').options == ['Low']


def test_pdf_uses_only_selected_inputs_and_updates_for_new_data():
    raw = pd.DataFrame([
        {'Partner': partner, 'FICO group': fico, 'PD band': pd_band, 'Strategy': arm,
         'Metric': metric, 'Sample mean': mean, 'Sample variance': variance, 'Sample number': 100}
        for partner, fico, pd_band in [('Holland', 'FICO A', 'PD low'), ('Princess', 'FICO B', 'PD high')]
        for arm in ['Control 1000', 'Test 2000', 'Unused 3000']
        for metric, mean, variance in [('Revolving Balance', 100, 400), ('RAR', 12, 25)]
    ])
    first_data = normalize_table(raw, grouping='FICO group')
    first = calculate_plan(first_data, ['Holland'], ['FICO A'], 'Control 1000', ['Test 2000'],
                           'Revolving Balance', None, 5, 'Absolute', .05, .8, True)
    first['input_grouping'] = 'FICO group'
    first_text = '\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(pdf_summary(first))).pages)
    assert all(value in first_text for value in ['Holland', 'FICO A', 'Revolving Balance', 'Control 1000', 'Test 2000'])
    assert all(value not in first_text for value in ['Princess', 'FICO B', 'PD low', 'RAR', 'Unused 3000'])
    assert {row['metric'] for row in first['history']} == {'Revolving Balance'}
    assert len(first['history']) == 2

    changed = raw.copy()
    changed['Sample mean'] = 4321
    changed['Sample variance'] = 2500
    second_data = normalize_table(changed, grouping='PD band')
    second = calculate_plan(second_data, ['Princess'], ['PD high'], 'Control 1000', ['Test 2000'],
                            'RAR', None, 5, 'Absolute', .05, .8, True)
    second['input_grouping'] = 'PD band'
    second_text = '\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(pdf_summary(second))).pages)
    assert all(value in second_text for value in ['Princess', 'PD high', 'RAR', '4,321', '2,500'])
    assert all(value not in second_text for value in ['Holland', 'FICO A', 'FICO B', 'Revolving Balance', 'Unused 3000'])
    assert first['data_fingerprint'] != second['data_fingerprint']
    assert first['arm_totals'] != second['arm_totals']
