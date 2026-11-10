import io
import math
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.experiment_platform.customer_plan_ui import demo_data, demo_overall_summary, normalize_table, grouping_columns, calculate_plan, calculate_group_plans, historical_metric_chart
from src.experiment_platform.customer_palette import strategy_styles
from src.experiment_platform.workspace_visuals import history_detail_charts
from src.experiment_platform.customer_html_report import build_html_summary
from src.experiment_platform.customer_grouping import ordered_groups


def _calculate(data):
    return calculate_plan(data, ['Holland','PriceLine'], list(data.segment.unique()), 'Credit line 1000', ['Credit line 2000','Credit line 3000'], 'Revolving Balance', 'NCL', 5, 'Relative %', .05, .8, True)


def test_shared_control_keeps_neyman_ratios_for_all_tests():
    rows = [('250', 120, 47729.07, 23627), ('2000', 199.34, 155682.07, 43962),
            ('3500', 323.23, 466854.65, 88513), ('7000', 405.80, 874489.39, 30874)]
    data = pd.DataFrame([{'partner':'Partner', 'segment':'Group', 'strategy':arm, 'metric':'Metric',
                          'sample_mean':mean, 'sample_variance':variance, 'sample_number':n, 'group_by':'FICO'}
                         for arm, mean, variance, n in rows])
    plan = calculate_plan(data, ['Partner'], ['Group'], '250', ['2000','3500','7000'], 'Metric', None, 20, 'Relative %', .05, .8, True)
    totals = {r['arm']: r['analyzable_n'] for r in plan['arm_totals']}
    control_n = max(r['pair_control_n'] for r in plan['pairwise'])
    assert control_n == totals['250'] == 4582
    for arm, _, variance, _ in rows[1:]:
        assert totals[arm] == math.ceil(control_n * math.sqrt(variance / rows[0][2]))
    assert sum(totals.values()) == 46802
    assert all(row['power'] >= .8 for row in plan['comparisons'])


def test_group_specific_strategies_use_separate_historical_cells_and_shared_family():
    from src.experiment_platform.customer_html_report import build_html_summary
    data = normalize_table(demo_data(), grouping='FICO group', allow_pool=True)
    segments = list(data.segment.unique())[:2]
    choices = {segments[0]: {'control': 'Credit line 1000', 'tests': ['Credit line 2000']},
               segments[1]: {'control': 'Credit line 2000', 'tests': ['Credit line 3000', 'Credit line 4000']}}
    plan = calculate_group_plans(data, ['Holland', 'PriceLine'], segments, choices, 'Revolving Balance', None, 5, 'Relative %', .05, .8, True)
    first, second = [group['plan'] for group in plan['group_plans']]
    assert first['family_size'] == second['family_size'] == 3
    assert first['planning_alpha'] == pytest.approx(.05 / 3)
    assert {r['arm'] for r in first['history']} == {'Credit line 1000', 'Credit line 2000'}
    assert {r['arm'] for r in second['history']} == {'Credit line 2000', 'Credit line 3000', 'Credit line 4000'}
    assert {r['segment'] for r in first['history']} == {segments[0]}
    assert {r['segment'] for r in second['history']} == {segments[1]}
    assert all(row['power'] >= .8 for group in (first, second) for row in group['comparisons'])
    assert first['comparisons'][0]['baseline'] != second['comparisons'][0]['baseline']
    plan['config']['name'] = '<script>bad</script>'
    report = build_html_summary(plan).decode()
    assert '&lt;script&gt;bad&lt;/script&gt;' in report and '<script>bad</script>' not in report
    assert all(segment in report for segment in segments)
    assert 'Credit line 4000' in report and 'Historical inputs used' in report
    missing = data[~((data.partner == 'Holland') & (data.segment == segments[1]) & (data.strategy == 'Credit line 4000'))]
    with pytest.raises(ValueError, match='Missing historical statistics'):
        calculate_group_plans(missing, ['Holland', 'PriceLine'], segments, choices, 'Revolving Balance', None, 5, 'Relative %', .05, .8, True)


def test_plan_survives_missing_package_metadata_and_does_not_require_reportlab(monkeypatch):
    from importlib.metadata import PackageNotFoundError
    from src.experiment_platform import business_plan, customer_plan_ui

    actual_version = business_plan.package_version
    def missing_metadata(name):
        if name == 'statsmodels':
            raise PackageNotFoundError(name)
        return actual_version(name)
    monkeypatch.setattr(business_plan, 'package_version', missing_metadata)
    app = AppTest.from_file('app.py', default_timeout=60).run()
    app.button(key='home_choose_customer').click().run()
    app.button(key='customer_demo_plan').click().run()
    next(b for b in app.button if b.label == 'Get plan').click().run(timeout=60)
    assert not app.exception
    assert any(m.label == 'Total customers needed' for m in app.metric)
    assert any('statsmodels' in w.value and 'version information' in w.value for w in app.warning)
    plan = app.session_state['customer_plan_result']
    assert plan['runtime_versions']['statsmodels'] == business_plan.VERSION_UNAVAILABLE
    assert 'reportlab' not in plan['runtime_versions']
    assert plan['runtime_versions']['numpy'] == actual_version('numpy')
    assert plan['arm_totals']

    assert any(button.proto.label == 'Download complete summary (HTML)' for button in app.get('download_button'))
    assert not any('PDF' in button.proto.label for button in app.get('download_button'))
    assert not any(m.label == 'Control customers needed' for m in app.metric)


def test_many_credit_lines_have_distinct_styles_and_plausible_demo():
    names = [f'Credit line {1000 * i}' for i in range(1, 31)]
    colors, patterns = strategy_styles(names)
    assert len(set(colors[name] for name in names[:24])) == 24
    assert colors[names[0]] == colors[names[24]] and patterns[names[0]] != patterns[names[24]]
    demo = demo_data()
    for group, cap in [('FICO 650–699', 5500), ('FICO 700–749', 8500), ('FICO 750+', 12000)]:
        expected = {f'Credit line {n}' for n in range(500, cap + 1, 500)}
        for _, cell in demo[demo['FICO group'].eq(group)].groupby(['Partner', 'PD score group', 'Metric']):
            assert set(cell.Strategy) == expected
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
    marked = historical_metric_chart(data, 'FICO group', 'Revolving Balance', ['Control', 'Test A', 'Test B'],
                                     {'High': {'control':'Test B', 'tests':['Control']},
                                      'Low': {'control':'Control', 'tests':['Test A']}})
    assert list(next(t for t in marked.data if t.name == 'High').text) == ['Test', 'Historical only', 'Control']
    assert list(next(t for t in marked.data if t.name == 'Low').text) == ['Control', 'Test', 'Historical only']
    assert high.connectgaps is False
    assert fig.layout.xaxis.title.text == 'Historical strategy'
    assert fig.layout.yaxis.title.text == 'Revolving Balance historical mean'


def test_overall_chart_weights_customers_and_partner_view_has_no_assignment_guides():
    from src.experiment_platform.customer_plan_ui import overall_metric_chart, pool_statistics
    history = pd.DataFrame([
        ('A', 'Low', '1000', 10, 4, 90), ('B', 'Low', '1000', 20, 8, 10),
        ('A', 'High', '1000', 31, 6, 100), ('A', 'Low', '2000', 25, 9, 20),
    ], columns=['partner','segment','strategy','sample_mean','sample_variance','sample_number'])
    pooled = pool_statistics(history, ['segment', 'strategy']).assign(partner='All selected Partners')
    fig = overall_metric_chart(history, pooled, 'FICO group', 'Balance', ['1000','2000'])
    assert list(fig.data[0].y) == [21, 25]
    assert list(fig.data[0].customdata) == [200, 20]
    by_partner = historical_metric_chart(history, 'FICO group', 'Balance', ['1000','2000'])
    for chart in (fig, by_partner):
        assert not chart.layout.shapes
        assert all(trace.text is None for trace in chart.data)


def test_group_charts_keep_each_groups_control_and_tests_separate():
    from src.experiment_platform.customer_plan_ui import group_metric_charts, pool_statistics
    options = ['1000', '2000', '3500', '7000']
    groups = ['750+', '650–699']
    history = pd.DataFrame([(p, group, strategy, mean, 9, n)
                            for group in groups for strategy in options
                            for p, mean, n in [('A', 10, 90), ('B', 30, 10)]],
                           columns=['partner','segment','strategy','sample_mean','sample_variance','sample_number'])
    pooled = pool_statistics(history, ['segment','strategy']).assign(partner='All selected Partners')
    selected = {'650–699': {'control':'1000', 'tests':['2000']},
                '750+': {'control':'2000', 'tests':['3500','7000']}}
    def panels():
        return dict(group_metric_charts(pooled, 'FICO group', 'Balance', options, selected))
    figures = panels()
    assert list(figures) == groups[::-1]
    for group, fig in figures.items():
        choice = selected[group]
        assert len(fig.data) == 1 and list(fig.data[0].y) == [12] * 4
        assert {(s.x0, s.line.color) for s in fig.layout.shapes} == (
            {(choice['control'], '#657887')} | {(x, '#006DE3') for x in choice['tests']})
    selected['750+']['tests'] = ['7000']
    assert [s.x0 for s in panels()['750+'].layout.shapes] == ['2000','7000']
    assert [s.x0 for s in panels()['650–699'].layout.shapes] == ['1000','2000']


def test_fico_bar_axes_are_categorical_and_ordered_with_sparse_partner_cells():
    data = pd.DataFrame([{'partner':p, 'segment':segment, 'strategy':strategy, 'sample_number':n}
                         for p, segment, strategy, n in [('A','750+','Control',10), ('A','650–699','Test',20),
                                                        ('B','700–749','Control',30), ('B','750+','Test',40)]])
    count, spread = history_detail_charts(data.assign(sample_variance=9), 'segment', 'FICO group', 'Balance', ['Control','Test'])
    assert count.to_dict()['encoding']['xOffset']['sort'] == ['650–699', '700–749', '750+']
    assert spread.to_dict()['encoding']['y']['sort'] == ['650–699', '700–749', '750+']
    assert spread.to_dict()['encoding']['color']['field'] == 'historical_sd'
    assert ordered_groups(['750+', '650–699', '700–749'], 'FICO group') == ['650–699', '700–749', '750+']
    assert ordered_groups(['High', 'Low'], 'PD score') == ['High', 'Low']


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
    contents = build_html_summary(plan).decode()
    for text in ['Experiment design summary', 'Credit line 1000', 'Comparison verification', 'Calculation method', 'Neyman', 'Historical inputs', 'Reproducibility']:
        assert text in contents
    assert 'Synthetic demo' in contents
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
    assert not next(c for c in app.checkbox if c.label == 'Protect against multiple comparisons (Bonferroni)').value
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
    chart_tabs = [tab for tab in app.tabs if tab.label in ('Overall', 'By FICO group', 'By Partner')]
    assert [tab.label for tab in chart_tabs] == ['Overall', 'By FICO group', 'By Partner']
    import json
    for tab in chart_tabs:
        assert not tab.get('expander')
        charts = tab.get('plotly_chart')
        details = [json.loads(chart.proto.spec) for chart in tab.get('arrow_vega_lite_chart')]
        assert [fig['mark']['type'] for fig in details] == ['bar', 'rect']
        assert [fig['encoding']['x']['field'] for fig in details] == ['strategy', 'strategy']
        assert details[0]['encoding']['y']['field'] == 'sample_number'
        assert details[1]['encoding']['color']['field'] == 'historical_sd'
        guides = [json.loads(chart.proto.spec)['layout'].get('shapes', []) for chart in charts]
        assert sum(bool(lines) for lines in guides) == (3 if tab.label == 'By FICO group' else 0)
    assert next(s for s in app.selectbox if s.label == 'Primary metric').options == ['Revolving Balance', 'RAR', 'NCL']
    next(s for s in app.selectbox if s.label == 'Group by').set_value('PD score group').run()
    assert not app.exception
    group_controls = [s for s in app.selectbox if s.label.startswith('Control — ')]
    group_controls[1].set_value('Credit line 2000').run()
    group_tests = [m for m in app.multiselect if m.label.startswith('Test options — ')]
    group_tests[1].set_value(['Credit line 3000']).run()
    next(b for b in app.button if b.label == 'Get plan').click().run(timeout=60)
    assert not app.exception
    group_plans = app.session_state['customer_plan_result']['group_plans']
    assert group_plans[1]['plan']['comparisons'][0]['control'] == 'Credit line 2000'
    assert group_plans[1]['plan']['comparisons'][0]['test'] == 'Credit line 3000'
    assert any('Total customers needed' == m.label for m in app.metric)
    expected = {g['segment']: sum(row['recruitment_n'] for row in g['plan']['arm_totals']) for g in group_plans}
    summary = next(d.value for d in app.dataframe if 'option_0' in d.value.columns).set_index('group')
    assert summary.total.to_dict() == {**expected, 'All selected groups': sum(expected.values())}
    arms = sorted({arm['arm'] for g in group_plans for arm in g['plan']['arm_totals']},
                  key=lambda name: int(name.split()[-1]))
    for i, arm in enumerate(arms):
        arm_total = 0
        for g in group_plans:
            roles = {row['name']: row['role'] for row in g['plan']['config']['arms']}
            count = next((row['recruitment_n'] for row in g['plan']['arm_totals'] if row['arm'] == arm), 0)
            assert summary.loc[g['segment'], f'option_{i}'] == (f'{count:,} · {roles[arm]}' if count else '—')
            arm_total += count
        assert summary.loc['All selected groups', f'option_{i}'] == f'{arm_total:,}'
    assert next(m.value for m in app.metric if m.label == 'Total customers needed') == f'{sum(expected.values()):,}'
    assert not any('Duration' in m.label for m in app.metric)
    app.text_input(key='customer_plan_name').set_value('Credit line test').run()
    assert not app.exception
    assert any(x.proto.label == 'Download complete summary (HTML)' for x in app.get('download_button'))
    next(b for b in app.button if b.label == 'Save to Your experiments').click().run()
    assert not app.exception
    assert saved[0]['sample_target'] > 0 and saved[0]['sample_collected'] == 0
    assert saved[0]['config_payload']['customer_plan']['config']['name'] == 'Credit line test'
    assert saved[0]['config_payload']['customer_plan']['config']['synthetic']
    next(b for b in app.button if b.label == 'Back to settings').click().run()
    assert not app.exception
    assert app.radio(key='customer_plan_source').value == 'Synthetic demo'
    assert [s.value for s in app.selectbox if s.label.startswith('Control — ')][1] == 'Credit line 2000'
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


def test_uploaded_group_candidates_and_existing_segment_column():
    raw = pd.DataFrame({
        'Partner':['A', 'A'], 'Region':['North', 'South'], 'PDScore':[1, 2],
        'FICOGroup':['700+', '700+'], 'Segment':['New', 'Existing'],
        'Customer ID':['a', 'b'], 'Historical Average':[3, 4], 'SampleSize':[10, 15],
        'Snapshot Date':['2024-01-01'] * 2, 'Empty':[None, None],
        'Strategy / Option':['1000'] * 2, 'Sample mean':[3, 4],
        'Sample variance':[1, 1], 'Sample number':[10, 15],
    })
    assert [label for label, _ in grouping_columns(raw)] == ['FICOGroup', 'PDScore', 'Segment', 'Region']
    result = normalize_table(raw, grouping='FICOGroup', allow_pool=True)
    assert result.segment.tolist() == ['700+']
    assert result.sample_number.tolist() == [25]


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


def test_html_uses_only_selected_inputs_and_updates_for_new_data():
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
    first_text = build_html_summary(first).decode()
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
    second_text = build_html_summary(second).decode()
    assert all(value in second_text for value in ['Princess', 'PD high', 'RAR', '4,321', '2,500'])
    assert all(value not in second_text for value in ['Holland', 'FICO A', 'FICO B', 'Revolving Balance', 'Unused 3000'])
    assert first['data_fingerprint'] != second['data_fingerprint']
    assert first['arm_totals'] != second['arm_totals']
