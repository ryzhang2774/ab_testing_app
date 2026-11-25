import numpy as np
import pandas as pd
import pytest

from src.experiment_platform.customer_trend import _fit_cached, fit_historical_trend, recommend_test_lines, historical_trend_regions
from src.experiment_platform.customer_trend_charts import historical_trend_chart


def history(amounts=range(500, 9000, 500)):
    rng = np.random.default_rng(39)
    return pd.DataFrame([
        dict(partner=partner, segment='FICO group', strategy=f'Credit line {x}',
             sample_mean=80 + 850 * (1 - np.exp(-x / 1300)) + rng.normal(0, 20) + offset,
             sample_variance=250000., sample_number=1500 + int(x / 10), metric='Balance')
        for partner, offset in [('A', 0), ('B', 150)] for x in amounts
    ])


def test_fixed_partner_mix_removes_changing_composition_and_uses_mean_precision():
    rows = pd.DataFrame([
        dict(partner=p, strategy=str(x), sample_mean=mean, sample_variance=var, sample_number=n)
        for x in [500, 1000, 2000, 4000, 8000]
        for p, mean, var, n in [('A', 10., 100., 900 if x == 500 else 100),
                                 ('B', 100., 400., 100 if x == 500 else 900)]
    ])
    fit = fit_historical_trend(rows, ['A', 'B'])
    assert fit['available']
    assert dict(fit['partner_weights']) == {'A': .9, 'B': .1}
    assert fit['mean'] == pytest.approx([19] * 5)
    assert fit['se'][0] ** 2 == pytest.approx(.9 ** 2 * 100 / 900 + .1 ** 2 * 400 / 100)
    assert fit['se'][1] ** 2 == pytest.approx(.9 ** 2 * 100 / 100 + .1 ** 2 * 400 / 900)
    assert np.ptp(fit['fitted']) < 1e-5


def test_robust_fit_resists_one_spike_and_retains_input_statistics():
    rows = history()
    original = rows.copy(deep=True)
    clean = fit_historical_trend(rows, ['A', 'B'])
    spike = rows.copy()
    spike.loc[spike.strategy.eq('Credit line 4500'), 'sample_mean'] += 900
    fitted = fit_historical_trend(spike, ['A', 'B'])
    assert fitted['available']
    assert np.mean(abs(np.array(fitted['fitted']) - clean['fitted'])) < 80
    i = fitted['labels'].index('Credit line 4500')
    assert fitted['huber'][i] < .5
    suggestions = recommend_test_lines(fitted, 'Credit line 500')
    assert 'Credit line 4500' not in suggestions[:-1]
    assert suggestions[-1] == 'Credit line 8500'
    assert len(suggestions) == 4
    pd.testing.assert_frame_equal(rows, original)


def test_irregular_lines_are_numeric_and_suggestions_never_invent_bins():
    rows = history([250, 1000, 2000, 3500, 7000, 10500, 12000])
    fit = fit_historical_trend(rows.sample(frac=1, random_state=2), ['B', 'A'])
    assert fit['available']
    assert fit['x'] == (250, 1000, 2000, 3500, 7000, 10500, 12000)
    selected = recommend_test_lines(fit, 'Credit line 250')
    assert selected[-1] == 'Credit line 12000'
    assert set(selected) <= set(rows.strategy)
    assert fit['grid'][0] == 250 and fit['grid'][-1] == 12000
    assert np.isfinite(np.r_[fit['grid'], fit['fitted']]).all()


def test_missing_partners_are_excluded_without_imputation_and_guardrails_limit_options():
    rows = history()
    rows = rows[~(rows.partner.eq('B') & rows.strategy.eq('Credit line 8500'))]
    fit = fit_historical_trend(rows, ['A', 'B'])
    assert fit['skipped'] == 1
    assert 'Credit line 8500' not in fit['labels']
    assert recommend_test_lines(fit, 'Credit line 500')[-1] == 'Credit line 8000'
    allowed = ['Credit line 1000', 'Credit line 2500', 'Credit line 7000']
    suggested = recommend_test_lines(fit, 'Credit line 500', allowed)
    assert suggested[-1] == 'Credit line 7000'
    assert set(suggested) <= set(allowed)


def test_small_noisy_history_withholds_intermediate_points():
    rows = history()
    rows['sample_number'] = 3
    rows['sample_variance'] = 1e8
    fit = fit_historical_trend(rows, ['A', 'B'])
    assert fit['available'] and fit['noise_ratio'] > .12
    assert recommend_test_lines(fit, 'Credit line 500') == ['Credit line 8500']
    assert not fit_historical_trend(history([500, 1000, 2000, 3000]), ['A', 'B'])['available']
    categorical = history([500, 1000, 2000, 3000, 4000]).replace({'strategy': {'Credit line 500': 'BAU'}})
    assert not fit_historical_trend(categorical, ['A', 'B'])['available']
    assert not fit_historical_trend(history(), ['Missing partner'])['available']
    flat = history().assign(sample_mean=100., sample_variance=0.)
    assert recommend_test_lines(fit_historical_trend(flat, ['A', 'B']), 'Credit line 500') == ['Credit line 8500']


def test_cached_math_is_reused_for_control_changes_but_new_data_refits():
    rows = history()
    _fit_cached.cache_clear()
    fit = fit_historical_trend(rows, ['A', 'B'])
    first = _fit_cached.cache_info()
    for control in fit['labels'][:4]:
        recommend_test_lines(fit_historical_trend(rows, ['A', 'B']), control)
    assert _fit_cached.cache_info().misses == first.misses
    assert _fit_cached.cache_info().hits == first.hits + 4
    changed = rows.copy()
    changed.loc[0, 'sample_mean'] += 25
    fit_historical_trend(changed, ['A', 'B'])
    assert _fit_cached.cache_info().misses == first.misses + 1


def test_plot_keeps_recommendations_separate_from_manual_tests():
    from src.experiment_platform.customer_plan_ui import pool_statistics
    rows = history()
    # Preserve a real zigzag and a changing Partner mix rather than plotting
    # either the fitted curve or the fixed-mix means used for recommendations.
    rows.loc[(rows.partner == 'B') & (rows.strategy == 'Credit line 2000'), 'sample_number'] = 10000
    rows.loc[rows.strategy == 'Credit line 4500', 'sample_mean'] += 600
    pooled = pool_statistics(rows, ['strategy'])
    fit = fit_historical_trend(rows, ['A', 'B'])
    recommended = recommend_test_lines(fit, 'Credit line 500')
    tests = ['Credit line 2000', 'Credit line 7000']
    fig = historical_trend_chart(fit, 'Balance', 'Credit line 500', tests, recommended, history=pooled)
    point = next(trace for trace in fig.data if trace.name == 'Recommended')
    assert point.marker.color == 'white' and point.marker.line.color == '#008d85'
    assert list(point.x) == [float(label.split()[-1]) for label in recommended]
    guides = [shape for shape in fig.layout.shapes if shape.type == 'line']
    assert {shape.x0 for shape in guides} == {500, 2000, 7000}
    assert all(shape.x0 == shape.x1 for shape in guides)
    assert fig.layout.xaxis.type == 'linear'
    assert not any(trace.name in ('Robust fit', 'Ordinary weighted fit') for trace in fig.data)
    raw = next(trace for trace in fig.data if trace.name == 'Historical mean')
    assert raw.mode == 'lines+markers'
    assert list(raw.x) == sorted(int(label.split()[-1]) for label in pooled.strategy)
    expected = pooled.set_index('strategy')
    for value, mean, data in zip(raw.x, raw.y, raw.customdata):
        label, variance, n = data
        assert mean == expected.loc[label, 'sample_mean']
        assert variance == expected.loc[label, 'sample_variance']
        assert n == expected.loc[label, 'sample_number']
    assert raw.y[list(raw.x).index(4500)] > max(fit['fitted']) + 300
    for mean, data in zip(point.y, point.customdata):
        assert mean == expected.loc[data[0], 'sample_mean']
    assert any(row[3] == 'Not selected' for row in point.customdata)
    empty = historical_trend_chart(fit, 'Balance', 'Credit line 500', [], recommended, history=pooled)
    assert [shape.x0 for shape in empty.layout.shapes if shape.type == 'line'] == [500]


def test_regions_leave_uncertain_transition_and_stop_when_growth_resumes():
    xs = tuple(range(500, 4500, 500))
    fitted = (100, 200, 300, 400, 420, 430, 435, 440)
    trend = dict(available=True, x=xs, grid=xs, fitted=fitted,
                 noise_ratio=.02, cv_ratio=.03, knee=2000,
                 knee_bounds=(2000, 2500), knee_support=.9)
    assert historical_trend_regions(trend) == [
        {'label':'Growth region', 'x0':500., 'x1':2000.},
        {'label':'Possible plateau', 'x0':2500., 'x1':4000.},
    ]
    rebound = {**trend, 'fitted':(100, 200, 300, 400, 420, 430, 630, 850)}
    assert historical_trend_regions(rebound)[-1] == {'label':'Possible plateau', 'x0':2500., 'x1':3000.}
    unstable = historical_trend_regions({**trend, 'knee_support':.3})
    assert [region['label'] for region in unstable] == ['Growth region']
    assert historical_trend_regions({**trend, 'noise_ratio':.4}) == []
    assert historical_trend_regions({**trend, 'cv_ratio':.5}) == []
    assert historical_trend_regions({**trend, 'fitted':tuple(-y for y in fitted)}) == []
    assert historical_trend_regions({**trend, 'fitted':(100,) * len(xs)}) == []
    linear = {**trend, 'fitted':tuple(range(100, 900, 100)), 'knee':None, 'knee_bounds':None}
    assert historical_trend_regions(linear) == [{'label':'Growth region', 'x0':500., 'x1':4000.}]
