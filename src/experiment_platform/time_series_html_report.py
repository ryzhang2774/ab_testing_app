"""Portable, self-contained reports for time-series planning and analysis."""
from html import escape
import math
import pandas as pd


def _e(value):
    return escape(str(value), quote=True)


def _num(value, digits=2):
    try:
        v = float(value)
        return f'{v:,.{digits}f}' if math.isfinite(v) else 'Not available'
    except (TypeError, ValueError):
        return 'Not available'


def _precise(value):
    try:
        v = float(value)
        return format(v, '.17g') if math.isfinite(v) else 'Not available'
    except (TypeError, ValueError):
        return 'Not available'


def _table(headers, rows):
    head = ''.join(f'<th>{_e(v)}</th>' for v in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{_e(v)}</td>' for v in row) + '</tr>' for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _document(title, content):
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{_e(title)}</title><style>
body{{font:15px/1.5 Arial,sans-serif;color:#173d55;max-width:1120px;margin:35px auto;padding:0 24px}}h1,h2{{color:#006b9b}}h2{{border-bottom:1px solid #d7e4eb;padding-bottom:8px;margin-top:32px}}.note{{background:#eaf6fb;border-left:4px solid #00aeef;padding:12px}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:7px;text-align:left;border-bottom:1px solid #d7e4eb}}th{{background:#eaf6fb}}.table-wrap{{overflow-x:auto}}@media print{{.table-wrap{{overflow:visible}}thead{{display:table-header-group}}tr{{break-inside:avoid}}}}</style></head><body><h1>{_e(title)}</h1>{content}</body></html>'''.encode('utf-8')


def build_analysis_html(bundle, cfg, source, group, label):
    sections = []
    for item in bundle:
        pp, model = item['prepost'], item['model']
        sections.append(f"<h2>{_e(item['name'])}</h2>")
        if pp is None:
            sections.append(f"<p>Not calculated: {_e(item['error'])}</p>")
            continue
        test = pp.get('significance_test', {})
        sig = ('Significant' if test.get('significant') else 'Not significant') if test.get('available') else 'Unavailable: ' + str(test.get('reason', 'not estimated'))
        p = ('<0.0001' if test['p_value'] < .0001 else _num(test['p_value'], 4)) if test.get('available') else 'Not available'
        sections.append(_table(['Before mean', 'After mean', 'Change', 'Lift', 'Exploratory test', 'p-value'], [[
            _num(pp['pre_value']), _num(pp['post_value']), _num(pp['absolute_change']),
            _num(pp['percent_change'] * 100, 2) + '%' if pd.notna(pp['percent_change']) and pp['pre_value'] > 0 else 'Not available', sig, p]]))
        sections.append(f"<p>Matching windows: {_e(pp['pre_start'].date())} to {_e(pp['pre_end'].date())} ({pp['pre_observations']} periods), and {_e(pp['post_start'].date())} to {_e(pp['post_end'].date())} ({pp['post_observations']} periods).</p>")
        if model:
            low, high = model['prediction_interval']
            sections.append('<h3>Actual versus modelled no-campaign outcome</h3>')
            sections.append(_table(['Actual total', 'Estimated no-campaign total', 'Difference', 'Lift', 'Model interval', 'Validation'], [[
                _num(model['observed_total']), _num(model['expected_total']), _num(model['cumulative_impact']),
                _num(model['relative_impact'] * 100, 2) + '%' if model['expected_total'] > 0 and pd.notna(model['relative_impact']) else 'Not available',
                f"{_num(low)} to {_num(high)} ({model['interval_level']:.0%})", model.get('reliability','Review')]]))
            if model.get('estimation') == 'Bayesian monthly local level':
                sections.append(f"<p>Bayesian monthly counterfactual fitted on {model['pre_observations']} pre-launch "
                                f"months and evaluated on {model['post_observations']} campaign months. "
                                f"{_e(model['uncertainty_note'])}</p>")
            spec = model.get('model_specification', {})
            sections.append(f"<p>Seasonal cycle: {_e(model.get('seasonal_period') or 'not included')} periods; trend: {_e(spec.get('level', 'Not recorded'))}; comparison signals: {_e(', '.join(model.get('covariates', [])) or 'none')}.</p>")
            for warning in model.get('warnings', []): sections.append(f'<p class="note">{_e(warning)}</p>')
            sections.append('<h3>Counterfactual by period</h3>')
            sections.append(_table(['Date', 'Actual', 'Estimated without campaign', 'Lower forecast bound', 'Upper forecast bound', 'Difference'],
                [[str(r['_date'].date()), _num(r['_outcome']), _num(r['counterfactual']), _num(r['lower']), _num(r['upper']), _num(r['impact'])] for r in model['result'].to_dict('records')]))
        else:
            sections.append(f"<p class='note'>Model unavailable: {_e(item.get('error') or 'not estimated')}</p>")
        sections.append('<h3>Aggregated observations used</h3>')
        signals = [c for c in getattr(cfg, 'selected_covariates', []) if c in item['data'].columns]
        sections.append(_table(['Date', label] + signals, [[str(r['_date'].date()), _precise(r['_outcome'])] + [_precise(r[c]) for c in signals] for r in item['data'].to_dict('records')]))
    body = f"<p>Source: {_e(source)} · Metric: {_e(label)} · Group by: {_e(group)} · Launch: {_e(cfg.intervention_date)} · Campaign end: {_e(cfg.campaign_end_date or 'latest observation')}.</p>"
    body += "<p class='note'>This is an exploratory observational time-series analysis. Neither method alone proves the campaign caused the change.</p>" + ''.join(sections)
    body += '''<h2>How the results are calculated</h2><p>Before and after are averages over equal-length windows. Difference = after average − before average; lift = difference / before average when the baseline is positive.</p>
<p>The exploratory two-sided significance test compares period means using a regression with Newey–West HAC standard errors, allowing some serial correlation. At least 8 periods in each window and a stable standard error are required. This test does not remove trend, seasonality, other events, or adjust across segments.</p>
<p>The structural state-space model fits the pre-period and forecasts a no-campaign baseline. Modelled impact = actual − predicted; cumulative impact is the sum over the campaign period; model lift = cumulative impact / predicted total when positive. Forecast intervals are conditional on the fitted model and assumptions. Comparison signals must not be affected by the campaign. Short history and unavailable backtesting reduce confidence. Review model diagnostics, seasonal coverage and external events.</p>'''
    return _document('Time Series Analysis Report', body)


def build_plan_html(plan, data, cfg):
    rows = []
    for r in plan.get('power', []):
        low, high = r.get('power_ci', (None,None))
        mde = r.get('minimum_detectable_effect')
        if mde is not None and math.isfinite(float(mde)) and plan.get('effect_type') == 'Relative Lift (%)': mde *= 100
        rows.append([r.get('duration'), _num(r.get('power'), 4), f'{_num(low,4)} to {_num(high,4)}', _num(mde,4), r.get('calibration_status'), r.get('status')])
    signals = [c for c in plan.get('predictors', []) if c in data.columns]
    inputs = [[str(r['_date'].date()), _precise(r['_outcome'])] + [_precise(r[c]) for c in signals] for r in data.to_dict('records')]
    body = f"<p>Metric: {_e(cfg.outcome_column)} · Frequency: {_e(cfg.frequency)} · Planned launch: {_e(cfg.planned_launch_date)} · Historical periods: {len(data)}.</p>"
    body += _table(['Setting','Value'], [['Target power', f"{plan.get('target_power', 0):.0%}"], ['False positive tolerance', f"{plan.get('alpha', 0):.0%}"],
        ['Effect scenario', f"{plan.get('effect')} {plan.get('effect_type')}"] if plan.get('effect') is not None else ['Effect scenario', 'Estimate detectable effect'],
        ['Ramp periods', plan.get('ramp')], ['Validation mode', plan.get('validation_mode')], ['Recommended duration', plan.get('recommended_duration') or 'Not established'],
        ['Signals', ', '.join(plan.get('predictors', [])) or 'None'], ['Seasonality used', plan.get('model_seasonality', 'Not recorded')]])
    body += '<h2>Duration and detectability</h2><p>MDE units follow the selected effect type. Power is per comparison; simulation intervals describe simulation precision.</p>'
    body += _table(['Duration (periods)', 'Estimated power', 'Simulation interval', 'MDE', 'Calibration', 'Status'], rows)
    body += '<h2>Historical outcome and signals used</h2>' + _table(['Date', cfg.outcome_column] + signals, inputs)
    body += '''<h2>How the plan is calculated</h2><p>Only observations before the planned launch are used. For each candidate duration, the app estimates whether a specified positive effect could be detected at the selected false-positive tolerance and power. Planning uses a one-sided increase, while the analysis model reports a two-sided interval; their detection probabilities need not match. The effect is a planning scenario, not a forecast.</p>
<p>When sufficient history is available, historical pseudo-launches are split into calibration and evaluation windows. A structural model produces no-campaign paths; the effect is added to those paths and detection frequency is measured. Candidate recommendations require the simulation power interval lower bound to reach target and calibration checks to pass.</p>
<p>When historical validation is unavailable, the planner uses a labelled assumption-based approximation based on detrended residual variability and an AR(1) autocorrelation estimate. It does not empirically validate the false-positive rate. In this mode, the target is assessed using point-estimated power.</p>
<p>A duration is a number of time periods, not customers. Results depend on the historical window, forecast assumptions, seasonal coverage, comparison signals and possible other events. No plan guarantees a causal conclusion.</p>'''
    return _document('Time Series Planning Report', body)
