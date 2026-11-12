"""Business-facing customer results: one dataset, overall and segment views."""
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .customer_grouping import credit_line_amount, historical_strategy_order

STRATEGIES_PER_PAGE = 10

def _p_text(p):
    return '<0.001' if p < .001 else f'{p:.3f}'


def average_chart(result, comparisons, binary, metric, control, colors, patterns):
    """Display average outcomes and the CI for each Test-Control difference.

    Difference intervals are shifted by the observed Control mean onto the
    average-outcome axis; they are not confidence intervals for each mean.
    """
    # For named offers, retain the existing order by average. For credit lines,
    # order the strategies by limit even when the observed response falls.
    rows = result.sort_values('Mean / Rate', kind='stable').set_index('Credit Line')
    rows.index = rows.index.astype(str)
    labels = historical_strategy_order(rows.index)
    rows = rows.loc[labels]
    scale = 100 if binary else 1
    means = rows['Mean / Rate'].to_numpy(dtype=float) * scale
    control_mean = float(rows.loc[control, 'Mean / Rate'])
    lookup = comparisons.set_index('Test option')
    lookup.index = lookup.index.astype(str)
    lift_labels, hover, upper_errors, lower_errors = [], [], [], []
    for name, mean in zip(labels, means):
        details = [f'<b>{escape(name)}</b>', f'Average: {mean:,.2f}', f'Customers: {int(rows.at[name, "N"]):,}']
        if name == control:
            lift_labels.append('Control')
            details.append('Reference strategy')
            upper_errors.append(0.)
            lower_errors.append(0.)
        else:
            comparison = lookup.loc[name]
            difference = comparison['Difference']
            low, high = comparison['CI lower'], comparison['CI upper']
            lower_errors.append(max(0., (difference - low) * scale))
            upper_errors.append(max(0., (high - difference) * scale))
            lift = f'{100 * difference / control_mean:+,.1f}%' if control_mean > 0 else None
            significant = comparison['Significant']
            lift_labels.append((lift or '—') + (' ★' if significant else ''))
            details.extend([f'Lift {lift}' if lift else 'Lift unavailable',
                            f'p {_p_text(comparison["Adjusted p-value"])} · ' +
                            ('Significant' if significant else 'Not significant'),
                            f'Difference interval: {low*scale:+,.2f} to {high*scale:+,.2f}' +
                            (' percentage points' if binary else '')])
        hover.append('<br>'.join(details))

    fig = go.Figure(go.Bar(x=labels, y=means, hovertext=hover, hovertemplate='%{hovertext}<extra></extra>',
        marker=dict(color=[colors[name] for name in labels], pattern_shape=[patterns[name] for name in labels]),
        error_y=dict(type='data', symmetric=False, array=upper_errors, arrayminus=lower_errors,
                     color='#314b5c', thickness=2, width=6), showlegend=False))
    low = min(0., min(means - lower_errors))
    high = max(0., max(means + upper_errors))
    span = max(high-low, abs(high)*.1, 1.)
    fig.add_hline(y=control_mean * scale, line_dash='dot', line_color='#728897', line_width=1)
    fig.add_trace(go.Scatter(x=labels, y=means + upper_errors + span*.045, text=lift_labels,
                             mode='text', textposition='top center', textfont=dict(size=14, color='#16384d'),
                             hoverinfo='skip', showlegend=False))
    amounts = [credit_line_amount(label) for label in labels]
    monetary = any(value is not None for value in amounts) and all(
        amount is not None or label == control for amount, label in zip(amounts, labels))
    ticks = []
    for name, value in zip(labels, amounts):
        if monetary and value is not None:
            ticks.append(f'{value:,.0f}' if value.is_integer() else f'{value:,}')
        else:
            ticks.append(name)
    page_size = STRATEGIES_PER_PAGE
    paged = len(labels) > page_size
    pages = []
    if paged:
        for start in range(0, len(labels), page_size):
            stop = min(start + page_size, len(labels))
            pages.append(dict(label=f'{ticks[start]} – {ticks[stop - 1]}', method='relayout',
                              args=[{'xaxis.range': [start - .5, stop - .5]}]))
    fig.update_layout(height=430 if paged else 360, bargap=.30, font=dict(size=14),
        margin=dict(l=20, r=24, t=62 if paged else 18, b=18), hoverlabel=dict(align='left', font_size=14),
        xaxis=dict(title='Credit line ($)' if monetary else 'Assigned strategy', type='category',
                   categoryorder='array', categoryarray=labels, tickmode='array', tickvals=labels, ticktext=ticks,
                   tickfont=dict(size=14), automargin=True, fixedrange=paged,
                   range=[-.5, page_size - .5] if paged else None),
        updatemenus=[dict(type='dropdown', buttons=pages, x=0, xanchor='left', y=1.04, yanchor='bottom',
                          bgcolor='white', bordercolor='#b8cfda')] if paged else [],
        yaxis_title='Rate (%)' if binary else str(metric), plot_bgcolor='white', paper_bgcolor='white')
    fig.update_yaxes(gridcolor='#e5eef2', range=[low-span*.05, high+span*.16], automargin=True)
    return fig


def _result_rows(result, scope, metric_type, frame, arm_col, outcome_col, control, group_col, alpha):
    control_row = result[result['Is Control']].iloc[0]
    rows = []
    for _, row in result[~result['Is Control']].iterrows():
        if 'Method' in row and isinstance(row['Method'], str):
            method = row['Method']
        elif metric_type == 'Continuous':
            method = 'Welch t-test'
        else:
            pair = frame[frame[arm_col].astype(str).str.strip().isin([control, str(row['Credit Line'])])]
            if scope != 'Overall' and group_col:
                pair = pair[pair[group_col].astype(str) == scope]
            rate = pd.to_numeric(pair[outcome_col], errors='coerce').mean()
            counts = pair.groupby(arm_col).size()
            method = 'Fisher exact test' if len(counts) < 2 or min(counts.min()*rate, counts.min()*(1-rate)) < 5 else 'Two-proportion z-test'
        rows.append({'Scope':scope, 'Test option':row['Credit Line'], 'Control N':int(control_row['N']), 'Test N':int(row['N']),
                     'Control mean / rate':control_row['Mean / Rate'], 'Test mean / rate':row['Mean / Rate'],
                     'Difference':row['Effect vs Control'], 'CI lower':row['CI Lower'], 'CI upper':row['CI Upper'],
                     'Test statistic':row['Statistic'], 'Raw p-value':row['Raw p-value'], 'Adjusted p-value':row['Adjusted p-value'],
                     'Significant':bool(row['Adjusted p-value'] < alpha), 'Method':method})
    frame = pd.DataFrame(rows).sort_values('Test option', kind='stable').set_index('Test option', drop=False)
    return frame.loc[historical_strategy_order(frame.index)].reset_index(drop=True)


def render_decision_card(insight):
    insight_rows = ''.join(
        '<div class="insight-row"><div class="insight-option">' + escape(item['test']) + '</div>'
        '<span class="insight-badge ' + escape(item['tone']) + '">' + escape(item['status']) + '</span>'
        '<div class="insight-detail">' + escape(item['detail']) + '</div></div>'
        for item in insight['comparisons']
    )
    observations = ''.join('<p class="insight-observation">' + escape(item) + '</p>' for item in insight['observations'])
    st.markdown(
        '<style>'
        '.insight-card{border:1px solid #d1e4ed;border-radius:10px;background:#f7fbfd;padding:1.2rem 1.35rem;margin:.3rem 0 1rem;color:#153449}'
        '.insight-heading{display:flex;align-items:center;gap:.65rem;color:#006b9b;font-size:.85rem;font-weight:750;letter-spacing:.02em}'
        '.insight-sparkle{display:grid;place-items:center;width:2rem;height:2rem;border-radius:8px;background:#d9f2fc;font-size:1.1rem}'
        '.insight-summary{margin:.8rem 0 1rem;line-height:1.55;color:#153449;font-size:.93rem}'
        '.insight-row{display:flex;align-items:center;gap:.65rem;flex-wrap:wrap;border-top:1px solid #e0edf2;padding:.7rem 0}'
        '.insight-option{font-weight:700;min-width:10rem;color:#153449}.insight-detail{font-size:.84rem;color:#526a79;flex:1;min-width:13rem}'
        '.insight-badge{font-size:.69rem;font-weight:700;padding:.22rem .5rem;border-radius:20px;background:#e9eff2;color:#526a79}'
        '.insight-badge.positive{background:#e0f3ee;color:#11675b}.insight-badge.negative{background:#fbeae8;color:#a03b32}'
        '.insight-observation{border-top:1px solid #e0edf2;padding:.8rem 0 0;margin:.15rem 0 0;color:#36596c;font-size:.84rem}'
        '</style><div class="insight-card"><div class="insight-heading"><span class="insight-sparkle">✦</span> RESULT INSIGHTS</div>'
        '<p class="insight-summary">' + escape(insight['primary']) + '</p>' + insight_rows + observations + '</div>',
        unsafe_allow_html=True,
    )


def simple_customer_analysis(render_evidence) -> None:
    st.session_state.customer_analysis_is_current = False
    st.markdown('### Analyze results')
    st.caption('Compare randomized Control and Test strategies using experiment results.')
    source = st.radio('Data source', ['Upload file', 'Databricks', 'Synthetic demo'], index=2, horizontal=True, key='simple_source')
    if source == 'Databricks':
        st.info('Databricks is not connected yet. Choose Upload file or Synthetic demo to continue.')
        return
    from .customer_summary_ui import render_summary_analysis
    render_summary_analysis(source)
