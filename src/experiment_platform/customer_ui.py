"""Business-facing customer results: one dataset, overall and segment views."""
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st



def _p_text(p):
    return '<0.001' if p < .001 else f'{p:.3f}'


def average_chart(result, comparisons, binary, metric, control, colors, patterns):
    """Display average outcomes and the CI for each Test-Control difference.

    Difference intervals are shifted by the observed Control mean onto the
    average-outcome axis; they are not confidence intervals for each mean.
    """
    rows = result.sort_values('Mean / Rate', kind='stable')
    scale = 100 if binary else 1
    labels = rows['Credit Line'].astype(str).tolist()
    means = rows['Mean / Rate'].to_numpy(dtype=float) * scale
    control_mean = float(rows.loc[rows['Is Control'], 'Mean / Rate'].iloc[0])
    lookup = comparisons.set_index('Test option')
    fig = go.Figure(go.Bar(x=labels, y=means,
        marker=dict(color=[colors[name] for name in labels], pattern_shape=[patterns[name] for name in labels]),
        customdata=np.stack([rows['N'], ['Control' if x == control else 'Test' for x in labels]], axis=-1),
        hovertemplate='%{x} (%{customdata[1]})<br>Average: %{y:.2f}<br>Customers: %{customdata[0]}<extra></extra>',
        showlegend=False))
    annotation_text, annotation_y, all_bounds = [], [], list(means) + [0.0]
    for name, mean in zip(labels, means):
        if name == control:
            annotation_text.append(f'Control<br>Average {mean:,.2f}')
            annotation_y.append(mean)
            continue
        comparison = lookup.loc[name]
        low = (control_mean + float(comparison['CI lower'])) * scale
        high = (control_mean + float(comparison['CI upper'])) * scale
        all_bounds += [low, high]
        lift = float(comparison['Difference']) / control_mean * 100 if control_mean > 0 else None
        lift_text = f'Lift {lift:+,.1f}%' if lift is not None and np.isfinite(lift) else 'Lift unavailable'
        significance = 'Significant' if comparison['Significant'] else 'Not significant'
        annotation_text.append(f'Average {mean:,.2f}<br>{lift_text}<br>p {_p_text(comparison["Adjusted p-value"])}<br>{significance}')
        annotation_y.append(max(mean, high))
        fig.add_trace(go.Scatter(x=[name, name], y=[low, high], mode='lines+markers',
            line=dict(color='#334e5d', width=2), marker=dict(symbol='line-ew', size=10, color='#334e5d'),
            hovertemplate='Comparison interval (shifted): %{y:.2f}<extra></extra>', showlegend=False))
    ymin, ymax = min(all_bounds), max(all_bounds)
    span = max(ymax-ymin, abs(ymax)*.1, 1.)
    fig.add_hline(y=control_mean * scale, line_dash='dot', line_color='#92a5af', line_width=1,
                  annotation_text='Control average', annotation_position='top left')
    fig.add_trace(go.Scatter(x=labels, y=[v + span*.025 for v in annotation_y], text=annotation_text,
                             mode='text', textposition='top center', textfont=dict(size=11, color='#334e5d'),
                             hoverinfo='skip', showlegend=False))
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=75),
        xaxis=dict(title='Assigned strategy', categoryorder='array', categoryarray=labels),
        yaxis_title='Rate (%)' if binary else str(metric), plot_bgcolor='white', paper_bgcolor='white')
    fig.update_yaxes(gridcolor='#e5eef2', range=[ymin-span*.06, ymax+span*.42])
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
    return pd.DataFrame(rows).sort_values('Test option', kind='stable').reset_index(drop=True)


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
