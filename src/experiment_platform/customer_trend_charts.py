"""Observed historical statistics with suggestions and manual assignments."""
import numpy as np
import plotly.graph_objects as go

from .customer_grouping import credit_line_amount
from .customer_trend import historical_trend_regions


def historical_trend_chart(trend, primary, control, tests, recommended, *, history):
    fig = go.Figure()
    rows = history.copy()
    rows['_amount'] = rows['strategy'].map(credit_line_amount)
    rows = rows.dropna(subset=['_amount', 'sample_mean']).sort_values('_amount')
    grid, curve = rows['_amount'].tolist(), rows['sample_mean'].tolist()
    observed = rows.set_index('strategy')
    def metadata(label):
        row = observed.loc[label]
        return [label, float(row['sample_variance']), int(row['sample_number'])]
    historical_hover = ('%{customdata[0]}<br>Historical mean: %{y:,.2f}'
                        '<br>Sample variance: %{customdata[1]:,.2f}'
                        '<br>Sample number: %{customdata[2]:,d}')
    for region in historical_trend_regions(trend):
        growth = region['label'] == 'Growth region'
        fig.add_shape(type='rect', x0=region['x0'], x1=region['x1'], y0=0, y1=1,
                      xref='x', yref='y domain', layer='below', line=dict(width=0),
                      fillcolor='rgba(0,126,184,0.08)' if growth else 'rgba(230,168,45,0.14)')
        label = region['label']
        if (region['x1'] - region['x0']) / (grid[-1] - grid[0]) < .25:
            label = label.replace(' ', '<br>', 1)
        fig.add_annotation(x=(region['x0'] + region['x1']) / 2, y=1.13, yref='paper',
                           text=label, showarrow=False, xanchor='center', yanchor='bottom',
                           font=dict(size=12, color='#006b9b' if growth else '#805817'))
    fig.add_scatter(x=grid, y=curve, mode='lines+markers', name='Historical mean',
                    line=dict(color='#007eb8', width=2), marker=dict(size=6, color='#007eb8'),
                    customdata=[metadata(label) for label in rows['strategy']],
                    hovertemplate=historical_hover + '<extra>Historical data</extra>')
    for label in tests:
        value = credit_line_amount(label)
        if value is None or label not in observed.index:
            continue
        fig.add_shape(type='line', x0=value, x1=value, y0=0, y1=1, xref='x', yref='y domain',
                      layer='below', line=dict(color='#315b86', width=1.4, dash='dash'))
        if label not in recommended:
            fig.add_scatter(x=[value], y=[float(observed.loc[label, 'sample_mean'])], mode='markers',
                            marker=dict(color='#315b86', size=8), name='Your Test', showlegend=False,
                            customdata=[metadata(label)], hovertemplate=historical_hover + '<extra>Your Test</extra>')
    fig.add_scatter(x=[None], y=[None], mode='lines', name='Your Tests',
                    line=dict(color='#315b86', width=1.4, dash='dash'), hoverinfo='skip')
    baseline = credit_line_amount(control)
    if baseline is not None and control in observed.index:
        fig.add_shape(type='line', x0=baseline, x1=baseline, y0=0, y1=1, xref='x', yref='y domain',
                      line=dict(color='#526b7b', width=1.5, dash='dot'))
        fig.add_scatter(x=[baseline], y=[float(observed.loc[control, 'sample_mean'])], mode='markers',
                        marker=dict(symbol='square', size=9, color='#526b7b'), showlegend=False,
                        customdata=[metadata(control)], hovertemplate=historical_hover + '<extra>Control</extra>')
        fig.add_annotation(x=baseline, y=1.02, yref='paper', text='Control', showarrow=False,
                           xanchor='left' if baseline < np.median(grid) else 'right', font=dict(size=12, color='#526b7b'))
    recommended = [label for label in recommended if label in observed.index]
    amounts = [credit_line_amount(label) for label in recommended]
    means = [float(observed.loc[label, 'sample_mean']) for label in recommended]
    for size, opacity in [(32, .06), (24, .12), (18, .18)]:
        fig.add_scatter(x=amounts, y=means, mode='markers', showlegend=False, hoverinfo='skip',
                        marker=dict(size=size, color=f'rgba(0,181,172,{opacity})', line=dict(width=0)))
    positions = []
    spread = max(float(np.ptp(curve)), 1e-8)
    for i, (value, mean) in enumerate(zip(amounts, means)):
        close = (i > 0 and value - amounts[i - 1] < .18 * (grid[-1] - grid[0])
                 and abs(mean - means[i - 1]) < .12 * spread)
        positions.append(('bottom left' if close else 'top left') if value == grid[-1]
                         else 'bottom center' if close else 'top center')
    fig.add_scatter(x=amounts, y=means, mode='markers+text', name='Recommended',
                    marker=dict(size=13, color='white', line=dict(color='#008d85', width=2.5)),
                    text=[f'{value:,.0f}' for value in amounts], textposition=positions,
                    textfont=dict(size=12, color='#31576e'), cliponaxis=False,
                    customdata=[metadata(label) + ['Selected' if label in tests else 'Not selected',
                                 'Upper endpoint' if label == recommended[-1] else 'Suggested intermediate'] for label in recommended],
                    hovertemplate=historical_hover + '<br>%{customdata[4]} · %{customdata[3]}<extra>Recommended</extra>')
    numeric = np.asarray(curve)
    span = max(float(np.ptp(numeric)), abs(float(np.mean(numeric))) * .1, 1e-6)
    low, high = float(np.min(numeric) - .08 * span), float(np.max(numeric) + .15 * span)
    axis_name = 'Credit line' if any('credit' in label.casefold() or '$' in label for label in trend['labels']) else 'Strategy value'
    fig.update_layout(height=380, margin=dict(t=64, b=64, l=48, r=16),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Geist, Segoe UI, sans-serif', size=12, color='#526b7b'),
                      legend=dict(orientation='h', y=-.25, x=0, font=dict(size=12)),
                      xaxis=dict(title=axis_name, type='linear', range=[grid[0], grid[-1]],
                                 nticks=6, tickformat=',.0f', gridcolor='#e4edf3', zeroline=False),
                      yaxis=dict(title=primary, range=[low, high], nticks=5,
                                 tickformat=',.2~f', gridcolor='#e4edf3', zeroline=False))
    return fig
