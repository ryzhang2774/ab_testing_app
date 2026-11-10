"""Presentation components for the desktop experiment workspace."""
from functools import lru_cache
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from .customer_grouping import ordered_groups

OCEAN = '#0075a5'
CYAN = '#00a7ce'
PALE = '#87cee4'
GRID = '#d7e7ee'
ASSETS = Path(__file__).parent / 'assets' / 'workspace'


@lru_cache(maxsize=1)
def workspace_css():
    return '\n'.join((ASSETS / name).read_text(encoding='utf-8')
                     for name in ('bulma.min.css', 'workspace.css'))


def location_trail(location):
    return (f'<div class="bulma-breadcrumb workspace-trail" aria-label="Location">'
            f'<ul><li>Experiment Platform</li><li>{escape(location)}</li></ul></div>')


def page_header(title, description, location=None):
    trail = location_trail(location) if location else ''
    st.markdown(
        f'<header class="workspace-mast">{trail}<h1 class="bulma-title">{escape(title)}</h1>'
        f'<p class="bulma-subtitle">{escape(description)}</p></header>', unsafe_allow_html=True)


def method_diagram(workspace):
    """Small schematics; coordinates are diagram positions, not portfolio data."""
    def chart(rows):
        return alt.Chart(pd.DataFrame(rows))

    x = alt.X('x:Q', scale=alt.Scale(domain=[0, 100], nice=False), axis=None)
    y = alt.Y('y:Q', scale=alt.Scale(domain=[0, 60], nice=False), axis=None)
    def line(points, color, **style):
        return chart([{'x': a, 'y': b, 'i': i} for i, (a,b) in enumerate(points)]).mark_line(
            color=color, strokeWidth=2, **style).encode(x=x, y=y, order='i:Q')

    def label(a, b, text):
        return chart([{'x': a, 'y': b, 'text': text}]).mark_text(
            color=OCEAN, fontSize=12, fontWeight=600).encode(x=x, y=y, text='text:N')

    layers = [chart([{'y': i} for i in range(8,60,11)]).mark_rule(color=GRID, opacity=.75).encode(y=y)]
    if workspace == 'Customer':
        layers += [label(18, 32, 'Accounts')]
        for level, color, role in [(43, OCEAN, 'Control'), (19, CYAN, 'Test')]:
            layers += [line([(31,32),(40,32),(40,level),(52,level)], color),
                       chart([{'x': 56+i*6, 'y': level} for i in range(6)]).mark_circle(
                           size=100, opacity=1, color=color).encode(x=x,y=y),
                       label(72,level+6,role)]
    elif workspace == 'Geography':
        for start, color, role in [(14,OCEAN,'Test markets'),(61,PALE,'Control markets')]:
            layers += [chart([{'x':start+8*i,'y':17+j*9} for i in range(4) for j in range(3)])
                       .mark_square(size=180, color=color, opacity=.95).encode(x=x, y=y),
                       label(start+12,47,role)]
    else:
        layers += [chart([{'x':51}]).mark_rule(color=CYAN,strokeDash=[4,3]).encode(x=x),
                   label(51,56,'Launch'),
                   line([(1,13),(10,16),(20,22),(32,26),(42,30),(51,30),(60,36),(70,40),(82,46),(98,50)],
                        OCEAN, interpolate='step-after'),
                   line([(51,30),(65,31),(80,32),(98,34)],CYAN,strokeDash=[5,4])]
    return (alt.layer(*layers).properties(height=170, description=f'{workspace} experiment design schematic')
            .configure_view(stroke=None).configure(background='transparent', padding=4))


def method_card(column, title, description, key, workspace, open_workspace):
    with column:
        st.altair_chart(method_diagram(workspace), use_container_width=True, theme=None)
        st.markdown(f'<article class="bulma-card method-copy"><div class="bulma-card-content">'
                    f'<h2 class="bulma-title">{escape(title)}</h2><p>{escape(description)}</p></div>'
                    f'</article>', unsafe_allow_html=True)
        st.button('Open workspace →', key=key, help=f'Open {title}', use_container_width=True,
                  on_click=open_workspace, args=(workspace,))


def history_detail_charts(data, dimension, dimension_label, metric, options):
    """Both panels use the same ordered strategy axis and the same underlying cells."""
    cells = data.assign(historical_sd=data.sample_variance.pow(.5)).copy()
    if dimension == 'strategy':
        cells['scope'] = 'Overall'
    else:
        cells['scope'] = cells[dimension]
    group_order = ordered_groups(cells.scope.unique(), dimension_label)
    x = alt.X('strategy:N', sort=list(options), title='Strategy',
              axis=alt.Axis(labelExpr="replace(datum.label, /^Credit line /i, '')", labelLimit=100))
    scope = alt.Tooltip('scope:N', title=dimension_label if dimension != 'strategy' else 'Scope')
    tips = [scope, alt.Tooltip('strategy:N', title='Strategy'),
            alt.Tooltip('sample_number:Q', title='Historical customers', format=',d'),
            alt.Tooltip('historical_sd:Q', title=f'{metric} standard deviation', format=',.2f')]
    base = alt.Chart(cells[['scope','strategy','sample_number','historical_sd']]).encode(x=x, tooltip=tips)
    count = base.mark_bar(color=OCEAN, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        y=alt.Y('sample_number:Q',title='Customers',axis=alt.Axis(format='~s')))
    if len(group_order) > 1:
        count = count.encode(xOffset=alt.XOffset('scope:N', sort=group_order),
                             color=alt.Color('scope:N', title=dimension_label, sort=group_order,
                                             scale=alt.Scale(scheme='tableau10'),
                                             legend=alt.Legend(orient='bottom', symbolType='square')))
    spread = base.mark_rect(stroke='white', strokeWidth=1).encode(
        y=alt.Y('scope:N',sort=group_order,title=None,axis=alt.Axis(labelLimit=150)),
        color=alt.Color('historical_sd:Q',title='Standard deviation',
                        scale=alt.Scale(scheme='tealblues',zero=True),legend=alt.Legend(orient='bottom')))
    return count.properties(height=180), spread.properties(height=alt.Step(30))


def render_history_details(data, dimension, axis_label, primary, options):
    count, spread = history_detail_charts(data, dimension, axis_label, primary, options)
    for title, chart in [('Historical sample count',count), (f'{primary} standard deviation',spread)]:
        st.markdown(f'**{title}**')
        chart = chart.configure_view(stroke=None).configure_axis(
            gridColor='#e3edf1', domainColor=GRID, tickColor=GRID, labelColor='#456272', titleColor='#456272'
        ).configure(background='transparent')
        st.altair_chart(chart, use_container_width=True, theme=None)
