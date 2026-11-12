"""Presentation components for the desktop experiment workspace."""
from base64 import b64encode
from html import escape
from pathlib import Path

import altair as alt
import streamlit as st

from .customer_grouping import ordered_groups
from .customer_palette import STRATEGY_COLORS

GRID = '#d5e4ed'
ASSETS = Path(__file__).parent / 'assets' / 'workspace'


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


def method_card(column, title, description, key, workspace, open_workspace):
    subject, _, noun = title.rpartition(" ")
    slug = workspace.lower().replace(" ", "-")
    image = b64encode((ASSETS / f"chooser-{slug}.jpg").read_bytes()).decode("ascii")
    with column:
        st.markdown(
            f'<article class="method-copy"><div class="method-photo">'
            f'<img src="data:image/jpeg;base64,{image}" alt=""></div>'
            f'<div class="method-body"><h2>{escape(subject)} <span>{escape(noun)}</span></h2>'
            f'<p>{escape(description)}</p></div></article>', unsafe_allow_html=True)
        st.button("Open workspace", key=key, use_container_width=True,
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
    count = base.mark_bar(color='#00689d', cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        y=alt.Y('sample_number:Q',title='Customers',axis=alt.Axis(format='~s')))
    if len(group_order) > 1:
        count = count.encode(xOffset=alt.XOffset('scope:N', sort=group_order),
                             color=alt.Color('scope:N', title=dimension_label, sort=group_order,
                                             scale=alt.Scale(scheme='tableau10'),
                                             legend=alt.Legend(orient='bottom', symbolType='square')))
    spread = base.mark_rect(stroke='white', strokeWidth=1).encode(
        y=alt.Y('scope:N',sort=group_order,title=None,axis=alt.Axis(labelLimit=150)),
        color=alt.Color('historical_sd:Q',title='Standard deviation',
                        scale=alt.Scale(range=['#edf6fb', '#72b9d8', '#005781'],zero=True),legend=alt.Legend(orient='bottom')))
    return count.properties(height=180), spread.properties(height=alt.Step(30))


def render_history_details(data, dimension, axis_label, primary, options):
    count, spread = history_detail_charts(data, dimension, axis_label, primary, options)
    for title, chart in [('Historical sample count',count), (f'{primary} standard deviation',spread)]:
        st.markdown(f'**{title}**')
        chart = chart.configure_view(stroke=None).configure_axis(
            gridColor='#e1ebf2', domainColor=GRID, tickColor=GRID, labelColor='#496476', titleColor='#496476'
        ).configure(background='transparent')
        st.altair_chart(chart, use_container_width=True, theme=None)


def allocation_bars(labels, counts, denominator, share_label, roles=None):
    """Label each percentage against its stated denominator, including small allocations."""
    rows = []
    for i, (label, count) in enumerate(zip(labels, counts)):
        share = count / denominator
        percentage = '<0.1%' if 0 < share < .001 else f'{share:.1%}'
        role = roles.iloc[i] if roles is not None else ''
        role_text = f'<small>{escape(role)}</small>' if role else ''
        rows.append(
            f'<tr><th scope="row">{escape(str(label))}{role_text}</th>'
            f'<td class="alloc-track-cell"><div class="alloc-track" aria-hidden="true">'
            f'<i class="{"alloc-test" if role == "Test" else "alloc-base"}" '
            f'style="width:{share:.6%}"></i></div></td>'
            f'<td>{count:,}</td><td>{escape(percentage)}</td></tr>'
        )
    st.markdown(
        '<table class="allocation-chart"><thead><tr>'
        f'<th scope="col">{"Strategy" if roles is not None else "Group"}</th>'
        '<th scope="col" aria-label="Allocation bar"></th><th scope="col">Customers</th>'
        f'<th scope="col">{escape(share_label)}</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>', unsafe_allow_html=True)


def recruitment_composition(groups, counts, total, grouping):
    bands, legend = [], []
    for index, (group, count) in enumerate(zip(groups, counts)):
        fraction = count / total
        ink = STRATEGY_COLORS[index % len(STRATEGY_COLORS)]
        label = escape(str(group))
        percent = '&lt;0.1%' if 0 < fraction < .001 else f'{fraction:.1%}'
        bands.append(f'<i style="width:{fraction:.8%};background:{ink}"></i>')
        legend.append(f'<li><span class="cohort-label"><i style="background:{ink}"></i>{label}</span>'
                      f'<strong>{count:,}</strong><span>{percent}</span></li>')
    st.markdown(
        f'<section class="cohort-figure"><h2>Recruitment by {escape(str(grouping))}</h2>'
        f'<div class="cohort-band" aria-hidden="true">{"".join(bands)}</div>'
        '<div class="cohort-labels" aria-hidden="true"><span>Group</span><span>Customers</span><span>Share of plan</span></div>'
        f'<ul>{"".join(legend)}</ul></section>', unsafe_allow_html=True)
