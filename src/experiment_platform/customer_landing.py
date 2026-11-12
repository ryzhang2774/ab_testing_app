"""Credit-line example for the Customer experiment landing page."""
from html import escape

import altair as alt
import streamlit as st

from .customer_demo import historical_demo
from .customer_plan_ui import pool_statistics


@st.cache_data(show_spinner=False)
def credit_line_preview():
    history = historical_demo().rename(columns={
        'FICO group': 'segment', 'Strategy': 'strategy', 'Metric': 'metric',
        'Sample mean': 'sample_mean', 'Sample number': 'sample_number',
        'Sample variance': 'sample_variance',
    })
    rows = pool_statistics(history, ['segment', 'strategy', 'metric'])
    rows['limit'] = rows.strategy.str.removeprefix('Credit line ').astype(int)
    return rows


def credit_line_example():
    left, right = st.columns([2, 1], gap='medium')
    preview = credit_line_preview()
    with left:
        with st.container():
            st.markdown('<h2 class="customer-figure-title">Historical outcomes by credit line</h2>', unsafe_allow_html=True)
            metric = st.radio('Demo metric', preview.metric.unique().tolist(), horizontal=True,
                              key='customer_landing_metric')
            rows = preview.loc[preview.metric.eq(metric)]
            segments = sorted(rows.segment.unique())
            palette = ['#006ba2', '#008679', '#ad650e']
            color = alt.Color('segment:N', title=None, sort=segments,
                              scale=alt.Scale(domain=segments, range=palette),
                              legend=alt.Legend(orient='bottom', columns=3))
            highlight = alt.selection_point(fields=['segment'], bind='legend')
            chart = alt.Chart(rows).mark_line(point=True, strokeWidth=2.5).encode(
                x=alt.X('limit:Q', title='Credit line ($)', axis=alt.Axis(format='~s', tickCount=6)),
                y=alt.Y('sample_mean:Q', title=f'Mean {metric} ($)'),
                color=color, opacity=alt.condition(highlight, alt.value(1), alt.value(.15)),
                tooltip=[alt.Tooltip('segment:N', title='Group'), alt.Tooltip('limit:Q', title='Credit line', format='$,.0f'),
                         alt.Tooltip('sample_mean:Q', title=metric, format='$,.2f'),
                         alt.Tooltip('sample_number:Q', title='Historical customers', format=',d')],
            ).add_params(highlight).properties(height=240).configure_view(stroke=None).configure_axis(
                domain=False, ticks=False, gridColor='#dce8ee', labelColor='#486477', titleColor='#31566b')

            st.altair_chart(chart, use_container_width=True, theme=None)
            st.caption('Synthetic history across four partners. Each point is a customer-weighted mean. '
                       'Use the randomized test to estimate the effect of changing the limit.')
    with right:
        with st.container():
            st.markdown('<h2 class="customer-figure-title">Credit lines by FICO group</h2>', unsafe_allow_html=True)
            st.caption('Available historical limits in the demo. Each group can test its own set of limits.')
            ranges = rows.groupby('segment', sort=True)['limit'].agg(['min', 'max'])
            ceiling = ranges["max"].max()
            for (group, limits), ink in zip(ranges.iterrows(), palette):
                low, high = limits["min"], limits["max"]
                st.markdown(
                    f'<div class="credit-range"><strong>{escape(group)}</strong>'
                    f'<span>${low:,.0f} – ${high:,.0f}</span><div aria-hidden="true">'
                    f'<i style="left:{low/ceiling:.3%};width:{(high-low)/ceiling:.3%};'
                    f'background:{ink}"></i></div></div>', unsafe_allow_html=True)
