"""Public-facing introductions to the three internal experiment workspaces."""
from base64 import b64encode
from html import escape
from os import getenv
from urllib.parse import quote, urlencode

import streamlit as st

from .workspace_visuals import ASSETS

COMPARISONS = {
    "Customer": (
        "Customer experiment",
        "Randomly assign customers to strategies and compare each test with Control.",
        "A random split assigns eligible customers to a $500 Control limit or a $1,000 Test limit.",
        "customer",
    ),
    "Geography": (
        "Geographic experiment",
        "Compare changes in Test and Control markets over the same period.",
        "Example map with blue Test markets and green Control markets.",
        "geo",
    ),
    "Time Series": (
        "Time series experiment",
        "Measure what changed after launch against the trend expected from history.",
        "At launch, the solid actual outcome rises above the dashed expected outcome.",
        "ts",
    ),
}


def comparison_image(workspace):
    filename = "comparison-" + workspace.lower().replace(" ", "-") + ".webp"
    payload = b64encode((ASSETS / filename).read_bytes()).decode("ascii")
    return f'<img src="data:image/webp;base64,{payload}" alt="{escape(COMPARISONS[workspace][2])}">'


HOME_ART = {
    "Customer": ("home-customer.svg", "Credit limit example. Control $2,500 and Test $5,000."),
    "Geography": ("home-markets.svg", "Example US market assignment. Test is blue and Control is orange."),
    "Time Series": ("home-timeline.svg", "Actual outcomes rise above the expected trend after launch."),
}


@st.cache_data(show_spinner=False)
def asset_url(filename, mime):
    payload = b64encode((ASSETS / filename).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def method_card(column, key, workspace, open_workspace):
    title, description, _, _ = COMPARISONS[workspace]
    filename, alt = HOME_ART[workspace]
    with column:
        st.markdown(f'<div class="method-art"><img src="{asset_url(filename, "image/svg+xml")}" alt="{escape(alt)}"></div>', unsafe_allow_html=True)
        st.button(title, key=key, use_container_width=True, on_click=open_workspace, args=(workspace,))
        st.markdown(f'<p class="method-description">{description}</p>', unsafe_allow_html=True)


def workspace_intro(workspace, start_workflow):
    title, description, _, key = COMPARISONS[workspace]
    copy, graphic = st.columns([1.2, 1], gap="large")
    with copy:
        st.markdown(f'<header class="work-intro"><h1>{title}</h1><p>{description}</p></header>',
                    unsafe_allow_html=True)
        plan, analyze = st.columns(2)
        plan.button("Plan an experiment", key=f"{key}_demo_plan", type="primary",
                    use_container_width=True, on_click=start_workflow, args=("plan",))
        analyze.button("Analyze results", key=f"{key}_demo_analyze", use_container_width=True,
                       on_click=start_workflow, args=("analyze",))
    graphic.markdown(f'<figure class="intro-art">{comparison_image(workspace)}</figure>', unsafe_allow_html=True)


def booking_action():
    recipient = getenv("EXPERIMENT_DEMO_EMAIL", "").strip() or "ruyue.zhang@barclays.com"
    query = urlencode({
        "subject": "Experiment Platform demo request",
        "body": "Hello,\n\nI'd like to book a demo of the Experiment Platform.\n\nMy team: \nPreferred times: ",
    }, quote_via=quote)
    href = f"mailto:{quote(recipient, safe='@.')}?{query}"
    return f'<a class="book-demo" href="{escape(href, quote=True)}" target="_self">Book a demo <span aria-hidden="true">↗</span></a>'


def overview_intro():
    photo = asset_url("credit-strategies.jpg", "image/jpeg")
    st.markdown(
        '<section class="intro-spread"><header class="home-intro">'
        '<h1>Plan experiments.<br><span>Analyze results.</span></h1>'
        '<p>Choose who to test, how long to run, and how to measure what changed after launch.</p>'
        f'<div class="intro-actions">{booking_action()}</div></header>'
        f'<figure class="home-cover"><img src="{photo}" alt="Blue credit cards in a clear display stand."></figure>'
        '</section><h2 class="explore-title">Choose an experiment</h2>',
        unsafe_allow_html=True)
