"""Decision ledger presentation tokens and desktop shell.

Statistical routines and role-specific strategy colors remain in their own modules.
"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from .workspace_visuals import workspace_css

HEADING = "#00395D"
ACTION = "#006DE3"
HIGHLIGHT = "#00AEEF"
BODY = "#16384d"
SECONDARY = "#5B6B78"
BORDER = "#ccdde6"
SURFACE = "#F5F7F9"
CONTROL = "#7A8793"
TESTS = ("#006ba2", "#008679", "#cc7c0e")

BUSINESS_STYLE = """
<style>
:root, .stApp {
  --heading:#00395D; --action:#006DE3; --cyan:#00AEEF; --ink:#1F2A33;
  --muted:#5B6B78; --line:#DDE3E8; --soft:#F5F7F9; --blue:#00395D;
  --wash:#F5F7F9; --coral:#006DE3; --teal:#00395D; --gold:#5B6B78;
  --display:Inter,Arial,Helvetica,sans-serif;
  --favorable:#1E7B4F; --adverse:#B3382C;
}
.stApp {background:#fff;color:var(--ink);font:14px/1.5 Inter,Arial,Helvetica,sans-serif;
  font-variant-numeric:tabular-nums;min-width:1280px}
.stApp .block-container {max-width:1240px;padding:32px 24px 64px}
.stApp :is(h1,h2,h3,h4,p,label,button,input,textarea,select) {font-family:var(--display)}
.stApp h1,.stApp [data-testid="stHeading"] h1 {font-size:28px;font-weight:600;line-height:1.25;color:var(--heading);letter-spacing:-.02em}
.stApp h2,.stApp [data-testid="stHeading"] h2 {font-size:20px;font-weight:600;color:var(--heading);letter-spacing:-.01em}
.stApp h3,.stApp h4 {font-size:16px;font-weight:600;color:var(--heading)}
.stApp p,.stApp label {font-size:14px;color:var(--ink)}
.stApp [data-testid="stCaptionContainer"] p {font-size:12px;color:var(--muted)}
[data-testid="stMainMenu"],[data-testid="stToolbar"],[data-testid="stDecoration"],footer {display:none}
[data-testid="stHeader"] {background:transparent;height:0}
.stApp [data-testid="stSidebar"] {
  width:220px!important;min-width:220px!important;max-width:220px!important;
  background:#fff;border-right:1px solid var(--line)}
.stApp [data-testid="stSidebarContent"] {padding:24px 12px}
.stApp [data-testid="stSidebarUserContent"] {padding:0}
.stApp [data-testid="stSidebarHeader"],[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] {display:none}
.ledger-product {color:var(--heading);font-size:15px;font-weight:600;padding:8px 10px 28px}
.stApp [data-testid="stSidebar"] :is(.st-key-ledger_navigation,.st-key-ledger_context) {gap:4px}
.stApp [data-testid="stSidebar"] :is(.st-key-ledger_navigation,.st-key-ledger_context) button {
  width:100%;justify-content:flex-start;border:0;border-left:3px solid transparent;
  background:#fff;border-radius:0;box-shadow:none;padding:10px;min-height:42px;transform:none}
.stApp [data-testid="stSidebar"] :is(.st-key-ledger_navigation,.st-key-ledger_context) button p {
  font-size:13px;color:var(--muted);font-weight:400;text-align:left}
.stApp [data-testid="stSidebar"] :is(.st-key-ledger_navigation,.st-key-ledger_context) button[kind="primary"] {
  border-left-color:var(--cyan);background:var(--soft)}
.stApp [data-testid="stSidebar"] :is(.st-key-ledger_navigation,.st-key-ledger_context) button[kind="primary"] p {
  color:var(--heading);font-weight:600}
.stApp [data-testid="stSidebar"] :is(.st-key-ledger_navigation,.st-key-ledger_context) button:hover {background:var(--soft)}
.stApp .st-key-ledger_context {margin-top:24px;padding-top:20px;border-top:1px solid var(--line)}
.ledger-context-title {font-size:12px;color:var(--heading);font-weight:600;padding:0 10px 8px}
.stApp .stButton button {border-radius:4px;box-shadow:none;transform:none;font-size:14px}
.stApp .stButton button[kind="primary"] {background:var(--action);border-color:var(--action)}
.stApp .stButton button[kind="primary"] p {color:white}
.stApp .stButton button:hover {box-shadow:none;transform:none}
.stApp button:focus-visible {outline:2px solid var(--cyan);outline-offset:2px}
.stApp :is(.customer-page-heading,.geo-page-heading,.ts-page-heading,.overview-hero) {
  display:block;padding:0 0 24px;margin:0 0 24px;background:white;
  border:0;border-bottom:1px solid var(--line);border-radius:0}
.stApp :is(.customer-page-heading,.geo-page-heading,.ts-page-heading,.overview-hero)::after {display:none}
.stApp :is(.customer-page-heading,.geo-page-heading,.ts-page-heading,.overview-hero) h1 {
  font:600 28px/1.25 var(--display);color:var(--heading);letter-spacing:-.02em}
.stApp :is(.customer-page-heading,.geo-page-heading,.ts-page-heading,.overview-hero) p {
  font-size:14px;line-height:1.5;color:var(--muted);margin:8px 0 0}
.stApp :is(.customer-explainer,.command-kpis,.lifecycle-rail,.attention-list) {
  background:white;border:1px solid var(--line);border-radius:6px;box-shadow:none}
.stApp .customer-flow-card {background:var(--soft);border-color:var(--line);border-radius:4px;box-shadow:none}
.stApp [data-testid="stMetric"] {background:var(--soft);border:1px solid var(--line);border-radius:4px;padding:16px;box-shadow:none}
.stApp [data-testid="stMetricValue"] {font-size:24px;color:var(--heading)}
.stApp [data-baseweb="tab-list"] {background:white;border:0;border-bottom:1px solid var(--line);border-radius:0}
.stApp [data-testid="stExpander"],.stApp [data-testid="stDataFrame"] {border-color:var(--line);border-radius:4px;background:white;box-shadow:none}
.stApp [data-baseweb="input"],.stApp [data-baseweb="select"] > div {background:white;border-radius:4px;border-color:var(--line)}
</style>
"""


def apply_theme() -> None:
    """Apply the shared CSS after workspace styles and set Plotly base defaults."""
    pio.templates["decision_ledger"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Segoe UI, Source Sans Pro, Arial, sans-serif", size=13, color=BODY),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            colorway=[CONTROL, *TESTS],
            xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
            yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        )
    )
    pio.templates.default = "plotly_white+decision_ledger"
    st.markdown(BUSINESS_STYLE + "<style>" + workspace_css() + "</style>", unsafe_allow_html=True)
