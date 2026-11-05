"""Decision ledger presentation tokens and desktop shell.

Statistical routines and role-specific strategy colors remain in their own modules.
"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

HEADING = "#00395D"
ACTION = "#006DE3"
HIGHLIGHT = "#00AEEF"
BODY = "#1F2A33"
SECONDARY = "#5B6B78"
BORDER = "#DDE3E8"
SURFACE = "#F5F7F9"
CONTROL = "#7A8793"
TESTS = ("#00395D", "#006DE3", "#00AEEF")

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
.stApp .studio-lead {display:block;padding:0 0 24px}
.stApp .studio-lead h1 {font:600 28px/1.25 var(--display);color:var(--heading);letter-spacing:-.02em;margin:0}
.stApp .studio-lead h1 br {display:none}
.stApp .studio-lead p {font:400 14px/1.5 var(--display);color:var(--muted);max-width:80ch;margin:8px 0 0}


.stApp .st-key-experiment_choices button {
  display:block;width:100%;padding:28px;border:1px solid var(--line);border-top:3px solid var(--cyan);
  border-radius:6px;background:#fff;color:var(--ink);box-shadow:none;
  transition:transform .2s ease,border-color .2s ease,background-color .2s ease,box-shadow .2s ease;
}
.stApp .st-key-experiment_choices button:hover {
  transform:translateY(-4px);border-color:var(--action);background:#f8fcff;
  box-shadow:0 8px 20px rgba(0,57,93,.08)
}
.stApp .st-key-experiment_choices button > div {width:100%;min-width:0}
.stApp .st-key-experiment_choices button p {
  display:flex;flex-direction:column;align-items:flex-start;gap:0;
  min-height:354px;width:100%;margin:0;text-align:left;color:var(--ink);white-space:normal
}
.stApp .st-key-experiment_choices button strong {
  display:block;font-size:22px;line-height:1.3;font-weight:600;color:var(--heading);margin:28px 0 16px
}
.stApp .st-key-experiment_choices button em {
  display:block;font-size:14px;line-height:1.7;font-style:normal;color:var(--muted);white-space:normal;min-width:0
}
.stApp .st-key-experiment_choices button p::before {
  content:"";width:88px;height:88px;flex-shrink:0;display:block;border-radius:6px;
  background-color:#eaf6fc;background-repeat:no-repeat;background-position:center;background-size:52px;
  transition:transform .2s ease,background-color .2s ease
}
.stApp .st-key-experiment_choices button p::after {
  content:"Explore experiment →";display:block;width:100%;padding-top:20px;margin-top:auto;
  border-top:1px solid var(--line);font-size:14px;font-weight:600;color:var(--action)
}
.stApp .st-key-experiment_choices button:hover p::before {transform:translateY(-2px);background-color:#ddf1fb}
.stApp .st-key-home_choose_customer button p::before {background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%23006DE3%22%20stroke-width%3D%221.25%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Ccircle%20cx%3D%2212%22%20cy%3D%228%22%20r%3D%223%22%2F%3E%3Cpath%20d%3D%22M6%2021v-2a6%206%200%200%201%2012%200v2%22%2F%3E%3Cpath%20d%3D%22M5%205a3%203%200%200%200%200%206m-3%208v-2a5%205%200%200%201%203-4M19%205a3%203%200%200%201%200%206m3%208v-2a5%205%200%200%200-3-4%22%2F%3E%3C%2Fsvg%3E")}
.stApp .st-key-home_choose_geo button p::before {background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%23006DE3%22%20stroke-width%3D%221.25%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Ccircle%20cx%3D%2212%22%20cy%3D%2212%22%20r%3D%229%22%2F%3E%3Cellipse%20cx%3D%2212%22%20cy%3D%2212%22%20rx%3D%224%22%20ry%3D%229%22%2F%3E%3Cpath%20d%3D%22M3%2012h18M5%206.5a15%2015%200%200%200%2014%200M5%2017.5a15%2015%200%200%201%2014%200%22%2F%3E%3C%2Fsvg%3E")}
.stApp .st-key-home_choose_timeseries button p::before {background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%23006DE3%22%20stroke-width%3D%221.25%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M3%203v18h18%22%2F%3E%3Cpath%20d%3D%22M6%2016v-2m4%202v-4m4%204v-6m4%206v-8%22%20stroke%3D%22%239ACFE7%22%2F%3E%3Cpath%20d%3D%22m6%2011%204-4%204%202%206-5%22%2F%3E%3Ccircle%20cx%3D%226%22%20cy%3D%2211%22%20r%3D%221%22%20fill%3D%22%23006DE3%22%2F%3E%3Ccircle%20cx%3D%2210%22%20cy%3D%227%22%20r%3D%221%22%20fill%3D%22%23006DE3%22%2F%3E%3Ccircle%20cx%3D%2214%22%20cy%3D%229%22%20r%3D%221%22%20fill%3D%22%23006DE3%22%2F%3E%3C%2Fsvg%3E")}
@media(prefers-reduced-motion:reduce){.stApp .st-key-experiment_choices button,.stApp .st-key-experiment_choices button p::before {transition:none;transform:none}}
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
            font=dict(family="Inter, Arial, Helvetica, sans-serif", size=13, color=BODY),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            colorway=[CONTROL, *TESTS],
            xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
            yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        )
    )
    pio.templates.default = "plotly_white+decision_ledger"
    st.markdown(BUSINESS_STYLE, unsafe_allow_html=True)
