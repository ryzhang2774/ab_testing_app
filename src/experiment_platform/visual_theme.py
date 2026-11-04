"""Desktop presentation layer for the experiment workspace."""
BUSINESS_STYLE = """
<style>
.stApp {
  --ink:#142f47; --muted:#54687a; --line:#ced9e1; --soft:#f3f6f8;
  --coral:#12547b; --teal:#226b79; --gold:#536b81; --cyan:#00aeef;
  --heading-blue:#007bac; --section-blue:#00729f;
  background:#f5f7f9; color:var(--ink);
  font-family:Inter,"Segoe UI",Arial,sans-serif;
}
.stApp .block-container {max-width:1320px;padding:2rem clamp(2rem,4vw,4rem) 4rem}
.stApp [data-testid="stSidebar"] {background:#ecf1f4;border-right:1px solid var(--line)}
.stApp [data-testid="stSidebar"] h1 {font-size:1.35rem;font-weight:700;letter-spacing:-.035em}
.stApp :is(.overview-hero,.customer-page-heading,.ts-page-heading,.geo-page-heading) {
  padding:clamp(1.5rem,3vw,2.75rem); background:#eff7fb; color:var(--ink);
  border:1px solid #cddfea;border-left:4px solid var(--cyan);border-radius:8px;margin:1rem 0 1.5rem;
}
.stApp .overview-hero::after {display:none}
.stApp :is(.overview-hero,.customer-page-heading,.ts-page-heading,.geo-page-heading) h1 {
  color:var(--heading-blue);font-family:inherit;font-weight:630;font-size:clamp(2rem,3.2vw,3rem);
  letter-spacing:-.045em;line-height:1.12;margin:0 0 .65rem;
}
.stApp :is(.overview-hero,.customer-page-heading,.ts-page-heading,.geo-page-heading) p {
  color:#425e71;font-size:1rem;line-height:1.55;margin:0;max-width:72ch
}
.stApp :is(h1,h2,h3,.overview-section-title,.ts-panel-title,.geo-panel-title) {
  color:var(--section-blue);
}
.stApp .overview-section-title {font-size:1.18rem;letter-spacing:-.02em}
.stApp .overview-section-copy {font-size:.9rem;margin-bottom:1rem;color:var(--muted)}
.stApp .workflow-card {
  border:1px solid var(--line); border-top:3px solid #77b4ce; border-radius:8px;
  padding:1.4rem; background:white; box-shadow:0 2px 6px #142f4708;
}
.stApp .workflow-card:hover {transform:none;border-color:#62b5d9}
.stApp .workflow-card-head {padding-bottom:1rem;border-bottom:1px solid var(--line)}
.stApp .workflow-card .workflow-mark {background:#e6f5fc;color:var(--section-blue);border-radius:4px;padding:.35rem;width:auto;height:auto;min-width:2.8rem;letter-spacing:.06em}
.stApp .workflow-method {color:#476478;font-size:.7rem;letter-spacing:.06em}
.stApp .workflow-card h3 {font-size:1.32rem;font-weight:620;letter-spacing:-.025em;line-height:1.25;margin:1.1rem 0 .6rem}
.stApp .workflow-card p {font-size:.94rem;color:#445a6b;line-height:1.5}
.stApp .workflow-examples,.stApp .workflow-outcome {font-size:.83rem;line-height:1.45}
.stApp .workflow-outcome {margin-top:auto;padding-top:.85rem;color:#284960}
.stApp :is(.st-key-home_choose_customer,.st-key-home_choose_geo,.st-key-home_choose_timeseries) button {
  color:white;background:var(--section-blue);border-color:var(--section-blue);border-radius:6px;min-height:2.75rem
}
.stApp .stButton button:hover {transform:none}
.stApp [data-testid="stMetric"] {border:1px solid var(--line);border-radius:6px;padding:1rem 1.15rem;box-shadow:none;background:white}
.stApp [data-testid="stMetricValue"],.stApp .command-kpi-value {font-variant-numeric:tabular-nums;letter-spacing:-.045em}
.stApp :is(.command-kpis,.attention-list,.lifecycle-rail,.customer-explainer,.insight-card) {border-color:var(--line);border-radius:8px;background:white}
.stApp :is(.customer-product-bar,.geo-product-bar,.ts-product-bar) {border-bottom:1px solid var(--line);background:transparent}
.stApp :is(.customer-product-mark,.ts-product-mark,.geo-product-mark) {background:#e3f2f8;color:#155876;border-radius:5px}
.stApp [data-baseweb="tab-list"] {background:transparent;border:0;border-bottom:1px solid var(--line);border-radius:0;padding:0}
.stApp [data-baseweb="input"],.stApp [data-baseweb="select"] > div {background:#fff;border-radius:5px}
.stApp [data-testid="stExpander"] {background:white;border-radius:6px}
.stApp :is(.customer-flow-card,.business-guide,.strategy-banner) {border-color:#cddae3;background:#f2f6f8}
.stApp .stButton button {white-space:normal}
.stApp .workflow-card-head {flex-wrap:wrap}
.stApp .workflow-method {text-align:left}
@media(max-width:1000px){
  .stApp .block-container {padding-left:1.4rem;padding-right:1.4rem}
  .stApp .workflow-card {padding:1rem}
  .stApp .workflow-card h3 {font-size:1.1rem}
}
@media(prefers-reduced-motion:reduce){.stApp .workflow-card,.stApp .stButton button{transition:none}}
</style>
"""
