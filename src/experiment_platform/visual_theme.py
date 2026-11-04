"""Shared desktop styling for the experiment platform."""
BUSINESS_STYLE = """
<style>
.stApp {
  --blue:#0079b6; --cyan:#00aeef; --ink:#263644; --muted:#536a79;
  --line:#c9dce5; --wash:#edf7fb; --display:Georgia,"Times New Roman",serif;
  color:var(--ink); background:white; font-family:"Segoe UI",Arial,sans-serif;
}
.stApp .block-container {max-width:1440px;padding:2.6rem clamp(2rem,5vw,5rem) 4rem}
.stApp [data-testid="stSidebar"] {background:#f5fafc;border-right:1px solid var(--line)}
.studio-brand {display:flex;align-items:center;gap:.8rem;color:var(--ink);
  font:600 clamp(1.25rem,2vw,1.65rem) var(--display);letter-spacing:-.045em}
.studio-brand i {display:block;width:1.45rem;height:1rem;transform:skew(-19deg);
  background:repeating-linear-gradient(90deg,var(--cyan) 0 4px,transparent 4px 8px)}
.st-key-studio_masthead {border-bottom:1px solid var(--blue);padding-bottom:.6rem}
.stApp .st-key-studio_masthead button {border:0;border-bottom:2px solid transparent;
  border-radius:0;background:white;box-shadow:none;transform:none;min-height:2.8rem}
.stApp .st-key-studio_masthead button p {color:var(--muted);font-weight:550}
.stApp .st-key-studio_masthead button:is([kind="primary"],:hover) {background:white;border-color:var(--cyan)}
.stApp .st-key-studio_masthead button:is([kind="primary"],:hover) p {color:var(--blue)}
.studio-lead {display:grid;grid-template-columns:52% 1fr;align-items:center;
  padding:clamp(2rem,4vw,4rem) 1.2rem clamp(1.8rem,4vw,3.5rem)}
.stApp .studio-lead h1 {padding:0;color:var(--blue);font:normal clamp(3.7rem,6.6vw,6.7rem)/.96 var(--display);letter-spacing:-.056em}
.stApp .studio-lead p {margin:0;max-width:25ch;color:var(--ink);
  font:normal clamp(1.35rem,2.15vw,2.2rem)/1.24 var(--display);letter-spacing:-.025em}
.st-key-design_directory {--tracks:3em minmax(0,1.15fr) minmax(0,1.4fr) minmax(0,.9fr) 1.5em;
  gap:.5rem;border-bottom:1px solid var(--blue);padding-bottom:1.35rem}
.studio-index,.stApp .st-key-design_directory button p {
  display:grid;grid-template-columns:var(--tracks);align-items:center;column-gap:1rem;
  text-align:left;padding:0 clamp(.8rem,1.7vw,1.65rem)}
.studio-index {padding-bottom:.8rem;border-bottom:1px solid var(--blue);
  color:var(--muted);font-size:.83rem;letter-spacing:.03em}
.studio-index span:first-child {grid-column:2}
.stApp .st-key-design_directory button {display:block;width:100%;padding:0;border:0;border-radius:0;
  background:var(--wash);box-shadow:none;transform:none}
.stApp .st-key-design_directory button:hover {background:#dff2fa}
.stApp .st-key-design_directory button:focus-visible {outline:2px solid var(--blue);outline-offset:2px}
.stApp .st-key-design_directory button p {min-height:clamp(5.3rem,7vw,7rem);
  color:var(--ink);font-size:clamp(.92rem,1.18vw,1.15rem);line-height:1.4}
.stApp .st-key-design_directory button p code {color:var(--blue);background:none;padding:0;
  font:normal 1.8rem var(--display);border:0;border-right:1px solid #a4d5e9;border-radius:0}
.stApp .st-key-design_directory button p strong {color:var(--blue);
  font:normal clamp(1.55rem,2.8vw,2.9rem)/1.1 var(--display);letter-spacing:-.045em}
.stApp .st-key-design_directory button p em {font-style:normal;border-left:1px solid #bdddea;padding-left:1rem}
.stApp .st-key-design_directory button p::after {content:"→";font:normal 1.6rem Arial;color:var(--blue)}
.stApp :is(.customer-page-heading,.geo-page-heading,.ts-page-heading) {
  display:grid;grid-template-columns:52% 1fr;align-items:center;padding:2rem 0;
  border-bottom:1px solid var(--blue);margin-bottom:1.2rem}
.stApp :is(.customer-page-heading,.geo-page-heading,.ts-page-heading)::after {display:none}
.stApp :is(.customer-page-heading,.geo-page-heading,.ts-page-heading) h1,
.stApp [data-testid="stHeading"] h1 {color:var(--blue);font:normal clamp(2.4rem,4vw,4.5rem)/1.06 var(--display);letter-spacing:-.05em}
.stApp :is(.customer-page-heading,.geo-page-heading,.ts-page-heading) p {color:var(--muted);font-size:1rem;line-height:1.5}
.stApp :is(.customer-product-mark,.geo-product-mark,.ts-product-mark) {border-radius:0;color:var(--blue);background:var(--wash)}
.stApp :is(.customer-explainer,.command-kpis,.lifecycle-rail,.attention-list) {
  border-radius:0;box-shadow:none;background:white;border:0;border-block:1px solid var(--line)}
.stApp .customer-flow-card {border:none;border-left:1px solid var(--line);border-radius:0;background:white}
.stApp :is(.ts-panel-title,.geo-panel-title,h2,h3) {color:#0073a6}
.stApp [data-testid="stMetric"] {background:var(--wash);border:0;border-top:1px solid var(--blue);border-radius:0;box-shadow:none}
.stApp [data-testid="stMetricValue"] {font-variant-numeric:tabular-nums}
.stApp [data-baseweb="tab-list"] {background:white;border:0;border-bottom:1px solid var(--line);border-radius:0}
.stApp [data-testid="stExpander"] {border-radius:3px;background:white}
</style>
"""
