"""Four-step Customer Business V2 workbench. Statistics stay in pure modules."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import html
import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .business_config import default_config, validate_config, control_name, fingerprint, utc_now
from .business_data import read_table, validate_history, prepare_records, prepare_summaries, frame_fingerprint
from .business_plan import build_plan, clean_records, population_weights, exploratory_protocol, setup_memo
from .business_analysis import analyze, monitor
from .business_store import code_fingerprint, import_protocol, next_version, save_protocol, saved_protocols
from .business_demo import demo_config, demo_history, demo_records

STEPS = ["Set up experiment", "Build test plan", "Monitor experiment", "Analyze results"]
CSS = """
<style>
.stApp:has(.bv2-shell){background:#f1f6fa;color:#10203e;--coral:#007f89}
.stApp:has(.bv2-shell) .block-container{max-width:1440px;padding-top:1.5rem}
.stApp:has(.bv2-shell) [data-testid="stSidebar"]{background:#fff;border-right:1px solid #dce5ed}
.stApp:has(.bv2-shell) h1,.stApp:has(.bv2-shell) h2,.stApp:has(.bv2-shell) h3{color:#10203e;letter-spacing:-.025em}
.stApp:has(.bv2-shell) h1{font-size:2rem;font-weight:750;padding-top:0}
.stApp:has(.bv2-shell) h3{font-size:1.3rem;padding-top:.1rem}
.stApp:has(.bv2-shell) [data-testid="stVerticalBlockBorderWrapper"]>div{background:#fff;border-color:#dce5ed;border-radius:10px}
.stApp:has(.bv2-shell) [data-testid="stBaseButton-primary"],.stApp:has(.bv2-shell) [data-testid="baseButton-primaryFormSubmit"]{background:#007f89;border-color:#007f89}
.stApp:has(.bv2-shell) [data-testid="stMetricValue"]{font-size:1.8rem;color:#10203e}
.bv2-shell{display:flex;align-items:center;gap:12px;border-bottom:1px solid #dce5ed;padding-bottom:18px;margin-bottom:24px;color:#10203e;font-size:18px;font-weight:700}
.bv2-mark{width:34px;height:34px;border-radius:6px;background:#007f89;color:white;display:grid;place-items:center;font-size:23px}
.bv2-shell small{font-size:13px;color:#64748b;font-weight:400;margin-left:auto}
.bv2-subtitle{color:#65768c;margin-top:-10px;margin-bottom:12px;font-size:14px}
.bv2-summary{padding:4px 2px}.bv2-summary h3{font-size:20px;margin:0 0 14px}.bv2-summary-row{border-top:1px solid #e1e8ef;padding:17px 0;font-size:14px;color:#64748b}.bv2-summary-row b{display:block;font-size:16px;color:#10203e;margin-bottom:4px}
.bv2-note{border-left:3px solid #007f89;padding:10px 15px;background:#edf7f7;color:#294a59;font-size:14px;line-height:1.5;margin:12px 0 18px}
.stApp:has(.bv2-shell) [data-testid="stHorizontalBlock"]{gap:1.1rem}
.stApp:has(.bv2-shell) .stButton>button{border-radius:7px;font-size:14px;min-height:40px}
.stApp:has(.bv2-shell) [role="radiogroup"]{gap:1rem;flex-wrap:wrap}
.stApp:has(.bv2-shell) label{font-size:14px}
.stApp:has(.bv2-shell) label[data-baseweb="radio"]:has(input:checked)>div:first-child{background:#007f89;border-color:#007f89}
.stApp:has(.bv2-shell) label[data-baseweb="checkbox"]:has(input:checked)>span:first-child{background:#007f89;border-color:#007f89}
.stApp:has(.bv2-shell) [data-baseweb="tag"]{background:#007f89}
.stApp:has(.bv2-shell) .stButton>button:hover{border-color:#007f89;color:#007f89;box-shadow:0 2px 7px rgba(0,127,137,.1)}
.stApp:has(.bv2-shell) .stButton>button[kind="primary"]:hover{color:white;background:#006c75}
@media(max-width:760px){.stApp:has(.bv2-shell) .block-container{padding:1rem}.bv2-shell small{display:none}.stApp:has(.bv2-shell) h1{font-size:1.65rem}}
</style>
"""


def _state():
    if "business_v2" not in st.session_state:
        c = default_config()
        st.session_state.business_v2 = {"config": c, "setup_saved": False, "step": STEPS[0], "demo": False,
                                       "history": pd.DataFrame(), "plan": None, "candidate": None, "bundle": None,
                                       "snapshot": None, "arm_seed": c["arms"], "metric_seed": c["metrics"], "record_seed": pd.DataFrame()}
    return st.session_state.business_v2


def _go(step):
    st.session_state.bv2_step = step


def _load_demo():
    for key in list(st.session_state):
        if key.startswith("bv2_"):
            del st.session_state[key]
    c = demo_config()
    st.session_state.business_v2 = {"config": c, "setup_saved": True, "step": STEPS[0], "demo": True,
                                   "history": demo_history(), "plan": None, "candidate": None, "bundle": None,
                                   "snapshot": None, "arm_seed": deepcopy(c["arms"]), "metric_seed": deepcopy(c["metrics"]),
                                   "record_seed": demo_records()}


def _new_experiment():
    for key in list(st.session_state):
        if key.startswith("bv2_") or key == "business_v2":
            del st.session_state[key]


def _freeze_candidate():
    b=_state()
    plan=b.get("candidate")
    if not plan:
        return
    try:
        save_protocol(plan)
    except (OSError,ValueError) as exc:
        b["save_error"]=str(exc)
        return
    b["plan"]=deepcopy(plan)
    b["config"]=deepcopy(plan["config"])
    b["snapshot"]=b["bundle"]=None
    b["frozen_input_fp"]=b.get("candidate_fp")
    b["draft_dirty"]=False
    st.session_state.bv2_step=STEPS[2]


def _open_plan(p):
    _new_experiment()
    b=_state()
    is_draft=p.get("kind")=="setup_draft"
    b.update(config=p["config"],setup_saved=True,plan=None if is_draft else p,history=pd.DataFrame(p["history"]),
             arm_seed=p["config"]["arms"],metric_seed=p["config"]["metrics"],setup_saved_at=p["saved_at"],
             demo=p["config"].get("synthetic",False),record_seed=demo_records() if p["config"].get("synthetic") else pd.DataFrame())
    st.session_state.bv2_pending_step=STEPS[0] if is_draft else STEPS[2]


def _side_summary(c, stage):
    with st.container(border=True):
        primary = next(m["name"] for m in c["metrics"] if m["role"] == "Primary")
        st.markdown('<div class="bv2-summary"><h3>Your experiment</h3>' + "".join(
            f'<div class="bv2-summary-row"><b>{html.escape(a)}</b>{html.escape(b)}</div>' for a,b in [
                (f"{len(c['arms'])} arms", "Control and test options"), ("1 primary metric", primary),
                ("Fixed observation window", f"{c['window_days']} days + {c['maturity_delay_days']} days maturity delay"),
                (c["decision_scope"] + " decision", "Primary metric with guardrails"),
                ("Next step", {STEPS[0]: "Add historical partner statistics to calculate the sample you need.", STEPS[1]: "Freeze the plan before enrolling customers.", STEPS[2]: "Check assignment, outcome maturity and guardrails.", STEPS[3]: "Read the effect together with its uncertainty."}[stage])]) + '</div>', unsafe_allow_html=True)


def _setup(b):
    c = deepcopy(b["config"])
    left, right = st.columns([3.15, 1.3])
    with right:
        _side_summary(c, STEPS[0])
    with left, st.form("bv2_setup_form"):
        with st.container(border=True):
            st.subheader("What are you testing?")
            st.caption("Define the decision before collecting results.")
            a, z = st.columns(2)
            c["name"] = a.text_input("Experiment name", c["name"], key="bv2_name")
            c["owner"] = z.text_input("Owner", c["owner"], key="bv2_owner", placeholder="Team or accountable owner")
            c["question"] = st.text_area("Business question", c["question"], height=78, key="bv2_question")
            arms = st.data_editor(pd.DataFrame(b["arm_seed"]), num_rows="dynamic", hide_index=True, use_container_width=True, key="bv2_arms",
                column_config={"name": st.column_config.TextColumn("Arm name", required=True), "definition": st.column_config.TextColumn("What customers receive", required=True, width="large"), "role": st.column_config.SelectboxColumn("Role", options=["Control", "Test"], required=True)})
            c["arms"]=clean_records(arms)
        with st.container(border=True):
            st.subheader("What improvement would make this worthwhile?")
            st.caption("One primary outcome drives the decision. Guardrails protect the business.")
            st.caption("Use a customer-level amount or a 0/1 result. Ratios of totals, time-to-event outcomes and correlated repeated records need a different design.")
            metrics = st.data_editor(pd.DataFrame(b["metric_seed"]), num_rows="dynamic", hide_index=True, use_container_width=True, key="bv2_metrics",
                column_config={"name": st.column_config.TextColumn("Metric / result column", required=True),
                    "role": st.column_config.SelectboxColumn("Role", options=["Primary", "Guardrail", "Secondary"], required=True),
                    "type": st.column_config.SelectboxColumn("Type", options=["Continuous", "Binary"], required=True),
                    "direction": st.column_config.SelectboxColumn("Better when", options=["Higher", "Lower"], required=True),
                    "effect": st.column_config.NumberColumn("Worthwhile change", min_value=0.),
                    "effect_scale": st.column_config.SelectboxColumn("Change unit", options=["Relative %", "Absolute", "Percentage points"]),
                    "margin": st.column_config.NumberColumn("Guardrail tolerance", help="Absolute metric units. For binary: .01 = 1 percentage point.", min_value=0.),
                    "alert_threshold": st.column_config.NumberColumn("Operational alert", help="Absolute metric units; 0 disables the point-estimate alert.", min_value=0.),
                    "window_days": st.column_config.NumberColumn("Window (days)", min_value=1, step=1),
                    "definition": st.column_config.TextColumn("Calculation definition", width="large"),
                    "unit": st.column_config.TextColumn("Unit"), "include_in_plan": st.column_config.CheckboxColumn("Size for this metric")},
                column_order=["name", "role", "type", "direction", "effect", "effect_scale", "margin", "include_in_plan", "unit", "definition", "window_days", "alert_threshold"])
            st.caption("Scroll the metric table horizontally for definitions, windows and operational alerts. Binary history requires exact successes and N; a binary margin of 0.01 means 1 percentage point.")
        with st.expander("Population, timing & advanced settings"):
            a,z = st.columns(2)
            c["unit"] = a.selectbox("Randomization unit", ["Customer", "Account"], index=["Customer","Account"].index(c["unit"]))
            c["decision_scope"] = z.selectbox("Where do you need an independent decision?", ["Overall", "Each segment", "Each partner"], index=["Overall","Each segment","Each partner"].index(c["decision_scope"]))
            c["start_date"] = a.date_input("Planned start", date.fromisoformat(c["start_date"])).isoformat()
            c["max_recruitment_weeks"] = z.number_input("Maximum recruitment weeks", 1, 520, int(c["max_recruitment_weeks"]))
            flow_frequency = a.selectbox("Eligible traffic frequency", ["Weekly", "Monthly"])
            traffic = z.number_input("Eligible customers per period", min_value=1., value=float(c["eligible_per_week"]), step=100.)
            c["eligible_per_week"] = traffic if flow_frequency == "Weekly" else traffic*12/52
            c["window_days"] = a.number_input("Default outcome window (days)", 1, 3650, int(c["window_days"]))
            c["maturity_delay_days"] = z.number_input("Additional maturity delay (days)", 0, 3650, int(c["maturity_delay_days"]))
            c["alpha"] = a.number_input("False-positive budget (alpha)", .001, .25, float(c["alpha"]), step=.005, format="%.3f")
            c["power"] = z.number_input("Detection chance per comparison (power)", .5, .99, float(c["power"]), step=.05)
            c["attrition"] = a.number_input("Expected missing outcome fraction", 0., .9, float(c["attrition"]), step=.01)
            c["multiplicity"] = z.checkbox("Protect multiple comparisons", bool(c["multiplicity"]))
            c["srm_alpha"] = a.number_input("SRM diagnostic threshold", .00001, .1, float(c["srm_alpha"]), format="%.5f")
            c["smd_threshold"] = z.number_input("SMD reference line", .01, 2., float(c["smd_threshold"]), step=.01)
            c["allocation"] = st.selectbox("Allocation method", ["Recommended", "Business fixed"], index=["Recommended","Business fixed"].index(c["allocation"]), help="Recommended: pairwise Neyman, largest shared Control, then solve each Test again.")
            # Always render assumptions so changing a selectbox inside a form needs only one submit.
            allocation_rows = st.data_editor(pd.DataFrame([{"arm": a["name"], "allocation_weight": c["fixed_shares"].get(a["name"],1.), "test_sd_multiplier": c.get("test_sd_multipliers",{}).get(a["name"],1.)} for a in c["arms"]]), hide_index=True, key="bv2_allocation_grid", disabled=["arm"], use_container_width=True)
            c["fixed_shares"] = dict(zip(allocation_rows.arm, allocation_rows.allocation_weight))
            c["test_sd_multipliers"] = dict(zip(allocation_rows.arm, allocation_rows.test_sd_multiplier))
            c["assumption_only"] = st.checkbox("Assumption-only planning (historical N unavailable)", c["assumption_only"])
            c["historical_start"] = a.text_input("Historical start (YYYY-MM-DD, optional)", c["historical_start"])
            c["historical_end"] = z.text_input("Historical end (YYYY-MM-DD, optional)", c["historical_end"])
            st.caption("A new Test uses Control variability unless you supply arm-specific history or change its SD multiplier. Fixed-horizon inference is the default.")
        saved = st.form_submit_button("Save setup & continue", type="primary", use_container_width=True)
        if saved:
            try:
                c["arms"], c["metrics"] = clean_records(arms), clean_records(metrics)
                c = validate_config(c)
                changed = fingerprint(c) != fingerprint(b["config"])
                b["config"], b["setup_saved"] = c, True
                b["arm_seed"]=deepcopy(c["arms"])
                b["metric_seed"]=deepcopy(c["metrics"])
                b["setup_saved_at"]=utc_now()
                if changed:
                    b["candidate"] = b["snapshot"] = None
                    b["bundle"] = None
                st.session_state.bv2_pending_step = STEPS[1]
                st.rerun()
            except (ValueError, TypeError, KeyError) as exc:
                st.error(str(exc))
    if b["setup_saved"]:
        st.button("I already have results — go to analysis", on_click=_go, args=(STEPS[3],), key="bv2_skip_plan")
        _downloads(setup_memo(b["config"],b.setdefault("setup_saved_at",utc_now())),None,STEPS[0],"setup")


def _import_table(label, key, seed=None, template=None):
    synthetic=_state()["demo"]
    mode = st.radio(label, ["Upload CSV", "Paste from Excel", "Edit table"], horizontal=True, key=f"{key}_mode")
    if template is not None:
        st.download_button("Download template", template.to_csv(index=False).encode(), file_name=f"{key}_template.csv", mime="text/csv", key=f"{key}_template")
    if mode == "Upload CSV":
        uploaded = st.file_uploader("Choose a CSV file", type="csv", key=f"{key}_upload",disabled=synthetic)
        if uploaded:
            return read_table(uploaded.getvalue())
        if seed is not None and not seed.empty:
            st.caption("Showing the current working data. A new upload replaces this snapshot.")
            return seed.copy()
    elif mode == "Paste from Excel":
        value = st.text_area("Paste column headers and rows", height=170, placeholder="partner\tsegment\tmetric\tsample_mean\tsample_variance\tsample_number", key=f"{key}_paste",disabled=synthetic)
        if synthetic and seed is not None:
            st.caption("Synthetic mode uses the demo snapshot. Start a new experiment to upload or paste your data.")
            return seed.copy()
        if value.strip():
            return read_table(value)
    else:
        data = seed if seed is not None and not seed.empty else template if template is not None else pd.DataFrame()
        return st.data_editor(data, num_rows="dynamic", hide_index=True, use_container_width=True, key=f"{key}_editor")
    return pd.DataFrame()


def _plan(b):
    if not b["setup_saved"]:
        st.info("Start by saving the experiment setup.")
        st.button("Go to setup", on_click=_go, args=(STEPS[0],))
        return
    c = b["config"]
    st.subheader("How many customers do we need?")
    st.caption("Bring one row per Partner × Segment × Metric. New Tests do not need invented historical results.")
    template = pd.DataFrame(columns=["partner","segment","metric","sample_mean","sample_variance","sample_number","successes"])
    try:
        with st.container(border=True):
            raw = _import_table("Historical data", "bv2_history", b["history"], template)
            with st.expander("Field mapping & data definitions"):
                st.caption("VAR.S / ddof=1 is sample variance; STDEV.S / SD is converted automatically. N counts independent customers, not customer-month rows. Binary rows require successes and sample_number; leave the mean blank if it is calculated from counts.")
                if not raw.empty:
                    target = ["partner", "segment", "metric", "sample_mean", "sample_variance", "sample_number", "sample_sd", "successes", "arm"]
                    mappings = {}
                    for col in target:
                        choices = ["(not supplied)"] + raw.columns.tolist()
                        selected = st.selectbox(col, choices, index=choices.index(col) if col in choices else 0, key=f"bv2_hist_map_{col}")
                        if selected != "(not supplied)":
                            if selected in mappings:
                                raise ValueError("Each source field can map to only one statistical field.")
                            mappings[selected] = col
                    raw = raw.rename(columns=mappings)
        if raw.empty:
            if b["plan"]:
                b["draft_dirty"]=True
                b["snapshot"]=b["bundle"]=None
            st.info("Upload, paste or enter historical statistics. Use the explicit demo from the sidebar to explore.")
            return
        history = validate_history(raw, c)
        st.success(f"✓ Validated {len(history):,} rows · {history.partner.nunique()} Partners · {history.segment.nunique()} Segments · {history.metric.nunique()} metrics")
        a,z,q = st.columns(3)
        pp = a.multiselect("Partners to include", history.partner.unique().tolist(), default=history.partner.unique().tolist(), key="bv2_history_partners")
        ss = z.multiselect("Segments to include", history.segment.unique().tolist(), default=history.segment.unique().tolist(), key="bv2_history_segments")
        mm = q.selectbox("Preview metric", history.metric.unique().tolist(), key="bv2_history_metric")
        chosen = history[history.partner.isin(pp) & history.segment.isin(ss)].copy()
        st.dataframe(chosen[chosen.metric.eq(mm)], hide_index=True, use_container_width=True)
        if chosen.empty:
            if b["plan"]:
                b["draft_dirty"]=True
                b["snapshot"]=b["bundle"]=None
            st.warning("Select at least one population.")
            return
        with st.container(border=True):
            st.markdown("#### Which customer mix should the decision represent?")
            st.caption("Starting weights reflect historical Control volume. Confirm or edit the expected future mix; the same weights will be used in every arm and in the final analysis.")
            wconfig = deepcopy(c)
            selected_pairs=set(zip(chosen.partner,chosen.segment))
            saved_weights=[r for r in c.get("weights",[]) if (r["partner"],r["segment"]) in selected_pairs]
            wconfig["weights"] = saved_weights if {(r["partner"],r["segment"]) for r in saved_weights}==selected_pairs else []
            default_weights = population_weights(chosen, wconfig)
            signature = fingerprint(default_weights)[:12]
            weights = st.data_editor(pd.DataFrame(default_weights), hide_index=True, disabled=["partner", "segment"], use_container_width=True, key=f"bv2_weights_{signature}", column_config={"weight":st.column_config.NumberColumn("Population weight", min_value=.000001, format="%.5f")})
            accepted = st.checkbox("These weights represent the intended decision population", key="bv2_accept_weights")
        calc_config = deepcopy(c)
        calc_config["weights"] = clean_records(weights)
        calc_config["weights"] = population_weights(chosen,calc_config)
        current_fp = fingerprint({"config":calc_config,"history":clean_records(chosen)})
        if b["plan"]:
            frozen_fp=b.get("frozen_input_fp") or fingerprint({"config":b["plan"]["config"],"history":b["plan"]["history"]})
            b["draft_dirty"]=current_fp!=frozen_fp
            if b["draft_dirty"]:
                b["snapshot"]=b["bundle"]=None
                st.warning("Draft inputs differ from the frozen plan. Recalculate and freeze a new version, or reopen the saved version to use its original assumptions.")
        if b.get("candidate_fp") != current_fp:
            b["candidate"] = None
        if st.button("Calculate sample requirements", type="primary", disabled=not accepted, key="bv2_calculate"):
            with st.spinner("Solving the shared Control and checking every planned comparison…"):
                version = next_version(c["experiment_id"])
                b["candidate"] = build_plan(calc_config, chosen, version, code_fingerprint())
                b["candidate_fp"] = current_fp
                b["history"] = raw.copy()
        plan = b["candidate"]
        if not plan:
            return
        _plan_summary(plan)
        a,z = st.columns(2)
        a.button("Freeze plan & continue", type="primary", use_container_width=True, key="bv2_freeze",on_click=_freeze_candidate)
        if b.get("save_error"):
            st.error("Could not freeze this version: "+b.pop("save_error")+". Calculate again to obtain a new version if another session saved it.")
        z.caption(f"Freezing saves protocol v{plan['version']}. Later edits create a new version.")
        _downloads(plan, None, STEPS[1], "candidate")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        if b["plan"]:
            b["draft_dirty"]=True
            b["snapshot"]=b["bundle"]=None
        st.error(f"Check the input: {exc}")


def _plan_summary(plan):
    a,z,q = st.columns(3)
    a.metric("Required analyzable sample", f"{sum(r['analyzable_n'] for r in plan['arm_totals']):,}")
    z.metric("Required recruitment sample", f"{sum(r['recruitment_n'] for r in plan['arm_totals']):,}")
    q.metric("Estimated recruitment", f"{plan['recruitment_weeks']:.1f} weeks")
    if not plan["within_max_duration"]:
        st.warning("△ This plan exceeds the maximum recruitment period. Review traffic or the smallest useful change.")
    st.markdown(f"**Recruitment end:** {plan['recruitment_end']} · **Earliest mature readout:** {plan['earliest_readout']}")
    totals = pd.DataFrame(plan["arm_totals"])
    st.dataframe(totals, hide_index=True, use_container_width=True)
    fig = px.bar(totals, x="arm", y=["analyzable_n","recruitment_n"], barmode="group", color_discrete_sequence=["#007f89","#bdcfdc"], labels={"value":"Customers / accounts", "arm":"Arm", "variable":"Target"})
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True, key="bv2_plan_targets")
    binding = min(plan["comparisons"], key=lambda r:r["power"])
    st.info(f"Binding comparison: {binding['scope']} · {binding['metric']} · {binding['test']} vs {binding['control']}. Verified power: {binding['power']:.1%}. This is power per comparison.")
    with st.expander("Allocation, historical assumptions & calculation details"):
        st.write(plan["estimator"])
        st.dataframe(pd.DataFrame(plan["targets"]), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame(plan["pairwise"]), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame(plan["comparisons"]), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame(plan["variance_assumptions"]), hide_index=True, use_container_width=True)
        for note in plan["limitations"]:
            st.caption(note)


def _get_bundle(b, plan):
    c = plan["config"]
    with st.expander("Bring or replace experiment data", expanded=b["bundle"] is None):
        mode = st.radio("Data level", ["Customer records", "Aggregate summaries"], horizontal=True, key="bv2_data_level")
        as_of = st.date_input("Data as of (source snapshot date)", date.today(), key="bv2_data_asof").isoformat() + "T23:59:59+00:00"
        st.caption("Replacements do not accumulate counts. Saving or downloading does not change the source date.")
        if mode == "Customer records":
            primary = next(m["name"] for m in c["metrics"] if m["role"] == "Primary")
            template = pd.DataFrame(columns=["id","partner","segment","arm"] + [m["name"] for m in c["metrics"]] + ["baseline_age","outcome_window_complete"])
            frame = _import_table("Experiment data", "bv2_records", b["record_seed"], template)
            mappings = {}
            if not frame.empty:
                with st.popover("Map ID, labels and result columns",use_container_width=True):
                    choices = ["(not supplied)"] + frame.columns.tolist()
                    for target in ["id", "arm", "partner", "segment"] + [m["name"] for m in c["metrics"]]:
                        match = next((col for col in frame.columns if col.lower() == target.lower() or (target == primary and col.lower() == "outcome")), "(not supplied)")
                        value = st.selectbox(target, choices, index=choices.index(match), key=f"bv2_record_map_{target}")
                        if value != "(not supplied)":
                            if value in mappings:
                                raise ValueError("A source column cannot be mapped to multiple fields.")
                            mappings[value] = target
                    baseline = st.multiselect("Pre-treatment covariates only", [x for x in frame.columns if x not in mappings], default=[x for x in frame.columns if x.startswith("baseline_")], key="bv2_baselines")
                    base_types = {}
                    for col in baseline:
                        base_types[col] = st.selectbox(f"{col}: feature type", ["Auto", "Continuous", "Categorical"], key=f"bv2_baseline_type_{col}")
                c = deepcopy(c)
                c["baseline_types"] = base_types
                c["baseline_selection_explicit"] = True
            raw_fp = fingerprint({"raw":frame_fingerprint(frame),"mapping":mappings,"date":as_of,"baseline":c.get("baseline_types",{}),"level":mode,"protocol":plan["config_fingerprint"]})
            if b.get("bundle_input_fp") != raw_fp:
                b["snapshot"] = b["bundle"] = None
            if st.button("Validate & use this snapshot", type="primary", disabled=frame.empty, key="bv2_validate_records"):
                b["bundle"] = prepare_records(frame, c, as_of, mappings)
                b["bundle_input_fp"] = raw_fp
                b["snapshot"] = None
                st.success("✓ Snapshot validated. Assigned customers are counted once across all metrics.")
        else:
            st.caption("Separate tables preserve assignment counts. Do not repeat assignment N once per metric.")
            results = _import_table("Outcome summaries", "bv2_summary_outcomes", template=pd.DataFrame(columns=["partner","segment","arm","metric","sample_mean","sample_variance","sample_number","successes"]))
            counts = _import_table("Assigned counts (for SRM)", "bv2_summary_counts", template=pd.DataFrame(columns=["partner","segment","arm","assigned_n","mature_n"]))
            baseline = _import_table("Baseline summaries (optional)", "bv2_summary_baseline", template=pd.DataFrame(columns=["partner","segment","arm","feature","kind","category","sample_mean","sample_variance","sample_number","missing_rate"]))
            raw_fp = fingerprint({"results":frame_fingerprint(results),"counts":frame_fingerprint(counts),"baseline":frame_fingerprint(baseline),"date":as_of,"level":mode,"protocol":plan["config_fingerprint"]})
            if b.get("bundle_input_fp") != raw_fp:
                b["snapshot"] = b["bundle"] = None
            if st.button("Validate & use these summaries", type="primary", disabled=results.empty, key="bv2_validate_summaries"):
                b["bundle"] = prepare_summaries(results, counts, baseline, c, as_of)
                b["bundle_input_fp"] = raw_fp
                b["snapshot"] = None
                st.success("✓ Aggregate snapshot validated.")
    return b["bundle"]


def _filter_controls(bundle):
    summary = bundle["summaries"]
    a,z = st.columns(2)
    partner = a.selectbox("Partner", ["All partners"]+summary.partner.unique().tolist(), key="bv2_filter_partner")
    segment = z.selectbox("Segment", ["All segments"]+summary.segment.unique().tolist(), key="bv2_filter_segment")
    return partner, segment


def _monitor_view(plan, bundle, snapshot):
    partner, segment = _filter_controls(bundle)
    counts = bundle["counts"]
    selected = counts.copy()
    outcomes = bundle["summaries"].copy()
    for col,value,all_value in [("partner",partner,"All partners"),("segment",segment,"All segments")]:
        if value != all_value:
            if not selected.empty:
                selected = selected[selected[col].eq(value)]
            outcomes = outcomes[outcomes[col].eq(value)]
    st.caption(f"Source updated: {snapshot['updated_at']} · {snapshot['source']}. Filters show operational slices; Overall statistical checks retain the frozen population.")
    primary = next(m["name"] for m in plan["config"]["metrics"] if m["role"] == "Primary")
    a,z,q = st.columns(3)
    a.metric("Assigned customers", f"{int(selected.assigned_n.sum()):,}" if not selected.empty else "Not provided")
    z.metric("Mature customers", f"{int(selected.mature_n.sum()):,}" if not selected.empty and selected.mature_n.notna().all() else "Not checked")
    q.metric("Observed primary results", f"{int(outcomes.loc[outcomes.metric.eq(primary),'sample_number'].sum()):,}")
    st.dataframe(pd.DataFrame(snapshot["progress"]), hide_index=True, use_container_width=True)
    a,z = st.columns(2)
    with a, st.container(border=True):
        st.markdown("#### Is assignment following the plan?")
        st.write("△ " + snapshot["srm"]["status"] if snapshot["srm"]["status"] != "No mismatch detected" else "✓ No mismatch detected")
        with st.expander("SRM details"):
            st.caption(snapshot["srm"]["reason"])
            st.dataframe(pd.DataFrame(snapshot["srm"]["rows"]), use_container_width=True, hide_index=True)
    with z, st.container(border=True):
        st.markdown("#### Are the groups comparable?")
        st.write(snapshot["balance"]["status"])
        with st.expander("Pre-treatment balance & SMD"):
            st.caption(snapshot["balance"]["reason"])
            st.dataframe(pd.DataFrame(snapshot["balance"]["rows"]), use_container_width=True, hide_index=True)
    st.markdown("#### Guardrail metrics & missing results")
    guard_names = [m["name"] for m in plan["config"]["metrics"] if m["role"] == "Guardrail"]
    guard = outcomes[outcomes.metric.isin(guard_names)]
    if guard.empty:
        st.info("Guardrail results were not provided for this slice.")
    else:
        st.dataframe(guard, hide_index=True, use_container_width=True)
        st.caption("Snapshot values are descriptive. Operational alert thresholds can trigger investigation; repeated ordinary p-values cannot authorize an early efficacy win.")
    if snapshot.get("guardrails"):
        st.dataframe(pd.DataFrame(snapshot["guardrails"]),use_container_width=True,hide_index=True)
    trend = pd.DataFrame(snapshot.get("trends",[]))
    if not trend.empty and guard_names:
        trend = trend[trend.metric.isin(guard_names)]
        for col,value,all_value in [("partner",partner,"All partners"),("segment",segment,"All segments")]:
            if value != all_value:
                trend=trend[trend[col].eq(value)]
        if not trend.empty and trend.snapshot_date.nunique()>1:
            metric=st.selectbox("Guardrail trend",guard_names,key="bv2_guardrail_trend_metric")
            trend=trend[trend.metric.eq(metric)].copy()
            trend["sum"]=trend["mean"]*trend.observed_n
            summary=trend.groupby(["snapshot_date","arm"],as_index=False)[["sum","observed_n"]].sum()
            summary["mean"]=summary["sum"]/summary.observed_n
            fig=px.line(summary,x="snapshot_date",y="mean",color="arm",markers=True,labels={"mean":"Descriptive observed rate","snapshot_date":"Source snapshot"},color_discrete_sequence=["#007f89","#4565a9","#ad6c26"])
            fig.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True,key="bv2_guardrail_trend")
            st.caption("Each date is a separate observed snapshot. This descriptive trend uses its actual observed mix; it is not the frozen-weight final effect estimator.")
        else:
            st.caption("A single snapshot has no trend. Upload multiple dated snapshots to see descriptive guardrail movement; customers are counted once in the latest snapshot.")
    with st.expander("Partner × Segment recruitment and outcome coverage"):
        st.dataframe(pd.DataFrame(snapshot.get("cell_progress",[])),use_container_width=True,hide_index=True)


def _analysis_view(plan, bundle, snapshot):
    if snapshot["blockers"]:
        st.warning("△ " + snapshot["decision_status"] + ". " + " ".join(snapshot["blockers"]))
    else:
        st.info("Evaluate the primary improvement together with all guardrails. A small p-value alone does not choose a winner.")
    if snapshot.get("guardrail_summary"):
        with st.expander("Guardrail decision summary",expanded=True):
            st.dataframe(pd.DataFrame([r for r in snapshot["guardrail_summary"] if r["confirmatory"]] or snapshot["guardrail_summary"]),hide_index=True,use_container_width=True)
    results = pd.DataFrame(snapshot["results"])
    a,z = st.columns(2)
    scope = a.selectbox("View results for", results.scope.unique().tolist(), key="bv2_results_scope")
    metric = z.selectbox("Result metric", [m["name"] for m in plan["config"]["metrics"]], key="bv2_results_metric")
    view = results[results.scope.eq(scope) & results.metric.eq(metric)]
    estimable=view[view.effect.notna() & view.ci_lower.notna() & view.ci_upper.notna()] if "effect" in view else pd.DataFrame()
    if not estimable.empty:
        fig=go.Figure(go.Scatter(x=estimable.effect,y=estimable.test,mode="markers",marker=dict(size=10,color="#007f89"),
            error_x=dict(type="data",symmetric=False,array=estimable.ci_upper-estimable.effect,arrayminus=estimable.effect-estimable.ci_lower,color="#45647d",thickness=2),
            hovertemplate="%{y}: %{x:.4g}<extra></extra>"))
        fig.add_vline(x=0,line_dash="dash",line_color="#a4b2c2",annotation_text="No change")
        m=next(m for m in plan["config"]["metrics"] if m["name"]==metric)
        if m["role"]=="Guardrail":
            fig.add_vline(x=(-m["margin"] if m["direction"]=="Higher" else m["margin"]),line_dash="dot",line_color="#a87824",annotation_text="Tolerance boundary")
        fig.update_layout(height=max(200,65*len(estimable)+90),margin=dict(l=5,r=30,t=30,b=20),xaxis_title=f"{metric}: Test minus Control ({m['unit']})",yaxis_title=None,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True,key="bv2_effect_intervals")
        st.caption("Dots show estimated changes. Lines show the reported confidence intervals; crossing the dashed line means no change remains plausible.")
    for r in view.to_dict("records"):
        with st.container(border=True):
            st.markdown(f"#### {r['test']} vs {r['control']}")
            st.write(r["interpretation"])
            if pd.notna(r.get("effect")):
                a,z,q = st.columns(3)
                a.metric("Change per customer", f"{r['effect']:+.4g}")
                z.metric("Relative change", f"{r['relative_lift']:+.1%}" if pd.notna(r.get("relative_lift")) else "Not defined")
                q.metric(f"{r['ci_level']:.1%} interval", f"{r['ci_lower']:.4g} to {r['ci_upper']:.4g}" if pd.notna(r.get("ci_lower")) else "Insufficient data")
                if pd.notna(r.get("percentage_points")):
                    st.caption(f"Binary change: {r['percentage_points']:+.3f} percentage points.")
            st.caption("Predeclared decision" if r["confirmatory"] else "Exploratory slice · no independent power guarantee")
    with st.expander("Statistical details, group statistics & all comparisons"):
        st.write(snapshot["multiplicity"])
        st.dataframe(view, hide_index=True, use_container_width=True)
        st.dataframe(bundle["summaries"], hide_index=True, use_container_width=True)
        st.caption("Continuous: Welch t-test. Binary: pooled two-proportion z-test when adequate, Fisher exact for a single sparse stratum. Weighted sparse Binary tests are withheld, with cell-level exact p-values in the data export. Segment significance differences are not interaction tests.")


def _downloads(plan, snapshot, stage, key):
    with st.expander("Download calculation memo & reproducible inputs"):
        st.caption("Uses the same saved calculation as this page. Individual customer IDs are excluded.")
        cache_key = fingerprint({"plan":plan,"snapshot":snapshot,"stage":stage})
        if st.button("Prepare PDF memo", key=f"bv2_pdf_prepare_{key}"):
            try:
                from .business_report import build_pdf
                with st.spinner("Building the calculation memorandum…"):
                    st.session_state.bv2_pdf = {"key":cache_key,"bytes":build_pdf(plan,snapshot,stage)}
            except (ImportError, ValueError) as exc:
                st.error(f"PDF generation failed: {exc}. Install the pinned requirements on the app server.")
        pdf = st.session_state.get("bv2_pdf",{})
        if pdf.get("key") == cache_key:
            st.download_button("Download PDF memo",pdf["bytes"],file_name=f"customer-v{plan['version']}-{key}.pdf",mime="application/pdf",key=f"bv2_pdf_download_{key}")
        label="Download setup JSON" if plan.get("kind")=="setup_draft" else "Download exploratory protocol JSON" if plan.get("retrospective") else "Download frozen protocol JSON"
        st.download_button(label,json.dumps(plan,ensure_ascii=False,allow_nan=False,indent=2).encode(),file_name=f"customer-protocol-v{plan['version']}.json",mime="application/json",key=f"bv2_json_{key}")
        rows = snapshot.get("results",snapshot["summaries"]) if snapshot else plan["targets"]
        st.download_button("Download aggregate CSV",pd.DataFrame(rows).to_csv(index=False).encode(),file_name=f"customer-{key}.csv",mime="text/csv",key=f"bv2_csv_{key}")
        if snapshot:
            st.download_button("Download snapshot JSON",json.dumps(snapshot,ensure_ascii=False,allow_nan=False,indent=2).encode(),file_name=f"customer-snapshot-{key}.json",mime="application/json",key=f"bv2_snapshot_{key}")


def render_business_workspace():
    st.markdown(CSS,unsafe_allow_html=True)
    b = _state()
    with st.sidebar:
        st.divider()
        st.caption("CUSTOMER BUSINESS V2")
        st.button("Explore synthetic demo",on_click=_load_demo,use_container_width=True,key="bv2_demo")
        st.button("New customer experiment",on_click=_new_experiment,use_container_width=True,key="bv2_new")
        with st.expander("Open a saved protocol"):
            saved=saved_protocols()
            if saved:
                option=st.selectbox("Saved versions",range(len(saved)),format_func=lambda i:f"{'DEMO · ' if saved[i]['synthetic'] else ''}{saved[i]['name']} · v{saved[i]['version']} · {saved[i]['owner']}",key="bv2_saved_versions")
                if st.button("Open saved version",key="bv2_open_saved"):
                    try:
                        _open_plan(import_protocol(Path(saved[option]["path"]).read_text(encoding="utf-8")))
                        st.rerun()
                    except (ValueError,KeyError,OSError) as exc:
                        st.error(str(exc))
            upload = st.file_uploader("Protocol JSON",type="json",key="bv2_protocol_upload")
            if upload and st.button("Open protocol",key="bv2_open_protocol"):
                try:
                    p = import_protocol(upload.getvalue())
                    _open_plan(p)
                    st.rerun()
                except (ValueError,KeyError,TypeError) as exc:
                    st.error(str(exc))
    st.markdown('<div class="bv2-shell"><span class="bv2-mark">C</span>Experiment Platform<small>Customer experiments</small></div>',unsafe_allow_html=True)
    a,z = st.columns([5,1])
    with a:
        st.title(b["config"]["name"])
        st.markdown(f'<div class="bv2-subtitle">Owner: {html.escape(b["config"]["owner"] or "Not assigned")}</div>',unsafe_allow_html=True)
    with z:
        st.caption(f"Frozen · v{b['plan']['version']}" if b["plan"] else "Draft · Not frozen")
        st.button("All experiments",on_click=lambda:setattr(st.session_state,"main_page","Overview"),key="bv2_all")
    if b["demo"]:
        st.info("Synthetic demo · All inputs and outcomes are illustrative. Use New customer experiment before bringing real data.")
    if "bv2_pending_step" in st.session_state:
        st.session_state.bv2_step=st.session_state.pop("bv2_pending_step")
    step=st.radio("Experiment stages",STEPS,horizontal=True,format_func=lambda s:f"{STEPS.index(s)+1}  {s}",label_visibility="collapsed",key="bv2_step")
    st.divider()
    if step==STEPS[0]:
        _setup(b)
    elif step==STEPS[1]:
        _plan(b)
    else:
        plan=b["plan"]
        exploratory = plan is None and step==STEPS[3] and b["setup_saved"]
        if plan is None and not exploratory:
            st.info("Freeze a sample plan first. Its population, definitions and expected assignment ratios carry into monitoring and analysis.")
            st.button("Build a test plan",on_click=_go,args=(STEPS[1],))
            return
        if exploratory:
            st.warning("Exploratory existing-data analysis · ID, Arm and Outcome are enough to estimate an effect. No planned-power or SRM claim is possible without a prospective protocol.")
            plan=exploratory_protocol(b["config"])
        if fingerprint(b["config"])!=plan["config_fingerprint"] or b.get("draft_dirty"):
            st.warning("Setup changed. Recalculate and freeze a new protocol version before monitoring or analyzing these settings.")
            return
        st.subheader("Are the groups comparable?" if step==STEPS[2] else "What changed, and how certain are we?")
        try:
            bundle=_get_bundle(b,plan)
            if bundle is None:
                st.info("Add a snapshot to see assignment, maturity and result coverage.")
                return
            if exploratory:
                plan=exploratory_protocol(b["config"],bundle)
            if step==STEPS[2]:
                snap=b["snapshot"]
                if snap is None or snap["stage"]!=STEPS[2] or snap["data_fingerprint"]!=bundle["fingerprint"]:
                    snap=monitor(plan,bundle)
                    b["snapshot"]=snap
                _monitor_view(plan,bundle,snap)
                _downloads(plan,snap,step,"monitor")
                st.button("Continue to analysis",type="primary",on_click=_go,args=(STEPS[3],),key="bv2_continue_analysis")
            else:
                if st.button("Calculate final readout",type="primary",key="bv2_analyze"):
                    b["snapshot"]=analyze(plan,bundle)
                result=b["snapshot"]
                if result and result["stage"]==STEPS[3] and result["data_fingerprint"]==bundle["fingerprint"]:
                    _analysis_view(plan,bundle,result)
                    _downloads(plan,result,step,"analysis")
                else:
                    st.caption("This is a fixed-horizon readout. Finish recruitment and the observation window before acting.")
        except (ValueError,KeyError,TypeError,OSError) as exc:
            b["snapshot"]=None
            st.error(f"Check the input: {exc}")
