"""Business-facing three-column customer result workflow."""
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from .core import MetricConfig, normalize_uploaded_dataset
from .customer import analyze_customer_records
from .portfolio import configuration_fingerprint
from .timeseries import humanize_column_name


def simple_customer_analysis(render_evidence) -> None:
    st.session_state.customer_analysis_is_current = False
    st.markdown("**Analyze a test with just three columns**")
    st.caption("ID = one randomized customer/account · Arm = assigned option · Outcome = actual result over the same observation window. Every option is compared with your chosen control.")
    template = "ID,Arm,Outcome\nC001,Control,100\nC002,Control,110\nC003,Option A,115\nC004,Option A,120\nC005,Option B,108\nC006,Option B,125\n"
    st.download_button("Download 3-column CSV template", template, "customer_results_template.csv", "text/csv", key="simple_template")
    source = st.radio("Result source", ["Upload CSV", "Try an example"], horizontal=True, key="simple_source")
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload ID, arm label, and actual outcome", type=["csv"], key="simple_upload")
        if uploaded is None:
            st.info("Only three columns are required. For conversion or retention, use Outcome = 1 for yes and 0 for no. Add Partner only if you want separate Partner results.")
            return
        try:
            # IDs and numeric-looking labels are identifiers, not quantities.
            # Preserve leading zeroes; only the outcome is converted later.
            frame = normalize_uploaded_dataset(pd.read_csv(uploaded, dtype=str))
        except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
            st.error(f"Could not read this CSV: {exc}")
            return
    else:
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({"ID": [f"C{i:04d}" for i in range(600)],
                              "Arm": np.repeat(["Control", "Option A", "Option B"], 200),
                              "Outcome": np.concatenate([rng.normal(mean, sd, 200) for mean, sd in [(100, 20), (108, 25), (101, 30)]])})
        st.info("Example results — these are synthetic customers, not live business data.")
    if len(frame.columns) < 3 or frame.empty:
        st.error("Provide a non-empty CSV with ID, arm label, and actual outcome columns.")
        return
    columns = list(frame.columns)
    saved = st.session_state.get("customer_simple_settings", {})
    def guess(names, fallback):
        return next((col for name in names for col in columns if col.lower() == name), fallback)
    id_col = guess(["id", "customer_id", "account_id", "user_id"], columns[0])
    arm_col = guess(["arm", "label", "treatment", "group", "assigned_treatment"], columns[1])
    outcome_col = guess(["outcome", "value", "result", "actual", "revenue", "conversion"], columns[2])
    id_col = saved.get("id_column") if saved.get("id_column") in columns else id_col
    arm_col = saved.get("arm_column") if saved.get("arm_column") in columns else arm_col
    outcome_col = saved.get("outcome_column") if saved.get("outcome_column") in columns else outcome_col
    design = st.session_state.design_config
    with st.expander("Column mapping and advanced settings", expanded=False):
        a, b, c = st.columns(3)
        id_col = a.selectbox("ID column", columns, index=columns.index(id_col), key="simple_id")
        arm_col = b.selectbox("Arm label column", columns, index=columns.index(arm_col), key="simple_arm")
        outcome_col = c.selectbox("Actual outcome column", columns, index=columns.index(outcome_col), key="simple_outcome")
        partner_options = ["No separate Partner analysis"] + [col for col in columns if col not in {id_col, arm_col, outcome_col}]
        partner = st.selectbox("Optional Partner column", partner_options, index=partner_options.index(saved["partner_column"]) if saved.get("partner_column") in partner_options else 0, key="simple_partner")
        metric_type = st.selectbox("Outcome type", ["Auto", "Continuous", "Binary"], index=["Auto", "Continuous", "Binary"].index(saved.get("metric_type", "Auto")), key="simple_type")
        direction = st.selectbox("Better outcome", ["Higher is Better", "Lower is Better"], index=1 if saved.get("direction") == "Lower is Better" else 0, key="simple_direction")
        alpha = st.number_input("False-positive rate", min_value=.001, max_value=.2, value=float(design.alpha), key="simple_alpha")
        protection = st.selectbox("Comparison protection", ["Holm", "None"], index=0 if design.multiplicity_method == "Holm" else 1, key="simple_protection")
        joint = st.checkbox("Protect one decision across all Partner comparisons", value=True, key="simple_joint")
    labels = frame[arm_col].dropna().astype(str).str.strip().drop_duplicates().tolist()
    if len(labels) < 2:
        st.error("Provide at least two arm labels: a control and one test option.")
        return
    preferred = next((label for label in labels if label.lower() in {"control", "bau", "current"}), labels[0])
    preferred = saved["control"] if saved.get("control") in labels else preferred
    control = st.selectbox("Which label is the current / control experience?", labels, index=labels.index(preferred), key="simple_control")
    outcome_name = st.text_input("Business outcome name", value=saved.get("outcome_name", humanize_column_name(outcome_col)), key="simple_name")
    st.caption(f"Detected: {len(frame):,} records · {len(labels) - 1} test options versus {control}. Confirm that assignment was randomized; three columns cannot prove randomization or equal Partner mix.")
    expected = None
    planned = design.arm_sample_sizes
    matching_plan = all(float(planned.get(label, 0)) > 0 for label in labels)
    allocation_key = configuration_fingerprint({"labels": labels, "plan": planned})[:10]
    with st.expander("Check planned traffic allocation (optional)", expanded=False):
        check_allocation = st.checkbox("I have a planned traffic split", value=matching_plan, key=f"simple_check_allocation_{allocation_key}")
        st.caption("Use the intended randomization split, not the observed split. Without it, allocation health is left unchecked.")
        if check_allocation:
            expected = {}
            total = sum(float(planned.get(label, 0)) for label in labels) if matching_plan else len(labels)
            for index, label in enumerate(labels):
                value = float(planned[label]) / total if matching_plan else 1 / len(labels)
                expected[label] = st.number_input(f"{label}: planned traffic (%)", min_value=.001, max_value=100., value=100 * value, key=f"simple_share_{allocation_key}_{index}")
            if not np.isclose(sum(expected.values()), 100, atol=.1):
                st.error("Planned traffic percentages must total 100%.")
                return
    settings = dict(id_column=id_col, arm_column=arm_col, outcome_column=outcome_col,
                    control=control, alpha=alpha, metric_type=metric_type,
                    multiplicity_method=protection, partner_column=None if partner == partner_options[0] else partner,
                    joint_decision=joint, expected_shares=expected)
    fingerprint = configuration_fingerprint({"data": frame.to_json(orient="split"), "settings": settings, "outcome": outcome_name, "direction": direction})
    current = st.session_state.get("simple_fingerprint") == fingerprint
    if st.button("Calculate Results", type="primary", key="simple_calculate"):
        try:
            results, audit = analyze_customer_records(frame, **settings)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state.simple_results = results
        st.session_state.simple_audit = audit
        st.session_state.simple_timestamp = datetime.now().isoformat(timespec="seconds")
        st.session_state.simple_fingerprint = fingerprint
        current = True
    if not current:
        if st.session_state.get("simple_fingerprint"):
            st.warning("Inputs changed. Calculate again to update the result.")
        return
    results, audit = st.session_state.simple_results, st.session_state.simple_audit
    metric = MetricConfig(outcome_name, outcome_col, "Primary", metric_type=audit["metric_type"], direction=direction)
    design.alpha, design.multiplicity_method = alpha, protection
    design.audience_decision_scope = "One joint decision across audiences" if joint else "Separate decision per audience"
    st.session_state.metrics_config = {"Primary": metric}
    st.session_state.group_analysis_results = {group: {"Primary": result} for group, result in results.items()}
    st.session_state.analysis_results_by_role = {"Primary": next(iter(results.values()))}
    st.session_state.customer_analysis_timestamp = st.session_state.simple_timestamp
    st.session_state.customer_analysis_mode = "Simple results"
    st.session_state.customer_analysis_is_current = True
    st.session_state.customer_evidence_note = "Customer records checked; confirm randomized assignment, outcome maturity, and Partner balance before acting"
    st.session_state.customer_simple_settings = {**settings, "outcome_name": outcome_name, "direction": direction}
    health = pd.DataFrame(audit["health"])
    st.markdown("**Data checks**")
    st.success(f"{audit['accounts']:,} unique IDs · numeric outcomes complete · at least two accounts per option")
    st.dataframe(health, hide_index=True, use_container_width=True)
    if (health["Allocation check"] == "Review").any():
        st.warning("Observed allocation differs from the planned split. Investigate assignment and missing customers before using these effects for a decision.")
    combined = []
    for index, (group, result) in enumerate(results.items()):
        st.markdown(f"**{group} · What changed?**")
        for _, row in result[~result["Is Control"]].iterrows():
            sign = 1 if direction == "Higher is Better" else -1
            low, high = sorted([sign * row["CI Lower"], sign * row["CI Upper"]])
            status = "Evidence of improvement" if low > 0 and row["Significant"] else "Evidence of worse outcomes" if high < 0 and row["Significant"] else "Evidence is not sufficient to establish a difference"
            scale, unit = (100, " percentage points") if metric.metric_type == "Binary" else (1, " outcome units")
            st.write(f"**{row['Credit Line']}**: {status}. Difference from {control}: {row['Effect vs Control'] * scale:+,.2f}{unit}; uncertainty range {row['CI Lower'] * scale:+,.2f} to {row['CI Upper'] * scale:+,.2f}.")
        render_evidence(result, metric, f"simple_evidence_{index}")
        combined.append(result.assign(Partner=group))
    st.caption("An inconclusive result does not prove no effect. Check the planned sample target and observation window before considering more data. Repeated early checks require a separately reviewed sequential-testing plan.")
    st.download_button("Download analysis results", pd.concat(combined).to_csv(index=False), "customer_analysis_results.csv", "text/csv", key="simple_results_download")
