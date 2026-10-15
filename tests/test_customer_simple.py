import numpy as np
import pandas as pd
import pytest
from scipy import stats
from streamlit.testing.v1 import AppTest

from src.experiment_platform.core import DesignConfig, MetricConfig
from src.experiment_platform.customer import (
    analyze_customer_records, combine_partner_summaries, plan_from_arm_summaries,
    power_continuous_unequal, split_partner_targets, validate_arm_summaries,
)


def summaries():
    return pd.DataFrame({"Partner": ["A"] * 3, "Arm": ["Control", "Low", "High"],
                         "Mean": [100., 100., 100.], "SD": [30., 15., 70.]})


def test_shared_control_is_pairwise_max_and_treatments_are_resolved():
    metric = MetricConfig("Revenue", "", "Primary", effect_type="Absolute", effect_value=8.)
    design = DesignConfig()
    rows, comparisons = plan_from_arm_summaries(summaries(), {"A": "Control"}, {"A": 1000}, metric, design)
    targets = {row["Arm"]: row["Required Accounts"] for row in rows}
    nc = targets["Control"]
    assert nc == max(row["Pairwise Control N"] for row in comparisons)
    assert any(row["Treatment N"] < row["Pairwise Treatment N"] for row in comparisons)
    assert sum(row["Traffic Allocation"] for row in rows) == pytest.approx(1.)
    for row in comparisons:
        assert row["Control N"] == nc
        assert row["Planned Power"] >= .8
        assert power_continuous_unequal(.025, 8, row["Control SD"], row["Treatment SD"], nc, row["Treatment N"] - 1) < .8
        assert "vs Control" in row["Comparison"]


def test_sd_and_variance_inputs_are_equivalent_and_conflicts_rejected():
    data = summaries()
    expected = data.assign(Variance=data.SD ** 2).drop(columns="SD")
    pd.testing.assert_frame_equal(validate_arm_summaries(data, True), validate_arm_summaries(expected, True))
    with pytest.raises(ValueError, match="disagree"):
        validate_arm_summaries(data.assign(Variance=1), True)


def test_fixed_partner_mix_includes_between_partner_variance_and_exact_targets():
    a = summaries().assign(Mean=0., SD=1.)
    b = summaries().assign(Partner="B", Mean=100., SD=1.)
    combined = combine_partner_summaries(pd.concat([a, b]), {"A": 1, "B": 3})
    assert np.allclose(combined.Mean, 75.)
    assert np.allclose(combined.Variance, 1876.)
    rows, _ = plan_from_arm_summaries(combined, {"Combined partners": "Control"}, {"Combined partners": 100},
                                     MetricConfig("Revenue", "", "Primary", effect_type="Absolute", effect_value=10), DesignConfig())
    split = split_partner_targets(rows, {"A": 1, "B": 3})
    for row in rows:
        arm_split = split[split.Arm == row["Arm"]]
        assert arm_split["Required Accounts"].sum() == row["Required Accounts"]
        assert abs(arm_split.iloc[0]["Required Accounts"] - row["Required Accounts"] / 4) <= 1
    with pytest.raises(ValueError, match="same arms"):
        combine_partner_summaries(pd.concat([a, b.iloc[:2]]), {"A": 1, "B": 1})


def records():
    return pd.DataFrame({"ID": [f"c{i}" for i in range(12)],
                         "Arm": ["Control"] * 4 + ["Option A"] * 4 + ["Option B"] * 4,
                         "Outcome": [8., 9., 11., 12., 12., 13., 15., 16., 9., 12., 13., 14.]})


def test_three_column_analysis_matches_welch_and_does_not_invent_allocation():
    frame = records()
    results, audit = analyze_customer_records(frame, "ID", "Arm", "Outcome", "Control")
    result = results["All customers"].set_index("Credit Line")
    expected = stats.ttest_ind(frame[frame.Arm == "Option A"].Outcome, frame[frame.Arm == "Control"].Outcome, equal_var=False)
    assert result.loc["Option A", "Raw p-value"] == pytest.approx(expected.pvalue)
    assert result.loc["Option A", "Effect vs Control"] == 4
    assert result.loc["Option A", "Adjusted p-value"] >= expected.pvalue
    assert audit["health"][0]["Allocation check"].startswith("Not checked")


@pytest.mark.parametrize("change, message", [
    (lambda df: df.assign(ID="duplicate"), "unique"),
    (lambda df: df.assign(ID=" "), "blank"),
    (lambda df: df.assign(Outcome=np.nan), "finite"),
    (lambda df: df.assign(Outcome=np.inf), "finite"),
])
def test_three_column_analysis_blocks_invalid_records(change, message):
    with pytest.raises(ValueError, match=message):
        analyze_customer_records(change(records()), "ID", "Arm", "Outcome", "Control")


def test_three_column_binary_zero_baseline_has_finite_uncertainty():
    frame = records().assign(Outcome=[0] * 4 + [1, 0, 1, 0] + [0] * 4)
    results, audit = analyze_customer_records(frame, "ID", "Arm", "Outcome", "Control")
    result = results["All customers"].set_index("Credit Line")
    assert audit["metric_type"] == "Binary"
    assert np.isnan(result.loc["Option A", "Relative Lift"])
    assert np.isfinite(result.loc["Option A", "Adjusted p-value"])
    assert result.loc["Option B", "CI Lower"] < 0 < result.loc["Option B", "CI Upper"]
    assert result.loc["Option B", "Adjusted p-value"] == 1


def test_optional_partner_analysis_and_explicit_allocation_check():
    frame = pd.concat([records().assign(Partner="A"), records().assign(Partner="B", ID=lambda df: "b" + df.ID)])
    results, audit = analyze_customer_records(frame, "ID", "Arm", "Outcome", "Control", partner_column="Partner",
                                             expected_shares={"Control": .98, "Option A": .01, "Option B": .01})
    assert set(results) == {"A", "B"}
    assert audit["partner_checked"]
    assert all(row["Allocation check"] == "Review" for row in audit["health"])


def test_removing_partner_preserves_remaining_partners_traffic():
    app = AppTest.from_file("app.py", default_timeout=45).run()
    app.session_state.main_page = "Customer"
    app.session_state.customer_intent = "plan"
    app.session_state.customer_input_mode = "Arm summaries"
    app.run()
    flow = next(item for item in app.number_input if item.label == "Partner B: eligible accounts per period")
    flow.set_value(777.).run()
    app.multiselect(key="customer_summary_plan_partners").set_value(["Partner B"]).run()
    assert not app.exception
    remaining_flow = next(item for item in app.number_input if item.label == "Partner B: eligible accounts per period")
    assert remaining_flow.value == 777.
