import numpy as np
import pandas as pd
import pytest

from src.experiment_platform.customer import automatic_matching_distance, assign_group_labels, build_customer_preview, calculate_group_design, calculate_variant_group_design, default_numeric_groups, planning_comparison_count, suggest_group_designs
from src.experiment_platform.core import load_customer_demo
from src.experiment_platform.core import DesignConfig, MetricConfig
from src.experiment_platform.customer import continuous_summary_results, continuous_results, plan_from_arm_summaries, validate_arm_summaries


def planning_history(seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for line, mean, sd in [(3000, 900, 250), (5000, 1400, 400), (8000, 2050, 620)]:
        for _ in range(160):
            rows.append({"current_credit_line": line, "outcome": rng.normal(mean, sd)})
    return pd.DataFrame(rows)


def test_default_fico_groups_and_assignment():
    values = pd.Series([580, 650, 680, 719, 740, 800])
    definitions = default_numeric_groups(values, "fico")
    labels = assign_group_labels(values, definitions)
    assert definitions[0]["label"] == "Below 660"
    assert definitions[-1]["label"] == "720+"
    assert labels.tolist() == ["Below 660", "Below 660", "660 to 719", "660 to 719", "720+", "720+"]


def test_customer_preview_is_one_row_per_account_with_business_context():
    raw = load_customer_demo("General Customer Test")
    preview = build_customer_preview(
        raw,
        "account_id",
        "primary_outcome",
        "cpc",
        "risk_segment",
        "historical_experience",
        "booking_date",
        "mob",
    )
    assert preview["unique_accounts"] == raw["account_id"].nunique()
    assert len(preview["accounts"]) == raw["account_id"].nunique()
    assert preview["product_count"] == raw["cpc"].nunique()
    assert preview["group_count"] == raw["risk_segment"].nunique()
    assert preview["accounts"]["Customer / Account"].is_unique
    assert {"Historical Strategy", "Average Outcome", "Periods"}.issubset(preview["accounts"].columns)


def test_group_plan_allocates_flow_to_finish_lines_together():
    metric = MetricConfig(
        "Outcome",
        "outcome",
        "Primary",
        effect_type="Relative %",
        effect_value=0.15,
        source_column="outcome",
        processed_column="outcome",
    )
    design = DesignConfig(sample_size_basis="Primary Metric Only")
    rows, comparisons, _ = calculate_group_design(
        planning_history(),
        "current_credit_line",
        {"Primary": metric},
        design,
        5000,
        [3000, 8000],
        eligible_flow=300,
        group_label="Prime",
    )
    assert len(rows) == 3
    assert comparisons
    assert abs(sum(row["Traffic Allocation"] for row in rows) - 1) < 1e-9
    assert max(row["Test Duration"] for row in rows) - min(row["Test Duration"] for row in rows) < 1e-9


def test_suggestions_stay_inside_business_range_and_report_duration():
    metric = MetricConfig(
        "Outcome",
        "outcome",
        "Primary",
        effect_type="Relative %",
        effect_value=0.15,
        source_column="outcome",
        processed_column="outcome",
    )
    suggestions = suggest_group_designs(
        planning_history(),
        "current_credit_line",
        {"Primary": metric},
        DesignConfig(sample_size_basis="Primary Metric Only"),
        control=5000,
        minimum=3000,
        maximum=8000,
        treatment_count=2,
        eligible_flow=300,
        max_periods=12,
    )
    assert not suggestions.empty
    assert (suggestions["Test Duration"] > 0).all()
    assert all(3000 <= value <= 8000 for values in suggestions["Suggested Treatments"] for value in values)


def test_named_offer_plan_uses_group_level_history_without_strategy_column():
    history = pd.DataFrame({"outcome": np.linspace(80, 120, 300)})
    metric = MetricConfig(
        "Revenue",
        "outcome",
        "Primary",
        effect_type="Relative %",
        effect_value=0.1,
        source_column="outcome",
        processed_column="outcome",
    )
    rows, comparisons, binding = calculate_variant_group_design(
        history,
        {"Primary": metric},
        DesignConfig(sample_size_basis="Primary Metric Only"),
        "BAU Offer",
        ["Cash Bonus", "Rewards Bundle"],
        eligible_flow=600,
        group_label="Prime",
    )
    assert [row["Arm"] for row in rows] == ["BAU Offer", "Cash Bonus", "Rewards Bundle"]
    assert all(row["Support"] == "Group-level pooled history" for row in rows)
    assert abs(sum(row["Traffic Allocation"] for row in rows) - 1) < 1e-9
    assert comparisons and "Revenue" in binding


def test_automatic_matching_distance_excludes_remote_history():
    assert automatic_matching_distance([3000, 8000]) == 2500
    history = pd.DataFrame(
        {
            "current_credit_line": [-5000, 2500, 3000, 3500, 7500, 8000, 8500, 20000],
            "outcome": [10, 80, 100, 120, 180, 200, 220, 500],
        }
    )
    metric = MetricConfig(
        "Outcome",
        "outcome",
        "Primary",
        effect_type="Absolute",
        effect_value=20,
        source_column="outcome",
        processed_column="outcome",
    )
    rows, _, _ = calculate_group_design(
        history,
        "current_credit_line",
        {"Primary": metric},
        DesignConfig(sample_size_basis="Primary Metric Only"),
        3000,
        [8000],
        eligible_flow=100,
        group_label="All customers",
    )
    assert sum(row["Historical N"] for row in rows) == 6
    assert all(row["Historical Match Limit"] == 2500 for row in rows)


def test_joint_audience_scope_protects_all_treatment_comparisons():
    assert planning_comparison_count(2, 3, "Separate decision per audience") == 2
    assert planning_comparison_count(2, 3, "One joint decision across audiences") == 6

    metric = MetricConfig(
        "Outcome",
        "outcome",
        "Primary",
        effect_type="Relative %",
        effect_value=0.15,
        source_column="outcome",
        processed_column="outcome",
    )
    design = DesignConfig(sample_size_basis="Primary Metric Only")
    separate, _, _ = calculate_group_design(
        planning_history(),
        "current_credit_line",
        {"Primary": metric},
        design,
        5000,
        [3000, 8000],
        eligible_flow=300,
        group_label="Prime",
        family_comparisons=2,
    )
    joint, _, _ = calculate_group_design(
        planning_history(),
        "current_credit_line",
        {"Primary": metric},
        design,
        5000,
        [3000, 8000],
        eligible_flow=300,
        group_label="Prime",
        family_comparisons=6,
    )
    assert sum(row["Required Accounts"] for row in joint) > sum(row["Required Accounts"] for row in separate)


def summary_fixture():
    return pd.DataFrame({"Group": ["All customers"] * 3, "Arm": ["Control", "A", "B"],
                         "Mean": [100., 106., 110.], "Variance": [400., 900., 1600.], "N": [100, 80, 120]})


def test_summary_analysis_matches_raw_welch_results():
    rng = np.random.default_rng(123)
    raw = pd.DataFrame({"arm": ["Control"] * 90 + ["A"] * 110 + ["B"] * 70,
                        "outcome": np.r_[rng.normal(100, 20, 90), rng.normal(107, 30, 110), rng.normal(112, 40, 70)]})
    summaries = raw.groupby("arm")["outcome"].agg(Mean="mean", Variance="var", N="count").reset_index().rename(columns={"arm": "Arm"})
    summaries["Group"] = "All customers"
    expected = continuous_results(raw, "arm", "outcome", "Control", ["A", "B"], .05, "Holm").set_index("Credit Line").sort_index()
    actual = continuous_summary_results(summaries, "Control", .05, "Holm").set_index("Credit Line").sort_index()
    for column in ["Mean / Rate", "Effect vs Control", "Relative Lift", "CI Lower", "CI Upper", "p-value", "Adjusted p-value"]:
        np.testing.assert_allclose(actual[column], expected[column], equal_nan=True, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("column,value", [("Variance", -1), ("Mean", np.inf), ("N", 1), ("N", 2.5), ("Arm", ""), ("Group", None)])
def test_summary_validation_rejects_invalid_values(column, value):
    frame = summary_fixture().astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_arm_summaries(frame)


def test_summary_validation_rejects_duplicates_and_missing_arms():
    frame = summary_fixture()
    with pytest.raises(ValueError):
        validate_arm_summaries(pd.concat([frame, frame.iloc[:1]]))
    with pytest.raises(ValueError):
        validate_arm_summaries(frame.iloc[:1])


def test_summary_plan_uses_arm_specific_sd_and_finishes_together():
    metric = MetricConfig("Revenue", "", "Primary", effect_value=.15)
    design = DesignConfig()
    rows, comparisons = plan_from_arm_summaries(summary_fixture(), {"All customers": "Control"}, {"All customers": 300}, metric, design)
    assert {row["Historical SD"] for row in rows} == {20, 30, 40}
    assert len({row["Required Accounts"] for row in rows}) > 1
    assert sum(row["Traffic Allocation"] for row in rows) == pytest.approx(1)
    assert sum(row["Flow per Period"] for row in rows) == pytest.approx(300)
    assert len({round(row["Test Duration"], 8) for row in rows}) == 1
    assert min(row["Planned Power"] for row in comparisons) >= .8


def test_summary_plan_supports_separate_audiences_and_zero_baseline():
    frame = summary_fixture()
    second = frame.assign(Group="Prime")
    metric = MetricConfig("Revenue", "", "Primary", effect_value=.15)
    rows, _ = plan_from_arm_summaries(pd.concat([frame, second]), {"All customers": "Control", "Prime": "Control"}, {"All customers": 300, "Prime": 150}, metric, DesignConfig())
    assert len(rows) == 6
    assert rows[-1]["Test Duration"] == pytest.approx(2 * rows[0]["Test Duration"])
    frame.loc[0, "Mean"] = 0
    with pytest.raises(ValueError, match="absolute effect"):
        plan_from_arm_summaries(frame, {"All customers": "Control"}, {"All customers": 300}, metric, DesignConfig())


def test_summary_analysis_handles_zero_control_mean_without_infinite_lift():
    frame = summary_fixture()
    frame.loc[0, "Mean"] = 0
    result = continuous_summary_results(frame, "Control")
    assert result.loc[~result["Is Control"], "Relative Lift"].isna().all()


def test_summary_planning_accepts_assumed_variance_without_historical_counts():
    frame = summary_fixture().drop(columns="N")
    rows, detail = plan_from_arm_summaries(frame, {"All customers": "Control"}, {"All customers": 300}, MetricConfig("Revenue", "", "Primary", effect_value=.15), DesignConfig())
    assert all(row["Historical N"] == 0 for row in rows)
    assert min(row["Planned Power"] for row in detail) >= .8
    with pytest.raises(ValueError):
        continuous_summary_results(frame, "Control")
