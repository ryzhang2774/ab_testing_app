from streamlit.testing.v1 import AppTest
from unittest.mock import patch


def summary_app(intent):
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.session_state.main_page = "Customer"
    app.session_state.customer_intent = intent
    app.session_state.customer_input_mode = "Arm summaries"
    app.session_state.customer_summary_drafts = {intent: {
        "outcome": "Revenue", "rows": [
            {"Group": "All customers", "Arm": "Control", "Mean": 100., "Variance": 400., "N": 100},
            {"Group": "All customers", "Arm": "Offer A", "Mean": 110., "Variance": 900., "N": 120},
        ]}}
    return app.run()


def test_summary_plan_runs_and_hands_off_to_analysis():
    app = summary_app("plan")
    assert not app.exception
    app.button(key="customer_summary_plan_run").click().run()
    assert not app.exception
    assert app.session_state.design_config.total_sample_size > 0
    assert len(app.session_state.design_config.arm_sample_sizes) == 2
    app.button(key="customer_analyze_this_plan").click().run()
    assert not app.exception
    assert app.session_state.customer_intent == "analyze"
    assert not any(tab.label == "2 · Recommendation" for tab in app.tabs)


def test_summary_analysis_is_invalidated_when_input_changes():
    app = summary_app("analyze")
    app.button(key="customer_summary_analyze_run").click().run()
    assert not app.exception
    assert app.session_state.customer_analysis_is_current
    assert app.session_state.analysis_results_by_role["Primary"].iloc[1]["Effect vs Control"] == 10
    stamp = app.session_state.customer_analysis_timestamp
    app.number_input(key="customer_summary_analyze_alpha").set_value(.1).run()
    assert not app.exception
    assert not app.session_state.customer_analysis_is_current
    assert app.session_state.customer_analysis_timestamp == stamp
    assert any("Inputs changed" in warning.value for warning in app.warning)


def test_summary_save_preserves_analysis_time_and_does_not_claim_completion():
    app = summary_app("analyze")
    app.button(key="customer_summary_analyze_run").click().run()
    stamp = app.session_state.customer_analysis_timestamp
    with patch("src.experiment_platform.portfolio.load_experiments", return_value=[]), patch("src.experiment_platform.portfolio.upsert_experiment", side_effect=lambda record: {**record, "id": "test-summary"}) as saved:
        app.button(key="customer_save_experiment").click().run()
        assert not app.exception
        record = saved.call_args.args[0]
        assert record["analysis_timestamp"] == stamp
        assert record["data_updated_at"] == ""
        assert record["status"] != "Ready to decide"
        assert record["progress"] == 0  # No recruitment target was entered.
        assert record["sample_collected"] == 220
        assert record["latest_result"]["results"]["All customers"]["Primary"]
        assert record["config_payload"]["customer_summary_drafts"]["analyze"]["rows"]
        app.number_input(key="customer_summary_analyze_alpha").set_value(.1).run()
        app.button(key="customer_save_experiment").click().run()
        assert not app.exception
        stale = saved.call_args.args[0]
        assert stale["latest_result"] == {}
        assert stale["primary_direction"] == "No current result"
        assert stale["analysis_timestamp"] == ""


def test_summary_results_survive_switching_modes_without_becoming_raw_results():
    app = summary_app("analyze")
    app.button(key="customer_summary_analyze_run").click().run()
    stamp = app.session_state.customer_analysis_timestamp
    app.radio(key="customer_input_mode").set_value("Customer records").run()
    assert not app.session_state.customer_analysis_is_current
    app.radio(key="customer_input_mode").set_value("Arm summaries").run()
    assert not app.exception
    assert app.session_state.customer_analysis_is_current
    assert app.session_state.customer_analysis_timestamp == stamp


def test_saved_group_plan_routes_to_group_analysis():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.session_state.main_page = "Customer"
    app.session_state.customer_intent = "analyze"
    app.session_state.customer_input_mode = "Customer records"
    app.session_state.strategy_config.group_arms = {"Prime": {"control": 3000, "treatments": [5000]}}
    app.session_state.design_config.group_plan_rows = [
        {"Group": "Prime", "Arm": 3000, "Role": "Control", "Required Accounts": 100, "Traffic Allocation": .4},
        {"Group": "Prime", "Arm": 5000, "Role": "Treatment", "Required Accounts": 150, "Traffic Allocation": .6},
    ]
    app.run()
    assert not app.exception
    assert app.radio(key="group_analysis_source_v1").value == "Upload File"
    assert not any(tab.label == "2 · Recommendation" for tab in app.tabs)
