from streamlit.testing.v1 import AppTest
from datetime import date


def start_demo():
    a=AppTest.from_file("app.py",default_timeout=60).run()
    a.button(key="home_choose_customer").click().run()
    assert not a.exception
    a.button(key="bv2_demo").click().run()
    return a


def test_business_four_stage_workflow_and_stale_snapshot(tmp_path,monkeypatch):
    monkeypatch.setenv("CUSTOMER_PROTOCOL_DIR",str(tmp_path))
    a=start_demo()
    a.button(key="FormSubmitter:bv2_setup_form-Save setup & continue").click().run()
    assert not a.exception
    assert a.radio(key="bv2_step").value=="Build test plan"
    a.checkbox(key="bv2_accept_weights").check().run()
    a.button(key="bv2_calculate").click().run()
    assert not a.exception
    assert a.session_state.business_v2["candidate"]["targets"]
    a.button(key="bv2_freeze").click().run()
    assert not a.exception
    assert a.radio(key="bv2_step").value=="Monitor experiment"
    a.button(key="bv2_validate_records").click().run()
    assert not a.exception
    assert a.session_state.business_v2["bundle"]["counts"].assigned_n.sum()==1600
    monitor_time=a.session_state.business_v2["snapshot"]["calculated_at"]
    a.button(key="bv2_pdf_prepare_monitor").click().run()
    assert not a.exception
    assert a.session_state.business_v2["snapshot"]["calculated_at"]==monitor_time
    a.button(key="bv2_continue_analysis").click().run()
    a.button(key="bv2_analyze").click().run()
    assert not a.exception
    snapshot=a.session_state.business_v2["snapshot"]
    assert snapshot["results"] and snapshot["blockers"]
    a.button(key="bv2_pdf_prepare_analysis").click().run()
    assert not a.exception
    assert a.session_state.bv2_pdf["bytes"].startswith(b"%PDF")
    a.date_input(key="bv2_data_asof").set_value(date(2026,10,25)).run()
    assert not a.exception
    assert a.session_state.business_v2["snapshot"] is None
    assert a.session_state.business_v2["bundle"] is None


def test_editing_setup_does_not_overwrite_frozen_protocol(tmp_path,monkeypatch):
    monkeypatch.setenv("CUSTOMER_PROTOCOL_DIR",str(tmp_path))
    a=start_demo()
    a.radio(key="bv2_step").set_value("Build test plan").run()
    a.checkbox(key="bv2_accept_weights").check().run()
    a.button(key="bv2_calculate").click().run()
    a.button(key="bv2_freeze").click().run()
    original=a.session_state.business_v2["plan"]["config_fingerprint"]
    a.radio(key="bv2_step").set_value("Set up experiment").run()
    a.text_input(key="bv2_owner").set_value("New owner")
    a.button(key="FormSubmitter:bv2_setup_form-Save setup & continue").click().run()
    a.radio(key="bv2_step").set_value("Monitor experiment").run()
    assert not a.exception
    assert a.session_state.business_v2["plan"]["config_fingerprint"]==original
    assert any("Setup changed" in x.value for x in a.warning)


def test_frozen_weights_survive_navigation_and_population_edits_invalidate(tmp_path,monkeypatch):
    monkeypatch.setenv("CUSTOMER_PROTOCOL_DIR",str(tmp_path))
    a=start_demo()
    a.session_state.business_v2["config"]["weights"]=[
        {"partner":"Partner A","segment":"New","weight":.1},
        {"partner":"Partner A","segment":"Existing","weight":.2},
        {"partner":"Partner B","segment":"New","weight":.3},
        {"partner":"Partner B","segment":"Existing","weight":.4}]
    a.radio(key="bv2_step").set_value("Build test plan").run()
    a.checkbox(key="bv2_accept_weights").check().run()
    a.button(key="bv2_calculate").click().run()
    a.button(key="bv2_freeze").click().run()
    frozen=a.session_state.business_v2["plan"]["weights"]
    a.radio(key="bv2_step").set_value("Build test plan").run()
    assert not a.exception
    assert not a.session_state.business_v2.get("draft_dirty")
    assert a.session_state.business_v2["config"]["weights"]==frozen
    a.multiselect(key="bv2_history_partners").set_value(["Partner B"]).run()
    assert not a.exception
    assert a.session_state.business_v2["draft_dirty"]
    assert a.session_state.business_v2["plan"]["weights"]==frozen
    a.radio(key="bv2_step").set_value("Monitor experiment").run()
    assert any("Setup changed" in x.value for x in a.warning)


def test_edited_arm_definitions_and_metric_changes_survive_returning_to_setup():
    a=start_demo()
    a.session_state["bv2_arms"]={"edited_rows":{1:{"definition":"Revised business option"}},"added_rows":[],"deleted_rows":[]}
    a.session_state["bv2_metrics"]={"edited_rows":{0:{"effect":7.}},"added_rows":[],"deleted_rows":[]}
    a.button(key="FormSubmitter:bv2_setup_form-Save setup & continue").click().run()
    a.radio(key="bv2_step").set_value("Set up experiment").run()
    assert not a.exception
    arms=next(d.value for d in a.dataframe if "definition" in d.value and "role" in d.value and "effect" not in d.value)
    metrics=next(d.value for d in a.dataframe if "effect" in d.value)
    assert arms.iloc[1]["definition"]=="Revised business option"
    assert metrics.iloc[0]["effect"]==7.
