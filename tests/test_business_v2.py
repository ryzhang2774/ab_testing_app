from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from pypdf import PdfReader
from io import BytesIO

from src.experiment_platform.business_config import default_config, fingerprint, validate_config
from src.experiment_platform.business_demo import demo_config, demo_history, demo_records
from src.experiment_platform.business_data import read_table, validate_history, prepare_records, prepare_summaries
from src.experiment_platform.business_plan import build_plan, exploratory_protocol, variance_df
from src.experiment_platform.business_analysis import analyze, monitor, srm_check, balance_check
from src.experiment_platform.business_report import build_pdf
from src.experiment_platform.business_store import save_protocol, next_version, import_protocol


def simple_config():
    c=default_config()
    c.update(name="Simple test",owner="Growth",question="Does the new offer improve revenue?")
    c["metrics"][0].update(effect=5.,effect_scale="Absolute")
    return c


def simple_records():
    return pd.DataFrame({"ID":[f"{i:04}" for i in range(12)],"Arm":["Control"]*6+["Test A"]*6,
                         "Outcome":[80,90,95,105,110,120,90,100,104,116,120,130]})


@pytest.fixture(scope="module")
def demo_plan():
    return build_plan(demo_config(),demo_history())


def test_csv_and_excel_paste_preserve_ids_and_identical_historical_moments():
    h=demo_history()
    csv=validate_history(read_table(h.to_csv(index=False)),demo_config())
    paste=validate_history(read_table(h.to_csv(index=False,sep="\t")),demo_config())
    pd.testing.assert_frame_equal(csv,paste)
    assert read_table("ID,Arm,Outcome\n0001,Control,5\n").iloc[0].ID=="0001"


def test_partner_and_segment_are_independent_and_duplicates_rejected():
    h=validate_history(demo_history(),demo_config())
    assert h.partner.nunique()==h.segment.nunique()==2
    with pytest.raises(ValueError,match="Duplicate"):
        validate_history(pd.concat([h,h.iloc[:1]]),demo_config())
    with pytest.raises(ValueError,match="sample_variance"):
        validate_history(demo_history().assign(sample_variance=-2),demo_config())


def test_sd_conversion_and_missing_n_assumption_mode_are_explicit():
    c=simple_config()
    h=pd.DataFrame([dict(partner="A",metric="revenue",sample_mean=100,sample_sd=30,sample_number=50)])
    assert validate_history(h,c).sample_variance.iloc[0]==900
    with pytest.raises(ValueError,match="sample_number"):
        validate_history(h.drop(columns="sample_number"),c)
    c["assumption_only"]=True
    p=build_plan(c,h.drop(columns="sample_number"))
    assert "ASSUMPTION-ONLY" in p["limitations"][0]
    assert p["variance_assumptions"]


def test_binary_requires_real_success_counts_and_valid_observation_window():
    c=demo_config()
    h=demo_history()
    with pytest.raises(ValueError,match="successes"):
        validate_history(h.drop(columns="successes"),c)
    with pytest.raises(ValueError,match="disagrees"):
        validate_history(h.assign(sample_mean=.125),c)
    with pytest.raises(ValueError,match="windows"):
        validate_history(h.assign(window_days=14),c)


def test_shared_control_pairwise_max_and_final_power(demo_plan):
    p=demo_plan
    assert len(p["arm_totals"])==3 and len(p["targets"])==12
    control=next(r for r in p["arm_totals"] if r["arm"]=="Control")["analyzable_n"]
    assert control==max(r["pair_control_n"] for r in p["pairwise"])
    assert all(r["power"]>=p["config"]["power"] for r in p["comparisons"])
    assert p["family_size"]==4
    for _,g in pd.DataFrame(p["targets"]).groupby(["partner","segment"]):
        assert g.expected_share.sum()==pytest.approx(1.)
        assert g.analyzable_n.min()>=2


def test_segment_independent_plans_have_separate_power_and_protected_family():
    c=demo_config()
    c["decision_scope"]="Each segment"
    p=build_plan(c,demo_history())
    assert p["family_size"]==8
    assert {r["scope"] for r in p["comparisons"]}=={"Segment: New","Segment: Existing"}
    assert all(r["power"]>=.8 for r in p["comparisons"])


def test_business_fixed_ratios_are_frozen_without_replacing_them_with_rounding():
    c=simple_config()
    c.update(allocation="Business fixed",fixed_shares={"Control":1.,"Test A":3.})
    h=pd.DataFrame([dict(partner="A",metric="revenue",sample_mean=100,sample_variance=400,sample_number=100)])
    p=build_plan(c,h)
    assert {r["arm"]:r["expected_share"] for r in p["targets"]}=={"Control":.25,"Test A":.75}
    assert all(r["power"]>=.8 for r in p["comparisons"])


def test_minimal_three_column_readout_matches_scipy_welch_without_inventing_srm():
    c=simple_config()
    frame=simple_records()
    bundle=prepare_records(frame,c,"2026-10-26")
    p=exploratory_protocol(c,bundle)
    result=analyze(p,bundle)
    row=next(r for r in result["results"] if r["scope"]=="Overall")
    gold=stats.ttest_ind(frame[frame.Arm=="Test A"].Outcome,frame[frame.Arm=="Control"].Outcome,equal_var=False)
    assert row["effect"]==10
    assert row["raw_p_value"]==pytest.approx(gold.pvalue)
    assert row["degrees_of_freedom"]==pytest.approx(gold.df)
    assert result["srm"]["status"]==result["balance"]["status"]=="Not checked"
    assert result["progress"][0]["mature_n"] is None
    assert not row["confirmatory"]


def test_raw_and_same_definition_aggregate_summaries_produce_identical_results(demo_plan):
    c=demo_plan["config"]
    raw=prepare_records(demo_records(),c,"2026-10-26")
    summary=prepare_summaries(raw["summaries"],raw["counts"],raw["baselines"],c,raw["updated_at"])
    a,b=analyze(demo_plan,raw),analyze(demo_plan,summary)
    for r1,r2 in zip(a["results"],b["results"]):
        for key in ("effect","raw_p_value","ci_lower","ci_upper","adjusted_p_value"):
            assert r1.get(key)==pytest.approx(r2.get(key)) if r1.get(key) is not None else r2.get(key) is None
    assert a["progress"]==b["progress"]


def test_frozen_weighted_estimate_uses_within_stratum_variance_not_mixture_variance():
    c=simple_config()
    c["weights"]=[dict(partner="A",segment="S",weight=.25),dict(partner="B",segment="S",weight=.75)]
    h=pd.DataFrame([dict(partner=p,segment="S",metric="revenue",sample_mean=mean,sample_variance=100,sample_number=100) for p,mean in [("A",10),("B",1000)]])
    p=build_plan(c,h)
    rows=[]
    for partner,base,effect in [("A",10,2),("B",1000,6)]:
        for arm,n,mean,var in [("Control",100,base,100),("Test A",80,base+effect,144)]:
            rows.append(dict(partner=partner,segment="S",arm=arm,metric="revenue",sample_mean=mean,sample_variance=var,sample_number=n))
    s=prepare_summaries(pd.DataFrame(rows),pd.DataFrame(),pd.DataFrame(),c,"2026-10-26")
    r=next(r for r in analyze(p,s)["results"] if r["scope"]=="Overall")
    variance=(.25**2+.75**2)*(100/100+144/80)
    assert r["effect"]==5
    assert r["standard_error"]**2==pytest.approx(variance)
    assert r["control_mean"]==pytest.approx(752.5)
    assert max(x["variance_of_difference"] for x in p["comparisons"])<5


def test_missing_frozen_stratum_blocks_overall_instead_of_reweighting(demo_plan):
    d=demo_records()
    bundle=prepare_records(d[d.partner=="Partner A"],demo_plan["config"],"2026-10-26")
    overall=[r for r in analyze(demo_plan,bundle)["results"] if r["scope"]=="Overall"]
    assert all(r["status"]=="Insufficient data" and "not renormalized" in r["reason"] for r in overall)


def test_assigned_mature_observed_are_distinct_and_snapshots_do_not_accumulate():
    c=simple_config()
    r=simple_records().rename(columns={"ID":"id","Arm":"arm","Outcome":"revenue"})
    r["outcome_window_complete"]=[False,True,True,True,True,True]*2
    r.loc[1,"revenue"]=np.nan
    r["snapshot_date"]="2026-10-20"
    earlier=r.assign(snapshot_date="2026-10-19")
    bundle=prepare_records(pd.concat([earlier,r]),c,"2026-10-26")
    assert bundle["counts"].assigned_n.sum()==12
    assert bundle["counts"].mature_n.sum()==10
    assert bundle["summaries"].sample_number.sum()==9
    assert bundle["trends"].snapshot_date.nunique()==2
    with pytest.raises(ValueError,match="duplicate snapshots"):
        prepare_records(pd.concat([r,r]),c,"2026-10-26")
    with pytest.raises(ValueError,match="Duplicate ID"):
        prepare_records(r.drop(columns="snapshot_date").assign(id="same"),c,"2026-10-26")


def test_sparse_binary_uses_fisher_and_zero_variance_continuous_is_not_a_pass():
    c=simple_config()
    c["metrics"][0].update(type="Binary",effect_scale="Percentage points")
    r=simple_records().assign(Outcome=[0]*6+[0,0,0,1,1,0])
    bundle=prepare_records(r,c,"2026-10-26")
    result=analyze(exploratory_protocol(c,bundle),bundle)["results"][0]
    assert "Fisher exact" in result["method"]
    assert result["relative_lift"] is None
    assert result["ci_lower"]<result["ci_upper"]
    c["metrics"][0].update(type="Continuous",effect_scale="Absolute")
    bundle=prepare_records(r.assign(Outcome=1),c,"2026-10-26")
    assert analyze(exploratory_protocol(c,bundle),bundle)["results"][0]["status"]=="Insufficient data"


def test_srm_uses_unequal_frozen_ratios_and_stratum_specific_expectations():
    counts=pd.DataFrame([dict(partner="A",segment="S",arm=a,assigned_n=n) for a,n in [("Control",100),("Test A",300)]])
    targets=[dict(partner="A",segment="S",arm=a,expected_share=p) for a,p in [("Control",.25),("Test A",.75)]]
    r=srm_check(counts,targets,["Control","Test A"])
    assert r["status"]=="No mismatch detected" and r["rows"][0]["p_value"]==1
    flipped=counts.copy()
    flipped["assigned_n"]=[200,200]
    assert srm_check(flipped,targets,["Control","Test A"])["status"]=="Investigate"
    counts=pd.concat([counts,counts.assign(partner="B",assigned_n=[300,100])])
    targets += [dict(partner="B",segment="S",arm=a,expected_share=p) for a,p in [("Control",.75),("Test A",.25)]]
    r=srm_check(counts,targets,["Control","Test A"])
    assert "stratified" in r["rows"][0]["method"]
    assert r["rows"][0]["expected"]=={"Control":400.,"Test A":400.}
    assert r["status"]=="No mismatch detected"


def test_smd_is_baseline_only_handles_categorical_and_missing_covariates():
    c=demo_config()
    bundle=prepare_records(demo_records(),c,"2026-10-26")
    balance=balance_check(bundle["baselines"],c)
    assert {r["feature"] for r in balance["rows"]}=={"baseline_age","baseline_channel"}
    assert all(np.isfinite(r["smd"]) for r in balance["rows"])
    c["baseline_types"]={"revenue":"Continuous"}
    with pytest.raises(ValueError,match="cannot be used as baseline"):
        prepare_records(demo_records(),c,"2026-10-26")
    assert balance_check(pd.DataFrame(),c)["status"]=="Not checked"


def test_append_only_versions_import_recomputes_and_report_matches_snapshot(demo_plan,tmp_path):
    p=deepcopy(demo_plan)
    p["config"]["name"]="合作伙伴实验"
    p["config_fingerprint"]=fingerprint(p["config"])
    path=save_protocol(p,tmp_path)
    assert next_version(p["config"]["experiment_id"],tmp_path)==2
    with pytest.raises(FileExistsError):
        save_protocol(p,tmp_path)
    assert import_protocol(path.read_text())["targets"]==p["targets"]
    changed=deepcopy(p)
    changed["targets"][0]["recruitment_n"]+=10
    with pytest.raises(ValueError,match="disagree"):
        import_protocol(json.dumps(changed))
    bundle=prepare_records(demo_records(),p["config"],"2026-10-26")
    snapshot=analyze(p,bundle)
    stamp=snapshot["updated_at"]
    pdf=build_pdf(p,snapshot,"Analyze results")
    reader=PdfReader(BytesIO(pdf))
    text="\n".join(page.extract_text() for page in reader.pages)
    assert "合作伙伴实验" in text
    assert str(p["arm_totals"][0]["analyzable_n"]) in text
    assert snapshot["data_fingerprint"] in text.replace("\n","")
    assert len(reader.pages)>2 and snapshot["updated_at"]==stamp
    assert "DEMO-A" not in text


def test_guardrail_excluded_from_sizing_stays_in_confirmatory_family():
    c=demo_config()
    c["metrics"][1]["include_in_plan"]=False
    p=build_plan(c,demo_history())
    assert p["family_size"]==4
    assert {r["metric"] for r in p["comparisons"]}=={"revenue"}
    b=prepare_records(demo_records(),c,"2026-10-26")
    result=analyze(p,b)
    guard=[r for r in result["results"] if r["scope"]=="Overall" and r["role"]=="Guardrail"]
    assert all(r["confirmatory"] for r in guard)
    assert all(r["ci_level"]==pytest.approx(.9875) for r in guard)


def test_maturity_uses_source_date_and_cannot_be_advanced_by_download_date():
    c=simple_config()
    r=simple_records().assign(assignment_date="2026-09-25",snapshot_date="2026-10-20")
    b=prepare_records(r,c,"2026-10-26")
    assert b["counts"].mature_n.sum()==0
    assert b["summaries"].sample_number.sum()==0
    with pytest.raises(ValueError,match="completed-window flag"):
        prepare_records(r.assign(outcome_window_complete=True),c,"2026-10-26")
