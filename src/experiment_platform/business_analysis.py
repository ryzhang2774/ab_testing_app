"""Monitoring and fixed-horizon inference from the same independent-unit summaries."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep

from .business_config import control_name, effect_size, fingerprint, utc_now
from .business_plan import clean_records, planned_metrics, scopes, variance_df


def srm_check(counts: pd.DataFrame, targets: list[dict], arms: list[str], threshold=.001) -> dict:
    """Expected counts respect each stratum's frozen allocation, including unequal ratios."""
    if counts.empty or not targets:
        return {"status": "Not checked", "reason": "Both Assigned N and frozen expected allocation are required.", "rows": []}
    target = pd.DataFrame(targets)
    available = set(zip(target.partner, target.segment))
    if not set(zip(counts.partner, counts.segment)).issubset(available):
        return {"status": "Not checked", "reason": "Uploaded populations are outside the frozen plan.", "rows": []}
    all_observed, all_expected = np.zeros(len(arms)), np.zeros(len(arms))
    rows = []
    strata = list(counts.groupby(["partner", "segment"], sort=False))
    local_threshold = threshold / (len(strata) + 1)

    def check(observed, expected, name):
        if sum(observed) == 0:
            return {"scope": name, "status": "Insufficient data", "method": "Not calculated", "p_value": None}
        if min(expected) < 5:
            if len(arms) == 2:
                p = float(stats.binomtest(int(observed[0]), int(sum(observed)), expected[0]/sum(expected)).pvalue)
                method = "Exact binomial"
            else:
                return {"scope": name, "status": "Insufficient data", "method": "Sparse multi-arm counts; exact multinomial not implemented", "p_value": None}
        else:
            p = float(stats.chisquare(observed, expected).pvalue)
            method = "Pearson goodness-of-fit"
        return {"scope": name, "status": "Investigate" if p < local_threshold else "No mismatch detected",
                "method": method, "p_value": p, "threshold": local_threshold,
                "observed": dict(zip(arms, map(int, observed))), "expected": dict(zip(arms, map(float, expected)))}

    for (partner, segment), group in strata:
        tg = target[target.partner.eq(partner) & target.segment.eq(segment)].set_index("arm")
        if set(tg.index) != set(arms):
            return {"status": "Not checked", "reason": "The frozen plan does not contain every arm in every selected stratum.", "rows": []}
        observed = np.array([group.loc[group.arm.eq(a), "assigned_n"].sum() for a in arms], dtype=float)
        expected = observed.sum() * np.array([tg.loc[a, "expected_share"] for a in arms])
        all_observed += observed
        all_expected += expected
        rows.append(check(observed, expected, f"{partner} / {segment}"))
    # Conditioning on enrolled stratum totals does not produce a multinomial
    # when allocation probabilities differ by stratum. Calibrate that case
    # against its actual sum-of-multinomials distribution.
    shares = []
    for (partner, segment), group in strata:
        tg = target[target.partner.eq(partner) & target.segment.eq(segment)].set_index("arm")
        shares.append((int(group.assigned_n.sum()), np.array([tg.loc[a,"expected_share"] for a in arms], dtype=float)))
    if len(shares) > 1 and any(not np.allclose(p,shares[0][1]) for _,p in shares[1:]) and all_expected.sum() > 0:
        rng = np.random.default_rng(142857)
        simulations = np.zeros((49999,len(arms)))
        for n,p in shares:
            simulations += rng.multinomial(n,p,size=len(simulations))
        statistic = float(np.sum((all_observed-all_expected)**2/all_expected))
        simulated = np.sum((simulations-all_expected)**2/all_expected,axis=1)
        pvalue = float((1+np.sum(simulated >= statistic-1e-12))/(len(simulations)+1))
        insufficient=pvalue<=1/len(simulations) and local_threshold<1/(len(simulations)+1)
        overall = {"scope":"Overall","status":"Insufficient data" if insufficient else "Investigate" if pvalue < local_threshold else "No mismatch detected",
                   "method":"Conditional stratified multinomial Monte Carlo (49,999 draws; seed 142857)","p_value":pvalue,"threshold":local_threshold,
                   "observed":dict(zip(arms,map(int,all_observed))),"expected":dict(zip(arms,map(float,all_expected)))}
        if insufficient:
            overall["note"]="Monte Carlo resolution is insufficient at this family-adjusted diagnostic threshold; investigate the extreme mismatch with a higher-precision procedure."
    else:
        overall = check(all_observed,all_expected,"Overall")
    rows.insert(0, overall)
    statuses = {r["status"] for r in rows}
    status = "Investigate" if "Investigate" in statuses else "Insufficient data" if "Insufficient data" in statuses else "No mismatch detected"
    return {"status": status, "rows": rows, "reason": "Assigned N versus frozen, stratum-specific allocation; diagnostic family threshold divided across Overall and observed strata."}


def balance_check(baselines: pd.DataFrame, config: dict) -> dict:
    if baselines.empty:
        return {"status": "Not checked", "rows": [], "reason": "No pre-treatment covariates were provided."}
    control = control_name(config)
    rows = []
    for key, group in baselines.groupby(["partner", "segment", "feature", "category"], sort=False, dropna=False):
        cb = group[group.arm.eq(control)]
        for arm in [a["name"] for a in config["arms"] if a["name"] != control]:
            tb = group[group.arm.eq(arm)]
            row = dict(zip(["partner", "segment", "feature", "category"], key)) | {"test": arm, "control": control, "smd": None, "status": "Insufficient data"}
            if len(cb) == 1 and len(tb) == 1:
                c, t = cb.iloc[0], tb.iloc[0]
                row.update({"control_mean": c.sample_mean, "test_mean": t.sample_mean,
                            "control_missing_rate": c.missing_rate, "test_missing_rate": t.missing_rate})
                if min(c.sample_number, t.sample_number) >= 2 and pd.notna(c.sample_variance) and pd.notna(t.sample_variance):
                    # Categorical levels use Bernoulli population variance, continuous uses unbiased sample variance.
                    vc = c.sample_mean*(1-c.sample_mean) if c.kind == "Categorical" else c.sample_variance
                    vt = t.sample_mean*(1-t.sample_mean) if t.kind == "Categorical" else t.sample_variance
                    denominator = math.sqrt((vc+vt)/2)
                    difference = float(t.sample_mean-c.sample_mean)
                    if denominator > 0:
                        smd = difference/denominator
                        row.update(smd=smd, status="Review balance" if abs(smd) > config["smd_threshold"] else "Within reference line")
                    elif difference == 0:
                        row.update(smd=0., status="Within reference line", note="Both groups constant and identical.")
                    else:
                        row.update(status="Review balance", note="Zero denominator with different constants; SMD is undefined.")
            rows.append(row)
    status = "Review balance" if any(r["status"] == "Review balance" for r in rows) else "Insufficient data" if any(r["status"] == "Insufficient data" for r in rows) else "Within reference line"
    return {"status": status, "rows": clean_records(pd.DataFrame(rows)),
            "reason": "SMD = (Test mean - Control mean) / sqrt((Control variance + Test variance)/2). Categorical levels are indicator proportions. The reference line is diagnostic, not proof of randomization failure. Checks are within Partner × Segment."}


def monitor(plan: dict, bundle: dict) -> dict:
    config = plan["config"]
    counts, summaries = bundle["counts"], bundle["summaries"]
    arms = [a["name"] for a in config["arms"]]
    progress = []
    primary = next(m["name"] for m in config["metrics"] if m["role"] == "Primary")
    for arm in arms:
        ac = counts[counts.arm.eq(arm)] if not counts.empty else counts
        outcome = summaries[summaries.arm.eq(arm) & summaries.metric.eq(primary)]
        target = [r for r in plan.get("targets", []) if r["arm"] == arm]
        assigned = int(ac.assigned_n.sum()) if not ac.empty else None
        mature = int(ac.mature_n.sum()) if not ac.empty and ac.mature_n.notna().all() else None
        observed = int(outcome.sample_number.sum()) if not outcome.empty else 0
        progress.append({"arm": arm, "assigned_n": assigned, "mature_n": mature, "observed_n": observed,
                         "recruitment_target": sum(t["recruitment_n"] for t in target) if target else None,
                         "analyzable_target": sum(t["analyzable_n"] for t in target) if target else None,
                         "missing_result_rate": 1-observed/mature if mature else None})
    srm = srm_check(counts, plan.get("targets", []), arms, config["srm_alpha"])
    balance = balance_check(bundle["baselines"], config)
    cell_progress = []
    for target in plan.get("targets",[]):
        for metric in planned_metrics(config):
            cells = summaries[summaries.partner.eq(target["partner"]) & summaries.segment.eq(target["segment"]) & summaries.arm.eq(target["arm"]) & summaries.metric.eq(metric["name"])]
            n = int(cells.sample_number.sum()) if not cells.empty else 0
            cell_progress.append({**target,"metric":metric["name"],"observed_n":n,"target_met":n >= target["analyzable_n"]})
    guardrails = []
    control = control_name(config)
    for scope in scopes(config,plan["weights"]):
        if not scope["confirmatory"]:
            continue
        for metric in config["metrics"]:
            if metric["role"] != "Guardrail":
                continue
            for arm in arms:
                if arm == control:
                    continue
                differences = []
                for w in scope["weights"]:
                    cell = summaries[summaries.partner.eq(w["partner"]) & summaries.segment.eq(w["segment"]) & summaries.metric.eq(metric["name"])]
                    cb,tb = cell[cell.arm.eq(control)],cell[cell.arm.eq(arm)]
                    if cb.empty or tb.empty or cb.sample_mean.isna().any() or tb.sample_mean.isna().any():
                        break
                    differences.append(w["weight"]*float(tb.iloc[0].sample_mean-cb.iloc[0].sample_mean))
                valid = len(differences)==len(scope["weights"])
                effect = sum(differences) if valid else None
                deterioration = (-1 if metric["direction"]=="Higher" else 1)*effect if valid else None
                guardrails.append({"scope":scope["name"],"metric":metric["name"],"test":arm,"effect":effect,
                                   "alert_threshold":metric["alert_threshold"],"status":"Not checked" if not valid else "Operational alert" if metric["alert_threshold"]>0 and deterioration>metric["alert_threshold"] else "No point-estimate alert",
                                   "note":"Descriptive business alert; not a final safety or non-inferiority conclusion."})
    return {"stage": "Monitor experiment", "calculated_at": utc_now(), "updated_at": bundle["updated_at"],
            "data_fingerprint": bundle["fingerprint"], "config_fingerprint": plan["config_fingerprint"],
            "progress": progress, "srm": srm, "balance": balance, "source": bundle["source"],
            "limitations": bundle["limitations"], "maturity_known": bundle["maturity_known"],
            "summaries": clean_records(summaries), "cell_progress":cell_progress, "guardrails":guardrails,
            "trends": clean_records(bundle.get("trends",pd.DataFrame()))}


def _compare(summary: pd.DataFrame, scope: dict, metric: dict, arm: str, config: dict, interval_alpha: float) -> dict:
    control = control_name(config)
    row = {"scope": scope["name"], "scope_kind": scope["kind"], "metric": metric["name"], "role": metric["role"],
           "test": arm, "control": control, "confirmatory": scope["confirmatory"] and (metric["role"] in {"Primary","Guardrail"} or metric["include_in_plan"]),
           "status": "Insufficient data", "method": "Not calculated", "raw_p_value": None, "adjusted_p_value": None,
           "ci_lower": None, "ci_upper": None, "ci_level": 1-interval_alpha, "guardrail_status": "Not applicable"}
    by = summary[summary.metric.eq(metric["name"])].set_index(["partner", "segment", "arm"])
    cells, terms, ns = [], [], []
    mc = mt = 0.
    nc_total = nt_total = 0
    missing = []
    for w in scope["weights"]:
        ck, tk = (w["partner"], w["segment"], control), (w["partner"], w["segment"], arm)
        if ck not in by.index or tk not in by.index:
            missing.append(f"{w['partner']} / {w['segment']}")
            continue
        cb, tb = by.loc[ck], by.loc[tk]
        if min(cb.sample_number, tb.sample_number) < 2 or pd.isna(cb.sample_variance) or pd.isna(tb.sample_variance):
            missing.append(f"{w['partner']} / {w['segment']}")
            continue
        nc, nt = int(cb.sample_number), int(tb.sample_number)
        terms += [w["weight"]**2 * cb.sample_variance/nc, w["weight"]**2 * tb.sample_variance/nt]
        ns += [nc, nt]
        mc += w["weight"]*cb.sample_mean
        mt += w["weight"]*tb.sample_mean
        nc_total += nc
        nt_total += nt
        cells.append((w, cb, tb))
    if missing:
        row["reason"] = "Missing or insufficient frozen strata: " + "; ".join(missing) + ". Weights were not renormalized."
        return row
    effect = float(mt-mc)
    variance, df = variance_df(terms, ns)
    row.update(control_n=nc_total, test_n=nt_total, control_mean=float(mc), test_mean=float(mt), effect=effect,
               relative_lift=effect/mc if mc != 0 else None, percentage_points=effect*100 if metric["type"] == "Binary" else None,
               control_within_variance=sum(w["weight"]*cb.sample_variance for w,cb,tb in cells),
               test_within_variance=sum(w["weight"]*tb.sample_variance for w,cb,tb in cells))
    row["control_within_sd"]=math.sqrt(row["control_within_variance"])
    row["test_within_sd"]=math.sqrt(row["test_within_variance"])
    if metric["type"] == "Binary":
        sparse = False
        null_var = 0.
        alt_var = 0.
        lower = upper = 0.
        # Bonferroni over strata + comparison family gives a conservative weighted CI.
        for w, cb, tb in cells:
            nc, nt, xc, xt = int(cb.sample_number), int(tb.sample_number), int(cb.successes), int(tb.successes)
            pooled = (xc+xt)/(nc+nt)
            expected = [nc*pooled, nc*(1-pooled), nt*pooled, nt*(1-pooled)]
            sparse |= min(expected) < 5
            null_var += w["weight"]**2 * pooled*(1-pooled)*(1/nc+1/nt)
            alt_var += w["weight"]**2 * (cb.sample_mean*(1-cb.sample_mean)/nc+tb.sample_mean*(1-tb.sample_mean)/nt)
            lo, hi = confint_proportions_2indep(xt, nt, xc, nc, method="newcomb", compare="diff", alpha=interval_alpha/len(cells))
            lower += w["weight"]*lo
            upper += w["weight"]*hi
        row.update(ci_lower=float(lower), ci_upper=float(upper), ci_method="Newcombe-Wilson; Bonferroni over strata and confirmatory comparisons", standard_error=math.sqrt(alt_var), degrees_of_freedom=None)
        if not sparse and alt_var > 0:
            # Same fixed-weight risk-difference normal model as the planner.
            # The small-cell fallback below retains conservative cell intervals.
            critical=stats.norm.ppf(1-interval_alpha/2)
            row.update(ci_lower=max(-1.,effect-critical*math.sqrt(alt_var)),
                       ci_upper=min(1.,effect+critical*math.sqrt(alt_var)),
                       ci_method="Fixed-weight risk-difference normal interval; Bonferroni comparison-family correction" if config["multiplicity"] else "Fixed-weight risk-difference normal interval")
        if sparse:
            if len(cells) == 1:
                _, cb, tb = cells[0]
                result = stats.fisher_exact([[int(tb.successes), int(tb.sample_number-tb.successes)], [int(cb.successes), int(cb.sample_number-cb.successes)]])
                p = float(result.pvalue)
                statistic = None
                row["method"] = "Fisher exact (two-sided); Newcombe interval is a separate approximate procedure"
            else:
                row["reason"] = "Sparse Binary strata: weighted normal inference is unreliable. Exact per-stratum results are available in the appendix; no overall significance claim."
                row["method"] = "Weighted risk difference with conservative interval; overall test not calculated"
                row["cell_exact_results"] = [dict(partner=w["partner"], segment=w["segment"], p_value=float(stats.fisher_exact([[int(tb.successes), int(tb.sample_number-tb.successes)], [int(cb.successes), int(cb.sample_number-cb.successes)]]).pvalue)) for w,cb,tb in cells]
                return row
        elif null_var > 0:
            statistic = effect/math.sqrt(null_var)
            p = float(2*stats.norm.sf(abs(statistic)))
            row["method"] = "Two-proportion z-test" if len(cells) == 1 else "Fixed-weight stratified two-proportion z-test"
        else:
            row["reason"] = "No estimable Binary variation."
            return row
        se, distribution = math.sqrt(alt_var), stats.norm
    else:
        if variance <= 0:
            row["reason"] = "Both groups have zero estimable variance; uncertainty cannot be estimated."
            return row
        se = math.sqrt(variance)
        statistic = effect/se
        p = float(2*stats.t.sf(abs(statistic), df))
        critical = stats.t.ppf(1-interval_alpha/2, df)
        row.update(ci_lower=effect-critical*se, ci_upper=effect+critical*se,
                   standard_error=se, degrees_of_freedom=df if math.isfinite(df) else None,
                   ci_method="Welch-Satterthwaite; Bonferroni family interval" if config["multiplicity"] else "Welch-Satterthwaite unadjusted interval")
        row["method"] = "Welch unequal-variance t-test" if len(cells) == 1 else "Fixed-weight stratified Welch t-test"
        distribution = stats.t(df)
    row.update(status="Calculated", raw_p_value=p, statistic=statistic, efficacy_p_value=p)
    sign = 1 if metric["direction"] == "Higher" else -1
    benefit_low, benefit_high = sorted([sign*row["ci_lower"], sign*row["ci_upper"]])
    if metric["role"] == "Guardrail":
        margin = metric["margin"]
        # This is a non-inferiority hypothesis, separate from the two-sided efficacy p-value.
        if metric["type"] == "Binary":
            # Wald non-inferiority inference is available only for adequate cells.
            row["raw_p_value"] = float(stats.norm.sf((sign*effect+margin)/se)) if not sparse and se>0 else None
            row["guardrail_status"] = "Within tolerance" if benefit_low > -margin else "Beyond tolerance" if benefit_high < -margin else "Inconclusive"
            row["guardrail_method"] = "Fixed-weight normal non-inferiority test and family interval versus an absolute-rate margin" if not sparse else "Sparse-cell Newcombe interval versus an absolute-rate margin; no non-inferiority p-value asserted"
        else:
            row["raw_p_value"] = float(distribution.sf((sign*effect+margin)/se))
            row["guardrail_status"] = "Within tolerance" if benefit_low > -margin else "Beyond tolerance" if benefit_high < -margin else "Inconclusive"
            row["guardrail_method"] = "One-sided Welch non-inferiority p-value; decision uses conservative two-sided family interval"
        row["operational_alert"] = -sign*effect > metric["alert_threshold"] if metric["alert_threshold"] > 0 else None
        row["margin"] = margin
    return row


def analyze(plan: dict, bundle: dict) -> dict:
    config = plan["config"]
    family = plan["family_size"]
    all_scopes = scopes(config,plan["weights"],include_exploratory=True)
    total_rows = len(all_scopes)*len(config["metrics"])*(len(config["arms"])-1)
    exploratory_family = max(total_rows-family,1)
    rows = []
    for scope in all_scopes:
        for metric in config["metrics"]:
            for a in config["arms"]:
                if a["role"] == "Control":
                    continue
                confirmatory = scope["confirmatory"] and (metric["role"] in {"Primary","Guardrail"} or metric["include_in_plan"]) and not plan.get("retrospective")
                interval_family = family if confirmatory else (total_rows if plan.get("retrospective") else exploratory_family)
                interval_alpha = config["alpha"]/interval_family if config["multiplicity"] else config["alpha"]
                row = _compare(bundle["summaries"], scope, metric, a["name"], config, interval_alpha)
                if plan.get("retrospective"):
                    row["confirmatory"] = False
                rows.append(row)
    # Missing tests retain a p-value of 1 in the predeclared family, so exclusions cannot reduce protection.
    confirm = [r for r in rows if r["confirmatory"]]
    values = [r["raw_p_value"] if r["raw_p_value"] is not None else 1. for r in confirm]
    if values:
        adjusted = multipletests(values, alpha=config["alpha"], method="holm")[1] if config["multiplicity"] else values
        for row, p in zip(confirm, adjusted):
            row["adjusted_p_value"] = float(p) if row["raw_p_value"] is not None else None
    exploratory = [r for r in rows if not r["confirmatory"] and r["raw_p_value"] is not None]
    if exploratory:
        adjusted = multipletests([r["raw_p_value"] for r in exploratory], method="holm")[1] if config["multiplicity"] else [r["raw_p_value"] for r in exploratory]
        for row, p in zip(exploratory, adjusted):
            row["adjusted_p_value"] = float(p)
    health = monitor(plan, bundle)
    blockers = []
    if health["srm"]["status"] != "No mismatch detected":
        blockers.append("Resolve SRM or supply frozen allocation / assignment counts before trusting an experimental decision.")
    if not bundle["maturity_known"]:
        blockers.append("Confirm the observation window is complete.")
    if any(r["mature_n"] is not None and r["assigned_n"] != r["mature_n"] for r in health["progress"]):
        blockers.append("Some assigned customers are not yet mature; this is an interim readout.")
    if any(r["analyzable_target"] is not None and r["observed_n"] < r["analyzable_target"] for r in health["progress"]):
        blockers.append("Observed sample is below the frozen target; do not claim the fixed-horizon experiment is complete.")
    if any(not r["target_met"] for r in health["cell_progress"]):
        blockers.append("Some Partner × Segment × Arm cells have not reached their analyzable target.")
    if plan.get("retrospective"):
        blockers.append("No prospective protocol was supplied. This is an exploratory readout, with no SRM or planned-power guarantee.")
    if any(r["missing_result_rate"] is not None and r["missing_result_rate"] > 0 for r in health["progress"]):
        blockers.append("Mature customers have missing results. Investigate differential missingness before a causal decision.")
    if health["balance"]["status"] == "Review balance":
        blockers.append("Review the pre-treatment balance flags.")
    for r in rows:
        if r["status"] != "Calculated":
            r["interpretation"] = r.get("reason", "Insufficient data")
        elif r["role"] == "Guardrail":
            r["interpretation"] = {"Within tolerance": "The interval supports no deterioration beyond the saved tolerance.", "Beyond tolerance": "The interval supports deterioration beyond the saved tolerance.", "Inconclusive": "The data do not yet rule out unacceptable deterioration."}[r["guardrail_status"]]
        else:
            m = next(m for m in config["metrics"] if m["name"] == r["metric"])
            sign = 1 if m["direction"] == "Higher" else -1
            benefit_low, benefit_high = sorted([sign*r["ci_lower"], sign*r["ci_upper"]])
            r["interpretation"] = "Evidence of improvement." if benefit_low > 0 else "Evidence of deterioration." if benefit_high < 0 else "The interval includes no change; useful gains or losses may still be plausible."
            try:
                r["worthwhile_threshold"]=effect_size(m,r["control_mean"])
                if benefit_low>0 and sign*r["effect"]<r["worthwhile_threshold"]:
                    r["interpretation"]+=" The estimated gain is below the saved worthwhile-change threshold."
            except ValueError:
                r["worthwhile_threshold"]=None
        if not r["confirmatory"]:
            r["interpretation"] += " Exploratory: no independent power claim."
    guardrail_summary=[{"scope":r["scope"],"metric":r["metric"],"test":r["test"],"status":r["guardrail_status"] if r["status"]=="Calculated" else "Insufficient data","confirmatory":r["confirmatory"]} for r in rows if r["role"]=="Guardrail"]
    if any(r["confirmatory"] and r["status"]!="Within tolerance" for r in guardrail_summary):
        blockers.append("At least one planned Guardrail has not ruled out deterioration beyond its tolerance.")
    return {**health, "stage": "Analyze results", "results": rows, "blockers": blockers, "guardrail_summary":guardrail_summary,
            "decision_status": "Review required" if blockers else "Evaluate primary effect and all guardrails",
            "multiplicity": f"Holm across {family} predeclared comparisons; conservative Bonferroni family intervals. Exploratory results form a separate Holm family." if config["multiplicity"] else "No multiplicity protection (advanced override).",
            "limitations": bundle["limitations"] + ["A significant result in one segment and a non-significant result in another is not evidence of interaction.", "No automatic winner selection or early efficacy stopping."]}
