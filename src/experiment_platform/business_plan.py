"""Fixed-weight stratified planning, pairwise Neyman with a shared control.

The target estimand is sum(w_h * mean_h). Its sampling variance is
sum(w_h**2 * s_h**2 / n_h), without a between-stratum variance term.
"""
from __future__ import annotations

from datetime import date, timedelta
from importlib.metadata import version as package_version
import math
import platform
import numpy as np
import pandas as pd
from scipy import stats

from .business_config import control_name, effect_size, fingerprint, utc_now, validate_config
from .business_data import validate_history

LIMIT = 20_000_000


def clean_records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def planned_metrics(config: dict) -> list[dict]:
    return [m for m in config["metrics"] if m["role"] == "Primary" or m["include_in_plan"]]


def confirmatory_metrics(config:dict) -> list[dict]:
    return [m for m in config["metrics"] if m["role"] in {"Primary","Guardrail"} or m["include_in_plan"]]


def population_weights(history: pd.DataFrame, config: dict) -> list[dict]:
    primary = next(m["name"] for m in config["metrics"] if m["role"] == "Primary")
    base = history[(history.arm == control_name(config)) & history.metric.eq(primary)]
    if base.empty:
        raise ValueError("Every selected population needs Control history for the Primary metric.")
    available = set(zip(base.partner, base.segment))
    supplied = config.get("weights")
    if supplied:
        rows = [{"partner": str(r["partner"]), "segment": str(r["segment"]), "weight": float(r["weight"])} for r in supplied]
        if len({(r["partner"], r["segment"]) for r in rows}) != len(rows):
            raise ValueError("Population weights contain duplicate Partner × Segment rows.")
        if not rows or any(not math.isfinite(r["weight"]) or r["weight"] <= 0 or (r["partner"], r["segment"]) not in available for r in rows):
            raise ValueError("Every selected Partner × Segment needs a positive weight and Control history.")
    else:
        values = base.sample_number.fillna(1).to_numpy(dtype=float)
        rows = [{"partner": r.partner, "segment": r.segment, "weight": float(n)} for r, n in zip(base.itertuples(), values)]
    total = sum(r["weight"] for r in rows)
    divisor=1. if math.isclose(total,1.,rel_tol=0,abs_tol=1e-12) else total
    return [{**r, "weight": r["weight"] / divisor} for r in rows]


def scopes(config: dict, weights: list[dict], include_exploratory=False) -> list[dict]:
    output = [{"name": "Overall", "kind": "Overall", "confirmatory": config["decision_scope"] == "Overall", "weights": weights}]
    dimensions = ["segment"] if config["decision_scope"] == "Each segment" else ["partner"] if config["decision_scope"] == "Each partner" else []
    if include_exploratory:
        dimensions = list(dict.fromkeys(dimensions + ["segment", "partner"]))
    for dim in dimensions:
        labels=list(dict.fromkeys(r[dim] for r in weights))
        if len(labels)<2 and config["decision_scope"]!=f"Each {dim}":
            continue  # A single category would duplicate the Overall test.
        for label in labels:
            subset = [r for r in weights if r[dim] == label]
            total = sum(r["weight"] for r in subset)
            output.append({"name": f"{dim.title()}: {label}", "kind": dim.title(),
                           "confirmatory": config["decision_scope"] == f"Each {dim}",
                           "weights": [{**r, "weight": r["weight"] / total} for r in subset]})
    return output


def split_integer(total: int, weights: list[dict], minimum: int = 2) -> list[int]:
    total = max(int(total), minimum * len(weights))
    raw = np.array([r["weight"] for r in weights], dtype=float) * (total - minimum * len(weights))
    counts = np.floor(raw).astype(int) + minimum
    extra = total - int(counts.sum())
    for i in np.argsort(-(raw - np.floor(raw)), kind="stable")[:extra]:
        counts[i] += 1
    return counts.tolist()


def variance_df(parts: list[float], ns: list[int]) -> tuple[float, float]:
    variance = float(sum(parts))
    denom = sum(v * v / (n - 1) for v, n in zip(parts, ns) if n > 1)
    return variance, variance ** 2 / denom if denom > 0 else math.inf


def build_plan(config: dict, raw_history: pd.DataFrame, version: int = 1, code_version: str = "unavailable") -> dict:
    c = validate_config(config)
    history = validate_history(raw_history, c)
    weights = population_weights(history, c)
    if c["decision_scope"]=="Each segment" and {r["segment"] for r in weights}=={"All customers"}:
        raise ValueError("Provide actual Segment labels for independent segment decisions, or choose Overall.")
    if c["decision_scope"]=="Each partner" and {r["partner"] for r in weights}=={"Not provided"}:
        raise ValueError("Provide Partner labels for independent Partner decisions, or choose Overall.")
    c["weights"] = weights
    arm_names = [a["name"] for a in c["arms"]]
    control = control_name(c)
    tests = [a for a in arm_names if a != control]
    metrics = planned_metrics(c)
    blocks = [s for s in scopes(c, weights) if s["confirmatory"]]
    family = len(tests) * len(confirmatory_metrics(c)) * len(blocks)
    alpha = c["alpha"] / family if c["multiplicity"] else c["alpha"]
    lookup = history.set_index(["partner", "segment", "arm", "metric"])
    assumed = set()
    moment_cache = {}

    def moment(w, arm, metric):
        key = (w["partner"], w["segment"], arm, metric["name"])
        if key in moment_cache:
            return moment_cache[key]
        if key in lookup.index:
            row = lookup.loc[key]
            provenance = "Provided historical arm variance; historical mean difference is not an effect"
            variance = float(row.sample_variance)
        else:
            key = (w["partner"], w["segment"], control, metric["name"])
            if key not in lookup.index:
                raise ValueError(f"Missing Control history: {w['partner']} / {w['segment']} / {metric['name']}.")
            row = lookup.loc[key]
            multiplier = float(c.get("test_sd_multipliers", {}).get(arm, 1.))
            variance = float(row.sample_variance) * multiplier ** 2
            provenance = f"Control variance × {multiplier:g}² (assumed Test SD multiplier)"
            assumed.add((arm, w["partner"], w["segment"], metric["name"], provenance))
        control_key = (w["partner"], w["segment"], control, metric["name"])
        if control_key not in lookup.index:
            raise ValueError(f"Missing Control history for {metric['name']}.")
        # For Binary planning, means under the alternative come from the planned effect.
        result = variance, float(lookup.loc[control_key].sample_mean), provenance
        moment_cache[(w["partner"],w["segment"],arm,metric["name"])]=result
        return result

    targets, comparisons, pairwise = [], [], []
    for block in blocks:
        bw = block["weights"]
        min_total = 2 * len(bw)

        def details(counts):
            cell_counts = {arm: split_integer(counts[arm], bw) for arm in arm_names}
            out = []
            for metric in metrics:
                baseline = sum(w["weight"] * moment(w, control, metric)[1] for w in bw)
                delta = effect_size(metric, baseline)
                for arm in tests:
                    terms, ns, adequate, null_variance = [], [], True, 0.
                    for i, w in enumerate(bw):
                        vc, rate, _ = moment(w, control, metric)
                        vt, _, _ = moment(w, arm, metric)
                        nc, nt = cell_counts[control][i], cell_counts[arm][i]
                        if metric["type"] == "Binary":
                            shift = 0 if metric["role"] == "Guardrail" else (delta if metric["direction"] == "Higher" else -delta)
                            alt = rate + shift
                            if not 0 < rate < 1 or not 0 < alt < 1:
                                raise ValueError(f"{metric['name']}: Binary normal-approximation planning needs baseline and proposed rates strictly between 0 and 1. Use defensible rate assumptions or a specialist exact design.")
                            vc, vt = rate * (1-rate), alt * (1-alt)
                            adequate &= min(nc * rate, nc * (1-rate), nt * alt, nt * (1-alt)) >= 5
                            pooled = (nc*rate + nt*alt)/(nc+nt)
                            null_variance += w["weight"]**2 * pooled*(1-pooled)*(1/nc+1/nt)
                        terms += [w["weight"] ** 2 * vc / nc, w["weight"] ** 2 * vt / nt]
                        ns += [nc, nt]
                    var, df = variance_df(terms, ns)
                    if var <= 0:
                        raise ValueError(f"{metric['name']}: zero variance cannot support sample-size planning. Provide a defensible positive variance assumption.")
                    critical = stats.norm.ppf(1-alpha/2) if metric["type"] == "Binary" else stats.t.ppf(1-alpha/2, df)
                    ncp = delta / math.sqrt(var)
                    if metric["type"] == "Binary":
                        # Match the pooled-null variance used by the final z-test.
                        boundary = critical*math.sqrt(null_variance/var)
                        power = stats.norm.sf(boundary-ncp)
                        if metric["role"] != "Guardrail":
                            power += stats.norm.cdf(-boundary-ncp)
                    else:
                        power = stats.nct.sf(critical, df, ncp)
                        if metric["role"] != "Guardrail":
                            power += stats.nct.cdf(-critical, df, ncp)
                    out.append({"scope": block["name"], "metric": metric["name"], "role": metric["role"], "test": arm,
                                "control": control, "control_n": sum(cell_counts[control]), "test_n": sum(cell_counts[arm]),
                                "baseline": baseline, "detectable_change_or_margin": delta, "planning_alpha": alpha,
                                "power": float(power) if adequate else 0., "normal_cells_adequate": bool(adequate),
                                "variance_of_difference": var, "degrees_of_freedom": df if math.isfinite(df) else None})
            return out

        def find_min(predicate, start=4):
            low, high = min_total, max(min_total, int(start))
            while not predicate(high):
                high *= 2
                if high > LIMIT:
                    raise ValueError("Required sample exceeds 20 million per arm. Review the practical effect, variance or eligible population.")
            while low < high:
                mid = (low + high) // 2
                if predicate(mid):
                    high = mid
                else:
                    low = mid + 1
            return low

        primary = next(m for m in metrics if m["role"] == "Primary")
        effective_sd = {arm: math.sqrt(sum(w["weight"] * moment(w, arm, primary)[0] for w in bw)) for arm in arm_names}
        if min(effective_sd.values()) <= 0:
            raise ValueError("Primary planning variances must be positive.")
        if c["allocation"] == "Business fixed":
            shares = c["fixed_shares"]
            def fixed_counts(total):
                return {a: max(min_total, math.ceil(total*shares[a])) for a in arm_names}
            total = find_min(lambda n: all(r["power"] >= c["power"] for r in details(fixed_counts(n))), 32)
            counts = fixed_counts(total)
        else:
            pair_targets = {}
            for arm in tests:
                ratio = effective_sd[arm] / effective_sd[control]
                def satisfies(nc):
                    counts0 = {a: min_total for a in arm_names}
                    counts0[control], counts0[arm] = nc, max(min_total, math.ceil(nc*ratio))
                    return all(r["power"] >= c["power"] for r in details(counts0) if r["test"] == arm)
                nc = find_min(satisfies, 32)
                nt = max(min_total, math.ceil(nc*ratio))
                pair_targets[arm] = (nc, nt)
                pairwise.append({"scope": block["name"], "test": arm, "pair_control_n": nc, "pair_test_n": nt, "neyman_ratio": ratio})
            counts = {control: max(nc for nc, _ in pair_targets.values())}
            for arm, (_, nt) in pair_targets.items():
                def satisfies(n):
                    candidate = {a: min_total for a in arm_names}
                    candidate.update({control: counts[control], arm: n})
                    return all(r["power"] >= c["power"] for r in details(candidate) if r["test"] == arm)
                counts[arm] = find_min(satisfies, nt)
        verified = details(counts)
        if any(r["power"] + 1e-10 < c["power"] for r in verified):
            raise ValueError("Rounded allocation failed final power verification; no plan has been saved.")
        comparisons += verified
        for arm in arm_names:
            cells = split_integer(counts[arm], bw)
            for w, n in zip(bw, cells):
                targets.append({"partner": w["partner"], "segment": w["segment"], "arm": arm,
                                "analyzable_n": n, "recruitment_n": math.ceil(n/(1-c["attrition"])),
                                "scope": block["name"]})
    target_frame = pd.DataFrame(targets)
    for _, indices in target_frame.groupby(["partner", "segment"], sort=False).groups.items():
        total = target_frame.loc[indices, "recruitment_n"].sum()
        target_frame.loc[indices, "expected_share"] = (target_frame.loc[indices, "arm"].map(c["fixed_shares"]) if c["allocation"] == "Business fixed" else target_frame.loc[indices, "recruitment_n"] / total)
    arm_totals = target_frame.groupby("arm", sort=False)[["analyzable_n", "recruitment_n"]].sum().reset_index()
    recruited = int(arm_totals.recruitment_n.sum())
    weeks = recruited / c["eligible_per_week"]
    end = date.fromisoformat(c["start_date"]) + timedelta(days=math.ceil(weeks*7))
    readout = end + timedelta(days=max(m["window_days"] for m in c["metrics"]) + c["maturity_delay_days"])
    limitations = ["Power uses plug-in historical moments and an approximate fixed-weight stratified Welch / normal model; historical parameter uncertainty is not fully propagated.",
                   "Target power applies to each planned comparison, not to simultaneous success of every metric or arm.",
                   "Pairwise Neyman with shared control is intuitive and verified, but is not a global minimum-total-sample optimization.",
                   "Recruitment dates assume stable eligible traffic and adequate recruitment in every selected stratum.",
                   "Fixed-horizon inference: repeated monitoring does not authorize early efficacy stopping."]
    if any(m["type"]=="Binary" for m in metrics):
        limitations.append("Binary planning assumes a common absolute risk difference across strata at the design alternative, or zero true difference for Guardrail non-inferiority power; rates are never silently clipped.")
    if not c["historical_start"] or not c["historical_end"]:
        limitations.append("The historical period was not fully supplied. Confirm the source is representative and uses the configured metric observation windows.")
    if c["assumption_only"]:
        limitations.insert(0, "ASSUMPTION-ONLY PLAN: historical sample sizes are absent or unverified; evidence is limited.")
    return {"schema_version": 2, "version": version, "saved_at": utc_now(), "config": c,
            "config_fingerprint": fingerprint(c), "history": clean_records(history), "weights": weights,
            "data_fingerprint": fingerprint(clean_records(history)), "code_version": code_version,
            "runtime_versions":{"python":platform.python_version(),**{p:package_version(p) for p in ("numpy","pandas","scipy","statsmodels","streamlit","reportlab")}},
            "targets": clean_records(target_frame), "arm_totals": clean_records(arm_totals),
            "comparisons": comparisons, "pairwise": pairwise,
            "variance_assumptions": [{"arm": a, "partner": p, "segment": s, "metric": m, "source": source} for a,p,s,m,source in sorted(assumed)],
            "family_size": family, "planning_alpha": alpha, "recruitment_weeks": weeks,
            "recruitment_end": end.isoformat(), "earliest_readout": readout.isoformat(),
            "within_max_duration": weeks <= c["max_recruitment_weeks"], "limitations": limitations,
            "estimator": "Fixed-weight Partner × Segment mean difference; variance = sum(w² s²/n); Welch-Satterthwaite df. Binary: fixed-weight risk difference with normal inference when adequate, exact per-cell fallback."}


def exploratory_protocol(config: dict, bundle: dict | None = None) -> dict:
    """An honest three-column readout without inventing a prospective sample plan."""
    c = validate_config(config)
    weights = []
    if bundle:
        source = bundle["counts"]
        if source.empty:
            source = bundle["summaries"][["partner","segment"]].drop_duplicates().assign(assigned_n=1)
        group = source.groupby(["partner","segment"],sort=False).assigned_n.sum().reset_index()
        total = group.assigned_n.sum()
        if total <= 0:
            raise ValueError("At least one randomized customer is required.")
        weights = [{"partner":r.partner,"segment":r.segment,"weight":float(r.assigned_n/total)} for r in group.itertuples()]
    family = max(1,(len(c["arms"])-1)*len(confirmatory_metrics(c)))
    return {"schema_version":2,"retrospective":True,"version":0,"saved_at":utc_now(),"config":c,
            "config_fingerprint":fingerprint(c),"weights":weights,"history":[],"data_fingerprint":fingerprint([]),
            "code_version":"Exploratory protocol; no prospective plan", "targets":[],"arm_totals":[],"comparisons":[],"pairwise":[],"variance_assumptions":[],
            "family_size":family,"planning_alpha":c["alpha"]/family if c["multiplicity"] else c["alpha"],"recruitment_weeks":0,
            "recruitment_end":"Not planned","earliest_readout":"Not planned","within_max_duration":False,
            "limitations":["Exploratory existing-data readout. Weights reflect supplied assignment counts, not a prospective population definition. No sample targets or expected allocation were supplied."],
            "estimator":"Fixed-weight stratified mean difference with descriptive assignment-mix weights. No prospective power or SRM claim."}


def setup_memo(config:dict,saved_at:str) -> dict:
    p=exploratory_protocol(config)
    p.update(kind="setup_draft",retrospective=False,saved_at=saved_at,code_version="Setup only; sample requirements not calculated",
             limitations=["Setup draft. Historical inputs, population weights, sample targets and assignment ratios have not yet been frozen."],
             estimator="Not calculated. Save a sample plan before starting a confirmatory experiment.")
    return p
