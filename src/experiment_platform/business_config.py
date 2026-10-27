"""Versioned business protocol. No Streamlit or individual records in this module."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timezone
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fingerprint(value) -> str:
    def default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        raise TypeError(type(obj).__name__)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, default=default).encode()).hexdigest()


def default_config() -> dict:
    return {
        "schema_version": 2, "experiment_id": str(uuid4()), "synthetic":False,
        "name": "New customer experiment", "owner": "", "question": "",
        "unit": "Customer", "start_date": date.today().isoformat(),
        "max_recruitment_weeks": 12, "eligible_per_week": 1000.,
        "window_days": 28, "maturity_delay_days": 0, "decision_scope": "Overall",
        "arms": [{"name": "Control", "definition": "Current experience", "role": "Control"},
                 {"name": "Test A", "definition": "New experience", "role": "Test"}],
        "metrics": [{"name": "revenue", "role": "Primary", "type": "Continuous",
                     "unit": "currency / customer", "definition": "Total per randomized customer over the observation window",
                     "window_days": 28, "direction": "Higher", "effect": 5., "effect_scale": "Relative %",
                     "margin": 0., "alert_threshold": 0., "include_in_plan": True}],
        "alpha": .05, "power": .8, "multiplicity": True, "attrition": 0.,
        "srm_alpha": .001, "smd_threshold": .1,
        "allocation": "Recommended", "fixed_shares": {}, "test_sd_multipliers": {},
        "assumption_only": False, "historical_start": "", "historical_end": "",
        "weights": [], "baseline_types": {}, "design": "Fixed horizon",
    }


def validate_config(config: dict) -> dict:
    c = json.loads(json.dumps(config))
    for field in ("experiment_id", "name", "owner", "question"):
        if not str(c.get(field, "")).strip():
            raise ValueError(f"Enter {field.replace('_', ' ')} before saving the setup.")
        c[field] = str(c[field]).strip()
    date.fromisoformat(c["start_date"])
    if c["unit"] not in {"Customer", "Account"} or c["decision_scope"] not in {"Overall", "Each segment", "Each partner"}:
        raise ValueError("Choose a supported randomization unit and decision scope.")
    for field in ("eligible_per_week", "max_recruitment_weeks", "window_days"):
        if not math.isfinite(float(c[field])) or float(c[field]) <= 0:
            raise ValueError(f"{field.replace('_', ' ')} must be positive.")
    if int(c["window_days"]) != c["window_days"] or int(c["maturity_delay_days"]) != c["maturity_delay_days"] or c["maturity_delay_days"] < 0:
        raise ValueError("Observation and maturity delays must be whole days; delays cannot be negative.")
    if not 0 < c["alpha"] < .5 or not .5 <= c["power"] < 1 or not 0 <= c["attrition"] < 1:
        raise ValueError("Check alpha, power and expected missing outcomes.")
    if not 0 < c["srm_alpha"] < 1 or c["smd_threshold"] <= 0:
        raise ValueError("Check SRM and balance diagnostic thresholds.")
    arms = c["arms"]
    names = [str(a["name"]).strip() for a in arms]
    if len(names) < 2 or any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError("Use distinct, non-empty names for one Control and at least one Test.")
    if sum(a["role"] == "Control" for a in arms) != 1 or any(a["role"] not in {"Control", "Test"} for a in arms):
        raise ValueError("Define exactly one Control; all other arms must be Test.")
    for arm, name in zip(arms, names):
        arm["name"] = name
        if not str(arm["definition"]).strip():
            raise ValueError(f"Explain what customers receive in {name}.")
    metrics = c["metrics"]
    if sum(m["role"] == "Primary" for m in metrics) != 1:
        raise ValueError("Choose exactly one Primary metric.")
    if len({m["name"] for m in metrics}) != len(metrics):
        raise ValueError("Metric names must be unique.")
    for m in metrics:
        if not str(m["name"]).strip() or m["type"] not in {"Continuous", "Binary"}:
            raise ValueError("Use named Continuous or Binary metrics. Ratios, survival and correlated repeated records need a different analysis.")
        if m["name"] in {"id", "arm", "partner", "segment", "snapshot_date", "assignment_date", "outcome_window_complete"} or m["name"].startswith("baseline_"):
            raise ValueError("Metric names cannot use reserved ID, grouping, date or baseline field names.")
        if m["role"] not in {"Primary", "Guardrail", "Secondary"} or m["direction"] not in {"Higher", "Lower"}:
            raise ValueError("Check metric roles and improvement directions.")
        if not str(m.get("definition", "")).strip() or not str(m.get("unit", "")).strip():
            raise ValueError(f"{m['name']}: provide a definition and unit.")
        if int(m["window_days"]) != m["window_days"] or m["window_days"] <= 0:
            raise ValueError("Metric observation windows must be positive whole days.")
        if m["effect_scale"] not in {"Relative %", "Absolute", "Percentage points"}:
            raise ValueError("Choose Relative %, Absolute or Percentage points.")
        if m["effect_scale"] == "Percentage points" and m["type"] != "Binary":
            raise ValueError("Percentage points apply only to Binary metrics.")
        for f in ("effect", "margin", "alert_threshold"):
            if not math.isfinite(float(m[f])) or float(m[f]) < 0:
                raise ValueError(f"{m['name']}: {f} must be finite and non-negative.")
        if m["role"] == "Primary" and m["effect"] <= 0:
            raise ValueError("Enter a positive smallest worthwhile improvement.")
        if m["role"] == "Guardrail" and m["margin"] <= 0:
            raise ValueError("Enter a positive maximum tolerated deterioration for every Guardrail (in metric units; binary uses rate units).")
        if m["type"]=="Binary" and (m["margin"]>1 or m["alert_threshold"]>1):
            raise ValueError("Binary tolerance and alert thresholds use rate units between 0 and 1; 0.01 is one percentage point.")
        if m["include_in_plan"] and m["role"] == "Secondary" and m["effect"] <= 0:
            raise ValueError("Planned Secondary metrics need a positive detectable change.")
    for field in ("historical_start","historical_end"):
        if c[field]:
            date.fromisoformat(c[field])
    if c["historical_start"] and c["historical_end"] and date.fromisoformat(c["historical_start"]) > date.fromisoformat(c["historical_end"]):
        raise ValueError("Historical start must precede historical end.")
    if c["allocation"] == "Business fixed":
        shares = c["fixed_shares"]
        if set(shares) != set(names) or any(not math.isfinite(float(v)) or v <= 0 for v in shares.values()):
            raise ValueError("Supply a positive allocation weight for every arm.")
        total = sum(shares.values())
        divisor=1. if math.isclose(total,1.,rel_tol=0,abs_tol=1e-12) else total
        c["fixed_shares"] = {n: shares[n] / divisor for n in names}
    for arm, factor in c.get("test_sd_multipliers", {}).items():
        if arm not in names or not math.isfinite(float(factor)) or factor <= 0:
            raise ValueError("Test SD multipliers must be positive and refer to defined arms.")
    return c


def control_name(c: dict) -> str:
    return next(a["name"] for a in c["arms"] if a["role"] == "Control")


def effect_size(metric: dict, baseline: float) -> float:
    if metric["role"] == "Guardrail":
        return float(metric["margin"])
    value = float(metric["effect"])
    if metric["effect_scale"] == "Relative %":
        value *= abs(baseline) / 100
    elif metric["effect_scale"] == "Percentage points":
        value /= 100
    if value <= 0:
        raise ValueError(f"{metric['name']}: use an absolute change when the baseline is zero.")
    return value
