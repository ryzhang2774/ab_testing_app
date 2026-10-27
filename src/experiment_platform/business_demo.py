"""Explicitly opt-in synthetic business examples and template generation."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import pandas as pd
from .business_config import default_config


def demo_config() -> dict:
    c = default_config()
    c.update(name="Partner growth experiment", owner="Growth team", question="Does a simpler offer increase revenue without increasing cancellations?", eligible_per_week=2500.,synthetic=True)
    c["arms"] = [{"name": "Control", "definition": "Current offer", "role": "Control"},
                 {"name": "Test A", "definition": "Simplified offer", "role": "Test"},
                 {"name": "Test B", "definition": "Simplified offer with reminder", "role": "Test"}]
    c["metrics"].append({"name": "cancellation", "role": "Guardrail", "type": "Binary", "unit": "rate (0 to 1)",
                          "definition": "Cancellation within 28 days per randomized customer", "window_days": 28,
                          "direction": "Lower", "effect": 1., "effect_scale": "Percentage points", "margin": .03,
                          "alert_threshold": .02, "include_in_plan": True})
    return c


def demo_history() -> pd.DataFrame:
    rows = []
    for p, shift in [("Partner A", 0), ("Partner B", 25)]:
        for s, extra in [("New", 0), ("Existing", 20)]:
            rows += [{"partner": p, "segment": s, "metric": "revenue", "sample_mean": 100.+shift+extra,
                      "sample_variance": float((30+shift/5+extra/5)**2), "sample_number": 1200, "successes": None},
                     {"partner": p, "segment": s, "metric": "cancellation", "sample_mean": .1,
                      "sample_variance": None, "sample_number": 1200, "successes": 120}]
    return pd.DataFrame(rows)


def demo_records() -> pd.DataFrame:
    rng = np.random.default_rng(412)
    rows = []
    for p, shift in [("Partner A", 0), ("Partner B", 25)]:
        for s, extra in [("New", 0), ("Existing", 20)]:
            for arm, effect, n in [("Control", 0, 140), ("Test A", 7, 130), ("Test B", 11, 130)]:
                for i in range(n):
                    rows.append({"id": f"DEMO-{p[-1]}-{s}-{arm[-1]}-{i}", "partner": p, "segment": s, "arm": arm,
                                 "revenue": round(float(rng.normal(100+shift+extra+effect, 30+shift/5+extra/5)), 2),
                                 "cancellation": int(rng.random() < (.1 if arm != "Test B" else .14)),
                                 "baseline_age": int(rng.integers(21, 75)), "baseline_channel": rng.choice(["Online", "Branch"]),
                                 "outcome_window_complete": True, "snapshot_date": "2026-10-20"})
    return pd.DataFrame(rows)
