"""Strict, auditable input adapters for Customer Business V2."""
from __future__ import annotations

from io import StringIO
import hashlib
import numpy as np
import pandas as pd

from .business_config import control_name, fingerprint

KEYS = ["partner", "segment", "arm"]
ALIASES = {"sample mean": "sample_mean", "mean": "sample_mean", "sample variance": "sample_variance",
           "variance": "sample_variance", "sample number": "sample_number", "n": "sample_number",
           "observed_n": "sample_number", "sd": "sample_sd", "stdev.s": "sample_sd", "outcome": "outcome"}


def read_table(text: str | bytes) -> pd.DataFrame:
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    if not text.strip():
        return pd.DataFrame()
    sep = "\t" if "\t" in text.splitlines()[0] else ","
    return pd.read_csv(StringIO(text), sep=sep, dtype=str, keep_default_na=False)


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d.columns = [ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in d.columns]
    if d.columns.duplicated().any():
        raise ValueError("Column names become duplicates after mapping. Use distinct field names.")
    return d.replace(r"^\s*$", np.nan, regex=True)


def frame_fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, na_rep="").encode()).hexdigest()


def _text(d: pd.DataFrame, col: str, fallback: str | None = None):
    if col not in d:
        if fallback is None:
            raise ValueError(f"Missing column: {col}.")
        d[col] = fallback
    if d[col].isna().any():
        rows = (np.flatnonzero(d[col].isna()) + 2).tolist()[:8]
        raise ValueError(f"{col}: missing value at source rows {rows}.")
    d[col] = d[col].astype(str).str.strip()


def _number(d, col, minimum=None, integer=False, optional=False):
    if col not in d:
        if optional:
            d[col] = np.nan
            return
        raise ValueError(f"Missing column: {col}.")
    raw = d[col]
    values = pd.to_numeric(raw, errors="coerce")
    valid = np.isfinite(values)
    if minimum is not None:
        valid &= values >= minimum
    if integer:
        valid &= values.mod(1).eq(0)
    if optional:
        valid |= raw.isna()
    if not valid.all():
        rows = (np.flatnonzero(~valid) + 2).tolist()[:8]
        raise ValueError(f"{col}: invalid {'whole number' if integer else 'number'} at source rows {rows}.")
    d[col] = values


def validate_history(frame: pd.DataFrame, config: dict, *, outcomes=False) -> pd.DataFrame:
    d = canonical(frame)
    if d.empty:
        raise ValueError("Add historical statistics before calculating a plan." if not outcomes else "Add outcome summaries.")
    for col, fallback in [("partner", "Not provided"), ("segment", "All customers"),
                          ("arm", None if outcomes else control_name(config)), ("metric", None)]:
        _text(d, col, fallback)
    names = {m["name"]: m for m in config["metrics"]}
    if not set(d.metric).issubset(names):
        raise ValueError("Unknown metric names: " + ", ".join(sorted(set(d.metric) - names.keys())))
    if not set(d.arm).issubset({a["name"] for a in config["arms"]}):
        raise ValueError("Arm labels must match the saved setup.")
    duplicate = d.duplicated(KEYS + ["metric"], keep=False)
    if duplicate.any():
        raise ValueError(f"Duplicate Partner × Segment × Arm × Metric at source rows {(np.flatnonzero(duplicate) + 2).tolist()[:8]}.")
    _number(d, "sample_number", 2, integer=True, optional=config["assumption_only"] and not outcomes)
    if "sample_sd" in d:
        _number(d, "sample_sd", 0, optional=True)
        if "sample_variance" in d:
            supplied = pd.to_numeric(d.sample_variance, errors="coerce")
            both = d.sample_sd.notna() & d.sample_variance.notna()
            if not np.allclose(supplied[both], d.loc[both, "sample_sd"] ** 2):
                raise ValueError("SD and sample variance disagree; variance must equal SD squared.")
            d["sample_variance"] = supplied.fillna(d.sample_sd ** 2)
        else:
            d["sample_variance"] = d.sample_sd ** 2
    for col in ("sample_mean", "sample_variance", "successes"):
        if col not in d:
            d[col] = np.nan
    for name, metric in names.items():
        mask = d.metric.eq(name)
        part = d[mask].copy()
        if part.empty:
            continue
        if metric["type"] == "Binary":
            _number(part, "successes", 0, integer=True)
            if part.sample_number.isna().any() or (part.successes > part.sample_number).any():
                raise ValueError(f"{name}: Binary metrics require exact successes and valid N, with successes <= N.")
            rate = part.successes / part.sample_number
            given = pd.to_numeric(part.sample_mean, errors="coerce")
            if (given.notna() & ~np.isclose(given, rate, atol=1e-8)).any():
                raise ValueError(f"{name}: supplied mean disagrees with exact successes / N.")
            d.loc[mask, "successes"] = part.successes
            d.loc[mask, "sample_mean"] = rate
            d.loc[mask, "sample_variance"] = rate * (1 - rate) * part.sample_number / (part.sample_number - 1)
        else:
            _number(part, "sample_mean")
            _number(part, "sample_variance", 0)
            d.loc[mask, ["sample_mean", "sample_variance"]] = part[["sample_mean", "sample_variance"]]
    for col in ("sample_mean", "sample_variance", "successes"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if "window_days" in d:
        _number(d, "window_days", 1, integer=True)
        expected = d.metric.map({m["name"]: m["window_days"] for m in config["metrics"]})
        if not d.window_days.eq(expected).all():
            raise ValueError("Observation windows disagree with metric definitions in setup.")
    for field in ("historical_start", "historical_end"):
        if field in d:
            dates = pd.to_datetime(d[field], errors="coerce")
            if dates.isna().any() or (config.get(field) and not dates.eq(pd.Timestamp(config[field])).all()):
                raise ValueError(f"{field} disagrees with the configured historical period.")
    return d.reset_index(drop=True)


def prepare_records(frame: pd.DataFrame, config: dict, as_of: str, mapping: dict | None = None) -> dict:
    d = frame.rename(columns=mapping or {}).copy()
    d.columns = [str(c).strip() for c in d.columns]
    # The minimal ID / Arm / Outcome path remains valid.
    d = d.rename(columns={c: c.lower() for c in d.columns if c.lower() in {"id", "arm", "partner", "segment", "outcome"}})
    primary = next(m["name"] for m in config["metrics"] if m["role"] == "Primary")
    if primary not in d and "outcome" in d:
        d = d.rename(columns={"outcome": primary})
    if d.columns.duplicated().any():
        raise ValueError("Field mapping produces duplicate columns.")
    provided = {field: field in d for field in ("partner", "segment")}
    for col, fallback in [("id", None), ("arm", None), ("partner", "Not provided"), ("segment", "All customers")]:
        _text(d, col, fallback)
    if not set(d.arm).issubset({a["name"] for a in config["arms"]}):
        raise ValueError("Uploaded arm labels must match the saved setup. Map or rename the labels first.")
    trend_source = d.copy() if "snapshot_date" in d else pd.DataFrame()
    if d.id.duplicated().any():
        if "snapshot_date" not in d:
            raise ValueError("Duplicate ID: use one row per independent randomized unit, or dated snapshots.")
        for _, group in d.groupby("id", sort=False):
            if any(group[col].nunique() > 1 for col in KEYS):
                raise ValueError("The same ID changes arm, Partner or Segment across snapshots. Resolve assignment conflicts.")
        dates = pd.to_datetime(d.snapshot_date, errors="coerce", utc=True)
        if dates.isna().any() or d.assign(_snapshot=dates).duplicated(["id", "_snapshot"]).any():
            raise ValueError("Each ID needs a unique, valid snapshot_date; duplicate snapshots are ambiguous.")
        d = d.assign(_snapshot=dates).sort_values("_snapshot").drop_duplicates("id", keep="last")
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    updated = "Not provided"
    if "snapshot_date" in d:
        dates = pd.to_datetime(d.snapshot_date, errors="coerce", utc=True)
        if dates.isna().any() or (dates > as_of_ts).any():
            raise ValueError("Snapshot dates must be valid and cannot be after the selected data-as-of time.")
        updated = dates.max().isoformat()
    elif as_of:
        updated = as_of_ts.isoformat()
    maturity_known = False
    derived_complete = None
    if "assignment_date" in d:
        assigned=pd.to_datetime(d.assignment_date,errors="coerce",utc=True)
        if assigned.isna().any() or (assigned>as_of_ts).any():
            raise ValueError("Assignment dates must be valid and not after the data-as-of time.")
        if "snapshot_date" in d and (assigned>pd.to_datetime(d.snapshot_date,utc=True)).any():
            raise ValueError("An assignment cannot occur after its source snapshot.")
        days=max(m["window_days"] for m in config["metrics"])+config["maturity_delay_days"]
        source_cutoff=pd.to_datetime(d.snapshot_date,utc=True) if "snapshot_date" in d else as_of_ts
        derived_complete=assigned+pd.Timedelta(days=days)<=source_cutoff
    if "outcome_window_complete" in d:
        complete = d.outcome_window_complete.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        if complete.isna().any():
            raise ValueError("outcome_window_complete must contain true/false or 1/0 for every unit.")
        d["_mature"] = complete.astype(bool)
        if derived_complete is not None and (d._mature & ~derived_complete).any():
            raise ValueError("A completed-window flag conflicts with assignment date and the configured observation window.")
        maturity_known = True
    elif "assignment_date" in d:
        d["_mature"] = derived_complete
        maturity_known = True
    else:
        d["_mature"] = True  # Analysis allowed with an explicit unverified-maturity limitation.
    counts, summaries = [], []
    for key, group in d.groupby(KEYS, sort=False):
        base = dict(zip(KEYS, key))
        counts.append({**base, "assigned_n": len(group), "mature_n": int(group._mature.sum()) if maturity_known else None})
        for metric in config["metrics"]:
            col = metric["name"]
            if col not in group:
                continue
            raw = group[col].replace(r"^\s*$", np.nan, regex=True)
            values = pd.to_numeric(raw, errors="coerce")
            if (raw.notna() & (~np.isfinite(values))).any():
                raise ValueError(f"{col}: invalid numeric values. Leave missing results blank; do not replace them with zero.")
            if metric["type"] == "Binary" and not set(values.dropna()).issubset({0, 1}):
                raise ValueError(f"{col}: Binary results must be 0 or 1.")
            values = values[group._mature].dropna()
            summaries.append({**base, "metric": col, "sample_number": len(values),
                              "sample_mean": float(values.mean()) if len(values) else None,
                              "sample_variance": float(values.var(ddof=1)) if len(values) > 1 else None,
                              "successes": int(values.sum()) if metric["type"] == "Binary" else None})
    baselines = []
    baseline_types = config.get("baseline_types",{}) if config.get("baseline_selection_explicit") else {**{c: "Auto" for c in d.columns if c.startswith("baseline_")}, **config.get("baseline_types", {})}
    for feature, kind in baseline_types.items():
        if feature not in d:
            continue
        if feature in {m["name"] for m in config["metrics"]}:
            raise ValueError("Outcome metrics cannot be used as baseline covariates.")
        if kind == "Auto":
            nonblank = d[feature].replace(r"^\s*$", np.nan, regex=True).dropna()
            kind = "Continuous" if len(nonblank) and pd.to_numeric(nonblank, errors="coerce").notna().all() else "Categorical"
        categories = sorted(d[feature].dropna().astype(str).unique()) if kind == "Categorical" else [""]
        for key, group in d.groupby(KEYS, sort=False):
            raw = group[feature].replace(r"^\s*$", np.nan, regex=True)
            for category in categories:
                values = (raw.dropna().astype(str) == category).astype(float) if kind == "Categorical" else pd.to_numeric(raw, errors="coerce").dropna()
                if kind == "Continuous" and (raw.notna().sum() != len(values) or not np.isfinite(values).all()):
                    raise ValueError(f"{feature}: invalid continuous baseline values.")
                baselines.append({**dict(zip(KEYS, key)), "feature": feature, "kind": kind, "category": category,
                                  "sample_number": len(values), "sample_mean": float(values.mean()) if len(values) else None,
                                  "sample_variance": float(values.var(ddof=1)) if len(values) > 1 else None,
                                  "missing_rate": float(raw.isna().mean())})
    trends = []
    if not trend_source.empty:
        ts = pd.to_datetime(trend_source.snapshot_date,errors="coerce",utc=True)
        if ts.isna().any() or (ts > as_of_ts).any():
            raise ValueError("Every historical snapshot needs a valid date no later than data-as-of.")
        trend_source["snapshot_date"] = ts.dt.date.astype(str)
        # Snapshot means are descriptive and never summed across dates.
        for key,group in trend_source.groupby(["snapshot_date"]+KEYS,sort=True):
            if "outcome_window_complete" in group:
                complete=group.outcome_window_complete.astype(str).str.lower().isin(["true","1","yes"])
                group=group[complete]
            for metric in config["metrics"]:
                if metric["name"] not in group:
                    continue
                vals = pd.to_numeric(group[metric["name"]],errors="coerce").dropna()
                trends.append({**dict(zip(["snapshot_date"]+KEYS,key)),"metric":metric["name"],"observed_n":len(vals),"mean":float(vals.mean()) if len(vals) else None})
    return {"counts": pd.DataFrame(counts), "summaries": pd.DataFrame(summaries), "baselines": pd.DataFrame(baselines), "trends":pd.DataFrame(trends),
            "fingerprint": fingerprint({"source":frame_fingerprint(frame),"mapping":mapping or {},"as_of":as_of,"baseline_types":config.get("baseline_types",{})}), "updated_at": updated, "source": "Customer records",
            "maturity_known": maturity_known, "dimensions_provided": provided,
            "limitations": ([] if maturity_known else ["Outcome maturity was not provided; final decisions require confirmation of a complete observation window."]) + (["Baseline covariates are assumed to be measured before assignment; timing is not independently verified."] if baselines else [])}


def prepare_summaries(outcomes: pd.DataFrame, counts: pd.DataFrame, baselines: pd.DataFrame,
                      config: dict, as_of: str) -> dict:
    s = validate_history(outcomes, config, outcomes=True)
    c = canonical(counts) if not counts.empty else pd.DataFrame(columns=KEYS + ["assigned_n", "mature_n"])
    if not c.empty:
        for col, fallback in [("partner", "Not provided"), ("segment", "All customers"), ("arm", None)]:
            _text(c, col, fallback)
        _number(c, "assigned_n", 0, integer=True)
        _number(c, "mature_n", 0, integer=True, optional=True)
        if c.duplicated(KEYS).any() or (c.mature_n > c.assigned_n).any():
            raise ValueError("Counts must have one row per Partner × Segment × Arm and Mature N cannot exceed Assigned N.")
        if not set(c.arm).issubset({a["name"] for a in config["arms"]}):
            raise ValueError("Assigned count arms must match the saved setup.")
        merged = s.merge(c, on=KEYS, how="left", validate="many_to_one")
        if merged.assigned_n.isna().any() or (merged.sample_number > merged.assigned_n).any() or (merged.sample_number > merged.mature_n).fillna(False).any():
            raise ValueError("Every outcome group needs assignment counts and Observed N cannot exceed Mature / Assigned N.")
    b = canonical(baselines) if not baselines.empty else pd.DataFrame()
    if not b.empty:
        if "category" not in b:
            b["category"] = "(continuous)"
        b["category"] = b["category"].fillna("(continuous)")
        for col, fallback in [("partner", "Not provided"), ("segment", "All customers"), ("arm", None), ("feature", None), ("kind", None), ("category", None)]:
            _text(b, col, fallback)
        if not set(b.kind).issubset({"Continuous", "Categorical"}):
            raise ValueError("Baseline kind must be Continuous or Categorical.")
        for col, minimum in [("sample_number", 2), ("sample_mean", None), ("sample_variance", 0), ("missing_rate", 0)]:
            _number(b, col, minimum, integer=col == "sample_number")
        if (b.missing_rate > 1).any() or b.duplicated(KEYS + ["feature", "category"]).any():
            raise ValueError("Invalid baseline missing rate or duplicate baseline category.")
        if not set(b.arm).issubset({a["name"] for a in config["arms"]}) or ((b.kind=="Categorical") & ~b.sample_mean.between(0,1)).any():
            raise ValueError("Baseline arms must match the setup and categorical proportions must lie between 0 and 1.")
        if b.feature.isin([m["name"] for m in config["metrics"]]).any():
            raise ValueError("Only pre-treatment covariates belong in baseline summaries.")
    return {"counts": c, "summaries": s, "baselines": b, "fingerprint": fingerprint({"outcomes":frame_fingerprint(s),"assigned":frame_fingerprint(c),"baseline":frame_fingerprint(b),"as_of":as_of}),
            "updated_at": as_of, "source": "Aggregate summaries", "maturity_known": not c.empty and c.mature_n.notna().all(),
            "dimensions_provided": {"partner": "partner" in canonical(outcomes), "segment": "segment" in canonical(outcomes)},
            "limitations": ["Aggregate inputs cannot verify duplicate IDs, cross-arm customers or individual outliers."]}
