"""Append-only protocol versions. Individual customer data is never persisted here.

For Databricks configure CUSTOMER_PROTOCOL_DIR to an access-controlled persistent
Volume. The default is a local development store, not a shared multi-user DB.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import math

from .business_config import fingerprint


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("business_*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "customer-v2-sha256:" + digest.hexdigest()


def protocol_root() -> Path:
    return Path(os.environ.get("CUSTOMER_PROTOCOL_DIR", ".customer_protocols"))


def save_protocol(plan: dict, root: Path | None = None) -> Path:
    identity = plan["config"]["experiment_id"]
    if not re.fullmatch(r"[A-Za-z0-9-]{1,80}", identity):
        raise ValueError("Invalid experiment ID.")
    folder = (root or protocol_root()) / identity
    folder.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a protocol from a concurrent session.
    path = folder / f"v{int(plan['version']):04d}.json"
    payload = json.dumps(plan, ensure_ascii=False, allow_nan=False, indent=2)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return path


def next_version(identity: str, root: Path | None = None) -> int:
    if not re.fullmatch(r"[A-Za-z0-9-]{1,80}", identity):
        raise ValueError("Invalid experiment ID.")
    folder = (root or protocol_root()) / identity
    return max([int(p.stem[1:]) for p in folder.glob("v[0-9]*.json")] or [0]) + 1


def saved_protocols(root: Path | None = None) -> list[dict]:
    """List protocol metadata, never customer-level data."""
    output=[]
    for path in (root or protocol_root()).glob("*/v[0-9]*.json"):
        try:
            p=json.loads(path.read_text(encoding="utf-8"))
            output.append({"path":str(path),"name":p["config"]["name"],"owner":p["config"]["owner"],"version":p["version"],"saved_at":p["saved_at"],"synthetic":p["config"].get("synthetic",False)})
        except (OSError,ValueError,KeyError):
            continue
    return sorted(output,key=lambda x:x["saved_at"],reverse=True)


def import_protocol(payload: str | bytes) -> dict:
    from .business_config import validate_config
    from .business_plan import build_plan
    import pandas as pd
    plan = json.loads(payload)
    if plan.get("schema_version") != 2:
        raise ValueError("Choose a Customer Business V2 protocol JSON.")
    validate_config(plan["config"])
    if plan.get("kind")=="setup_draft":
        if fingerprint(plan["config"])!=plan["config_fingerprint"]:
            raise ValueError("Setup configuration fingerprint does not match.")
        return plan
    if fingerprint(plan["config"]) != plan["config_fingerprint"] or fingerprint(plan["history"]) != plan["data_fingerprint"]:
        raise ValueError("Protocol fingerprints do not match its configuration or historical inputs.")
    if plan.get("retrospective"):
        if plan.get("targets") or plan.get("comparisons") or not plan.get("weights") or any(r["weight"]<=0 for r in plan["weights"]) or not math.isclose(sum(r["weight"] for r in plan["weights"]),1.):
            raise ValueError("Exploratory protocols cannot contain planned targets and must contain a valid descriptive population mix.")
        return plan
    # Recalculate targets to detect modified derived numbers; do not trust arbitrary imported output.
    rebuilt = build_plan(plan["config"], pd.DataFrame(plan["history"]), plan["version"], plan["code_version"])
    def equivalent(a,b):
        if isinstance(a,dict) and isinstance(b,dict):
            return set(a)==set(b) and all(equivalent(a[k],b[k]) for k in a)
        if isinstance(a,list) and isinstance(b,list):
            return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):
            return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
        return a==b
    for field in ("targets", "arm_totals", "family_size", "weights", "comparisons", "pairwise", "variance_assumptions", "planning_alpha", "recruitment_weeks", "recruitment_end", "earliest_readout", "within_max_duration", "estimator"):
        if not equivalent(rebuilt[field],plan[field]):
            raise ValueError(f"Imported {field} disagree with a fresh calculation. Rebuild and save a new protocol.")
    return plan
