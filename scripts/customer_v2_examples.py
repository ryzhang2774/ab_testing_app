"""Generate synthetic templates, a reproducible demo protocol and PDF examples."""
from pathlib import Path
import json
import csv
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_platform.business_demo import demo_config, demo_history, demo_records
from src.experiment_platform.business_plan import build_plan
from src.experiment_platform.business_data import prepare_records
from src.experiment_platform.business_analysis import monitor, analyze
from src.experiment_platform.business_report import build_pdf
from src.experiment_platform.business_store import code_fingerprint


def main():
    root = Path(__file__).resolve().parents[1]
    samples = root / "sample_data"
    output = root / "output" / "pdf"
    output.mkdir(parents=True, exist_ok=True)
    c, h, r = demo_config(), demo_history(), demo_records()
    c["experiment_id"] = "customer-business-v2-demo"
    h.to_csv(samples / "customer_v2_history_demo.csv", index=False)
    h.to_csv(samples / "customer_v2_history_paste.tsv", index=False, sep="\t",quoting=csv.QUOTE_NONNUMERIC)
    r.to_csv(samples / "customer_v2_records_demo.csv", index=False)
    plan = build_plan(c, h, code_version=code_fingerprint())
    bundle = prepare_records(r, c, "2026-10-26T23:59:59+00:00")
    bundle["counts"].to_csv(samples / "customer_v2_counts_demo.csv", index=False)
    bundle["summaries"].to_csv(samples / "customer_v2_outcomes_demo.csv", index=False)
    bundle["baselines"].to_csv(samples / "customer_v2_baselines_demo.csv", index=False)
    (output / "customer-v2-demo-protocol.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshots = [("design", None, "Build test plan"), ("monitor", monitor(plan, bundle), "Monitor experiment"), ("analysis", analyze(plan, bundle), "Analyze results")]
    for name, snapshot, stage in snapshots:
        (output / f"customer-v2-{name}-demo.pdf").write_bytes(build_pdf(plan, snapshot, stage))
    print(f"Generated templates and {len(snapshots)} PDF examples in {output}")


if __name__ == "__main__":
    main()
