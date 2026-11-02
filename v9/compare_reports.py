"""Summarize frozen-policy E0..E3 reports after the user runs evaluations."""

import argparse
import json
from pathlib import Path
import statistics


FIXED = ("seed", "model_hash", "task_trace_hash", "exogenous_trace_hash",
         "topology_hash", "dependency_lock_hash", "candidate_mode",
         "warmup_days", "arrival_cutoff_sim", "tariff_mode")


def compare(paths):
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    baseline = reports[0]
    for report in reports:
        if report["status"] != "VALID":
            raise ValueError("All reports must be VALID")
        for key in FIXED:
            if report["metadata"].get(key) != baseline["metadata"].get(key):
                raise ValueError(f"Input mismatch: {key}")
    rows = {}
    for stage, report in enumerate(reports):
        decisions = report.get("decision_records", ())
        times = sorted(d["decision_wall_seconds"] for d in decisions
                       if d.get("decision_wall_seconds") is not None)
        p95 = times[min(len(times) - 1, int(0.95 * len(times)))] if times else None
        runtime = report["diagnostics"]["runtime_summary"]
        rows[f"E{stage}"] = {
            "metrics": report["metrics"],
            "candidate_generation_seconds": runtime["sections_seconds"].get("candidate_slot_generation"),
            "decision_mean_seconds": statistics.mean(times) if times else None,
            "decision_p95_seconds": p95,
            "search_stats": runtime.get("path_search_cache", {}).get("v9_search", {}),
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs=4, type=Path, metavar="E0..E3")
    args = parser.parse_args()
    print(json.dumps(compare(args.reports), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
