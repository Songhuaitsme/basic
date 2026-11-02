"""Validate and summarize paired V7/V8 evaluation reports."""

import argparse
import json
from pathlib import Path
import statistics


INPUT_FIELDS = (
    "seed", "model_hash", "task_trace_hash", "exogenous_trace_hash",
    "topology_hash", "dependency_lock_hash", "candidate_mode",
    "warmup_days", "arrival_cutoff_sim", "tariff_mode",
)


def summarize(pairs):
    rows = []
    for baseline_path, optimized_path in pairs:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        optimized = json.loads(Path(optimized_path).read_text(encoding="utf-8"))
        if baseline["status"] != "VALID" or optimized["status"] != "VALID":
            raise ValueError("Both evaluations must be VALID")
        for key in INPUT_FIELDS:
            if baseline["metadata"][key] != optimized["metadata"][key]:
                raise ValueError(f"Input mismatch: {key}")
        if baseline["metrics"] != optimized["metrics"]:
            raise ValueError("Exact metrics differ")
        original_decisions = baseline["decision_records"]
        new_decisions = optimized["decision_records"]
        if len(original_decisions) != len(new_decisions):
            raise ValueError("Decision count differs")
        for index, (a, b) in enumerate(zip(original_decisions, new_decisions)):
            for key in a:
                if key not in ("decision_wall_seconds", "theoretical_time_slots") and a[key] != b[key]:
                    raise ValueError(f"Decision {index} differs: {key}")
        a = baseline["diagnostics"]["runtime_summary"]
        b = optimized["diagnostics"]["runtime_summary"]
        if "candidate_detail" in a or "candidate_detail" in b:
            raise ValueError("Timing comparisons must disable detailed profiling")
        stats = b["path_search_cache"]["static_hard_pruning"]
        rows.append({
            "baseline": str(baseline_path), "optimized": str(optimized_path),
            "decisions": len(original_decisions),
            "baseline_candidate_seconds": a["sections_seconds"]["candidate_slot_generation"],
            "optimized_candidate_seconds": b["sections_seconds"]["candidate_slot_generation"],
            "baseline_total_seconds": a["total_wall_seconds"],
            "optimized_total_seconds": b["total_wall_seconds"],
            "baseline_theoretical_slots": sum(x["theoretical_time_slots"] for x in original_decisions),
            "optimized_search_slots": sum(x["theoretical_time_slots"] for x in new_decisions),
            **{key: stats[key] for key in (
                "static_cpu_nodes_rejected", "path_queries_avoided",
                "grid_preparations_avoided_empty_window",
            )},
        })
    baseline_times = [row["baseline_candidate_seconds"] for row in rows]
    optimized_times = [row["optimized_candidate_seconds"] for row in rows]
    return {
        "status": "VALID", "pairs": rows,
        "candidate_median_baseline_seconds": statistics.median(baseline_times),
        "candidate_median_optimized_seconds": statistics.median(optimized_times),
        "candidate_median_difference_seconds": statistics.median(baseline_times) - statistics.median(optimized_times),
        "candidate_paired_differences_seconds": [a - b for a, b in zip(baseline_times, optimized_times)],
        "slot_count_note": "baseline theoretical slots versus optimized post-pruning search slots",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="Alternating baseline and optimized reports")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.reports) % 2:
        parser.error("Provide an even number of reports")
    result = summarize(list(zip(args.reports[::2], args.reports[1::2])))
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
