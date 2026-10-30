"""Run/compare serial, fixed-model V6 candidate-profiler trials.

Full evaluations are only started by explicitly running this module.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PAIR_FIELDS = (
    "system_version", "seed", "warmup_days", "evaluation_start_sim",
    "arrival_cutoff_sim", "candidate_mode", "tariff_mode", "model_hash",
    "config_hash", "code_hash", "task_trace_hash", "exogenous_trace_hash",
    "topology_hash", "dependency_lock_hash",
)


def _read_trial(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "VALID":
        raise ValueError(f"Invalid evaluation: {path}")
    return report


def _decisions(report):
    # Check every recorded decision field except its inherently variable time.
    return [
        {key: value for key, value in record.items() if key != "decision_wall_seconds"}
        for record in report["decision_records"]
    ]


def summarize(output_dir, repeats):
    """Reject changed inputs/decisions before reporting any performance claim."""
    reference = None
    rows = []
    detail_reports = []
    detail_trials = []
    for trial in range(1, repeats + 1):
        for mode in ("off", "on"):
            path = output_dir / f"trial_{trial:02}_{mode}.json"
            report = _read_trial(path)
            conditions = {field: report["metadata"][field] for field in PAIR_FIELDS}
            signature = (conditions, report["metrics"], _decisions(report))
            if reference is None:
                reference = signature
            if signature != reference:
                differences = [
                    name for name, expected, actual in zip(
                        ("metadata", "metrics", "decisions"), reference, signature
                    ) if expected != actual
                ]
                raise ValueError(f"Equivalence check failed ({', '.join(differences)}): {path}")
            runtime = report["diagnostics"]["runtime_summary"]
            detail = runtime.get("candidate_detail")
            if mode == "on":
                if detail is None or detail["status"] != "VALID":
                    raise ValueError(f"Missing or overlapping detailed timers: {path}")
                detail_reports.append(detail)
                detail_trials.append(trial)
            elif detail is not None:
                raise ValueError(f"Baseline unexpectedly contains detailed timers: {path}")
            rows.append({
                "trial": trial, "mode": mode, "source": path.name,
                "total_wall_seconds": runtime["total_wall_seconds"],
                "candidate_seconds": runtime["sections_seconds"]["candidate_slot_generation"],
                "mean_decision_ms": runtime["decision_wall_seconds"]["mean"] * 1000,
                "p95_decision_ms": runtime["decision_wall_seconds"]["p95"] * 1000,
            })
    off = [row["total_wall_seconds"] for row in rows if row["mode"] == "off"]
    on = [row["total_wall_seconds"] for row in rows if row["mode"] == "on"]
    median_off, median_on = statistics.median(off), statistics.median(on)
    overhead = median_on / median_off - 1 if median_off else None
    # Select an actual median-duration detailed run; never synthesize additive
    # stage times from independently computed medians.
    representative = sorted(
        zip(detail_trials, detail_reports), key=lambda pair: pair[1]["pipeline_seconds"]
    )[len(detail_reports) // 2]
    result = {
        "status": "VALID", "equivalence": "all metrics and non-timing decision fields identical",
        "conditions": reference[0], "trials_per_mode": repeats,
        "baseline_median_seconds": median_off, "detailed_median_seconds": median_on,
        "instrumentation_overhead_ratio": overhead,
        "overhead_within_5_percent": overhead is not None and overhead <= 0.05,
        "cprofile_recommended": max(d["unattributed_percent"] for d in detail_reports) > 5,
        "representative_detail_source": f"trial_{representative[0]:02}_on.json",
        "representative_candidate_detail": representative[1], "trials": rows,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    with (output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from v1.candidate_profiling import write_candidate_detail_csv
    write_candidate_detail_csv(representative[1], output_dir / "candidate_detail.csv")
    return result


def _evaluation_arguments(args, output, detailed=False):
    command = [
        "-m", "v6.evaluate_v6", "--policy", "candidate_dqn",
        "--model-path", str(args.model_path), "--device", "cpu",
        "--seed", str(args.seed), "--warmup-days", str(args.warmup_days),
        "--arrival-cutoff", str(args.arrival_cutoff), "--output", str(output),
    ]
    if detailed:
        command.append("--candidate-profile-detail")
    return command


def run_cprofile(args):
    """Independent extra run, excluded from on/off timing comparison."""
    import pstats

    target = args.output_dir / "diagnostic.pstats"
    subprocess.run(
        [sys.executable, "-m", "cProfile", "-o", str(target)]
        + _evaluation_arguments(args, args.output_dir / "diagnostic_report.json"),
        cwd=ROOT, check=True,
    )
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(str(target)).stats.items():
        rows.append({
            "filename": filename, "line": line, "function": name,
            "primitive_calls": primitive, "calls": calls,
            "self_seconds": own, "cumulative_seconds": cumulative,
        })
    rows.sort(key=lambda row: row["self_seconds"], reverse=True)
    with (args.output_dir / "diagnostic_functions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "artifacts/v6/logs/candidate_dqn.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/v6/profiling/candidate_detail_seed42")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-days", type=float, default=2)
    parser.add_argument("--arrival-cutoff", type=float, default=288)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--cprofile-if-needed", action="store_true", help="Allow an extra independent cProfile evaluation if unattributed time exceeds 5%%")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.model_path = args.model_path.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.summarize_only:
        # Preflight every output before starting a potentially long run.
        expected = [args.output_dir / f"trial_{i:02}_{mode}.json" for i in range(1, args.repeats + 1) for mode in ("off", "on")]
        if any(path.exists() for path in expected):
            parser.error("Trial reports already exist; choose another --output-dir or use --summarize-only")
        model_hash = hashlib.sha256(args.model_path.read_bytes()).hexdigest()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for trial in range(1, args.repeats + 1):
            order = ("off", "on") if trial % 2 else ("on", "off")
            for mode in order:
                if hashlib.sha256(args.model_path.read_bytes()).hexdigest() != model_hash:
                    raise ValueError("Model checkpoint changed during benchmark")
                print(f"Trial {trial}/{args.repeats}: detail={mode}", flush=True)
                output = args.output_dir / f"trial_{trial:02}_{mode}.json"
                subprocess.run(
                    [sys.executable] + _evaluation_arguments(args, output, mode == "on"),
                    cwd=ROOT, check=True,
                )
    result = summarize(args.output_dir, args.repeats)
    print(json.dumps({key: result[key] for key in (
        "status", "baseline_median_seconds", "detailed_median_seconds",
        "instrumentation_overhead_ratio", "overhead_within_5_percent", "cprofile_recommended",
    )}, ensure_ascii=False), flush=True)
    if args.cprofile_if_needed and result["cprofile_recommended"]:
        print("Starting independent cProfile diagnostic; excluded from timing comparison", flush=True)
        run_cprofile(args)


if __name__ == "__main__":
    main()
