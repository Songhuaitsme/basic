"""Run paired full-system V7/V9 evaluations, then V9 E0-E3 ablations.

No evaluation is started on import. Run --phase overall first; inspect its
summary before running --phase ablation. Every seed uses one persisted task
trace and the same frozen DQN checkpoint in all versions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from shared import config
from v2.evaluate_five_policies import _prepare_fixed_task_trace


FIXED = ("seed", "model_hash", "task_trace_hash", "exogenous_trace_hash",
         "topology_hash", "dependency_lock_hash", "candidate_mode",
         "warmup_days", "arrival_cutoff_sim", "tariff_mode")


def _source_hash():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "shared" / "config.py"]
    for version in ("v1", "v3", "v4", "v5", "v6", "v7", "v8", "v9"):
        paths.extend(path for path in (root / version).rglob("*.py")
                     if "tests" not in path.parts and path.name != "evaluate_full_comparison.py")
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _paths(root, seed):
    base = root / f"seed_{seed}"
    return {"trace": base / "fixed_task_trace.json",
            "v7": base / "v7.json", "e3": base / "v9_e3.json",
            **{f"e{stage}": base / f"v9_e{stage}.json" for stage in range(3)}}


def _load(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "VALID":
        raise ValueError(f"Invalid evaluation: {path}")
    return report


def _check_pair(left, right):
    for key in FIXED:
        if left["metadata"].get(key) != right["metadata"].get(key):
            raise ValueError(f"Paired inputs differ at {key}")


def _metric(value):
    if isinstance(value, dict):
        return value.get("value") if value.get("status") == "VALID" else None
    return value


def _flatten(report):
    metrics = report["metrics"]
    runtime = report["diagnostics"]["runtime_summary"]
    result = {key: float(metrics[key]) for key in (
        "arrival_count", "completed_count", "expired_count", "failed_count",
        "total_economic_cost_yuan", "completed_cpu_hours",
    )}
    for key in ("completion_rate", "completed_task_green_coverage",
                "system_green_absorption_rate", "cost_yuan_per_completed_cpu_hour"):
        result[key] = _metric(metrics[key])
    for sla in ("Hard", "Soft", "Flexible"):
        for field in ("expired_rate", "start_delay_p95_sim"):
            result[f"{sla}_{field}"] = _metric(metrics["sla_metrics"][sla][field])
    result["mean_active_wait_sim"] = _metric(
        metrics["active_wait_metrics"]["mean_active_wait_sim"])
    result["candidate_generation_seconds"] = runtime["sections_seconds"]["candidate_slot_generation"]
    result["decision_mean_seconds"] = runtime["decision_wall_seconds"]["mean"]
    result["decision_p95_seconds"] = runtime["decision_wall_seconds"]["p95"]
    result["total_wall_seconds"] = runtime["total_wall_seconds"]
    result["search_totals"] = runtime.get("path_search_cache", {}).get("v9_search", {}).get("totals", {})
    return result


def summarize(root, seeds, phase):
    labels = ("v7", "e3") if phase == "overall" else ("e0", "e1", "e2", "e3")
    rows = []
    code_by_label = {}
    for seed in seeds:
        paths = _paths(root, seed)
        reports = {label: _load(paths[label]) for label in labels}
        reference = reports[labels[0]]
        trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
        for label, report in reports.items():
            _check_pair(reference, report)
            meta = report["metadata"]
            if meta["seed"] != seed or meta["task_trace_hash"] != trace["task_trace_hash"]:
                raise ValueError(f"Wrong seed or trace: {paths[label]}")
            signature = (meta["code_hash"], meta["config_hash"])
            if label in code_by_label and code_by_label[label] != signature:
                raise ValueError(f"Code/config changed across seeds for {label}")
            code_by_label[label] = signature
            rows.append({"seed": seed, "variant": label, **_flatten(report)})
    summary = {"phase": phase, "seeds": list(seeds), "rows": rows, "median": {},
               "paired_delta_from_reference": {}}
    reference_label = labels[0]
    for label in labels:
        group = [row for row in rows if row["variant"] == label]
        numeric = (key for key in group[0] if key not in ("seed", "variant", "search_totals"))
        summary["median"][label] = {}
        for key in numeric:
            values = [row[key] for row in group if row[key] is not None]
            summary["median"][label][key] = statistics.median(values) if values else None
        if label != reference_label:
            baseline = {row["seed"]: row for row in rows if row["variant"] == reference_label}
            summary["paired_delta_from_reference"][label] = {
                key: [row[key] - baseline[row["seed"]][key] for row in group
                      if row[key] is not None and baseline[row["seed"]][key] is not None]
                for key in summary["median"][label]
            }
    destination = root / f"{phase}_summary.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding="utf-8")
    print(f"Wrote {destination}", flush=True)
    return summary


def _run(label, path, trace, seed, args, model_hash):
    if path.exists():
        if not args.resume:
            raise FileExistsError(f"Report exists; use --resume or a new output directory: {path}")
        report = _load(path)
        payload = json.loads(trace.read_text(encoding="utf-8"))
        meta = report["metadata"]
        expected = {"seed": seed, "model_hash": model_hash,
                    "task_trace_hash": payload["task_trace_hash"],
                    "warmup_days": args.warmup_days,
                    "arrival_cutoff_sim": payload["arrival_cutoff_sim"]}
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Existing report has different inputs: {path}")
        print(f"Reusing {path}", flush=True)
        return
    if label == "v7":
        module, extra = "v7.evaluate_v7", []
    else:
        module, extra = "v9.evaluate_v9", ["--search-stage", label[1:]]
    command = [sys.executable, "-m", module, *extra,
               "--policy", "candidate_dqn", "--model-path", str(args.model_path),
               "--task-trace-input", str(trace), "--seed", str(seed),
               "--warmup-days", str(args.warmup_days),
               "--arrival-cutoff", str(args.arrival_cutoff),
               "--device", "cpu", "--audit", "periodic",
               "--report-mode", "compact", "--output", str(path)]
    print(f"Running {label}, seed {seed}: {path}", flush=True)
    subprocess.run(command, check=True)
    _load(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("overall", "ablation"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44, 45, 46))
    parser.add_argument("--warmup-days", type=float, default=2)
    parser.add_argument("--arrival-cutoff", type=float,
                        default=config.TRAFFIC_DAY_DURATION_IN_SIM)
    parser.add_argument("--model-path", type=Path,
                        default=Path("artifacts/v6/logs/candidate_dqn.pt"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v9/full_comparison"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if not args.summary_only:
        if not args.model_path.is_file():
            parser.error(f"checkpoint does not exist: {args.model_path}")
        model_hash = hashlib.sha256(args.model_path.read_bytes()).hexdigest()
        manifest = {"source_hash": _source_hash(), "model_hash": model_hash,
                    "seeds": list(args.seeds), "warmup_days": args.warmup_days,
                    "arrival_cutoff": args.arrival_cutoff}
        manifest_path = args.output_dir / "input_manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError(f"Code, model or evaluation inputs changed; use a new output directory: {manifest_path}")
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for seed in args.seeds:
            paths = _paths(args.output_dir, seed)
            trace = paths["trace"]
            if not trace.exists():
                _prepare_fixed_task_trace(trace, seed=seed,
                    arrival_cutoff=args.arrival_cutoff, candidate_chunk_size=65536,
                    warmup_days=args.warmup_days)
            else:
                payload = json.loads(trace.read_text(encoding="utf-8"))
                expected_end = args.warmup_days * config.TRAFFIC_DAY_DURATION_IN_SIM + args.arrival_cutoff
                if (payload["seed"] != seed or payload["warmup_days"] != args.warmup_days
                        or payload["arrival_cutoff_sim"] != expected_end):
                    raise ValueError(f"Existing trace has different inputs: {trace}")
            labels = ("v7", "e3") if args.phase == "overall" else ("e0", "e1", "e2")
            if args.phase == "ablation" and not paths["e3"].exists():
                raise FileNotFoundError(f"Run --phase overall first: {paths['e3']}")
            if args.phase == "ablation":
                prior = _load(paths["e3"])["metadata"]
                trace_hash = json.loads(trace.read_text(encoding="utf-8"))["task_trace_hash"]
                if (prior.get("seed") != seed or prior.get("model_hash") != model_hash
                        or prior.get("task_trace_hash") != trace_hash):
                    raise ValueError(f"Prior E3 report uses different inputs: {paths['e3']}")
            for label in labels:
                _run(label, paths[label], trace, seed, args, model_hash)
            paired = {label: _load(paths[label]) for label in
                      (("v7", "e3") if args.phase == "overall" else ("e0", "e1", "e2", "e3"))}
            first = paired[next(iter(paired))]
            for report in paired.values():
                _check_pair(first, report)
    summarize(args.output_dir, args.seeds, args.phase)


if __name__ == "__main__":
    main()
