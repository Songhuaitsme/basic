"""Serial, same-checkpoint V6/V7 performance and exact-decision comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from v6.profile_candidates import PAIR_FIELDS, _decisions


ROOT = Path(__file__).resolve().parents[1]
COMMON_FIELDS = tuple(k for k in PAIR_FIELDS if k not in {"system_version", "code_hash", "config_hash"})


def _current_receipt(version, args, model_hash):
    """Fingerprint current code/config without running a simulation.

    Keep this path list in step with run_evaluation's code_files. Runtime
    construction supplies the version-specific list (including V7 sources).
    This module itself is not part of evaluation code_hash, so adding resume
    does not invalidate already completed trials.
    """
    import importlib
    import networkx as nx
    import numpy as np
    import torch
    from shared import config
    from v1.evaluate_v1 import _canonical_hash

    factory = getattr(importlib.import_module(f"{version}.{version}_runtime"), f"create_{version}_runtime")
    runtime = factory(
        policy_name="candidate_dqn", random_seed=args.seed, device="cpu",
        forecast_end_sim=(args.warmup_days * config.TRAFFIC_DAY_DURATION_IN_SIM
                          + args.arrival_cutoff + config.V1_MAX_FORECAST_LOOKAHEAD_SIM),
        candidate_chunk_size=65536,
    )
    package = ROOT / "v1"
    code_files = tuple(package / path for path in (
        "v1_runtime.py", "evaluate_v1.py", "scheduler/v1_scheduler.py",
        "scheduler/candidate_generator.py", "scheduler/queue_manager.py",
        "scheduler/resource_calendar.py", "accounting/energy.py",
        "evaluation_v1/runner.py", "evaluation_v1/diagnostics.py",
        "profiling.py", "candidate_profiling.py", "learning/candidate_dqn.py",
        "simulation/state_machine.py",
    )) + tuple(runtime.evaluation_code_files)
    view = {
        name: getattr(config, name) for name in dir(config)
        if name.isupper() and isinstance(getattr(config, name),
            (str, int, float, bool, tuple, list, dict, type(None)))
    }
    view["V1_SOFT_TARDINESS_WEIGHT"] = config.V1_SOFT_TARDINESS_WEIGHT
    view["V1_FLEXIBLE_TARDINESS_WEIGHT"] = config.V1_FLEXIBLE_TARDINESS_WEIGHT
    view["SYSTEM_VERSION"] = {"v6": "6.0", "v7": "7.0"}[version]
    view.update(runtime.config_view_overrides)
    start = args.warmup_days * config.TRAFFIC_DAY_DURATION_IN_SIM
    return {
        "system_version": view["SYSTEM_VERSION"], "seed": args.seed,
        "warmup_days": args.warmup_days, "evaluation_start_sim": start,
        "arrival_cutoff_sim": start + args.arrival_cutoff,
        "model_hash": model_hash,
        "code_hash": hashlib.sha256(b"".join(path.read_bytes() for path in code_files)).hexdigest(),
        "config_hash": _canonical_hash(view),
        "topology_hash": _canonical_hash(nx.node_link_data(runtime.infrastructure.topo_manager.graph)),
        "dependency_lock_hash": _canonical_hash({"numpy": np.__version__, "torch": torch.__version__}),
    }


def prepare_resume(args, model_hash, receipt_factory=None):
    """Validate all existing trials before launching work or moving partials."""
    receipt_factory = receipt_factory or _current_receipt
    receipts, completed, partials = {}, set(), []
    common = None
    for trial in range(1, args.repeats + 1):
        for version in ("v6", "v7"):
            path = args.output_dir / f"trial_{trial:02}_{version}.json"
            if not path.exists():
                continue
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeError):
                partials.append(path)
                continue
            if not isinstance(report, dict) or any(
                name not in report for name in ("status", "metadata", "metrics", "decision_records", "diagnostics")
            ):
                partials.append(path)
                continue
            if report["status"] != "VALID":
                raise ValueError(f"Existing evaluation is invalid; inspect it before resuming: {path}")
            if version not in receipts:
                receipts[version] = receipt_factory(version, args, model_hash)
            changed = [k for k, value in receipts[version].items() if report["metadata"].get(k) != value]
            if changed:
                raise ValueError(f"Cannot reuse {path.name}: current {', '.join(changed)} differ; use a new --output-dir")
            runtime = report["diagnostics"].get("runtime_summary", {})
            if "candidate_detail" in runtime or runtime.get("path_search_cache", {}).get("enabled") != (version == "v7"):
                raise ValueError(f"Wrong profiling/cache mode in existing trial: {path}")
            # Read these now, rather than after hours of remaining evaluations.
            for name in ("total_wall_seconds", "sections_seconds", "decision_wall_seconds"):
                if name not in runtime:
                    raise ValueError(f"Incomplete runtime summary in existing trial: {path}")
            signature = ({k: report["metadata"][k] for k in COMMON_FIELDS}, report["metrics"], _decisions(report))
            if common is None:
                common = signature
            elif common != signature:
                raise ValueError(f"Existing trials differ in inputs, metrics or decisions: {path}")
            completed.add((trial, version))
    for path in partials:
        backup = path.with_name(path.name + f".interrupted-{time.time_ns()}")
        path.rename(backup)
        print(f"Preserved incomplete report: {backup.name}; this trial will rerun", flush=True)
    return completed


def summarize(output_dir, repeats):
    reference = None
    versions = {}
    rows = []
    cache_receipts = []
    for trial in range(1, repeats + 1):
        for version in ("v6", "v7"):
            path = output_dir / f"trial_{trial:02}_{version}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["status"] != "VALID":
                raise ValueError(f"Invalid evaluation: {path}")
            metadata = report["metadata"]
            if metadata["system_version"] != {"v6": "6.0", "v7": "7.0"}[version]:
                raise ValueError(f"Wrong version: {path}")
            full_conditions = {k: metadata[k] for k in PAIR_FIELDS}
            if version in versions and versions[version] != full_conditions:
                raise ValueError(f"Code/config/input changed between {version} repetitions: {path}")
            versions[version] = full_conditions
            signature = ({k: metadata[k] for k in COMMON_FIELDS}, report["metrics"], _decisions(report))
            if reference is None:
                reference = signature
            if signature != reference:
                changed = [name for name, a, b in zip(("inputs/model", "metrics", "decisions"), reference, signature) if a != b]
                raise ValueError(f"V6/V7 equivalence failed ({', '.join(changed)}): {path}")
            runtime = report["diagnostics"]["runtime_summary"]
            if "candidate_detail" in runtime:
                raise ValueError(f"Performance comparison must disable detailed profiling: {path}")
            cache = runtime["path_search_cache"]
            if cache["enabled"] != (version == "v7"):
                raise ValueError(f"Wrong path-cache state: {path}")
            if version == "v7":
                cache_receipts.append({"trial": trial, **cache})
            provider = cache.get("provider", {})
            rows.append({
                "trial": trial, "version": version, "source": path.name,
                "total_wall_seconds": runtime["total_wall_seconds"],
                "candidate_seconds": runtime["sections_seconds"]["candidate_slot_generation"],
                "mean_decision_ms": runtime["decision_wall_seconds"]["mean"] * 1000,
                "p95_decision_ms": runtime["decision_wall_seconds"]["p95"] * 1000,
                "path_cache_hits": provider.get("path_cache_hits", 0),
                "path_cache_misses": provider.get("path_cache_misses", 0),
                "filtered_graph_builds": provider.get("filtered_graph_builds", 0),
                "path_searches": provider.get("path_searches", 0),
                "single_source_runs": provider.get("single_source_runs", 0),
                "task_grid_hits": cache.get("retry_grids", {}).get("task_grid_hits", 0),
            })
    medians = {}
    for version in ("v6", "v7"):
        group = [row for row in rows if row["version"] == version]
        medians[version] = {key: statistics.median(row[key] for row in group) for key in (
            "total_wall_seconds", "candidate_seconds", "mean_decision_ms", "p95_decision_ms",
        )}
    result = {
        "status": "VALID", "equivalence": "all metrics and non-timing decision fields identical",
        "trials_per_version": repeats, "conditions_by_version": versions,
        "median": medians,
        "total_time_reduction_ratio": 1 - medians["v7"]["total_wall_seconds"] / medians["v6"]["total_wall_seconds"],
        "candidate_time_reduction_ratio": 1 - medians["v7"]["candidate_seconds"] / medians["v6"]["candidate_seconds"],
        "cache_statistics": cache_receipts, "trials": rows,
    }
    (output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "artifacts/v6/logs/candidate_dqn.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/v7/evaluation/comparison_v6_v7_seed42")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-days", type=float, default=2)
    parser.add_argument("--arrival-cutoff", type=float, default=288)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summarize-only", action="store_true")
    mode.add_argument("--resume", action="store_true", help="Validate and skip completed trials; run only missing/interrupted trials")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output_dir = args.output_dir.resolve()
    args.model_path = args.model_path.resolve()
    if not args.summarize_only:
        paths = [args.output_dir / f"trial_{i:02}_{version}.json" for i in range(1, args.repeats + 1) for version in ("v6", "v7")]
        if not args.resume and any(p.exists() for p in paths):
            parser.error("Trial reports already exist; use --resume, another --output-dir or --summarize-only")
        model_hash = hashlib.sha256(args.model_path.read_bytes()).hexdigest()
        completed = prepare_resume(args, model_hash) if args.resume else set()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for trial in range(1, args.repeats + 1):
            order = ("v6", "v7") if trial % 2 else ("v7", "v6")
            for version in order:
                if (trial, version) in completed:
                    print(f"Trial {trial}/{args.repeats}: {version} already VALID; skipping", flush=True)
                    continue
                if hashlib.sha256(args.model_path.read_bytes()).hexdigest() != model_hash:
                    raise ValueError("Model checkpoint changed during comparison")
                print(f"Trial {trial}/{args.repeats}: {version}", flush=True)
                subprocess.run([
                    sys.executable, "-m", f"{version}.evaluate_{version}",
                    "--policy", "candidate_dqn", "--model-path", str(args.model_path),
                    "--seed", str(args.seed), "--warmup-days", str(args.warmup_days),
                    "--arrival-cutoff", str(args.arrival_cutoff), "--device", "cpu",
                    "--output", str(args.output_dir / f"trial_{trial:02}_{version}.json"),
                ], cwd=ROOT, check=True)
    result = summarize(args.output_dir, args.repeats)
    print(json.dumps({key: result[key] for key in ("status", "median", "total_time_reduction_ratio", "candidate_time_reduction_ratio")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
