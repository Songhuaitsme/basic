"""Run the five frozen V2 policies on identical per-seed workloads."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from shared import config
from v1.evaluate_v1 import _canonical_hash, _generate_trace, _jsonable
from v1.v1_runtime import create_v1_runtime
from v2.evaluate_coordination import resolve_model_path
from v2.export_five_policy_metrics import (
    EvaluationDataError,
    POLICIES,
    discover_reports,
    export_policy_artifacts,
    export_reports,
)


DEFAULT_MODEL_PATH = Path(
    "artifacts/v2/formal/"
    "candidate_dqn_seed7_layered_pool_600000_tou_region_from_scratch.pt"
)
DEFAULT_REFERENCE_REPORT = Path(
    "artifacts/v2/evaluation/coordination_v2_seed42/evaluation_report.json"
)
REFERENCE_FIELDS = (
    "system_version",
    "requirements_version",
    "algorithm_version",
    "candidate_mode",
    "seed",
    "arrival_cutoff_sim",
    "config_hash",
    "topology_hash",
    "task_trace_hash",
    "exogenous_trace_hash",
    "dependency_lock_hash",
    "tariff_mode",
)


def evaluation_command(
    *,
    policy: str,
    seed: int,
    output: Path,
    model_path: Path,
    arrival_cutoff: float,
    safety_cap: int,
    device: str,
    candidate_chunk_size: int,
    audit: str,
    audit_interval: int,
    report_mode: str,
    task_trace_input: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "v2.evaluate_v2",
        "--policy",
        policy,
        "--arrival-cutoff",
        str(arrival_cutoff),
        "--seed",
        str(seed),
        "--safety-cap",
        str(safety_cap),
        "--device",
        device,
        "--candidate-chunk-size",
        str(candidate_chunk_size),
        "--audit",
        audit,
        "--audit-interval",
        str(audit_interval),
        "--report-mode",
        report_mode,
        "--task-trace-input",
        str(task_trace_input),
        "--output",
        str(output),
    ]
    if policy == "candidate_dqn":
        command.extend(("--model-path", str(model_path)))
    return command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_reusable_report(
    path: Path,
    *,
    policy: str,
    seed: int,
    arrival_cutoff: float,
    model_hash: str,
) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata = report.get("metadata", {})
    reusable = (
        report.get("status") == "VALID"
        and not report.get("unsettled_task_ids")
        and metadata.get("system_version") == "2.0"
        and metadata.get("seed") == seed
        and metadata.get("arrival_cutoff_sim") == arrival_cutoff
    )
    if policy == "candidate_dqn":
        reusable = reusable and metadata.get("model_hash") == model_hash
    return reusable


def _load_reference(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(f"cannot read reference report {path}: {exc}") from exc
    if report.get("status") != "VALID" or report.get("unsettled_task_ids"):
        raise EvaluationDataError(f"reference report is not settled and VALID: {path}")
    if not report.get("task_records"):
        raise EvaluationDataError(f"reference report has no task records: {path}")
    return report


def _validate_against_reference(
    report_path: Path,
    reference: dict,
    *,
    policy: str,
    model_hash: str,
) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata") or {}
    reference_metadata = reference.get("metadata") or {}
    mismatches = [
        field for field in REFERENCE_FIELDS
        if metadata.get(field) != reference_metadata.get(field)
    ]
    if policy == "candidate_dqn" and metadata.get("model_hash") != model_hash:
        mismatches.append("model_hash")
    reference_ids = [
        item.get("task_id") for item in reference.get("task_records") or ()
    ]
    report_ids = [item.get("task_id") for item in report.get("task_records") or ()]
    if sorted(report_ids) != sorted(reference_ids):
        mismatches.append("task_id_set")
    reference_count = len(reference_ids)
    arrival_count = (report.get("metrics") or {}).get("arrival_count")
    if arrival_count != reference_count:
        mismatches.append("arrival_count")
    if mismatches:
        raise EvaluationDataError(
            f"{policy} does not match the coordination reference: "
            + ", ".join(dict.fromkeys(mismatches))
        )


def _prepare_fixed_task_trace(
    path: Path,
    *,
    seed: int,
    arrival_cutoff: float,
    candidate_chunk_size: int,
    reference: dict,
) -> dict:
    runtime = create_v1_runtime(
        policy_name="earliest_feasible",
        forecast_end_sim=(
            arrival_cutoff + config.V1_MAX_FORECAST_LOOKAHEAD_SIM
        ),
        random_seed=seed,
        device="cpu",
        candidate_chunk_size=candidate_chunk_size,
    )
    trace = _generate_trace(runtime, arrival_cutoff, seed)
    trace_hash = _canonical_hash([asdict(task) for task in trace])
    reference_metadata = reference.get("metadata") or {}
    reference_ids = [
        item.get("task_id") for item in reference.get("task_records") or ()
    ]
    trace_ids = [task.task_id for task in trace]
    if trace_hash != reference_metadata.get("task_trace_hash"):
        raise EvaluationDataError(
            "generated fixed task trace does not match coordination_v2_seed42"
        )
    if sorted(trace_ids) != sorted(reference_ids):
        raise EvaluationDataError(
            "generated fixed task IDs do not match coordination_v2_seed42"
        )
    payload = {
        "source_reference": str(DEFAULT_REFERENCE_REPORT),
        "seed": seed,
        "arrival_cutoff_sim": arrival_cutoff,
        "task_count": len(trace),
        "task_trace_hash": trace_hash,
        "tasks": [_jsonable(asdict(task)) for task in trace],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all five frozen policies with paired V2 workloads"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(42,))
    parser.add_argument(
        "--arrival-cutoff",
        type=float,
        default=config.TRAFFIC_DAY_DURATION_IN_SIM,
    )
    parser.add_argument("--safety-cap", type=int, default=1_000_000)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=DEFAULT_REFERENCE_REPORT,
        help="VALID coordination report whose exact task trace must be reused",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--candidate-chunk-size", type=int, default=65_536)
    parser.add_argument("--audit", choices=("full", "periodic", "final"), default="periodic")
    parser.add_argument("--audit-interval", type=int, default=500)
    parser.add_argument("--report-mode", choices=("compact", "full"), default="compact")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v2/evaluation/five_policy"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing VALID reports with matching seed and cutoff",
    )
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = tuple(dict.fromkeys(args.seeds))
    if not seeds:
        parser.error("at least one seed is required")
    if args.arrival_cutoff <= 0.0:
        parser.error("--arrival-cutoff must be positive")
    if args.safety_cap <= 0:
        parser.error("--safety-cap must be positive")
    if args.candidate_chunk_size <= 0:
        parser.error("--candidate-chunk-size must be positive")
    if args.audit_interval <= 0:
        parser.error("--audit-interval must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    model_path = resolve_model_path(repo_root, args.model_path)
    reference_path = (
        args.reference_report
        if args.reference_report.is_absolute()
        else repo_root / args.reference_report
    ).resolve()
    reference = _load_reference(reference_path)
    reference_metadata = reference.get("metadata") or {}
    if len(reference.get("task_records") or ()) != 2416:
        parser.error("reference report must contain exactly 2416 task records")
    if seeds != (reference_metadata.get("seed"),):
        parser.error("--seeds must exactly match the reference report seed")
    model_hash = _sha256(model_path)
    if reference_metadata.get("model_hash") != model_hash:
        parser.error("model hash does not match coordination_v2_seed42")

    commands = []
    for seed in seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        task_trace_input = seed_dir / f"fixed_task_trace_seed{seed}.json"
        for policy in POLICIES:
            policy_dir = seed_dir / policy
            output = policy_dir / f"{policy}_seed{seed}.json"
            commands.append((seed, policy, output, evaluation_command(
                policy=policy,
                seed=seed,
                output=output,
                model_path=model_path,
                arrival_cutoff=args.arrival_cutoff,
                safety_cap=args.safety_cap,
                device=args.device,
                candidate_chunk_size=args.candidate_chunk_size,
                audit=args.audit,
                audit_interval=args.audit_interval,
                report_mode=args.report_mode,
                task_trace_input=task_trace_input,
            )))

    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN",
            "evaluation_count": len(commands),
            "reference_report": str(reference_path),
            "reference_task_count": len(reference.get("task_records") or ()),
            "reference_task_trace_hash": reference_metadata.get("task_trace_hash"),
            "commands": [command for _, _, _, command in commands],
        }, ensure_ascii=False, indent=2))
        return

    for seed in seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        trace_payload = _prepare_fixed_task_trace(
            seed_dir / f"fixed_task_trace_seed{seed}.json",
            seed=seed,
            arrival_cutoff=args.arrival_cutoff,
            candidate_chunk_size=args.candidate_chunk_size,
            reference=reference,
        )
        print(json.dumps({
            "status": "FIXED_TASK_TRACE_READY",
            "seed": seed,
            "task_count": trace_payload["task_count"],
            "task_trace_hash": trace_payload["task_trace_hash"],
        }, ensure_ascii=False))

    for seed, policy, output, command in commands:
        if output.exists():
            if args.resume and _is_reusable_report(
                output,
                policy=policy,
                seed=seed,
                arrival_cutoff=args.arrival_cutoff,
                model_hash=model_hash,
            ):
                print(json.dumps({
                    "status": "REUSED",
                    "seed": seed,
                    "policy": policy,
                    "output": str(output),
                }, ensure_ascii=False))
                _validate_against_reference(
                    output,
                    reference,
                    policy=policy,
                    model_hash=model_hash,
                )
                export_policy_artifacts(output, output.parent)
                continue
            if not args.overwrite:
                raise FileExistsError(
                    f"output already exists: {output}; use --resume or --overwrite"
                )
        output.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps({
            "status": "RUNNING",
            "seed": seed,
            "policy": policy,
            "output": str(output),
        }, ensure_ascii=False), flush=True)
        subprocess.run(command, check=True)
        _validate_against_reference(
            output,
            reference,
            policy=policy,
            model_hash=model_hash,
        )
        export_policy_artifacts(output, output.parent)

    result = {
        "status": "VALID",
        "seeds": list(seeds),
        "evaluation_count": len(commands),
        "output_dir": str(args.output_dir),
        "reference_report": str(reference_path),
        "reference_task_count": len(reference.get("task_records") or ()),
        "reference_task_trace_hash": reference_metadata.get("task_trace_hash"),
    }
    if not args.no_export:
        source_dir = args.output_dir / "source_data"
        validation = export_reports(discover_reports(args.output_dir), source_dir)
        result["source_data_dir"] = str(source_dir)
        result["long_metric_row_count"] = validation["long_metric_row_count"]
        result["wide_metric_row_count"] = validation["wide_metric_row_count"]
        result["granularity_coverage"] = validation["granularity_coverage"]
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
