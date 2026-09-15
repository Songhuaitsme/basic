"""Single-model V2 evaluation focused on compute-electricity coordination.

This entry point runs only Candidate DQN.  Its Earliest Feasible comparison is
captured from the same candidate set and reservation snapshot as each DQN
decision; it does not launch a second baseline simulation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable, Mapping, Sequence

import torch

from shared import config
from v1.evaluate_v1 import _jsonable, run_evaluation
from v2.coordination_metrics import (
    ADJUSTMENT_FIELDS,
    COUNTERFACTUAL_LABEL,
    GROUP_METRIC_FIELDS,
    SLA_METRIC_FIELDS,
    build_adjustment_metrics,
    build_green_coverage_gain_distribution,
    build_mutually_exclusive_coordination_groups,
    build_requested_coordination_groups,
    build_sla_type_metrics,
    enrich_adjustments,
)


DEFAULT_SEED = 42
REQUIRED_TARIFF_MODE = "tou_region"
MODEL_PATTERN = "*tou_region*.pt"
TASK_FIELDS = (
    "task_id",
    "source_node",
    "target_node",
    "arrival_time",
    "earliest_start",
    "selected_start",
    "compute_end",
    "active_wait",
    "cpu_demand",
    "execution_duration",
    "energy_consumption",
    "earliest_cost",
    "selected_cost",
    "cost_saving",
    "earliest_green_coverage",
    "selected_green_coverage",
    "green_coverage_gain",
    "earliest_green_energy",
    "selected_green_energy",
    "green_energy_gain",
    "earliest_green_absorption",
    "selected_green_absorption",
    "green_absorption_gain",
    "sla_type",
    "sla_satisfied",
    "earliest_sla_satisfied",
    "earliest_target_node",
    "selected_region",
    "earliest_region",
    *ADJUSTMENT_FIELDS,
    "comparison_basis",
    "final_state",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_policy_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise ValueError(f"checkpoint has no model_state_dict: {path}")
    if not isinstance(checkpoint.get("metadata"), Mapping):
        raise ValueError(f"checkpoint has no metadata: {path}")
    return dict(checkpoint)


def discover_v2_policy_model(repo_root: Path) -> Path:
    """Select the highest-step formal regional-pricing policy checkpoint."""

    formal_dir = repo_root / "artifacts" / "v2" / "formal"
    candidates = sorted(
        path for path in formal_dir.glob(MODEL_PATTERN)
        if path.is_file() and not path.name.endswith(".last.pt")
    )
    if not candidates:
        raise FileNotFoundError(
            f"no formal {MODEL_PATTERN} checkpoint found under {formal_dir}"
        )
    ranked = []
    failures = []
    for path in candidates:
        try:
            checkpoint = _load_policy_checkpoint(path)
            tariff_mode = (checkpoint.get("run_config") or {}).get(
                "V1_TARIFF_MODE"
            )
            if tariff_mode != REQUIRED_TARIFF_MODE:
                failures.append(
                    f"{path}: checkpoint tariff mode is {tariff_mode!r}"
                )
                continue
            steps = int(checkpoint.get("training_steps") or -1)
            ranked.append((steps, path.stat().st_mtime_ns, path.name, path))
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            failures.append(f"{path}: {exc}")
    if not ranked:
        raise ValueError(
            "formal V2 policy checkpoints are unreadable: " + "; ".join(failures)
        )
    return max(ranked)[-1].resolve()


def resolve_model_path(repo_root: Path, model_path: Path | None) -> Path:
    path = (
        discover_v2_policy_model(repo_root)
        if model_path is None
        else (model_path if model_path.is_absolute() else repo_root / model_path)
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {path}")
    if path.name.endswith(".last.pt"):
        raise ValueError("training-resume .last.pt files are not policy checkpoints")
    checkpoint = _load_policy_checkpoint(path)
    tariff_mode = (checkpoint.get("run_config") or {}).get("V1_TARIFF_MODE")
    if tariff_mode != REQUIRED_TARIFF_MODE:
        raise ValueError(
            f"model checkpoint uses {tariff_mode!r}; "
            f"{REQUIRED_TARIFF_MODE!r} is required"
        )
    return path


def _json_cell(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping], fields: Sequence[str]) -> None:
    extras = sorted({key for row in rows for key in row if key not in fields})
    fieldnames = tuple(fields) + tuple(extras)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: _json_cell(value) for key, value in row.items()}
            for row in rows
        )


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _difference(selected, earliest):
    if selected is None or earliest is None:
        return None
    return float(selected) - float(earliest)


def build_task_rows(report) -> list[dict]:
    rows = []
    for record in report.task_records:
        selected_start = record.compute_start_sim
        earliest_start = record.earliest_compute_start_sim
        active_wait = _difference(selected_start, earliest_start)
        selected_sla = (
            record.final_state == "Completed"
            and record.start_delay_sim is not None
            and record.start_delay_sim <= record.latest_start_limit_sim + 1e-12
        )
        earliest_sla = (
            earliest_start is not None
            and earliest_start - record.arrival_time_sim
            <= record.latest_start_limit_sim + 1e-12
        )
        earliest_cost = record.earliest_candidate_marginal_system_cost_yuan
        selected_cost = record.candidate_marginal_system_cost_yuan
        earliest_green = record.earliest_green_coverage
        selected_green = record.selected_green_coverage
        earliest_green_energy = (
            record.earliest_candidate_marginal_green_energy_mwh
        )
        selected_green_energy = record.candidate_marginal_green_energy_mwh
        earliest_absorption = record.earliest_green_absorption_delta
        selected_absorption = record.selected_green_absorption_delta
        row = {
            "task_id": record.task_id,
            "source_node": record.source_node,
            "target_node": record.target_node,
            "arrival_time": record.arrival_time_sim,
            "earliest_start": earliest_start,
            "selected_start": selected_start,
            "compute_end": record.compute_end_sim,
            "active_wait": active_wait,
            "cpu_demand": record.cpu_demand,
            "execution_duration": record.execution_duration_sim,
            "energy_consumption": record.task_energy_mwh,
            "earliest_cost": earliest_cost,
            "selected_cost": selected_cost,
            "cost_saving": (
                None if earliest_cost is None or selected_cost is None
                else earliest_cost - selected_cost
            ),
            "earliest_green_coverage": earliest_green,
            "selected_green_coverage": selected_green,
            "green_coverage_gain": _difference(selected_green, earliest_green),
            "earliest_green_energy": earliest_green_energy,
            "selected_green_energy": selected_green_energy,
            "green_energy_gain": _difference(
                selected_green_energy, earliest_green_energy
            ),
            "earliest_green_absorption": earliest_absorption,
            "selected_green_absorption": selected_absorption,
            "green_absorption_gain": _difference(
                selected_absorption, earliest_absorption
            ),
            "sla_type": record.sla_type,
            "sla_satisfied": selected_sla,
            "earliest_sla_satisfied": earliest_sla,
            "final_state": record.final_state,
            "terminal_reason": record.terminal_reason,
            "preferred_start_limit": record.preferred_start_limit_sim,
            "latest_start_limit": record.latest_start_limit_sim,
            "realized_cost": record.task_attributed_cost_yuan,
            "realized_green_energy": record.task_attributed_green_energy_mwh,
            "realized_green_coverage": (
                record.task_attributed_green_energy_mwh / record.task_energy_mwh
                if record.task_energy_mwh not in (None, 0.0) else None
            ),
            "earliest_target_node": record.earliest_target_node,
        }
        rows.append(row)
    return enrich_adjustments(rows)


def _mean(rows: Iterable[Mapping], field: str):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def _metric_value(metric):
    if metric is None:
        return None
    return metric.value


def build_system_metrics(report, task_rows: Sequence[Mapping]) -> dict:
    total = len(task_rows)
    completed = sum(row["final_state"] == "Completed" for row in task_rows)
    scheduled = [row for row in task_rows if row["selected_start"] is not None]
    sla_violations = sum(not row["sla_satisfied"] for row in task_rows)
    active_wait = [row for row in scheduled if row["temporal_adjustment"]]
    total_energy = math.fsum(
        float(row["energy_consumption"] or 0.0) for row in task_rows
    )
    total_green = (
        0.0 if report.accounting_report is None
        else report.accounting_report.total_task_attributed_green_energy_mwh
    )
    cost_savings = [
        float(row["cost_saving"])
        for row in scheduled if row["cost_saving"] is not None
    ]
    green_energy_gains = [
        float(row["green_energy_gain"])
        for row in scheduled if row["green_energy_gain"] is not None
    ]
    earliest_sla_rate = _ratio(
        sum(row["earliest_sla_satisfied"] for row in scheduled), len(scheduled)
    )
    selected_sla_rate = _ratio(
        sum(row["sla_satisfied"] for row in scheduled), len(scheduled)
    )
    metrics = {
        "total_tasks": total,
        "completed_tasks": completed,
        "completion_rate": _ratio(completed, total),
        "sla_violation_tasks": sla_violations,
        "sla_violation_rate": _ratio(sla_violations, total),
        "total_cost": (
            None if report.metrics is None
            else report.metrics.total_economic_cost_yuan
        ),
        "total_energy_consumption": total_energy,
        "total_green_energy_used": total_green,
        "task_green_coverage_rate": (
            None if report.metrics is None
            else _metric_value(report.metrics.completed_task_green_coverage)
        ),
        "system_green_absorption_rate": (
            None if report.metrics is None
            else _metric_value(report.metrics.system_green_absorption_rate)
        ),
        "active_wait_tasks": len(active_wait),
        "active_wait_ratio": _ratio(len(active_wait), len(scheduled)),
        "average_active_wait": _mean(active_wait, "active_wait"),
        "reservation_success_rate": _ratio(len(scheduled), total),
        "total_cost_saving": math.fsum(cost_savings),
        "average_cost_saving_per_task": (
            statistics.fmean(cost_savings) if cost_savings else None
        ),
        "average_green_coverage_gain": _mean(
            scheduled, "green_coverage_gain"
        ),
        "total_green_energy_gain": math.fsum(green_energy_gains),
        "average_green_absorption_gain": _mean(
            scheduled, "green_absorption_gain"
        ),
        "average_additional_wait": _mean(scheduled, "active_wait"),
        "sla_change": (
            None if selected_sla_rate is None or earliest_sla_rate is None
            else selected_sla_rate - earliest_sla_rate
        ),
        "scheduled_task_denominator": len(scheduled),
    }
    metrics.update(build_adjustment_metrics(task_rows))
    metrics["definitions"] = {
        "counterfactual": COUNTERFACTUAL_LABEL,
        "candidate_values": (
            "earliest/selected cost, green coverage, green energy and green "
            "absorption are decision-time candidate estimates"
        ),
        "realized_values": (
            "total cost, task energy and attributed green energy are finalized "
            "post-completion accounting values"
        ),
        "adjustment_ratio_denominator": "tasks with a committed reservation",
        "reservation_success_rate": "committed reservations / arrived tasks",
        "sla_violation": (
            "task did not complete within its absolute latest-start limit"
        ),
        "average_active_wait": "mean over tasks whose active_wait is positive",
        "average_additional_wait": "mean over all committed tasks",
    }
    return metrics


def select_representative_tasks(rows: Sequence[Mapping], limit: int = 5) -> list[dict]:
    scheduled = [row for row in rows if row.get("selected_start") is not None]

    def score(row):
        return (
            int(row["region_temporal_adjustment"]),
            int(row["node_temporal_adjustment"]),
            int(row["region_adjustment"]),
            int(row["node_adjustment"]),
            int(row["temporal_adjustment"]),
            int(row["sla_type"] == "Flexible"),
            float(row.get("green_coverage_gain") or 0.0),
            float(row.get("cpu_demand") or 0.0),
            str(row["task_id"]),
        )

    return [dict(row) for row in sorted(scheduled, key=score, reverse=True)[:limit]]


def _format_number(value, digits=6):
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _format_percent(value):
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.2f}%"


def print_summary(metrics: Mapping, representatives: Sequence[Mapping]) -> None:
    table = (
        ("Total Tasks", metrics["total_tasks"], "count"),
        ("Completion Rate", metrics["completion_rate"], "percent"),
        ("SLA Violation Rate", metrics["sla_violation_rate"], "percent"),
        ("Reservation Success Rate", metrics["reservation_success_rate"], "percent"),
        ("Total Cost", metrics["total_cost"], "number"),
        ("Total Energy Consumption", metrics["total_energy_consumption"], "number"),
        ("Total Green Energy Used", metrics["total_green_energy_used"], "number"),
        ("Task Green Coverage", metrics["task_green_coverage_rate"], "percent"),
        ("System Green Absorption", metrics["system_green_absorption_rate"], "percent"),
        ("Node Adjustment Tasks", metrics["node_adjustment_tasks"], "count"),
        ("Node Adjustment Ratio", metrics["node_adjustment_ratio"], "percent"),
        ("Region Adjustment Tasks", metrics["region_adjustment_tasks"], "count"),
        ("Region Adjustment Ratio", metrics["region_adjustment_ratio"], "percent"),
        ("Temporal Adjustment Tasks", metrics["temporal_adjustment_tasks"], "count"),
        ("Temporal Adjustment Ratio", metrics["temporal_adjustment_ratio"], "percent"),
        ("Region + Temporal Tasks", metrics["region_temporal_adjustment_tasks"], "count"),
        ("Region + Temporal Ratio", metrics["region_temporal_adjustment_ratio"], "percent"),
        ("Avg Green Coverage Gain (per-decision counterfactual)", metrics["average_green_coverage_gain"], "signed_percent"),
        ("Total Green Energy Gain (per-decision counterfactual)", metrics["total_green_energy_gain"], "signed"),
        ("Total Cost Saving (per-decision counterfactual)", metrics["total_cost_saving"], "number"),
        ("Average Additional Wait (per-decision counterfactual)", metrics["average_additional_wait"], "number"),
    )
    print("| Metric | Value |")
    print("| --- | ---: |")
    for label, value, kind in table:
        if kind == "percent":
            rendered = _format_percent(value)
        elif kind == "signed_percent":
            rendered = "N/A" if value is None else f"{100.0 * float(value):+.2f}%"
        elif kind == "signed":
            rendered = "N/A" if value is None else f"{float(value):+.6f}"
        elif kind == "count":
            rendered = str(int(value))
        else:
            rendered = _format_number(value)
        print(f"| {label} | {rendered} |")

    for row in representatives:
        if row["region_temporal_adjustment"]:
            coordination = "region + temporal adjustment"
        elif row["region_adjustment"]:
            coordination = "region adjustment"
        elif row["node_temporal_adjustment"]:
            coordination = "within-region node + temporal adjustment"
        elif row["node_adjustment"]:
            coordination = "within-region node adjustment"
        elif row["temporal_adjustment"]:
            coordination = "temporal adjustment"
        else:
            coordination = "no obvious adjustment"
        print()
        print(f"Task ID: {row['task_id']}")
        print(f"SLA Type: {row['sla_type']}")
        print(f"Source Node: {row['source_node']}")
        print(f"Selected Node: {row['target_node']}")
        print(f"Arrival Time: {_format_number(row['arrival_time'])}")
        print(f"Earliest Start: {_format_number(row['earliest_start'])}")
        print(f"Selected Start: {_format_number(row['selected_start'])}")
        print(f"Active Wait: {_format_number(row['active_wait'])}")
        print(f"Earliest Cost: {_format_number(row['earliest_cost'])}")
        print(f"Selected Cost: {_format_number(row['selected_cost'])}")
        print(f"Cost Saving: {_format_number(row['cost_saving'])}")
        print(f"Earliest Green Coverage: {_format_percent(row['earliest_green_coverage'])}")
        print(f"Selected Green Coverage: {_format_percent(row['selected_green_coverage'])}")
        print(f"Green Coverage Gain: {_format_percent(row['green_coverage_gain'])}")
        print(f"Earliest Green Energy: {_format_number(row['earliest_green_energy'])}")
        print(f"Selected Green Energy: {_format_number(row['selected_green_energy'])}")
        print(f"Green Energy Gain: {_format_number(row['green_energy_gain'])}")
        print(f"SLA Satisfied: {row['sla_satisfied']}")
        print(f"Coordination Type: {coordination}")


def _config_snapshot() -> dict:
    return {
        name: _jsonable(getattr(config, name))
        for name in dir(config)
        if name.isupper()
        and isinstance(
            getattr(config, name),
            (str, int, float, bool, tuple, list, dict, type(None)),
        )
    }


def _default_output_dir(seed: int) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts/v2/evaluation") / (
        f"coordination_v2_seed{seed}_{timestamp}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the formal V2 Candidate DQN for compute-electricity coordination"
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--arrival-cutoff",
        type=float,
        default=config.TRAFFIC_DAY_DURATION_IN_SIM,
    )
    parser.add_argument(
        "--warmup-days",
        type=float,
        default=config.WARMUP_DAYS,
    )
    parser.add_argument("--safety-cap", type=int, default=1_000_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=65_536,
    )
    parser.add_argument(
        "--audit", choices=("full", "periodic", "final"), default="periodic"
    )
    parser.add_argument("--audit-interval", type=int, default=500)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.arrival_cutoff <= 0.0:
        parser.error("--arrival-cutoff must be positive")
    if args.warmup_days < 0.0:
        parser.error("--warmup-days must be non-negative")
    if args.safety_cap <= 0:
        parser.error("--safety-cap must be positive")
    if args.candidate_chunk_size <= 0:
        parser.error("--candidate-chunk-size must be positive")
    if args.audit_interval <= 0:
        parser.error("--audit-interval must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    model_path = resolve_model_path(repo_root, args.model_path)
    checkpoint = _load_policy_checkpoint(model_path)
    output_dir = args.output_dir or _default_output_dir(args.seed)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    if output_dir.exists():
        raise FileExistsError(
            f"evaluation output already exists: {output_dir}; choose a new --output-dir"
        )
    output_dir.mkdir(parents=True)

    checkpoint_run_config = _jsonable(checkpoint.get("run_config") or {})
    manifest = {
        "status": "RUNNING",
        "generated_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, "-m", "v2.evaluate_coordination", *sys.argv[1:]],
        "seed": args.seed,
        "arrival_cutoff": args.arrival_cutoff,
        "warmup_days": args.warmup_days,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "model_training_steps": checkpoint.get("training_steps"),
        "model_training_seed": checkpoint.get("seed"),
        "checkpoint_metadata": _jsonable(checkpoint.get("metadata") or {}),
        "checkpoint_run_config": checkpoint_run_config,
        "evaluation_tariff_mode": config.V1_TARIFF_MODE,
        "training_tariff_mode": checkpoint_run_config.get("V1_TARIFF_MODE"),
        "epsilon": 0.0,
        "model_eval": True,
        "checkpoint_feature_scales_restored": False,
        "device": args.device,
        "candidate_chunk_size": args.candidate_chunk_size,
        "audit": args.audit,
        "audit_interval": args.audit_interval,
        "system_config": _config_snapshot(),
    }
    _write_json(output_dir / "evaluation_config.json", manifest)

    report = run_evaluation(
        "candidate_dqn",
        args.arrival_cutoff,
        args.seed,
        args.safety_cap,
        model_path,
        device=args.device,
        candidate_chunk_size=args.candidate_chunk_size,
        system_version="2.0",
        audit_mode=args.audit,
        audit_interval=args.audit_interval,
        warmup_days=args.warmup_days,
    )
    if report.status.value != "VALID" or report.unsettled_task_ids:
        manifest["status"] = report.status.value
        manifest["unsettled_task_ids"] = list(report.unsettled_task_ids)
        _write_json(output_dir / "evaluation_config.json", manifest)
        raise RuntimeError(
            f"evaluation is not valid: {report.status.value}; "
            f"unsettled={report.unsettled_task_ids}"
        )

    task_rows = build_task_rows(report)
    decision_rows = [_jsonable(asdict(item)) for item in report.decision_records]
    diagnostics = report.diagnostics or {}
    time_rows = list(diagnostics.get("time_records") or ())
    node_rows = list(diagnostics.get("node_records") or ())
    system_metrics = build_system_metrics(report, task_rows)
    coordination_groups = build_requested_coordination_groups(task_rows)
    exclusive_coordination_groups = (
        build_mutually_exclusive_coordination_groups(task_rows)
    )
    gain_distribution = build_green_coverage_gain_distribution(task_rows)
    sla_type_metrics = build_sla_type_metrics(task_rows)
    representatives = select_representative_tasks(task_rows)

    _write_csv(output_dir / "task_metrics.csv", task_rows, TASK_FIELDS)
    _write_csv(
        output_dir / "decision_metrics.csv",
        decision_rows,
        ("task_id", "status", "decision_id", "selected_candidate_id"),
    )
    _write_csv(
        output_dir / "time_metrics.csv",
        time_rows,
        (
            "time", "active_tasks", "computing_load", "total_energy_demand",
            "renewable_generation", "renewable_used", "electricity_price",
            "green_coverage_rate",
        ),
    )
    _write_csv(
        output_dir / "node_metrics.csv",
        node_rows,
        (
            "node_id", "assigned_tasks", "total_cpu_hours",
            "total_energy_consumption", "green_energy_available",
            "green_energy_used", "green_coverage_rate",
            "average_cpu_utilization", "peak_cpu_utilization",
            "electricity_cost",
        ),
    )
    _write_json(output_dir / "system_metrics.json", system_metrics)
    _write_csv(
        output_dir / "coordination_type_metrics.csv",
        coordination_groups,
        GROUP_METRIC_FIELDS,
    )
    _write_csv(
        output_dir / "coordination_type_metrics_mutually_exclusive.csv",
        exclusive_coordination_groups,
        GROUP_METRIC_FIELDS,
    )
    _write_csv(
        output_dir / "green_coverage_gain_distribution.csv",
        [gain_distribution],
        tuple(gain_distribution),
    )
    _write_csv(
        output_dir / "sla_type_metrics.csv",
        sla_type_metrics,
        SLA_METRIC_FIELDS,
    )
    _write_json(output_dir / "representative_tasks.json", representatives)

    report_payload = _jsonable(report)
    report_payload["cycle_result_count"] = len(report_payload["cycle_results"])
    report_payload["cycle_results"] = []
    _write_json(output_dir / "evaluation_report.json", report_payload)

    manifest.update({
        "status": "VALID",
        "completed_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir),
        "task_trace_hash": report.metadata.task_trace_hash,
        "config_hash": report.metadata.config_hash,
        "topology_hash": report.metadata.topology_hash,
        "exogenous_trace_hash": report.metadata.exogenous_trace_hash,
        "result_files": sorted(path.name for path in output_dir.iterdir()),
    })
    _write_json(output_dir / "evaluation_config.json", manifest)

    print_summary(system_metrics, representatives)
    print()
    print(json.dumps({
        "status": "VALID",
        "model_path": str(model_path),
        "seed": args.seed,
        "output_dir": str(output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
