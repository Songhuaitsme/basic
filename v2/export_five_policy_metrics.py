"""Export auditable, source-level metrics from V2 five-policy evaluations.

The raw JSON reports remain the source of truth.  This module validates that
the five reports for each seed are paired, then writes:

* ``metrics_long.csv``: one row per seed, policy, and metric;
* ``metrics_wide.csv``: one row per seed and metric, with five policy values;
* ``source_manifest.csv``: report provenance and pairing hashes; and
* ``task_metrics.csv`` / ``decision_metrics.csv``: task-level audit data;
* ``time_metrics.csv`` / ``node_metrics.csv`` / ``network_metrics.csv``:
  fine-grained resource observations;
* ``runtime_metrics.csv``: measured scheduling efficiency; and
* ``analysis_validation.json``: deterministic completeness checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping, Sequence

from v2.coordination_metrics import (
    ADJUSTMENT_FIELDS,
    GROUP_METRIC_FIELDS,
    SLA_METRIC_FIELDS,
    build_adjustment_metrics,
    build_green_coverage_gain_distribution,
    build_mutually_exclusive_coordination_groups,
    build_requested_coordination_groups,
    build_sla_type_metrics,
    enrich_adjustments,
)


POLICIES = (
    "earliest_feasible",
    "lowest_cost",
    "highest_green",
    "equal_weight",
    "candidate_dqn",
)

PAIRED_METADATA_FIELDS = (
    "system_version",
    "requirements_version",
    "algorithm_version",
    "task_schema_version",
    "candidate_schema_version",
    "model_schema_version",
    "metric_schema_version",
    "aggregation_schema_version",
    "candidate_mode",
    "seed",
    "arrival_cutoff_sim",
    "code_hash",
    "config_hash",
    "topology_hash",
    "task_trace_hash",
    "exogenous_trace_hash",
    "dependency_lock_hash",
    "tariff_mode",
    "gamma_per_second",
)


class EvaluationDataError(ValueError):
    """Raised when reports are incomplete or cannot be compared fairly."""


POLICY_SUMMARY_FIELDS = (
    "policy",
    "completion_rate",
    "sla_violation_rate",
    "total_cost",
    "total_green_energy_used",
    "task_green_coverage",
    "system_green_absorption",
    "active_wait_ratio",
)

SYSTEM_METRIC_FIELDS = (
    "policy",
    "seed",
    "total_tasks",
    "completed_tasks",
    "completion_rate",
    "sla_violation_tasks",
    "sla_violation_rate",
    "reservation_success_tasks",
    "reservation_success_rate",
    "total_cost",
    "total_energy_consumption",
    "total_green_energy_used",
    "task_green_coverage",
    "system_green_absorption",
    "active_wait_tasks",
    "active_wait_ratio",
    "average_active_wait",
    "average_cpu_utilization",
    "peak_cpu_utilization",
)


def _read_report(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(f"cannot read evaluation report {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise EvaluationDataError(f"evaluation report must be an object: {path}")
    return report


def _policy_from_name(path: Path, seed: int) -> str:
    suffix = f"_seed{seed}"
    stem = path.stem
    if not stem.endswith(suffix):
        raise EvaluationDataError(
            f"report name must end with {suffix}.json: {path}"
        )
    policy = stem[: -len(suffix)]
    if policy not in POLICIES:
        raise EvaluationDataError(f"unknown policy in report name: {path}")
    return policy


def discover_reports(input_dir: Path) -> list[Path]:
    """Return recognized policy reports beneath ``input_dir``."""

    found = []
    for path in sorted(input_dir.rglob("*.json")):
        if any(path.stem.startswith(f"{policy}_seed") for policy in POLICIES):
            found.append(path)
    if not found:
        raise EvaluationDataError(f"no five-policy evaluation reports found in {input_dir}")
    return found


def _finite_scalar(value, *, path: str):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise EvaluationDataError(f"non-finite metric at {path}")
        return value
    raise EvaluationDataError(f"metric value at {path} is not numeric or null")


def flatten_metrics(metrics: Mapping[str, object]) -> list[dict]:
    """Flatten all numeric metric leaves while preserving MetricValue status."""

    rows: list[dict] = []

    def visit(value, parts: tuple[str, ...]) -> None:
        metric_path = ".".join(parts)
        if isinstance(value, Mapping):
            if "value" in value and "status" in value:
                rows.append({
                    "metric_path": metric_path,
                    "value": _finite_scalar(value.get("value"), path=metric_path),
                    "status": value.get("status"),
                    "reason": value.get("reason"),
                    "numerator": _finite_scalar(
                        value.get("numerator"), path=f"{metric_path}.numerator"
                    ),
                    "denominator": _finite_scalar(
                        value.get("denominator"), path=f"{metric_path}.denominator"
                    ),
                })
                return
            for key, child in value.items():
                visit(child, parts + (str(key),))
            return
        if isinstance(value, (int, float, bool)) or value is None:
            rows.append({
                "metric_path": metric_path,
                "value": _finite_scalar(value, path=metric_path),
                "status": "VALID" if value is not None else "NOT_APPLICABLE",
                "reason": None if value is not None else "null scalar metric",
                "numerator": None,
                "denominator": None,
            })
            return
        # Descriptive leaves such as sla_type are not evaluation measures.
        if not isinstance(value, str):
            raise EvaluationDataError(
                f"unsupported metric value type at {metric_path}: {type(value).__name__}"
            )

    visit(metrics, ())
    return rows


def _validate_and_group(report_paths: Iterable[Path]):
    grouped: dict[int, dict[str, tuple[Path, dict]]] = {}
    for path in report_paths:
        report = _read_report(path)
        metadata = report.get("metadata")
        if not isinstance(metadata, dict) or "seed" not in metadata:
            raise EvaluationDataError(f"missing metadata.seed: {path}")
        seed = int(metadata["seed"])
        policy = _policy_from_name(path, seed)
        if policy in grouped.setdefault(seed, {}):
            raise EvaluationDataError(f"duplicate report for seed {seed}, policy {policy}")
        grouped[seed][policy] = (path, report)

    for seed, reports in grouped.items():
        missing = [policy for policy in POLICIES if policy not in reports]
        if missing:
            raise EvaluationDataError(
                f"seed {seed} is missing policies: {', '.join(missing)}"
            )
        reference = reports[POLICIES[0]][1]
        reference_metadata = reference["metadata"]
        reference_diagnostic_version = (
            (reference.get("diagnostics") or {}).get("schema_version")
        )
        for policy in POLICIES:
            path, report = reports[policy]
            if report.get("status") != "VALID":
                raise EvaluationDataError(f"report is not VALID: {path}")
            if report.get("unsettled_task_ids"):
                raise EvaluationDataError(f"report has unsettled tasks: {path}")
            if not isinstance(report.get("metrics"), dict):
                raise EvaluationDataError(f"report has no metrics object: {path}")
            metadata = report.get("metadata", {})
            mismatches = [
                field
                for field in PAIRED_METADATA_FIELDS
                if metadata.get(field) != reference_metadata.get(field)
            ]
            if mismatches:
                raise EvaluationDataError(
                    f"seed {seed}, policy {policy} has mismatched paired metadata: "
                    + ", ".join(mismatches)
                )
            diagnostic_version = (
                (report.get("diagnostics") or {}).get("schema_version")
            )
            if diagnostic_version != reference_diagnostic_version:
                raise EvaluationDataError(
                    f"seed {seed}, policy {policy} has mismatched diagnostic schema"
                )
    return grouped


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_cell(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _row_with_context(seed, policy, source_file, row):
    return {
        "seed": seed,
        "policy": policy,
        **{str(key): _json_cell(value) for key, value in row.items()},
        "source_file": source_file,
    }


def _write_dynamic_csv(path: Path, rows: Sequence[Mapping], base_fields: Sequence[str]):
    extra = sorted({key for row in rows for key in row if key not in base_fields})
    fields = tuple(base_fields) + tuple(extra)
    _write_csv(path, fields, rows)


def _distribution(values):
    items = sorted(float(value) for value in values if value is not None)
    if not items:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "maximum": None}

    def percentile(percent):
        position = (len(items) - 1) * percent / 100.0
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return items[lower]
        weight = position - lower
        return items[lower] * (1.0 - weight) + items[upper] * weight

    return {
        "count": len(items),
        "mean": statistics.fmean(items),
        "p50": percentile(50.0),
        "p95": percentile(95.0),
        "maximum": items[-1],
    }


def _fallback_diagnostics(report):
    """Derive the safe subset supported by pre-diagnostics reports."""

    tasks = tuple(report.get("task_records") or ())
    decisions = tuple(report.get("decision_records") or ())
    completed = tuple(item for item in tasks if item.get("final_state") == "Completed")
    remote = tuple(item for item in completed if item.get("transmission_start_sim") is not None)
    cpu_by_node = {}
    for item in completed:
        node = item.get("target_node")
        if node:
            cpu_by_node[node] = cpu_by_node.get(node, 0.0) + float(
                item.get("cpu_work_cpu_hours") or 0.0
            )
    allocations = tuple(cpu_by_node.values())
    allocation_mean = statistics.fmean(allocations) if allocations else 0.0
    return {
        "task_summary": {
            "scheduler_queue_delay_sim": _distribution(
                item.get("scheduler_queue_delay_sim") for item in tasks
            ),
            "start_delay_sim": _distribution(item.get("start_delay_sim") for item in tasks),
            "completion_delay_sim": _distribution(
                item.get("completion_delay_sim") for item in tasks
            ),
            "reservation_lead_sim": _distribution(
                item.get("reservation_lead_sim") for item in tasks
            ),
            "decision_wall_seconds": _distribution(
                item.get("decision_wall_seconds") for item in decisions
            ),
            "decision_count_per_arrived_task": (
                len(decisions) / len(tasks) if tasks else None
            ),
            "candidate_count_per_decision": _distribution(
                item.get("candidate_count") for item in decisions
            ),
        },
        "node_summary": {
            "used_node_count": len(cpu_by_node),
            "allocated_cpu_hours_cv": (
                statistics.pstdev(allocations) / allocation_mean
                if allocations and allocation_mean > 0.0 else 0.0
            ),
        },
        "network_summary": {
            "reservation_count": len(completed),
            "remote_reservation_count": len(remote),
            "remote_task_rate": len(remote) / len(completed) if completed else None,
            "transmission_duration_sim": _distribution(
                item["transmission_end_sim"] - item["transmission_start_sim"]
                for item in remote
            ),
        },
    }


def _diagnostic_metric_source(report):
    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        if not report.get("task_records") and not report.get("decision_records"):
            return {"diagnostics": {}}
        diagnostics = _fallback_diagnostics(report)
    return {
        "diagnostics": {
            key: diagnostics[key]
            for key in (
                "system_summary", "task_summary", "time_summary",
                "node_summary", "network_summary", "runtime_summary",
            )
            if isinstance(diagnostics.get(key), Mapping)
        }
    }


def _metric_value(value):
    if isinstance(value, Mapping):
        return value.get("value")
    return value


def build_unified_system_metrics(report: Mapping, policy: str) -> dict:
    """Build the common five-policy scorecard from one complete report."""

    metrics = report.get("metrics") or {}
    metadata = report.get("metadata") or {}
    tasks = tuple(report.get("task_records") or ())
    total = int(metrics.get("arrival_count") or len(tasks))
    completed = int(metrics.get("completed_count") or sum(
        task.get("final_state") == "Completed" for task in tasks
    ))
    reserved = int(metrics.get("reserved_ever_count") or sum(
        task.get("compute_start_sim") is not None for task in tasks
    ))
    violations = sum(
        task.get("final_state") != "Completed"
        or task.get("start_delay_sim") is None
        or float(task["start_delay_sim"])
        > float(task.get("latest_start_limit_sim") or 0.0) + 1e-12
        for task in tasks
    ) if tasks else max(0, total - completed)
    positive_waits = [
        float(task["active_wait_sim"])
        for task in tasks
        if task.get("compute_start_sim") is not None
        and task.get("active_wait_sim") is not None
        and float(task["active_wait_sim"]) > 1e-12
    ]
    accounting = report.get("accounting_report") or {}
    diagnostics = report.get("diagnostics") or {}
    node_summary = diagnostics.get("node_summary") or {}
    total_energy = math.fsum(
        float(task.get("task_energy_mwh") or 0.0) for task in tasks
    )
    system = {
        "policy": policy,
        "seed": metadata.get("seed"),
        "total_tasks": total,
        "completed_tasks": completed,
        "completion_rate": completed / total if total else None,
        "sla_violation_tasks": violations,
        "sla_violation_rate": violations / total if total else None,
        "reservation_success_tasks": reserved,
        "reservation_success_rate": reserved / total if total else None,
        "total_cost": metrics.get("total_economic_cost_yuan"),
        "total_energy_consumption": total_energy,
        "total_green_energy_used": accounting.get(
            "total_task_attributed_green_energy_mwh"
        ),
        "task_green_coverage": _metric_value(
            metrics.get("completed_task_green_coverage")
        ),
        "system_green_absorption": _metric_value(
            metrics.get("system_green_absorption_rate")
        ),
        "active_wait_tasks": len(positive_waits),
        "active_wait_ratio": len(positive_waits) / reserved if reserved else None,
        "average_active_wait": (
            statistics.fmean(positive_waits) if positive_waits else None
        ),
        "average_cpu_utilization": node_summary.get(
            "time_node_mean_cpu_utilization"
        ),
        "peak_cpu_utilization": node_summary.get(
            "maximum_node_cpu_utilization"
        ),
        "provenance": {
            "task_trace_hash": metadata.get("task_trace_hash"),
            "config_hash": metadata.get("config_hash"),
            "topology_hash": metadata.get("topology_hash"),
            "exogenous_trace_hash": metadata.get("exogenous_trace_hash"),
            "model_hash": metadata.get("model_hash"),
        },
        "definitions": {
            "sla_violation": (
                "task did not complete within its absolute latest-start limit"
            ),
            "reservation_success_rate": (
                "tasks ever reserved / total arrived tasks"
            ),
            "active_wait": (
                "selected compute start minus earliest feasible compute start"
            ),
            "active_wait_ratio": (
                "positive-active-wait tasks / tasks ever reserved"
            ),
            "average_active_wait": "mean over positive-active-wait tasks",
            "average_cpu_utilization": (
                "time-node mean CPU utilization over the evaluation span"
            ),
            "peak_cpu_utilization": (
                "maximum node CPU utilization over the evaluation span"
            ),
        },
    }
    return system


def _candidate_coordination_rows(tasks: Sequence[Mapping]) -> list[dict]:
    rows = []
    for original in tasks:
        row = dict(original)
        selected_start = row.get("compute_start_sim")
        earliest_start = row.get("earliest_compute_start_sim")
        earliest_cost = row.get("earliest_candidate_marginal_system_cost_yuan")
        selected_cost = row.get("candidate_marginal_system_cost_yuan")
        earliest_green = row.get("earliest_green_coverage")
        selected_green = row.get("selected_green_coverage")
        earliest_green_energy = row.get(
            "earliest_candidate_marginal_green_energy_mwh"
        )
        selected_green_energy = row.get("candidate_marginal_green_energy_mwh")
        row.update({
            "selected_start": selected_start,
            "earliest_start": earliest_start,
            "active_wait": (
                None if selected_start is None or earliest_start is None
                else float(selected_start) - float(earliest_start)
            ),
            "cost_saving": (
                None if earliest_cost is None or selected_cost is None
                else float(earliest_cost) - float(selected_cost)
            ),
            "green_coverage_gain": (
                None if earliest_green is None or selected_green is None
                else float(selected_green) - float(earliest_green)
            ),
            "green_energy_gain": (
                None
                if earliest_green_energy is None or selected_green_energy is None
                else float(selected_green_energy) - float(earliest_green_energy)
            ),
            "sla_satisfied": (
                row.get("final_state") == "Completed"
                and row.get("start_delay_sim") is not None
                and float(row["start_delay_sim"])
                <= float(row.get("latest_start_limit_sim") or 0.0) + 1e-12
            ),
        })
        rows.append(row)
    return enrich_adjustments(rows)


def export_policy_artifacts(report_path: Path, output_dir: Path) -> dict:
    """Write one policy's independent task/decision/time/node/system files."""

    report = _read_report(report_path)
    metadata = report.get("metadata") or {}
    seed = int(metadata["seed"])
    policy = _policy_from_name(report_path, seed)
    source_text = report_path.as_posix()
    tasks = [dict(item) for item in report.get("task_records") or ()]
    if policy == "candidate_dqn":
        tasks = _candidate_coordination_rows(tasks)
    decisions = [dict(item) for item in report.get("decision_records") or ()]
    diagnostics = report.get("diagnostics") or {}
    time_rows = [dict(item) for item in diagnostics.get("time_records") or ()]
    node_rows = [dict(item) for item in diagnostics.get("node_records") or ()]
    for rows in (tasks, decisions, time_rows, node_rows):
        for row in rows:
            row.update({"seed": seed, "policy": policy, "source_file": source_text})

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dynamic_csv(
        output_dir / "task_metrics.csv", tasks,
        ("seed", "policy", "task_id", "source_file"),
    )
    _write_dynamic_csv(
        output_dir / "decision_metrics.csv", decisions,
        ("seed", "policy", "task_id", "decision_id", "source_file"),
    )
    _write_dynamic_csv(
        output_dir / "time_metrics.csv", time_rows,
        ("seed", "policy", "start_sim", "end_sim", "source_file"),
    )
    _write_dynamic_csv(
        output_dir / "node_metrics.csv", node_rows,
        ("seed", "policy", "node", "source_file"),
    )

    system_metrics = build_unified_system_metrics(report, policy)
    if policy == "candidate_dqn":
        adjustment_metrics = build_adjustment_metrics(tasks)
        adjustment_definitions = adjustment_metrics.pop("definitions")
        system_metrics.update(adjustment_metrics)
        system_metrics["definitions"].update(adjustment_definitions)
        requested = build_requested_coordination_groups(tasks)
        exclusive = build_mutually_exclusive_coordination_groups(tasks)
        gain_distribution = build_green_coverage_gain_distribution(tasks)
        sla_metrics = build_sla_type_metrics(tasks)
        _write_csv(
            output_dir / "coordination_type_metrics.csv",
            GROUP_METRIC_FIELDS,
            requested,
        )
        _write_csv(
            output_dir / "coordination_type_metrics_mutually_exclusive.csv",
            GROUP_METRIC_FIELDS,
            exclusive,
        )
        _write_dynamic_csv(
            output_dir / "green_coverage_gain_distribution.csv",
            [gain_distribution],
            tuple(gain_distribution),
        )
        _write_csv(
            output_dir / "sla_type_coordination_metrics.csv",
            SLA_METRIC_FIELDS,
            sla_metrics,
        )

    (output_dir / "system_metrics.json").write_text(
        json.dumps(system_metrics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    system_csv_fields = SYSTEM_METRIC_FIELDS + tuple(
            field
            for adjustment in ADJUSTMENT_FIELDS
            for field in (f"{adjustment}_tasks", f"{adjustment}_ratio")
            if field in system_metrics
        )
    _write_csv(
        output_dir / "system_metrics.csv",
        system_csv_fields,
        [{field: system_metrics.get(field) for field in system_csv_fields}],
    )
    return system_metrics


def export_reports(report_paths: Iterable[Path], output_dir: Path) -> dict:
    """Validate reports and create long, wide, and provenance source tables."""

    grouped = _validate_and_group(Path(path) for path in report_paths)
    long_rows = []
    manifest_rows = []
    task_rows = []
    decision_rows = []
    time_rows = []
    node_rows = []
    network_rows = []
    runtime_rows = []
    policy_summary_rows = []

    for seed in sorted(grouped):
        for policy in POLICIES:
            source_path, report = grouped[seed][policy]
            metadata = report["metadata"]
            metrics = report["metrics"]
            source_text = source_path.as_posix()
            system_metrics = build_unified_system_metrics(report, policy)
            policy_summary_rows.append({
                field: system_metrics[field] for field in POLICY_SUMMARY_FIELDS
            })
            for metric in flatten_metrics(metrics):
                long_rows.append({
                    "seed": seed,
                    "policy": policy,
                    **metric,
                    "source_file": source_text,
                })
            for metric in flatten_metrics(_diagnostic_metric_source(report)):
                long_rows.append({
                    "seed": seed,
                    "policy": policy,
                    **metric,
                    "source_file": source_text,
                })

            accounting = report.get("accounting_report") or {}
            accounting_by_task = {
                item.get("task_id"): item
                for item in accounting.get("task_records") or ()
                if item.get("task_id")
            }
            decisions_by_task = {}
            for decision in report.get("decision_records") or ():
                decisions_by_task.setdefault(decision.get("task_id"), []).append(decision)
                decision_rows.append(_row_with_context(
                    seed, policy, source_text, decision
                ))
            for task in report.get("task_records") or ():
                task_id = task.get("task_id")
                task_decisions = decisions_by_task.get(task_id, ())
                decision_times = tuple(
                    float(item["decision_wall_seconds"])
                    for item in task_decisions
                    if item.get("decision_wall_seconds") is not None
                )
                enriched = dict(task)
                enriched.update({
                    f"accounting_{key}": value
                    for key, value in accounting_by_task.get(task_id, {}).items()
                    if key not in {"task_id", "target_node"}
                })
                enriched.update({
                    "decision_count": len(task_decisions),
                    "candidate_count_total": sum(
                        int(item.get("candidate_count") or 0)
                        for item in task_decisions
                    ),
                    "decision_wall_seconds_total": (
                        math.fsum(decision_times) if decision_times else None
                    ),
                    "decision_wall_seconds_p95": (
                        _distribution(decision_times)["p95"]
                        if decision_times else None
                    ),
                })
                task_rows.append(_row_with_context(
                    seed, policy, source_text, enriched
                ))

            diagnostics = report.get("diagnostics") or {}
            for item in diagnostics.get("time_records") or ():
                time_rows.append(_row_with_context(
                    seed, policy, source_text, item
                ))
            diagnostic_nodes = diagnostics.get("node_records") or ()
            if diagnostic_nodes:
                for item in diagnostic_nodes:
                    node_rows.append(_row_with_context(
                        seed, policy, source_text, item
                    ))
            else:
                fallback_nodes = {}
                for task in report.get("task_records") or ():
                    node = task.get("target_node")
                    if not node:
                        continue
                    values = fallback_nodes.setdefault(node, {
                        "node": node, "reservation_count": 0,
                        "allocated_cpu_hours": 0.0,
                        "task_attributed_cost_yuan": 0.0,
                        "task_attributed_green_energy_mwh": 0.0,
                    })
                    values["reservation_count"] += 1
                    values["allocated_cpu_hours"] += float(
                        task.get("cpu_work_cpu_hours") or 0.0
                    )
                    values["task_attributed_cost_yuan"] += float(
                        task.get("task_attributed_cost_yuan") or 0.0
                    )
                    values["task_attributed_green_energy_mwh"] += float(
                        task.get("task_attributed_green_energy_mwh") or 0.0
                    )
                total_cpu = math.fsum(
                    item["allocated_cpu_hours"] for item in fallback_nodes.values()
                )
                for item in fallback_nodes.values():
                    item["allocation_share"] = (
                        item["allocated_cpu_hours"] / total_cpu if total_cpu > 0.0 else None
                    )
                    node_rows.append(_row_with_context(
                        seed, policy, source_text, item
                    ))

            diagnostic_network = diagnostics.get("network_records") or ()
            if diagnostic_network:
                for item in diagnostic_network:
                    network_rows.append(_row_with_context(
                        seed, policy, source_text, {"resource_type": "edge", **item}
                    ))
            else:
                fallback_paths = {}
                for task in report.get("task_records") or ():
                    if task.get("transmission_start_sim") is None:
                        continue
                    path_id = task.get("path_id") or "unknown"
                    values = fallback_paths.setdefault(path_id, {
                        "resource_type": "path", "path_id": path_id,
                        "task_count": 0, "total_transmission_duration_sim": 0.0,
                    })
                    values["task_count"] += 1
                    values["total_transmission_duration_sim"] += (
                        float(task["transmission_end_sim"])
                        - float(task["transmission_start_sim"])
                    )
                for item in fallback_paths.values():
                    network_rows.append(_row_with_context(
                        seed, policy, source_text, item
                    ))

            runtime = diagnostics.get("runtime_summary")
            if isinstance(runtime, Mapping):
                flattened_runtime = {}

                def visit_runtime(value, parts):
                    if isinstance(value, Mapping):
                        for key, child in value.items():
                            visit_runtime(child, parts + (str(key),))
                    elif value is None or (
                        isinstance(value, (int, float)) and not isinstance(value, bool)
                    ):
                        flattened_runtime[".".join(parts)] = value

                visit_runtime(runtime, ())
                runtime_rows.append(_row_with_context(
                    seed, policy, source_text, flattened_runtime
                ))
            manifest_rows.append({
                "seed": seed,
                "policy": policy,
                "source_file": source_text,
                "status": report["status"],
                "system_version": metadata.get("system_version"),
                "tariff_mode": metadata.get("tariff_mode"),
                "diagnostic_schema_version": (
                    (report.get("diagnostics") or {}).get("schema_version")
                ),
                "arrival_cutoff_sim": metadata.get("arrival_cutoff_sim"),
                "model_hash": metadata.get("model_hash"),
                "code_hash": metadata.get("code_hash"),
                "config_hash": metadata.get("config_hash"),
                "topology_hash": metadata.get("topology_hash"),
                "task_trace_hash": metadata.get("task_trace_hash"),
                "exogenous_trace_hash": metadata.get("exogenous_trace_hash"),
                "dependency_lock_hash": metadata.get("dependency_lock_hash"),
                "arrival_count": metrics.get("arrival_count"),
                "completed_count": metrics.get("completed_count"),
                "unsettled_count": len(report.get("unsettled_task_ids", ())),
                "paired_metadata_match": True,
            })

    indexed = {
        (row["seed"], row["metric_path"], row["policy"]): row
        for row in long_rows
    }
    metric_keys = sorted({(row["seed"], row["metric_path"]) for row in long_rows})
    wide_rows = []
    for seed, metric_path in metric_keys:
        row = {"seed": seed, "metric_path": metric_path}
        for policy in POLICIES:
            metric = indexed.get((seed, metric_path, policy))
            row[policy] = None if metric is None else metric["value"]
            row[f"{policy}_status"] = "MISSING" if metric is None else metric["status"]
        wide_rows.append(row)

    long_fields = (
        "seed", "policy", "metric_path", "value", "status", "reason",
        "numerator", "denominator", "source_file",
    )
    wide_fields = (
        "seed", "metric_path", *POLICIES,
        *(f"{policy}_status" for policy in POLICIES),
    )
    manifest_fields = (
        "seed", "policy", "source_file", "status", "system_version",
        "tariff_mode",
        "diagnostic_schema_version",
        "arrival_cutoff_sim", "model_hash", "code_hash", "config_hash",
        "topology_hash", "task_trace_hash", "exogenous_trace_hash",
        "dependency_lock_hash", "arrival_count", "completed_count",
        "unsettled_count", "paired_metadata_match",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics_long.csv", long_fields, long_rows)
    _write_csv(output_dir / "metrics_wide.csv", wide_fields, wide_rows)
    _write_csv(output_dir / "source_manifest.csv", manifest_fields, manifest_rows)
    _write_csv(
        output_dir / "policy_summary.csv",
        POLICY_SUMMARY_FIELDS,
        policy_summary_rows,
    )
    _write_dynamic_csv(
        output_dir / "task_metrics.csv", task_rows,
        ("seed", "policy", "task_id", "source_file"),
    )
    _write_dynamic_csv(
        output_dir / "decision_metrics.csv", decision_rows,
        ("seed", "policy", "task_id", "decision_id", "source_file"),
    )
    _write_dynamic_csv(
        output_dir / "time_metrics.csv", time_rows,
        ("seed", "policy", "start_sim", "end_sim", "source_file"),
    )
    _write_dynamic_csv(
        output_dir / "node_metrics.csv", node_rows,
        ("seed", "policy", "node", "source_file"),
    )
    _write_dynamic_csv(
        output_dir / "network_metrics.csv", network_rows,
        ("seed", "policy", "resource_type", "source_file"),
    )
    _write_dynamic_csv(
        output_dir / "runtime_metrics.csv", runtime_rows,
        ("seed", "policy", "source_file"),
    )

    validation = {
        "status": "PASS",
        "seeds": sorted(grouped),
        "seed_count": len(grouped),
        "policies": list(POLICIES),
        "reports_per_seed": len(POLICIES),
        "source_report_count": len(manifest_rows),
        "long_metric_row_count": len(long_rows),
        "wide_metric_row_count": len(wide_rows),
        "task_row_count": len(task_rows),
        "decision_row_count": len(decision_rows),
        "time_row_count": len(time_rows),
        "node_row_count": len(node_rows),
        "network_row_count": len(network_rows),
        "runtime_row_count": len(runtime_rows),
        "policy_summary_row_count": len(policy_summary_rows),
        "granularity_coverage": {
            "task": bool(task_rows),
            "decision": bool(decision_rows),
            "time": bool(time_rows),
            "node": bool(node_rows),
            "network": bool(network_rows),
            "system": bool(long_rows),
            "runtime": bool(runtime_rows),
        },
        "all_reports_valid": True,
        "all_unsettled_counts_zero": True,
        "paired_metadata_fields": list(PAIRED_METADATA_FIELDS),
        "paired_metadata_match": True,
    }
    (output_dir / "analysis_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all metrics from paired V2 five-policy reports"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/v2/evaluation/five_policy"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir / "source_data"
    validation = export_reports(discover_reports(args.input_dir), output_dir)
    print(json.dumps({
        "status": validation["status"],
        "output_dir": str(output_dir),
        "source_report_count": validation["source_report_count"],
        "long_metric_row_count": validation["long_metric_row_count"],
        "wide_metric_row_count": validation["wide_metric_row_count"],
        "granularity_coverage": validation["granularity_coverage"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
