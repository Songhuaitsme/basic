"""Summarize existing five-policy reports without running any policy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from v2.export_five_policy_metrics import build_unified_system_metrics


POLICIES = (
    ("earliest_feasible", "Earliest Feasible"),
    ("lowest_cost", "Lowest Cost"),
    ("highest_green", "Highest Green"),
    ("equal_weight", "Equal Weight"),
    ("candidate_dqn", "Candidate DQN"),
)

CORE_FIELDS = (
    "Policy",
    "Total Tasks",
    "Completed Tasks",
    "Completion Rate",
    "SLA Violation Rate",
    "Reservation Success Rate",
    "Total Cost",
    "Total Energy Consumption",
    "Total Green Energy Used",
    "Task Green Coverage",
    "System Green Absorption",
    "Active Wait Tasks",
    "Active Wait Ratio",
    "Average Active Wait",
    "Average CPU Utilization",
    "Peak CPU Utilization",
)

PAIRED_FIELDS = (
    "seed",
    "arrival_cutoff_sim",
    "config_hash",
    "topology_hash",
    "task_trace_hash",
    "exogenous_trace_hash",
    "dependency_lock_hash",
    "tariff_mode",
)


def _write_csv(path: Path, fields, rows) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report_path(root: Path, policy: str) -> Path:
    return root / "seed_42" / policy / f"{policy}_seed42.json"


def _load_reports(root: Path) -> dict[str, dict]:
    reports = {}
    for policy, _ in POLICIES:
        path = _report_path(root, policy)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "VALID" or report.get("unsettled_task_ids"):
            raise ValueError(f"report is not settled and VALID: {path}")
        reports[policy] = report
    return reports


def _validate_pairing(reports: dict[str, dict], reference: dict) -> dict:
    baseline = reports[POLICIES[0][0]]["metadata"]
    mismatches = []
    for policy, _ in POLICIES:
        metadata = reports[policy]["metadata"]
        for field in PAIRED_FIELDS:
            if metadata.get(field) != baseline.get(field):
                mismatches.append(f"{policy}:{field}")
        metrics = reports[policy]["metrics"]
        if metrics.get("arrival_count") != 2416:
            mismatches.append(f"{policy}:arrival_count")
        if len(reports[policy].get("task_records") or ()) != 2416:
            mismatches.append(f"{policy}:task_record_count")
    reference_metadata = reference.get("metadata") or {}
    if baseline.get("task_trace_hash") != reference_metadata.get("task_trace_hash"):
        mismatches.append("reference:task_trace_hash")
    if mismatches:
        raise ValueError("five-policy pairing validation failed: " + ", ".join(mismatches))
    return {
        "status": "PASS",
        "policy_count": 5,
        "task_count_per_policy": 2416,
        "task_trace_hash": baseline["task_trace_hash"],
        "reference_task_trace_hash_match": True,
        "paired_fields": list(PAIRED_FIELDS),
    }


def _core_row(display_name: str, metrics: dict) -> dict:
    return {
        "Policy": display_name,
        "Total Tasks": metrics["total_tasks"],
        "Completed Tasks": metrics["completed_tasks"],
        "Completion Rate": metrics["completion_rate"],
        "SLA Violation Rate": metrics["sla_violation_rate"],
        "Reservation Success Rate": metrics["reservation_success_rate"],
        "Total Cost": metrics["total_cost"],
        "Total Energy Consumption": metrics["total_energy_consumption"],
        "Total Green Energy Used": metrics["total_green_energy_used"],
        "Task Green Coverage": metrics["task_green_coverage"],
        "System Green Absorption": metrics["system_green_absorption"],
        "Active Wait Tasks": metrics["active_wait_tasks"],
        "Active Wait Ratio": metrics["active_wait_ratio"],
        "Average Active Wait": metrics["average_active_wait"],
        "Average CPU Utilization": metrics["average_cpu_utilization"],
        "Peak CPU Utilization": metrics["peak_cpu_utilization"],
    }


def _relative_change(selected: float, baseline: float) -> float:
    return (selected - baseline) / baseline * 100.0


def _comparison_rows(earliest: dict, dqn: dict) -> list[dict]:
    comparison = "Candidate DQN vs Earliest Feasible Policy"
    return [
        {
            "Metric": "Cost change",
            "Value": _relative_change(dqn["total_cost"], earliest["total_cost"]),
            "Unit": "%",
            "Comparison": comparison,
        },
        {
            "Metric": "Green Energy Used change",
            "Value": _relative_change(
                dqn["total_green_energy_used"], earliest["total_green_energy_used"]
            ),
            "Unit": "%",
            "Comparison": comparison,
        },
        {
            "Metric": "Green Coverage change",
            "Value": (dqn["task_green_coverage"] - earliest["task_green_coverage"]) * 100.0,
            "Unit": "percentage points",
            "Comparison": comparison,
        },
        {
            "Metric": "Green Absorption change",
            "Value": (
                dqn["system_green_absorption"] - earliest["system_green_absorption"]
            ) * 100.0,
            "Unit": "percentage points",
            "Comparison": comparison,
        },
        {
            "Metric": "Completion Rate change",
            "Value": (dqn["completion_rate"] - earliest["completion_rate"]) * 100.0,
            "Unit": "percentage points",
            "Comparison": comparison,
        },
        {
            "Metric": "SLA Violation change",
            "Value": (
                dqn["sla_violation_rate"] - earliest["sla_violation_rate"]
            ) * 100.0,
            "Unit": "percentage points",
            "Comparison": comparison,
        },
        {
            "Metric": "Active Wait Tasks change",
            "Value": dqn["active_wait_tasks"] - earliest["active_wait_tasks"],
            "Unit": "tasks",
            "Comparison": comparison,
        },
        {
            "Metric": "Active Wait Ratio change",
            "Value": (dqn["active_wait_ratio"] - earliest["active_wait_ratio"]) * 100.0,
            "Unit": "percentage points",
            "Comparison": comparison,
        },
    ]


def _print_terminal_table(rows: list[dict]) -> None:
    print("| Policy | Completion | SLA Violation | Total Cost | Green Energy Used | Green Coverage | Green Absorption | Active Wait Ratio |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['Policy']} | {row['Completion Rate']:.2%} | "
            f"{row['SLA Violation Rate']:.2%} | {row['Total Cost']:.2f} | "
            f"{row['Total Green Energy Used']:.2f} | "
            f"{row['Task Green Coverage']:.2%} | "
            f"{row['System Green Absorption']:.2%} | "
            f"{row['Active Wait Ratio']:.2%} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/v2/evaluation/five_policy_formal_seed42"),
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=Path(
            "artifacts/v2/evaluation/coordination_v2_seed42/evaluation_report.json"
        ),
    )
    args = parser.parse_args()
    root = args.input_dir.resolve()
    reports = _load_reports(root)
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    validation = _validate_pairing(reports, reference)

    metrics_by_policy = {
        policy: build_unified_system_metrics(reports[policy], policy)
        for policy, _ in POLICIES
    }
    core_rows = [
        _core_row(display_name, metrics_by_policy[policy])
        for policy, display_name in POLICIES
    ]
    comparison_rows = _comparison_rows(
        metrics_by_policy["earliest_feasible"],
        metrics_by_policy["candidate_dqn"],
    )
    output_dir = root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "five_policy_core_summary.csv", CORE_FIELDS, core_rows)
    _write_csv(
        output_dir / "candidate_dqn_vs_earliest_feasible.csv",
        ("Metric", "Value", "Unit", "Comparison"),
        comparison_rows,
    )
    (output_dir / "summary_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_terminal_table(core_rows)
    print(json.dumps({
        "status": "VALID",
        "output_dir": str(output_dir),
        "task_trace_hash": validation["task_trace_hash"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
