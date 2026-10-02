"""Build reviewed source rows for the V3 Wait OFF five-policy comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


POLICIES = (
    (
        "earliest_feasible",
        "Earliest Feasible",
        "artifacts/v3/evaluation/wait_off_five_policy/"
        "earliest_feasible_seed42_warmup2d_wait_off.json",
    ),
    (
        "lowest_cost",
        "Lowest Cost",
        "artifacts/v3/evaluation/wait_off_five_policy/"
        "lowest_cost_seed42_warmup2d_wait_off.json",
    ),
    (
        "highest_green",
        "Highest Green",
        "artifacts/v3/evaluation/wait_off_five_policy/"
        "highest_green_seed42_warmup2d_wait_off.json",
    ),
    (
        "equal_weight",
        "Equal Weight",
        "artifacts/v3/evaluation/wait_off_five_policy/"
        "equal_weight_seed42_warmup2d_wait_off.json",
    ),
    (
        "candidate_dqn",
        "Candidate DQN",
        "artifacts/v3/evaluation/wait_ablation/formal_600k_dqn_wait_off.json",
    ),
)

PAIRED_FIELDS = (
    "seed",
    "task_trace_hash",
    "exogenous_trace_hash",
    "topology_hash",
    "code_hash",
    "config_hash",
    "warmup_days",
    "evaluation_start_sim",
    "arrival_cutoff_sim",
)


def value(metric):
    return metric.get("value") if isinstance(metric, dict) else metric


def pct(metric):
    number = value(metric)
    return None if number is None else number * 100.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_snapshot(root: Path) -> dict:
    reports = {}
    source_files = []
    source_rows = []

    for key, label, relative in POLICIES:
        path = root / relative
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "VALID":
            raise ValueError(f"report is not VALID: {path}")
        reports[key] = report
        source_files.append(relative)
        source_rows.append(
            {
                "policy_key": key,
                "policy": label,
                "source_file": relative,
                "sha256": sha256(path),
                "status": report["status"],
                "model_hash": report["metadata"].get("model_hash"),
                "config_hash": report["metadata"].get("config_hash"),
            }
        )

    reference = reports[POLICIES[0][0]]["metadata"]
    for field in PAIRED_FIELDS:
        observed = {reports[key]["metadata"].get(field) for key, _, _ in POLICIES}
        if observed != {reference.get(field)}:
            raise ValueError(f"paired field mismatch: {field}: {observed}")

    rows = []
    for key, label, relative in POLICIES:
        report = reports[key]
        metrics = report["metrics"]
        wait = metrics["active_wait_metrics"]
        hard = metrics["sla_metrics"]["Hard"]
        soft = metrics["sla_metrics"]["Soft"]
        flexible = metrics["sla_metrics"]["Flexible"]
        reserved = metrics["reserved_ever_count"]
        rows.append(
            {
                "policy_key": key,
                "policy": label,
                "status": report["status"],
                "arrival_count": metrics["arrival_count"],
                "reserved_count": reserved,
                "completed_count": metrics["completed_count"],
                "expired_count": metrics["expired_count"],
                "failed_count": metrics["failed_count"],
                "acceptance_pct": pct(metrics["acceptance_rate"]),
                "completion_pct": pct(metrics["completion_rate"]),
                "reservation_reliability_pct": pct(
                    metrics["reservation_reliability"]
                ),
                "total_cost_yuan": metrics["total_economic_cost_yuan"],
                "completed_cpu_hours": metrics["completed_cpu_hours"],
                "cost_per_completed_cpu_hour": value(
                    metrics["cost_yuan_per_completed_cpu_hour"]
                ),
                "cost_per_arrived_cpu_hour": value(
                    metrics["cost_yuan_per_arrived_cpu_hour"]
                ),
                "task_green_pct": pct(metrics["completed_task_green_coverage"]),
                "system_green_absorption_pct": pct(
                    metrics["system_green_absorption_rate"]
                ),
                "active_wait_count": wait["count"],
                "active_wait_pct": (
                    wait["count"] / reserved * 100.0 if reserved else None
                ),
                "mean_active_wait_sim": value(wait["mean_active_wait_sim"]),
                "p95_active_wait_sim": value(wait["p95_active_wait_sim"]),
                "positive_benefit_pct": pct(wait["positive_benefit_rate"]),
                "hard_count": hard["count"],
                "hard_expired_pct": pct(hard["expired_rate"]),
                "hard_start_delay_p50_sim": value(hard["start_delay_p50_sim"]),
                "hard_start_delay_p95_sim": value(hard["start_delay_p95_sim"]),
                "soft_count": soft["count"],
                "soft_on_time_pct": pct(soft["preferred_on_time_rate"]),
                "soft_tardy_pct": pct(soft["acceptable_tardy_rate"]),
                "soft_expired_pct": pct(soft["expired_rate"]),
                "soft_start_delay_p50_sim": value(soft["start_delay_p50_sim"]),
                "soft_start_delay_p95_sim": value(soft["start_delay_p95_sim"]),
                "soft_tardiness_p50_sim": value(
                    soft["preferred_start_tardiness_p50"]
                ),
                "soft_tardiness_p95_sim": value(
                    soft["preferred_start_tardiness_p95"]
                ),
                "flexible_count": flexible["count"],
                "flexible_on_time_pct": pct(flexible["preferred_on_time_rate"]),
                "flexible_tardy_pct": pct(flexible["acceptable_tardy_rate"]),
                "flexible_expired_pct": pct(flexible["expired_rate"]),
                "flexible_start_delay_p50_sim": value(
                    flexible["start_delay_p50_sim"]
                ),
                "flexible_start_delay_p95_sim": value(
                    flexible["start_delay_p95_sim"]
                ),
                "flexible_tardiness_p50_sim": value(
                    flexible["preferred_start_tardiness_p50"]
                ),
                "flexible_tardiness_p95_sim": value(
                    flexible["preferred_start_tardiness_p95"]
                ),
                "source_file": relative,
            }
        )

    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    evidence = {
        "title": "Formal V3 Wait OFF evaluation JSON",
        "detail": (
            "Direct extraction from five VALID reports. Paired fields were checked "
            "for exact equality; rates were converted from fractions to percent."
        ),
    }
    return {
        "id": "v3-wait-off-five-policy-source-table",
        "title": "V3 Wait OFF 五策略源数据对比",
        "surface": "dashboard",
        "status": "reviewed",
        "buildStatus": "creating",
        "generatedAt": generated_at,
        "filters": [],
        "queries": {
            "wait_off_policy_metrics": {
                "label": "Wait OFF 五策略源数据",
                "reportingField": "policy",
                "rows": rows,
                "methods": [
                    {
                        "language": "python",
                        "code": (
                            "Read metrics directly from each report; convert MetricValue "
                            "fractions to percent; compute active_wait_pct as "
                            "active_wait_count / reserved_count * 100."
                        ),
                    }
                ],
                "source": {
                    "label": "V3 Wait OFF five-policy evaluation reports",
                    "files": source_files,
                    "executedAt": generated_at,
                    "evidenceFlow": [evidence],
                    "metricDefinitions": [],
                },
            },
            "experiment_identity": {
                "label": "评估基础配置",
                "rows": [
                    {
                        **{field: reference.get(field) for field in PAIRED_FIELDS},
                        "report_count": len(POLICIES),
                        "all_reports_valid": True,
                        "paired_metadata_match": True,
                        "ablation_variant": "wait.off",
                    }
                ],
                "source": {
                    "label": "V3 Wait OFF paired evaluation metadata",
                    "files": source_files,
                    "executedAt": generated_at,
                    "evidenceFlow": [evidence],
                    "metricDefinitions": [],
                },
            },
            "source_manifest": {
                "label": "原始报告清单",
                "reportingField": "policy",
                "rows": source_rows,
                "source": {
                    "label": "V3 Wait OFF report provenance",
                    "files": source_files,
                    "executedAt": generated_at,
                    "evidenceFlow": [evidence],
                    "metricDefinitions": [],
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_snapshot(root), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
