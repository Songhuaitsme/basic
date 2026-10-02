"""Build reviewed data for the V4 quick five-policy comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


POLICIES = (
    ("earliest_feasible", "Earliest Feasible"),
    ("lowest_cost", "Lowest Cost"),
    ("highest_green", "Highest Green"),
    ("equal_weight", "Equal Weight"),
    ("candidate_dqn", "Candidate DQN"),
)

PAIRED_FIELDS = (
    "seed",
    "task_trace_hash",
    "exogenous_trace_hash",
    "topology_hash",
    "code_hash",
    "warmup_days",
    "evaluation_start_sim",
    "arrival_cutoff_sim",
)


def metric_value(value):
    return value.get("value") if isinstance(value, dict) else value


def source(files, label, definitions):
    return {
        "label": label,
        "files": files,
        "executedAt": "2026-09-30",
        "evidenceFlow": [
            {
                "title": "V4 quick evaluation JSON",
                "detail": (
                    "Five VALID reports generated with the same seed, task trace, "
                    "exogenous trace, topology, code, warm-up, and measurement window."
                ),
            }
        ],
        "metricDefinitions": definitions,
    }


def definition(field, label, text, components, files):
    return {
        "field": field,
        "label": label,
        "definition": text,
        "componentIds": components,
        "sourceLineage": [{"files": files}],
    }


def build_snapshot(root: Path) -> dict:
    input_dir = root / "artifacts" / "v4" / "evaluation" / "five_policy_quick"
    reports = {}
    files = []
    file_rows = []

    for key, label in POLICIES:
        path = input_dir / f"{key}_seed42_warmup2d_v4.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "VALID":
            raise ValueError(f"report is not VALID: {path}")
        reports[key] = payload
        relative = path.relative_to(root).as_posix()
        files.append(relative)
        file_rows.append(
            {
                "policy_key": key,
                "policy": label,
                "source_file": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "status": payload["status"],
                "model_hash": payload["metadata"].get("model_hash"),
                "config_hash": payload["metadata"].get("config_hash"),
            }
        )

    reference = reports[POLICIES[0][0]]["metadata"]
    for field in PAIRED_FIELDS:
        expected = reference.get(field)
        observed = {reports[key]["metadata"].get(field) for key, _ in POLICIES}
        if observed != {expected}:
            raise ValueError(f"paired field mismatch: {field}: {observed}")

    core_rows = []
    sla_rows = []
    wait_rows = []
    for key, label in POLICIES:
        metrics = reports[key]["metrics"]
        wait = metrics["active_wait_metrics"]
        core_rows.append(
            {
                "policy_key": key,
                "policy": label,
                "arrival_count": metrics["arrival_count"],
                "completed_count": metrics["completed_count"],
                "expired_count": metrics["expired_count"],
                "completion_pct": metric_value(metrics["completion_rate"]) * 100.0,
                "reservation_reliability_pct": metric_value(metrics["reservation_reliability"]) * 100.0,
                "total_cost_yuan": metrics["total_economic_cost_yuan"],
                "completed_cpu_hours": metrics["completed_cpu_hours"],
                "cost_per_completed_cpu_hour": metric_value(metrics["cost_yuan_per_completed_cpu_hour"]),
                "cost_per_arrived_cpu_hour": metric_value(metrics["cost_yuan_per_arrived_cpu_hour"]),
                "task_green_pct": metric_value(metrics["completed_task_green_coverage"]) * 100.0,
                "system_green_absorption_pct": metric_value(metrics["system_green_absorption_rate"]) * 100.0,
                "task_green_mwh": metrics["completed_task_green_coverage"].get("numerator"),
                "task_energy_mwh": metrics["completed_task_green_coverage"].get("denominator"),
                "system_green_used_mwh": metrics["system_green_absorption_rate"].get("numerator"),
                "system_green_supply_mwh": metrics["system_green_absorption_rate"].get("denominator"),
            }
        )

        reserved = metrics["reserved_ever_count"]
        wait_rows.append(
            {
                "policy_key": key,
                "policy": label,
                "reserved_count": reserved,
                "active_wait_count": wait["count"],
                "active_wait_pct": (wait["count"] / reserved * 100.0) if reserved else None,
                "mean_active_wait_sim": metric_value(wait["mean_active_wait_sim"]),
                "p95_active_wait_sim": metric_value(wait["p95_active_wait_sim"]),
                "positive_benefit_pct": (
                    metric_value(wait["positive_benefit_rate"]) * 100.0
                    if metric_value(wait["positive_benefit_rate"]) is not None
                    else None
                ),
            }
        )

        for sla_name in ("Hard", "Soft", "Flexible"):
            sla = metrics["sla_metrics"][sla_name]
            sla_rows.append(
                {
                    "policy_key": key,
                    "policy": label,
                    "sla_type": sla_name,
                    "task_count": sla["count"],
                    "preferred_on_time_pct": (
                        metric_value(sla["preferred_on_time_rate"]) * 100.0
                        if metric_value(sla["preferred_on_time_rate"]) is not None
                        else None
                    ),
                    "acceptable_tardy_pct": (
                        metric_value(sla["acceptable_tardy_rate"]) * 100.0
                        if metric_value(sla["acceptable_tardy_rate"]) is not None
                        else None
                    ),
                    "expired_pct": metric_value(sla["expired_rate"]) * 100.0,
                    "start_delay_p50_sim": metric_value(sla["start_delay_p50_sim"]),
                    "start_delay_p95_sim": metric_value(sla["start_delay_p95_sim"]),
                    "preferred_tardiness_p50_sim": metric_value(sla["preferred_start_tardiness_p50"]),
                    "preferred_tardiness_p95_sim": metric_value(sla["preferred_start_tardiness_p95"]),
                }
            )

    overview_ids = [
        "tradeoff-chart",
        "cost-chart",
        "green-chart",
        "core-table",
        "summary-cost",
        "summary-green",
        "summary-sla",
    ]
    core_definitions = [
        definition("completion_pct", "完成率", "完成任务数除以到达任务数。", overview_ids, files),
        definition(
            "cost_per_completed_cpu_hour",
            "单位完成 CPU 小时成本",
            "总经济成本除以完成 CPU 小时，单位为元/CPU·h；越低越好。",
            overview_ids,
            files,
        ),
        definition(
            "task_green_pct",
            "完成任务绿电覆盖率",
            "测量 cohort 完成任务在完整生命周期内的归因绿电量除以任务总能耗。",
            overview_ids,
            files,
        ),
        definition(
            "system_green_absorption_pct",
            "系统绿电吸收率",
            "正式测量窗口内全部运行负载使用的绿电量除以可用绿电量。",
            overview_ids,
            files,
        ),
    ]

    identity = {field: reference.get(field) for field in PAIRED_FIELDS}
    identity.update(
        {
            "report_count": len(POLICIES),
            "all_reports_valid": True,
            "paired_metadata_match": True,
        }
    )

    return {
        "title": "V4 五策略对比：Candidate DQN 同时降低成本并提高任务绿电覆盖",
        "surface": "report",
        "status": "reviewed",
        "buildStatus": "creating",
        "generatedAt": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "filters": [],
        "report": {
            "asOf": "2026-09-30",
            "originalQuestion": "可视化 V4 quick evaluation 的五种策略对比。",
        },
        "queries": {
            "policy_core": {
                "label": "五策略核心结果",
                "reportingField": "policy",
                "rows": core_rows,
                "methods": [
                    {
                        "language": "python",
                        "code": "Direct extraction from metrics and MetricValue fields in five VALID V4 JSON files; fractions converted to percent.",
                    }
                ],
                "source": source(files, "V4 quick five-policy evaluation reports", core_definitions),
            },
            "sla_detail": {
                "label": "分 SLA 指标",
                "reportingField": "policy",
                "rows": sla_rows,
                "source": source(files, "V4 quick evaluation SLA metrics", []),
            },
            "wait_detail": {
                "label": "主动等待指标",
                "reportingField": "policy",
                "rows": wait_rows,
                "source": source(files, "V4 quick evaluation active-wait metrics", []),
            },
            "experiment_identity": {
                "label": "实验一致性",
                "rows": [identity],
                "source": source(files, "V4 paired experiment metadata", []),
            },
            "source_manifest": {
                "label": "原始报告清单",
                "reportingField": "policy",
                "rows": file_rows,
                "source": source(files, "V4 evaluation report files", []),
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
