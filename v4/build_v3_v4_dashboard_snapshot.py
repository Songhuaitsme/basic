"""Build reviewed data for the full V3 versus V4 Candidate DQN dashboard."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "artifacts" / "cross_version_full"
OUTPUT = ROOT / "artifacts" / "cross_version_full" / "v3_v4_dashboard.reviewed.json"
SOURCES = {
    "V3": INPUT_DIR / "v3_candidate_dqn_seed42_warmup2d_measure1d.json",
    "V4": INPUT_DIR / "v4_candidate_dqn_seed42_warmup2d_measure1d.json",
}


def metric_value(value):
    return value.get("value") if isinstance(value, dict) else value


def pct(value):
    return None if value is None else 100.0 * value


def safe_change(current, baseline):
    if baseline in (None, 0) or current is None:
        return None
    return 100.0 * (current - baseline) / baseline


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(label, files, definitions=()):
    return {
        "label": label,
        "files": files,
        "executedAt": "2026-09-30",
        "evidenceFlow": [
            {
                "title": "Full one-day paired evaluation",
                "detail": (
                    "VALID V3 and V4 Candidate DQN reports with the same seed, task trace, "
                    "exogenous trace, topology, warm-up, and measurement window."
                ),
            }
        ],
        "metricDefinitions": list(definitions),
    }


def definition(field, label, text, component_ids, files):
    return {
        "field": field,
        "label": label,
        "definition": text,
        "componentIds": component_ids,
        "sourceLineage": [{"files": files}],
    }


def load_reports():
    reports = {}
    for version, path in SOURCES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "VALID":
            raise ValueError(f"report is not VALID: {path}")
        reports[version] = payload
    return reports


def completed_records(report):
    return {
        row["task_id"]: row
        for row in report["task_records"]
        if row["final_state"] == "Completed"
    }


def aggregate_pairs(rows, key_field):
    grouped = {}
    for row in rows:
        key = row[key_field]
        item = grouped.setdefault(
            key,
            {
                key_field: key,
                "task_count": 0,
                "cpu_hours": 0.0,
                "v3_cost_yuan": 0.0,
                "v4_cost_yuan": 0.0,
                "cost_delta_yuan": 0.0,
                "v3_green_mwh": 0.0,
                "v4_green_mwh": 0.0,
                "green_delta_mwh": 0.0,
            },
        )
        item["task_count"] += 1
        item["cpu_hours"] += row["cpu_hours"]
        item["v3_cost_yuan"] += row["v3_cost_yuan"]
        item["v4_cost_yuan"] += row["v4_cost_yuan"]
        item["cost_delta_yuan"] += row["cost_delta_yuan"]
        item["v3_green_mwh"] += row["v3_green_mwh"]
        item["v4_green_mwh"] += row["v4_green_mwh"]
        item["green_delta_mwh"] += row["green_delta_mwh"]
    for item in grouped.values():
        item["v3_cost_per_cpu_hour"] = item["v3_cost_yuan"] / item["cpu_hours"]
        item["v4_cost_per_cpu_hour"] = item["v4_cost_yuan"] / item["cpu_hours"]
        item["cost_delta_pct"] = safe_change(item["v4_cost_yuan"], item["v3_cost_yuan"])
    return list(grouped.values())


def build_snapshot():
    reports = load_reports()
    v3 = reports["V3"]
    v4 = reports["V4"]
    files = [path.relative_to(ROOT).as_posix() for path in SOURCES.values()]

    paired_fields = (
        "seed",
        "candidate_mode",
        "arrival_cutoff_sim",
        "evaluation_start_sim",
        "warmup_start_sim",
        "warmup_days",
        "topology_hash",
        "task_trace_hash",
        "exogenous_trace_hash",
        "dependency_lock_hash",
        "tariff_mode",
    )
    identity_rows = []
    for field in paired_fields:
        left = v3["metadata"].get(field)
        right = v4["metadata"].get(field)
        identity_rows.append(
            {
                "field": field,
                "v3": left,
                "v4": right,
                "matches": left == right,
                "role": "paired_control",
            }
        )
    for field in ("system_version", "model_hash", "config_hash", "code_hash"):
        identity_rows.append(
            {
                "field": field,
                "v3": v3["metadata"].get(field),
                "v4": v4["metadata"].get(field),
                "matches": v3["metadata"].get(field) == v4["metadata"].get(field),
                "role": "treatment_difference",
            }
        )
    paired_ok = all(row["matches"] for row in identity_rows if row["role"] == "paired_control")

    summary_rows = []
    for version, report in reports.items():
        metrics = report["metrics"]
        wait = metrics["active_wait_metrics"]
        summary_rows.append(
            {
                "version": version,
                "system_version": report["metadata"]["system_version"],
                "arrival_count": metrics["arrival_count"],
                "completed_count": metrics["completed_count"],
                "expired_count": metrics["expired_count"],
                "completion_pct": pct(metric_value(metrics["completion_rate"])),
                "total_cost_yuan": metrics["total_economic_cost_yuan"],
                "completed_cpu_hours": metrics["completed_cpu_hours"],
                "cost_per_completed_cpu_hour": metric_value(metrics["cost_yuan_per_completed_cpu_hour"]),
                "task_green_pct": pct(metric_value(metrics["completed_task_green_coverage"])),
                "system_green_absorption_pct": pct(metric_value(metrics["system_green_absorption_rate"])),
                "active_wait_count": wait["count"],
                "active_wait_pct": 100.0 * wait["count"] / metrics["reserved_ever_count"],
                "mean_active_wait_sim": metric_value(wait["mean_active_wait_sim"]),
                "p95_active_wait_sim": metric_value(wait["p95_active_wait_sim"]),
                "positive_wait_benefit_pct": pct(metric_value(wait["positive_benefit_rate"])),
                "final_settlement_time_sim": report["metadata"]["final_settlement_time_sim"],
            }
        )

    s3, s4 = summary_rows
    headline_rows = [
        {
            "metric": "单位完成 CPU 小时成本",
            "unit": "元/CPU·h",
            "direction": "lower",
            "v3": s3["cost_per_completed_cpu_hour"],
            "v4": s4["cost_per_completed_cpu_hour"],
        },
        {
            "metric": "任务绿电覆盖率",
            "unit": "%",
            "direction": "higher",
            "v3": s3["task_green_pct"],
            "v4": s4["task_green_pct"],
        },
        {
            "metric": "系统绿电吸收率",
            "unit": "%",
            "direction": "higher",
            "v3": s3["system_green_absorption_pct"],
            "v4": s4["system_green_absorption_pct"],
        },
        {
            "metric": "完成率",
            "unit": "%",
            "direction": "higher",
            "v3": s3["completion_pct"],
            "v4": s4["completion_pct"],
        },
        {
            "metric": "主动等待任务数",
            "unit": "个",
            "direction": "lower",
            "v3": s3["active_wait_count"],
            "v4": s4["active_wait_count"],
        },
        {
            "metric": "最终结算时间",
            "unit": "sim",
            "direction": "lower",
            "v3": s3["final_settlement_time_sim"],
            "v4": s4["final_settlement_time_sim"],
        },
    ]
    for row in headline_rows:
        row["delta"] = row["v4"] - row["v3"]
        row["delta_pct"] = safe_change(row["v4"], row["v3"])
        row["improved"] = row["delta"] < 0 if row["direction"] == "lower" else row["delta"] > 0

    sla_rows = []
    for version, report in reports.items():
        for sla_name, sla in report["metrics"]["sla_metrics"].items():
            sla_rows.append(
                {
                    "version": version,
                    "sla_type": sla_name,
                    "task_count": sla["count"],
                    "preferred_on_time_pct": pct(metric_value(sla["preferred_on_time_rate"])),
                    "acceptable_tardy_pct": pct(metric_value(sla["acceptable_tardy_rate"])),
                    "expired_pct": pct(metric_value(sla["expired_rate"])),
                    "start_delay_p50_sim": metric_value(sla["start_delay_p50_sim"]),
                    "start_delay_p95_sim": metric_value(sla["start_delay_p95_sim"]),
                    "preferred_tardiness_p95_sim": metric_value(sla["preferred_start_tardiness_p95"]),
                }
            )

    records = {version: completed_records(report) for version, report in reports.items()}
    if set(records["V3"]) != set(records["V4"]):
        raise ValueError("completed task cohorts differ between V3 and V4")

    task_pairs = []
    for task_id in sorted(records["V3"]):
        left = records["V3"][task_id]
        right = records["V4"][task_id]
        cpu_hours = float(left["cpu_work_cpu_hours"])
        if not math.isclose(cpu_hours, float(right["cpu_work_cpu_hours"]), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"CPU hours differ for task {task_id}")
        row = {
            "task_id": task_id,
            "sla_type": left["sla_type"],
            "cpu_hours": cpu_hours,
            "v3_cost_yuan": float(left["task_attributed_cost_yuan"]),
            "v4_cost_yuan": float(right["task_attributed_cost_yuan"]),
            "v3_green_mwh": float(left["task_attributed_green_energy_mwh"]),
            "v4_green_mwh": float(right["task_attributed_green_energy_mwh"]),
            "v3_active_wait_sim": float(left["active_wait_sim"]),
            "v4_active_wait_sim": float(right["active_wait_sim"]),
            "v3_start_delay_sim": float(left["start_delay_sim"]),
            "v4_start_delay_sim": float(right["start_delay_sim"]),
            "v3_target_node": left["target_node"],
            "v4_target_node": right["target_node"],
        }
        row["cost_delta_yuan"] = row["v4_cost_yuan"] - row["v3_cost_yuan"]
        row["green_delta_mwh"] = row["v4_green_mwh"] - row["v3_green_mwh"]
        row["wait_delta_sim"] = row["v4_active_wait_sim"] - row["v3_active_wait_sim"]
        row["start_delay_delta_sim"] = row["v4_start_delay_sim"] - row["v3_start_delay_sim"]
        row["node_changed"] = row["v3_target_node"] != row["v4_target_node"]
        task_pairs.append(row)

    sla_cost_rows = aggregate_pairs(task_pairs, "sla_type")
    total_cpu = sum(row["cpu_hours"] for row in sla_cost_rows)
    for row in sla_cost_rows:
        row["cpu_share_pct"] = 100.0 * row["cpu_hours"] / total_cpu

    savings = sorted(task_pairs, key=lambda row: row["cost_delta_yuan"])
    top_task_rows = savings[:15] + list(reversed(savings[-15:]))
    for rank, row in enumerate(savings[:15], 1):
        row["rank_group"] = "V4 节省最多"
        row["rank"] = rank
    for rank, row in enumerate(reversed(savings[-15:]), 1):
        row["rank_group"] = "V4 增加最多"
        row["rank"] = rank

    task_outcome_counts = {
        "v4_cheaper": sum(row["cost_delta_yuan"] < -1e-9 for row in task_pairs),
        "v3_cheaper": sum(row["cost_delta_yuan"] > 1e-9 for row in task_pairs),
        "equal": sum(abs(row["cost_delta_yuan"]) <= 1e-9 for row in task_pairs),
        "node_changed": sum(row["node_changed"] for row in task_pairs),
    }
    task_outcome_rows = [
        {"outcome": "V4 成本更低", "task_count": task_outcome_counts["v4_cheaper"]},
        {"outcome": "V3 成本更低", "task_count": task_outcome_counts["v3_cheaper"]},
        {"outcome": "成本相同", "task_count": task_outcome_counts["equal"]},
    ]

    file_rows = [
        {
            "version": version,
            "source_file": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "status": reports[version]["status"],
            "model_hash": reports[version]["metadata"]["model_hash"],
            "config_hash": reports[version]["metadata"]["config_hash"],
        }
        for version, path in SOURCES.items()
    ]

    common_definitions = [
        definition(
            "cost_per_completed_cpu_hour",
            "单位完成 CPU 小时成本",
            "测量 cohort 的总经济成本除以完成 CPU 小时；越低越好。",
            ["kpi-cost", "headline-comparison", "sla-cost-chart"],
            files,
        ),
        definition(
            "task_green_pct",
            "任务绿电覆盖率",
            "完成任务归因绿电量除以任务总能耗；越高越好。",
            ["kpi-green", "headline-comparison"],
            files,
        ),
        definition(
            "active_wait_count",
            "主动等待任务数",
            "最终提交候选相对最早可行候选存在正等待的已预留任务数。",
            ["kpi-wait", "wait-comparison"],
            files,
        ),
        definition(
            "cost_delta_yuan",
            "任务成本差额",
            "V4 任务归因成本减 V3；负值表示 V4 节省。",
            ["task-delta-chart", "task-detail-table"],
            files,
        ),
    ]

    return {
        "title": "V4 用更少等待换来更低成本和更高任务绿电覆盖",
        "surface": "dashboard",
        "status": "reviewed",
        "buildStatus": "creating",
        "generatedAt": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "filters": [
            {"field": "sla_type", "label": "SLA 类型", "type": "categorical", "default": "All"}
        ],
        "dashboard": {
            "asOf": "2026-09-30",
            "originalQuestion": "对比完整一日测量窗口下 Candidate DQN 的 V3 与 V4 表现。",
            "pairedComparisonValid": paired_ok,
            "taskOutcomeCounts": task_outcome_counts,
        },
        "queries": {
            "version_summary": {
                "label": "版本核心指标",
                "reportingField": "version",
                "rows": summary_rows,
                "source": source("V3/V4 full evaluation summaries", files, common_definitions),
            },
            "headline_comparison": {
                "label": "核心指标差异",
                "reportingField": "metric",
                "rows": headline_rows,
                "source": source("Derived paired metric comparison", files, common_definitions),
            },
            "sla_metrics": {
                "label": "SLA 表现",
                "reportingField": "sla_type",
                "rows": sla_rows,
                "source": source("SLA metrics from paired reports", files),
            },
            "sla_cost_contribution": {
                "label": "按 SLA 的成本与工作量贡献",
                "reportingField": "sla_type",
                "rows": sla_cost_rows,
                "source": source("Task-level paired cost attribution", files, common_definitions),
            },
            "task_outcomes": {
                "label": "逐任务成本胜负",
                "reportingField": "outcome",
                "rows": task_outcome_rows,
                "source": source("Task-level paired outcome counts", files),
            },
            "top_task_deltas": {
                "label": "成本变化最大的任务",
                "reportingField": "task_id",
                "rows": top_task_rows,
                "source": source("Largest paired task cost deltas", files, common_definitions),
            },
            "task_pairs": {
                "label": "逐任务配对明细",
                "reportingField": "task_id",
                "rows": task_pairs,
                "source": source("Completed-task paired records", files, common_definitions),
            },
            "experiment_identity": {
                "label": "实验一致性",
                "reportingField": "field",
                "rows": identity_rows,
                "source": source("Evaluation metadata", files),
            },
            "source_manifest": {
                "label": "原始文件清单",
                "reportingField": "version",
                "rows": file_rows,
                "source": source("Source report manifest", files),
            },
        },
    }


def main():
    snapshot = build_snapshot()
    OUTPUT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "VALID", "output": str(OUTPUT.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
