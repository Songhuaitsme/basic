"""Build auditable five-policy comparison tables from the raw evaluation JSON files."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
POLICIES = (
    "earliest_feasible",
    "lowest_cost",
    "highest_green",
    "equal_weight",
    "candidate_dqn",
)
LABELS = {
    "earliest_feasible": "Earliest feasible",
    "lowest_cost": "Lowest cost",
    "highest_green": "Highest green",
    "equal_weight": "Equal weight",
    "candidate_dqn": "Candidate DQN",
}


def metric_value(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"] if value.get("status") == "VALID" else None
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    reports = {}
    source_rows = []
    for policy in POLICIES:
        path = ROOT / f"{policy}_seed42.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[policy] = report
        source_rows.append({
            "policy": policy,
            "source_file": str(path.relative_to(ROOT.parents[2])).replace("\\", "/"),
            "status": report.get("status"),
            "seed": report["metadata"]["seed"],
            "task_trace_hash": report["metadata"]["task_trace_hash"],
            "exogenous_trace_hash": report["metadata"]["exogenous_trace_hash"],
            "arrival_count": report["metrics"]["arrival_count"],
            "completed_count": report["metrics"]["completed_count"],
            "unsettled_count": len(report["unsettled_task_ids"]),
        })

    reference = reports[POLICIES[0]]
    required_equal = (
        "seed",
        "arrival_cutoff_sim",
        "task_trace_hash",
        "exogenous_trace_hash",
        "topology_hash",
        "config_hash",
        "dependency_lock_hash",
    )
    problems = []
    for policy, report in reports.items():
        if report.get("status") != "VALID":
            problems.append(f"{policy}: status is not VALID")
        if report.get("unsettled_task_ids"):
            problems.append(f"{policy}: unsettled tasks remain")
        for key in required_equal:
            if report["metadata"].get(key) != reference["metadata"].get(key):
                problems.append(f"{policy}: mismatched metadata {key}")
    if problems:
        raise ValueError("; ".join(problems))

    summary_rows = []
    for policy, report in reports.items():
        metrics = report["metrics"]
        sla = metrics["sla_metrics"]
        wait = metrics["active_wait_metrics"]
        summary_rows.append({
            "policy": policy,
            "policy_label": LABELS[policy],
            "arrival_count": metrics["arrival_count"],
            "reserved_ever_count": metrics["reserved_ever_count"],
            "rejected_count": metrics["rejected_count"],
            "expired_count": metrics["expired_count"],
            "completed_count": metrics["completed_count"],
            "failed_count": metrics["failed_count"],
            "acceptance_rate": metric_value(metrics["acceptance_rate"]),
            "completion_rate": metric_value(metrics["completion_rate"]),
            "reservation_reliability": metric_value(metrics["reservation_reliability"]),
            "total_economic_cost_yuan": metrics["total_economic_cost_yuan"],
            "completed_cpu_hours": metrics["completed_cpu_hours"],
            "cost_yuan_per_completed_cpu_hour": metric_value(metrics["cost_yuan_per_completed_cpu_hour"]),
            "cost_yuan_per_arrived_cpu_hour": metric_value(metrics["cost_yuan_per_arrived_cpu_hour"]),
            "completed_task_green_coverage": metric_value(metrics["completed_task_green_coverage"]),
            "system_green_absorption_rate": metric_value(metrics["system_green_absorption_rate"]),
            "hard_start_delay_p95_sim": metric_value(sla["Hard"]["start_delay_p95_sim"]),
            "soft_preferred_on_time_rate": metric_value(sla["Soft"]["preferred_on_time_rate"]),
            "soft_start_delay_p95_sim": metric_value(sla["Soft"]["start_delay_p95_sim"]),
            "flexible_preferred_on_time_rate": metric_value(sla["Flexible"]["preferred_on_time_rate"]),
            "flexible_start_delay_p95_sim": metric_value(sla["Flexible"]["start_delay_p95_sim"]),
            "active_wait_count": wait["count"],
            "mean_active_wait_sim": metric_value(wait["mean_active_wait_sim"]),
            "p95_active_wait_sim": metric_value(wait["p95_active_wait_sim"]),
            "positive_wait_benefit_rate": metric_value(wait["positive_benefit_rate"]),
        })

    baseline = next(row for row in summary_rows if row["policy"] == "equal_weight")
    for row in summary_rows:
        row["cost_change_vs_equal_weight"] = (
            row["total_economic_cost_yuan"] - baseline["total_economic_cost_yuan"]
        ) / baseline["total_economic_cost_yuan"]
        row["green_coverage_change_vs_equal_weight"] = (
            row["completed_task_green_coverage"]
            - baseline["completed_task_green_coverage"]
        )
        row["soft_ontime_change_vs_equal_weight"] = (
            row["soft_preferred_on_time_rate"]
            - baseline["soft_preferred_on_time_rate"]
        )
        row["flexible_ontime_change_vs_equal_weight"] = (
            row["flexible_preferred_on_time_rate"]
            - baseline["flexible_preferred_on_time_rate"]
        )
        row["active_wait_change_vs_equal_weight"] = (
            row["active_wait_count"] - baseline["active_wait_count"]
        )

    sla_rows = []
    for policy, report in reports.items():
        for sla_type, values in report["metrics"]["sla_metrics"].items():
            sla_rows.append({
                "policy": policy,
                "policy_label": LABELS[policy],
                "sla_type": sla_type,
                "task_count": values["count"],
                "preferred_on_time_rate": metric_value(values["preferred_on_time_rate"]),
                "acceptable_tardy_rate": metric_value(values["acceptable_tardy_rate"]),
                "expired_rate": metric_value(values["expired_rate"]),
                "start_delay_p50_sim": metric_value(values["start_delay_p50_sim"]),
                "start_delay_p95_sim": metric_value(values["start_delay_p95_sim"]),
                "preferred_start_tardiness_p50": metric_value(values["preferred_start_tardiness_p50"]),
                "preferred_start_tardiness_p95": metric_value(values["preferred_start_tardiness_p95"]),
            })

    metric_specs = {
        "total_economic_cost_yuan": ("Cost", "CNY", "lower"),
        "cost_yuan_per_completed_cpu_hour": ("Cost", "CNY/CPU-hour", "lower"),
        "completed_task_green_coverage": ("Green", "fraction", "higher"),
        "system_green_absorption_rate": ("Green", "fraction", "higher"),
        "completion_rate": ("Reliability", "fraction", "higher"),
        "reservation_reliability": ("Reliability", "fraction", "higher"),
        "soft_preferred_on_time_rate": ("SLA", "fraction", "higher"),
        "flexible_preferred_on_time_rate": ("SLA", "fraction", "higher"),
        "hard_start_delay_p95_sim": ("SLA", "simulation time", "lower"),
        "active_wait_count": ("Waiting", "tasks", "lower diagnostic"),
        "p95_active_wait_sim": ("Waiting", "simulation time", "lower diagnostic"),
    }
    long_rows = []
    for row in summary_rows:
        for metric, (group, unit, direction) in metric_specs.items():
            value = row[metric]
            baseline_value = baseline[metric]
            long_rows.append({
                "policy": row["policy"],
                "policy_label": row["policy_label"],
                "metric_group": group,
                "metric": metric,
                "unit": unit,
                "direction": direction,
                "value": value,
                "equal_weight_value": baseline_value,
                "absolute_delta_vs_equal_weight": (
                    None if value is None or baseline_value is None else value - baseline_value
                ),
                "relative_delta_vs_equal_weight": (
                    None
                    if value is None or baseline_value in (None, 0)
                    else (value - baseline_value) / baseline_value
                ),
            })

    write_csv(OUT / "policy_metrics_wide.csv", summary_rows)
    write_csv(OUT / "policy_metrics_long.csv", long_rows)
    write_csv(OUT / "sla_metrics.csv", sla_rows)
    write_csv(OUT / "source_manifest.csv", source_rows)

    db_path = OUT / "strategy_comparison.sqlite"
    with sqlite3.connect(db_path) as connection:
        for table in ("policy_metrics", "metric_long", "sla_metrics", "source_manifest"):
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')

        def insert_table(name, rows):
            fields = list(rows[0])
            types = []
            for field in fields:
                sample = next((row[field] for row in rows if row[field] is not None), None)
                types.append("REAL" if isinstance(sample, (int, float)) else "TEXT")
            connection.execute(
                f'CREATE TABLE "{name}" ('
                + ", ".join(f'"{field}" {kind}' for field, kind in zip(fields, types))
                + ")"
            )
            connection.executemany(
                f'INSERT INTO "{name}" VALUES (' + ",".join("?" for _ in fields) + ")",
                [[row[field] for field in fields] for row in rows],
            )

        insert_table("policy_metrics", summary_rows)
        insert_table("metric_long", long_rows)
        insert_table("sla_metrics", sla_rows)
        insert_table("source_manifest", source_rows)
        connection.commit()

    checks = {
        "status": "PASS",
        "source_report_count": len(reports),
        "all_reports_valid": all(report["status"] == "VALID" for report in reports.values()),
        "paired_metadata_fields": list(required_equal),
        "paired_metadata_match": True,
        "all_unsettled_counts_zero": all(not report["unsettled_task_ids"] for report in reports.values()),
        "policy_row_count": len(summary_rows),
        "sla_row_count": len(sla_rows),
        "long_metric_row_count": len(long_rows),
        "single_seed_limitation": "Seed 42 only; no cross-seed confidence interval.",
    }
    (OUT / "analysis_validation.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    artifact_path = OUT / "artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    summary_source = {
        "id": "strategy_summary",
        "label": "Seed 42 five-policy reviewed aggregate metrics",
        "path": "artifacts/v1/evaluation/strategy_comparison_seed42/strategy_comparison.sqlite",
        "query": {
            "engine": "SQLite",
            "language": "SQL",
            "sql": "SELECT * FROM policy_metrics ORDER BY policy;",
            "description": "Read one reviewed aggregate metric row per policy after paired metadata validation.",
            "executed_at": "2026-08-17T00:00:00+08:00",
            "filters": [
                "seed=42",
                "evaluation status=VALID",
                "identical task and exogenous trace hashes",
                "zero unsettled tasks",
                "candidate_dqn epsilon=0",
            ],
            "metric_definitions": [
                "Economic cost is total attributed yuan; lower is better.",
                "Cost per completed CPU-hour divides total economic cost by completed CPU-hours; lower is better.",
                "Completed-task green coverage divides completed-task green energy by completed-task energy; higher is better.",
                "Completion rate divides completed tasks by arrivals; reservation reliability divides completed reservations by tasks ever reserved.",
                "All changes versus equal-weight are policy minus equal-weight.",
            ],
            "tables_used": ["strategy_comparison.policy_metrics"],
        },
    }
    sla_source = {
        "id": "sla_summary",
        "label": "Seed 42 policy-by-SLA reviewed metrics",
        "path": "artifacts/v1/evaluation/strategy_comparison_seed42/strategy_comparison.sqlite",
        "query": {
            "engine": "SQLite",
            "language": "SQL",
            "sql": "SELECT * FROM sla_metrics ORDER BY policy, sla_type;",
            "description": "Read the reviewed SLA metrics at policy and SLA-type grain.",
            "executed_at": "2026-08-17T00:00:00+08:00",
            "filters": ["seed=42", "evaluation status=VALID"],
            "metric_definitions": [
                "Preferred on-time rate is the share starting within the preferred window for Soft and Flexible tasks.",
                "Start-delay percentiles use simulation-time units.",
                "Hard tasks do not have a preferred-start deadline, so preferred on-time is not applicable.",
            ],
            "tables_used": ["strategy_comparison.sla_metrics"],
        },
    }
    manifest_source = {
        "id": "source_manifest",
        "label": "Five raw evaluation report audit manifest",
        "path": "artifacts/v1/evaluation/strategy_comparison_seed42/strategy_comparison.sqlite",
        "query": {
            "engine": "SQLite",
            "language": "SQL",
            "sql": "SELECT * FROM source_manifest ORDER BY policy;",
            "description": "List each raw evaluation JSON and the metadata used to verify comparability.",
            "executed_at": "2026-08-17T00:00:00+08:00",
            "filters": ["seed=42"],
            "metric_definitions": [
                "Comparable reports must be VALID and share seed, task trace, exogenous trace, topology, configuration, and dependency lock metadata.",
            ],
            "tables_used": ["strategy_comparison.source_manifest"],
        },
    }
    sources = [summary_source, sla_source, manifest_source]
    artifact["manifest"]["sources"] = sources
    artifact["sources"] = sources
    artifact["manifest"]["description"] = (
        "Seed 42 同一轨迹下五种冻结调度策略的完整源指标、成本、绿电、SLA 与等待队列比较。"
    )

    existing_charts = {
        chart["id"]: chart
        for chart in artifact["manifest"].get("charts", [])
        if chart["id"] in {"cost_per_cpu", "green_coverage"}
    }
    for chart in existing_charts.values():
        chart["sourceId"] = "strategy_summary"
        chart["dataset"] = "strategy_comparison"

    added_charts = [
        {
            "id": "soft_ontime",
            "title": "Soft 任务首选窗口准时率",
            "subtitle": "Seed 42，同一任务轨迹；比例越高越好，Soft 任务数为 1,429",
            "showDescription": True,
            "intent": "comparison",
            "question": "五种策略对 Soft 任务首选开始窗口的保障程度如何？",
            "rationale": "五个策略的一项同口径比例指标适合用横向条形图直接比较。",
            "comparisonContext": {"baseline": "equal_weight", "grain": "policy", "unit": "fraction", "semanticFamily": "Soft SLA"},
            "type": "horizontalBar",
            "dataset": "strategy_comparison",
            "sourceId": "strategy_summary",
            "encodings": {
                "x": {"field": "policy_label", "type": "nominal", "label": "策略"},
                "y": {"field": "soft_preferred_on_time_rate", "type": "quantitative", "format": "percent", "label": "Soft 准时率"},
            },
            "xAxisTitle": "策略",
            "yAxisTitle": "Soft 首选窗口准时率",
            "valueFormat": "percent",
            "layout": "full",
            "surface": {"palette": {"kind": "single", "root": "gold"}, "valueLabels": {"mode": "all"}},
        },
        {
            "id": "flexible_ontime",
            "title": "Flexible 任务首选窗口准时率",
            "subtitle": "Seed 42，同一任务轨迹；比例越高越好，Flexible 任务数为 353",
            "showDescription": True,
            "intent": "comparison",
            "question": "五种策略对 Flexible 任务首选开始窗口的保障程度如何？",
            "rationale": "五个策略的一项同口径比例指标适合用横向条形图直接比较。",
            "comparisonContext": {"baseline": "equal_weight", "grain": "policy", "unit": "fraction", "semanticFamily": "Flexible SLA"},
            "type": "horizontalBar",
            "dataset": "strategy_comparison",
            "sourceId": "strategy_summary",
            "encodings": {
                "x": {"field": "policy_label", "type": "nominal", "label": "策略"},
                "y": {"field": "flexible_preferred_on_time_rate", "type": "quantitative", "format": "percent", "label": "Flexible 准时率"},
            },
            "xAxisTitle": "策略",
            "yAxisTitle": "Flexible 首选窗口准时率",
            "valueFormat": "percent",
            "layout": "full",
            "surface": {"palette": {"kind": "single", "root": "orange"}, "valueLabels": {"mode": "all"}},
        },
        {
            "id": "active_wait_count",
            "title": "活跃等待任务数",
            "subtitle": "Seed 42，同一任务轨迹；用于诊断主动等待行为，不等同于失败任务",
            "showDescription": True,
            "intent": "comparison",
            "question": "各策略有多少任务主动等待未来时隙？",
            "rationale": "任务数为同单位离散策略比较，横向条形图便于查看差异幅度。",
            "comparisonContext": {"baseline": "equal_weight", "grain": "policy", "unit": "tasks", "semanticFamily": "active waiting"},
            "type": "horizontalBar",
            "dataset": "strategy_comparison",
            "sourceId": "strategy_summary",
            "encodings": {
                "x": {"field": "policy_label", "type": "nominal", "label": "策略"},
                "y": {"field": "active_wait_count", "type": "quantitative", "format": "number", "label": "活跃等待任务"},
            },
            "xAxisTitle": "策略",
            "yAxisTitle": "活跃等待任务数",
            "valueFormat": "number",
            "unit": "tasks",
            "layout": "full",
            "surface": {"palette": {"kind": "single", "root": "pink"}, "valueLabels": {"mode": "all"}},
        },
    ]
    artifact["manifest"]["charts"] = [
        existing_charts["cost_per_cpu"],
        existing_charts["green_coverage"],
        *added_charts,
    ]

    policy_table = next(
        table for table in artifact["manifest"]["tables"] if table["id"] == "policy_metrics"
    )
    policy_table["sourceId"] = "strategy_summary"
    policy_table["dataset"] = "strategy_comparison"
    policy_table["columns"] = [
        {"field": "policy_label", "label": "策略", "type": "string"},
        {"field": "total_economic_cost_yuan", "label": "总经济成本（元）", "format": "currency"},
        {"field": "cost_yuan_per_completed_cpu_hour", "label": "元/完成 CPU 小时", "format": "number"},
        {"field": "cost_change_vs_equal_weight", "label": "成本相对 Equal-weight", "format": "percent", "movement": True},
        {"field": "completed_task_green_coverage", "label": "任务绿电覆盖率", "format": "percent"},
        {"field": "green_coverage_change_vs_equal_weight", "label": "绿电覆盖变化", "format": "percent", "movement": True},
        {"field": "completion_rate", "label": "完成率", "format": "percent"},
        {"field": "reservation_reliability", "label": "预约可靠性", "format": "percent"},
        {"field": "soft_preferred_on_time_rate", "label": "Soft 准时率", "format": "percent"},
        {"field": "flexible_preferred_on_time_rate", "label": "Flexible 准时率", "format": "percent"},
        {"field": "active_wait_count", "label": "活跃等待任务", "format": "number"},
    ]
    sla_table = {
        "id": "sla_detail",
        "title": "策略与 SLA 类型明细",
        "subtitle": "Seed 42；Hard 的首选窗口指标不适用，延迟使用仿真时间单位",
        "showDescription": True,
        "dataset": "sla_comparison",
        "defaultSort": {"field": "policy_label", "direction": "asc"},
        "density": "dense",
        "sourceId": "sla_summary",
        "layout": "full",
        "columns": [
            {"field": "policy_label", "label": "策略", "type": "string"},
            {"field": "sla_type", "label": "SLA 类型", "type": "string"},
            {"field": "task_count", "label": "任务数", "format": "number"},
            {"field": "preferred_on_time_rate", "label": "首选窗口准时率", "format": "percent"},
            {"field": "acceptable_tardy_rate", "label": "可接受迟到率", "format": "percent"},
            {"field": "expired_rate", "label": "过期率", "format": "percent"},
            {"field": "start_delay_p50_sim", "label": "开始延迟 P50", "format": "number"},
            {"field": "start_delay_p95_sim", "label": "开始延迟 P95", "format": "number"},
        ],
    }
    source_table = {
        "id": "raw_source_manifest",
        "title": "五份原始评估文件",
        "subtitle": "Seed 42；状态、轨迹哈希与任务总量用于验证可比性",
        "showDescription": True,
        "dataset": "source_reports",
        "defaultSort": {"field": "policy", "direction": "asc"},
        "density": "dense",
        "sourceId": "source_manifest",
        "layout": "full",
        "columns": [
            {"field": "policy", "label": "策略", "type": "string"},
            {"field": "source_file", "label": "源文件", "type": "string"},
            {"field": "status", "label": "状态", "type": "string"},
            {"field": "seed", "label": "Seed", "format": "number"},
            {"field": "arrival_count", "label": "到达任务", "format": "number"},
            {"field": "completed_count", "label": "完成任务", "format": "number"},
            {"field": "unsettled_count", "label": "未结算任务", "format": "number"},
        ],
    }
    artifact["manifest"]["tables"] = [policy_table, sla_table, source_table]

    managed_block_ids = {
        "soft_chart",
        "flexible_finding",
        "flexible_chart",
        "waiting_finding",
        "waiting_chart",
        "sla_detail_intro",
        "sla_detail_table",
        "source_data_intro",
        "source_manifest_table",
    }
    old_blocks = [
        block for block in artifact["manifest"]["blocks"] if block["id"] not in managed_block_ids
    ]
    new_blocks = []
    for block in old_blocks:
        new_blocks.append(block)
        if block["id"] == "service_finding":
            new_blocks.extend([
                {"id": "soft_chart", "type": "chart", "chartId": "soft_ontime", "layout": "full"},
                {"id": "flexible_finding", "type": "markdown", "sourceId": "strategy_summary", "body": "## Flexible 准时率揭示局部服务权衡\n\nEarliest-feasible 的 Flexible 准时率为 **100%**，equal-weight 为 **96.32%**，DQN 为 **94.33%**。DQN 相比 equal-weight 下降 **1.98 个百分点**；这项退化没有影响总体完成率，但在正式多 seed 评估中必须作为独立护栏。", "layout": "full"},
                {"id": "flexible_chart", "type": "chart", "chartId": "flexible_ontime", "layout": "full"},
                {"id": "waiting_finding", "type": "markdown", "sourceId": "strategy_summary", "body": "## DQN 的主动等待少于 Equal-weight\n\nDQN 记录 **1,134** 个活跃等待任务，比 equal-weight 少 **813 个（-41.8%）**。Lowest-cost 为 1,063，Highest-green 为 1,580；Earliest-feasible 不主动等待未来时隙。等待数量只描述策略行为，不应单独解释为服务质量改善。", "layout": "full"},
                {"id": "waiting_chart", "type": "chart", "chartId": "active_wait_count", "layout": "full"},
            ])
        if block["id"] == "metrics_table_block":
            new_blocks.extend([
                {"id": "sla_detail_intro", "type": "markdown", "sourceId": "sla_summary", "body": "## SLA 源指标明细\n\n下面按策略与 Hard、Soft、Flexible 三类 SLA 展开任务数、首选窗口准时率、可接受迟到率、过期率和开始延迟分位数。Hard 没有首选开始窗口，对应准时率字段为空属于定义上的不适用，而不是缺失数据。", "layout": "full"},
                {"id": "sla_detail_table", "type": "table", "tableId": "sla_detail", "layout": "full"},
                {"id": "source_data_intro", "type": "markdown", "sourceId": "source_manifest", "body": "## 原始评估文件与可比性\n\n五份 JSON 均为 `VALID`，seed、任务轨迹、外生轨迹、拓扑、配置和依赖锁一致；每份均包含 2,416 个到达任务、2,401 个完成任务和 0 个未结算任务。由此支持 seed 42 内的公平横向比较。", "layout": "full"},
                {"id": "source_manifest_table", "type": "table", "tableId": "raw_source_manifest", "layout": "full"},
            ])
    artifact["manifest"]["blocks"] = new_blocks
    artifact["snapshot"]["datasets"] = {
        "strategy_comparison": summary_rows,
        "sla_comparison": sla_rows,
        "source_reports": source_rows,
    }
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## tl;dr\n",
                    "Seed 42 上，Candidate-DQN 相比 Equal-weight 降低总成本 1.29%，提高完成任务绿电覆盖 2.04 个百分点，完成率不变；但 Flexible 准时率下降 1.98 个百分点。单一 seed 只能用于诊断。",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Context & Methods\n",
                    "读取五份冻结策略评估 JSON，验证状态与配对轨迹元数据，然后生成宽表、长表、SLA 表及相对 Equal-weight 的差值。\n",
                    "### Key Assumptions\n",
                    "五份报告的 seed、任务轨迹、外生轨迹、拓扑、配置和依赖锁哈希必须一致。",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "ROOT = Path('.')\n",
                    "DATA = ROOT / 'artifacts/v1/evaluation/strategy_comparison_seed42'\n",
                    "wide = pd.read_csv(DATA / 'policy_metrics_wide.csv')\n",
                    "sla = pd.read_csv(DATA / 'sla_metrics.csv')\n",
                    "wide",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Data\n", "宽表每行一个策略；SLA 表每行一个策略与 SLA 类型。"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "wide[['policy_label', 'cost_yuan_per_completed_cpu_hour', 'completed_task_green_coverage', 'completion_rate', 'soft_preferred_on_time_rate', 'flexible_preferred_on_time_rate', 'active_wait_count']]",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Results\n", "分别绘制成本、绿电、SLA 与等待任务，避免把不同单位放在同一坐标轴。"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(2, 2, figsize=(13, 9))\n",
                    "wide.plot.barh(x='policy_label', y='cost_yuan_per_completed_cpu_hour', ax=axes[0,0], legend=False, title='Cost per completed CPU-hour')\n",
                    "wide.plot.barh(x='policy_label', y='completed_task_green_coverage', ax=axes[0,1], legend=False, title='Completed-task green coverage')\n",
                    "wide.plot.barh(x='policy_label', y=['soft_preferred_on_time_rate','flexible_preferred_on_time_rate'], ax=axes[1,0], title='Preferred on-time rate')\n",
                    "wide.plot.barh(x='policy_label', y='active_wait_count', ax=axes[1,1], legend=False, title='Active-wait tasks')\n",
                    "plt.tight_layout()",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Takeaways\n",
                    "- DQN 在成本与绿电上同时优于 Equal-weight。\n",
                    "- Lowest-cost 与 Highest-green 是明显的单目标极端策略。\n",
                    "- DQN 的 Flexible 准时率退化需要在多 seed 上复核。\n",
                    "- 下一步应做至少 10 个未见 seed 的配对评估。",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUT / "five_policy_analysis.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
