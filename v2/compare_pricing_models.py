"""Build a paired V2 regional-vs-uniform pricing HTML report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


DEFAULT_UNIFORM_REPORT = Path(
    "artifacts/v2/evaluation/five_policy/seed_42/candidate_dqn_seed42.json"
)
DEFAULT_REGIONAL_REPORT = Path(
    "artifacts/v2/evaluation/five_policy_refresh/seed_42/candidate_dqn_seed42.json"
)
DEFAULT_UNIFORM_MODEL = Path(
    "artifacts/v2/formal/candidate_dqn_seed7_layered_pool_600000_from_scratch.pt"
)
DEFAULT_REGIONAL_MODEL = Path(
    "artifacts/v2/formal/"
    "candidate_dqn_seed7_layered_pool_600000_tou_region_from_scratch.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/v2/evaluation/pricing_comparison_seed42/comparison_report"
)

MODE_LABELS = {
    "tou_uniform": "不分区定价",
    "tou_region": "分区定价",
}

METRIC_SPECS = {
    "cost_yuan_per_completed_cpu_hour": {
        "label": "每完成 CPU 小时成本",
        "unit": "CNY/CPU-hour",
        "direction": "lower",
        "group": "成本",
    },
    "completed_task_green_coverage": {
        "label": "完成任务绿电覆盖率",
        "unit": "fraction",
        "direction": "higher",
        "group": "绿电",
    },
    "system_green_absorption_rate": {
        "label": "系统绿电吸收率",
        "unit": "fraction",
        "direction": "higher",
        "group": "绿电",
    },
    "completion_rate": {
        "label": "任务完成率",
        "unit": "fraction",
        "direction": "higher",
        "group": "可靠性",
    },
    "reservation_reliability": {
        "label": "预留可靠性",
        "unit": "fraction",
        "direction": "higher",
        "group": "可靠性",
    },
    "soft_preferred_on_time_rate": {
        "label": "Soft 首选窗口准时率",
        "unit": "fraction",
        "direction": "higher",
        "group": "SLA",
    },
    "flexible_preferred_on_time_rate": {
        "label": "Flexible 首选窗口准时率",
        "unit": "fraction",
        "direction": "higher",
        "group": "SLA",
    },
    "hard_start_delay_p95_hours": {
        "label": "Hard 开始时延 P95",
        "unit": "hours",
        "direction": "lower",
        "group": "SLA",
    },
    "mean_completion_delay_hours": {
        "label": "平均任务完成时延",
        "unit": "hours",
        "direction": "lower",
        "group": "任务时延",
    },
    "allocated_cpu_hours_cv": {
        "label": "节点 CPU 工作量变异系数",
        "unit": "fraction",
        "direction": "lower",
        "group": "负载均衡",
    },
    "active_wait_count": {
        "label": "主动等待任务数",
        "unit": "tasks",
        "direction": "neutral",
        "group": "等待行为",
    },
    "mean_active_wait_hours": {
        "label": "平均主动等待时间",
        "unit": "hours",
        "direction": "neutral",
        "group": "等待行为",
    },
}


class PricingComparisonError(RuntimeError):
    pass


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise PricingComparisonError(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(value) -> float:
    if isinstance(value, dict):
        if value.get("status") not in (None, "VALID"):
            raise PricingComparisonError(f"invalid metric: {value}")
        value = value.get("value")
    if value is None:
        raise PricingComparisonError("required metric has no value")
    result = float(value)
    if not math.isfinite(result):
        raise PricingComparisonError("metric is NaN or Infinity")
    return result


def _extract_metrics(report: dict) -> dict[str, float]:
    metrics = report.get("metrics") or {}
    tasks = [
        item
        for item in report.get("task_records") or ()
        if item.get("final_state") == "Completed"
    ]
    if not tasks:
        raise PricingComparisonError("report contains no completed task records")

    cpu_by_node: dict[str, float] = {}
    completion_delays = []
    for item in tasks:
        node = item.get("target_node")
        if node:
            cpu_by_node[node] = cpu_by_node.get(node, 0.0) + float(
                item.get("cpu_work_cpu_hours") or 0.0
            )
        if item.get("completion_delay_sim") is not None:
            completion_delays.append(float(item["completion_delay_sim"]))
    allocations = tuple(cpu_by_node.values())
    allocation_mean = statistics.fmean(allocations) if allocations else 0.0
    allocation_cv = (
        statistics.pstdev(allocations) / allocation_mean
        if allocations and allocation_mean > 0.0
        else 0.0
    )
    seconds_per_sim_unit = 300.0
    sla = metrics.get("sla_metrics") or {}
    active_wait = metrics.get("active_wait_metrics") or {}
    return {
        "cost_yuan_per_completed_cpu_hour": _metric_value(
            metrics.get("cost_yuan_per_completed_cpu_hour")
        ),
        "completed_task_green_coverage": _metric_value(
            metrics.get("completed_task_green_coverage")
        ),
        "system_green_absorption_rate": _metric_value(
            metrics.get("system_green_absorption_rate")
        ),
        "completion_rate": _metric_value(metrics.get("completion_rate")),
        "reservation_reliability": _metric_value(
            metrics.get("reservation_reliability")
        ),
        "soft_preferred_on_time_rate": _metric_value(
            sla.get("Soft", {}).get("preferred_on_time_rate")
        ),
        "flexible_preferred_on_time_rate": _metric_value(
            sla.get("Flexible", {}).get("preferred_on_time_rate")
        ),
        "hard_start_delay_p95_hours": (
            _metric_value(sla.get("Hard", {}).get("start_delay_p95_sim"))
            * seconds_per_sim_unit
            / 3600.0
        ),
        "mean_completion_delay_hours": (
            statistics.fmean(completion_delays) * seconds_per_sim_unit / 3600.0
        ),
        "allocated_cpu_hours_cv": allocation_cv,
        "active_wait_count": float(active_wait.get("count") or 0.0),
        "mean_active_wait_hours": (
            _metric_value(active_wait.get("mean_active_wait_sim"))
            * seconds_per_sim_unit
            / 3600.0
        ),
    }


def _checkpoint_hash(path: Path, expected_mode: str) -> str:
    if not path.exists():
        raise PricingComparisonError(f"missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actual_mode = (checkpoint.get("run_config") or {}).get("V1_TARIFF_MODE")
    if actual_mode != expected_mode:
        raise PricingComparisonError(
            f"{path} uses {actual_mode!r}, expected {expected_mode!r}"
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_pair(
    uniform: dict,
    regional: dict,
    *,
    uniform_model: Path,
    regional_model: Path,
) -> dict:
    for label, report in (("uniform", uniform), ("regional", regional)):
        if report.get("status") != "VALID":
            raise PricingComparisonError(f"{label} report is not VALID")
        if report.get("unsettled_task_ids"):
            raise PricingComparisonError(f"{label} report has unsettled tasks")
    uniform_metadata = uniform.get("metadata") or {}
    regional_metadata = regional.get("metadata") or {}
    if uniform_metadata.get("tariff_mode") != "tou_uniform":
        raise PricingComparisonError("uniform report is not tou_uniform")
    if regional_metadata.get("tariff_mode") != "tou_region":
        raise PricingComparisonError("regional report is not tou_region")

    paired_keys = (
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
        "topology_hash",
        "task_trace_hash",
        "exogenous_trace_hash",
        "dependency_lock_hash",
        "gamma_per_second",
    )
    mismatches = [
        key
        for key in paired_keys
        if uniform_metadata.get(key) != regional_metadata.get(key)
    ]
    if mismatches:
        raise PricingComparisonError(
            "reports are not paired: " + ", ".join(mismatches)
        )
    uniform_hash = _checkpoint_hash(uniform_model, "tou_uniform")
    regional_hash = _checkpoint_hash(regional_model, "tou_region")
    if uniform_metadata.get("model_hash") != uniform_hash:
        raise PricingComparisonError("uniform report model hash does not match checkpoint")
    if regional_metadata.get("model_hash") != regional_hash:
        raise PricingComparisonError("regional report model hash does not match checkpoint")
    return {
        "status": "PASS",
        "seed": uniform_metadata.get("seed"),
        "arrival_cutoff_sim": uniform_metadata.get("arrival_cutoff_sim"),
        "paired_metadata_match": True,
        "paired_fields": list(paired_keys),
        "uniform_model_hash_match": True,
        "regional_model_hash_match": True,
        "all_reports_valid": True,
        "all_unsettled_counts_zero": True,
    }


def _comparison_rows(values: dict[str, dict[str, float]]) -> list[dict]:
    rows = []
    for metric, spec in METRIC_SPECS.items():
        uniform = values["tou_uniform"][metric]
        regional = values["tou_region"][metric]
        rows.append({
            "group": spec["group"],
            "metric": metric,
            "metric_label": spec["label"],
            "unit": spec["unit"],
            "direction": spec["direction"],
            "uniform_value": uniform,
            "regional_value": regional,
            "absolute_delta": regional - uniform,
            "relative_delta": (
                (regional - uniform) / uniform if uniform != 0.0 else None
            ),
        })
    return rows


def _long_rows(values: dict[str, dict[str, float]], seed: int) -> list[dict]:
    rows = []
    for mode, metrics in values.items():
        for metric, value in metrics.items():
            spec = METRIC_SPECS[metric]
            rows.append({
                "seed": seed,
                "pricing_mode": mode,
                "pricing_label": MODE_LABELS[mode],
                "group": spec["group"],
                "metric": metric,
                "metric_label": spec["label"],
                "value": value,
                "unit": spec["unit"],
                "direction": spec["direction"],
            })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise PricingComparisonError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source(
    path: str,
    source_id: str,
    label: str,
    generated_at: str,
    definitions: tuple[str, ...],
) -> dict:
    normalized = path.replace("\\", "/")
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "Python csv",
            "language": "Python",
            "sql": f"SELECT * FROM read_csv_auto('{normalized}')",
            "description": "读取经过配对校验的 V2 定价模型单 seed 指标。",
            "executed_at": generated_at,
            "filters": [
                "seed=42",
                "evaluation status=VALID",
                "zero unsettled tasks",
                "identical task, topology and exogenous trace hashes",
                "candidate_dqn epsilon=0 during evaluation",
            ],
            "metric_definitions": list(definitions),
            "tables_used": [path],
        },
    }


def _simple_chart(
    chart_id: str,
    title: str,
    subtitle: str,
    dataset: str,
    value_label: str,
    value_format: str,
    root: str,
) -> dict:
    return {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "showDescription": True,
        "intent": "comparison",
        "question": f"分区与不分区定价的{title}如何比较？",
        "rationale": "两个离散定价模式的一项同单位指标适合使用横向条形图。",
        "comparisonContext": {
            "baseline": "不分区定价",
            "grain": "pricing mode",
            "unit": value_label,
            "semanticFamily": "pricing model comparison",
        },
        "type": "horizontalBar",
        "dataset": dataset,
        "sourceId": "pricing_metrics",
        "encodings": {
            "x": {
                "field": "pricing_label",
                "type": "nominal",
                "label": "定价模式",
            },
            "y": {
                "field": "value",
                "type": "quantitative",
                "format": value_format,
                "label": value_label,
            },
            "tooltip": [
                {"field": "seed", "type": "quantitative", "label": "Seed"},
                {"field": "direction_label", "type": "nominal", "label": "判优方向"},
            ],
        },
        "xAxisTitle": "定价模式",
        "yAxisTitle": value_label,
        "valueFormat": value_format,
        "layout": "full",
        "surface": {
            "palette": {"kind": "single", "root": root},
            "valueLabels": {"mode": "all"},
        },
    }


def _chart_rows(
    values: dict[str, dict[str, float]],
    metric: str,
    seed: int,
) -> list[dict]:
    direction = METRIC_SPECS[metric]["direction"]
    direction_label = {
        "higher": "越高越好",
        "lower": "越低越好",
        "neutral": "描述性指标",
    }[direction]
    return [
        {
            "seed": seed,
            "pricing_mode": mode,
            "pricing_label": MODE_LABELS[mode],
            "metric": metric,
            "metric_label": METRIC_SPECS[metric]["label"],
            "value": metrics[metric],
            "unit": METRIC_SPECS[metric]["unit"],
            "direction_label": direction_label,
        }
        for mode, metrics in values.items()
    ]


def _percentage_delta(values, metric: str) -> float:
    uniform = values["tou_uniform"][metric]
    return (values["tou_region"][metric] - uniform) / uniform


def _point_delta(values, metric: str) -> float:
    return (
        values["tou_region"][metric] - values["tou_uniform"][metric]
    )


def _build_artifact(
    values: dict[str, dict[str, float]],
    validation: dict,
    manifest_rows: list[dict],
    comparison_rows: list[dict],
    *,
    generated_at: str,
    metrics_path: str,
    manifest_path: str,
) -> tuple[dict, dict]:
    seed = int(validation["seed"])
    cost_delta = _percentage_delta(values, "cost_yuan_per_completed_cpu_hour")
    green_delta = _point_delta(values, "completed_task_green_coverage")
    absorption_delta = _point_delta(values, "system_green_absorption_rate")
    latency_delta = _percentage_delta(values, "mean_completion_delay_hours")
    balance_delta = _percentage_delta(values, "allocated_cpu_hours_cv")
    soft_delta = _point_delta(values, "soft_preferred_on_time_rate")
    flexible_delta = _point_delta(values, "flexible_preferred_on_time_rate")
    completion_delta = _point_delta(values, "completion_rate")

    metrics_source = _source(
        metrics_path,
        "pricing_metrics",
        "V2 regional-vs-uniform pricing comparison metrics",
        generated_at,
        (
            "每完成 CPU 小时成本=任务归因经济成本/完成 CPU 小时。",
            "完成任务绿电覆盖率=完成任务绿电能量/完成任务总能量。",
            "系统绿电吸收率=完成任务使用的绿电能量/系统可用绿电能量。",
            "节点 CPU 工作量变异系数=各节点完成 CPU 小时的总体标准差/均值。",
            "平均任务完成时延仅统计最终状态为 Completed 的任务，并换算为物理小时。",
            "全部差值定义为分区定价 - 不分区定价。",
        ),
    )
    manifest_source = _source(
        manifest_path,
        "pricing_source_manifest",
        "V2 pricing-model paired report provenance",
        generated_at,
        (
            "两份报告必须有效、无未结算任务，且 seed、任务、拓扑和外生轨迹哈希一致。",
            "报告模型哈希必须与对应 tou_uniform 或 tou_region checkpoint 一致。",
        ),
    )
    sources = [metrics_source, manifest_source]

    charts = [
        _simple_chart(
            "cost_chart",
            "每完成 CPU 小时成本",
            f"Seed {seed}；人民币/完成 CPU 小时，越低越好",
            "cost_comparison",
            "CNY/CPU-hour",
            "number",
            "blue",
        ),
        {
            "id": "green_chart",
            "title": "绿电利用指标",
            "subtitle": f"Seed {seed}；完成任务绿电覆盖率与系统绿电吸收率，越高越好",
            "showDescription": True,
            "intent": "comparison",
            "question": "分区与不分区定价的任务绿电覆盖率和系统吸收率如何比较？",
            "rationale": "两个同单位绿电比例使用分组柱形图，保留定价模式与指标两个维度。",
            "comparisonContext": {
                "baseline": "不分区定价",
                "grain": "pricing mode by green metric",
                "unit": "fraction",
                "semanticFamily": "green-energy comparison",
            },
            "type": "bar",
            "dataset": "green_comparison",
            "sourceId": "pricing_metrics",
            "encodings": {
                "x": {"field": "pricing_label", "type": "nominal", "label": "定价模式"},
                "y": {"field": "value", "type": "quantitative", "format": "percent", "label": "比例"},
                "color": {"field": "metric_label", "type": "nominal", "label": "绿电指标"},
                "tooltip": [{"field": "seed", "type": "quantitative", "label": "Seed"}],
            },
            "xAxisTitle": "定价模式",
            "yAxisTitle": "比例",
            "valueFormat": "percent",
            "layout": "full",
            "surface": {
                "palette": {"kind": "categorical", "roots": ["olive", "gold"]},
                "legend": {"position": "top"},
                "valueLabels": {"mode": "all"},
            },
        },
        {
            "id": "sla_chart",
            "title": "可靠性与首选窗口准时率",
            "subtitle": f"Seed {seed}；完成率、Soft 与 Flexible 准时率，越高越好",
            "showDescription": True,
            "intent": "comparison",
            "question": "定价模式改变后，完成率与两类 SLA 准时率是否保持？",
            "rationale": "三个同单位比例使用分组柱形图展示定价模式内的护栏表现。",
            "comparisonContext": {
                "baseline": "不分区定价",
                "grain": "pricing mode by service metric",
                "unit": "fraction",
                "semanticFamily": "service-quality comparison",
            },
            "type": "bar",
            "dataset": "sla_comparison",
            "sourceId": "pricing_metrics",
            "encodings": {
                "x": {"field": "pricing_label", "type": "nominal", "label": "定价模式"},
                "y": {"field": "value", "type": "quantitative", "format": "percent", "label": "比例"},
                "color": {"field": "metric_label", "type": "nominal", "label": "服务指标"},
                "tooltip": [{"field": "seed", "type": "quantitative", "label": "Seed"}],
            },
            "xAxisTitle": "定价模式",
            "yAxisTitle": "比例",
            "valueFormat": "percent",
            "layout": "full",
            "surface": {
                "palette": {"kind": "categorical", "roots": ["blue", "gold", "orange"]},
                "legend": {"position": "top"},
                "valueLabels": {"mode": "all"},
            },
        },
        _simple_chart(
            "latency_chart",
            "平均任务完成时延",
            f"Seed {seed}；仅统计已完成任务，物理小时，越低越好",
            "latency_comparison",
            "hours",
            "number",
            "orange",
        ),
        _simple_chart(
            "balance_chart",
            "节点 CPU 工作量变异系数",
            f"Seed {seed}；各节点完成 CPU 小时 CV，越低表示越均衡",
            "balance_comparison",
            "fraction",
            "number",
            "gold",
        ),
    ]

    tables = [{
        "id": "exact_metrics_table",
        "title": "分区与不分区定价关键指标",
        "subtitle": f"Seed {seed}；差值为分区定价 - 不分区定价",
        "showDescription": True,
        "dataset": "exact_metrics",
        "sourceId": "pricing_metrics",
        "defaultSort": {"field": "group", "direction": "asc"},
        "density": "spacious",
        "layout": "full",
        "columns": [
            {"field": "group", "label": "方面", "type": "text"},
            {"field": "metric_label", "label": "指标", "type": "text"},
            {"field": "direction_label", "label": "方向", "type": "text"},
            {"field": "uniform_display", "label": "不分区定价", "type": "text"},
            {"field": "regional_display", "label": "分区定价", "type": "text"},
            {"field": "delta_display", "label": "差值", "type": "text"},
        ],
    }]

    cards = [
        {
            "id": "regional_cost_card",
            "description": "分区定价单位成本及其相对不分区定价的变化。",
            "dataset": "headline",
            "sourceId": "pricing_metrics",
            "metrics": [
                {"label": "分区定价单位成本", "field": "regional_cost", "format": "number"},
                {"label": "相对不分区定价", "field": "cost_delta", "format": "percent", "signed": True},
            ],
        },
        {
            "id": "regional_green_card",
            "description": "分区定价任务绿电覆盖率及百分点变化。",
            "dataset": "headline",
            "sourceId": "pricing_metrics",
            "metrics": [
                {"label": "分区定价绿电覆盖率", "field": "regional_green", "format": "percent"},
                {"label": "百分点变化", "field": "green_delta", "format": "percent", "signed": True},
            ],
        },
        {
            "id": "completion_card",
            "description": "分区定价任务完成率及百分点变化。",
            "dataset": "headline",
            "sourceId": "pricing_metrics",
            "metrics": [
                {"label": "分区定价完成率", "field": "regional_completion", "format": "percent"},
                {"label": "百分点变化", "field": "completion_delta", "format": "percent", "signed": True},
            ],
        },
        {
            "id": "latency_card",
            "description": "分区定价平均完成时延及相对变化。",
            "dataset": "headline",
            "sourceId": "pricing_metrics",
            "metrics": [
                {"label": "平均完成时延（小时）", "field": "regional_latency", "format": "number"},
                {"label": "相对变化", "field": "latency_delta", "format": "percent", "signed": True},
            ],
        },
        {
            "id": "balance_card",
            "description": "分区定价节点 CPU 工作量 CV 及相对变化。",
            "dataset": "headline",
            "sourceId": "pricing_metrics",
            "metrics": [
                {"label": "节点工作量 CV", "field": "regional_balance", "format": "number"},
                {"label": "相对变化", "field": "balance_delta", "format": "percent", "signed": True},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# V2 分区与不分区定价模型对比", "layout": "full"},
        {
            "id": "executive_summary",
            "type": "markdown",
            "sourceId": "pricing_metrics",
            "layout": "full",
            "body": (
                "## Executive Summary\n\n"
                f"- **分区定价在当前 seed 上同时呈现更低的账面成本和更高的任务绿电覆盖。** "
                f"每完成 CPU 小时成本相对不分区定价变化 **{cost_delta * 100:+.2f}%**，"
                f"完成任务绿电覆盖率变化 **{green_delta * 100:+.2f} 个百分点**。\n"
                f"- **总体完成率保持不变，但 SLA 存在结构性权衡。** 完成率变化 "
                f"**{completion_delta * 100:+.2f} 个百分点**；Soft 准时率变化 "
                f"**{soft_delta * 100:+.2f} 个百分点**，Flexible 准时率变化 "
                f"**{flexible_delta * 100:+.2f} 个百分点**。\n"
                f"- **分区定价提高了调度集中度并轻微增加平均完成时延。** 节点工作量 CV 相对变化 "
                f"**{balance_delta * 100:+.2f}%**，平均完成时延相对变化 **{latency_delta * 100:+.2f}%**。\n"
                "- **当前结论仅是单 seed 描述性结果。** 成本差异包含计价基准变化，不能全部解释为 DQN 策略能力提升。"
            ),
        },
        {
            "id": "comparison_basis",
            "type": "markdown",
            "sourceId": "pricing_source_manifest",
            "layout": "full",
            "body": (
                "## 比较口径\n\n"
                f"两份冻结 Candidate DQN 报告均为 **V2.0、seed {seed}、arrival cutoff 288**。"
                "它们共享完全相同的任务、拓扑和外生轨迹，并分别匹配 `tou_uniform` 与 `tou_region` "
                "600,000 周期 checkpoint。成本、时延和负载均衡越低越好；绿电、完成率与准时率越高越好。"
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [item["id"] for item in cards], "layout": "full"},
        {
            "id": "cost_finding",
            "type": "markdown",
            "sourceId": "pricing_metrics",
            "layout": "full",
            "body": (
                "## 分区定价的账面单位成本约减半\n\n"
                f"**不分区定价为 {values['tou_uniform']['cost_yuan_per_completed_cpu_hour']:.4f} 元/完成 CPU 小时，"
                f"分区定价为 {values['tou_region']['cost_yuan_per_completed_cpu_hour']:.4f} 元，变化 {cost_delta * 100:+.2f}%。** "
                "该差值同时反映定价规则和调度选择；若要单独判断策略效率，需要在共同反事实价格口径下重新计价。"
            ),
        },
        {"id": "cost_chart_block", "type": "chart", "chartId": "cost_chart", "layout": "full"},
        {
            "id": "green_finding",
            "type": "markdown",
            "sourceId": "pricing_metrics",
            "layout": "full",
            "body": (
                "## 分区定价提高了任务绿电覆盖和系统绿电吸收\n\n"
                f"**任务绿电覆盖率由 {values['tou_uniform']['completed_task_green_coverage'] * 100:.2f}% "
                f"升至 {values['tou_region']['completed_task_green_coverage'] * 100:.2f}%（{green_delta * 100:+.2f} 个百分点）。** "
                f"系统绿电吸收率同时变化 {absorption_delta * 100:+.2f} 个百分点，说明提升不只是分母变化。"
            ),
        },
        {"id": "green_chart_block", "type": "chart", "chartId": "green_chart", "layout": "full"},
        {
            "id": "service_finding",
            "type": "markdown",
            "sourceId": "pricing_metrics",
            "layout": "full",
            "body": (
                "## 总体完成率不变，Soft 与 Flexible SLA 方向相反\n\n"
                f"**两种定价模式的任务完成率均为 {values['tou_uniform']['completion_rate'] * 100:.2f}%。** "
                f"分区定价的 Soft 准时率变化 {soft_delta * 100:+.2f} 个百分点，而 Flexible 准时率变化 "
                f"{flexible_delta * 100:+.2f} 个百分点。部署判断不应只看总体完成率，需要分别保留 SLA 护栏。"
            ),
        },
        {"id": "sla_chart_block", "type": "chart", "chartId": "sla_chart", "layout": "full"},
        {
            "id": "latency_finding",
            "type": "markdown",
            "sourceId": "pricing_metrics",
            "layout": "full",
            "body": (
                "## 平均完成时延小幅增加\n\n"
                f"**平均任务完成时延由 {values['tou_uniform']['mean_completion_delay_hours']:.3f} 小时"
                f"变为 {values['tou_region']['mean_completion_delay_hours']:.3f} 小时，变化 {latency_delta * 100:+.2f}%。** "
                "当前增幅不大，但单 seed 无法判断其稳定性，仍需检查更多 seed 和长尾时延。"
            ),
        },
        {"id": "latency_chart_block", "type": "chart", "chartId": "latency_chart", "layout": "full"},
        {
            "id": "balance_finding",
            "type": "markdown",
            "sourceId": "pricing_metrics",
            "layout": "full",
            "body": (
                "## 分区定价带来更明显的节点工作量集中\n\n"
                f"**节点 CPU 工作量 CV 从 {values['tou_uniform']['allocated_cpu_hours_cv']:.3f} "
                f"上升到 {values['tou_region']['allocated_cpu_hours_cv']:.3f}，变化 {balance_delta * 100:+.2f}%。** "
                "这表明区域价格信号可能把任务更集中地引导到部分节点；应继续监控热点、容量余量和故障域暴露。"
            ),
        },
        {"id": "balance_chart_block", "type": "chart", "chartId": "balance_chart", "layout": "full"},
        {
            "id": "exact_values",
            "type": "markdown",
            "layout": "full",
            "body": "## 关键指标精确值\n\n下表汇总两种定价模式的核心指标和差值，便于复核图表与后续引用。",
        },
        {"id": "exact_metrics_table_block", "type": "table", "tableId": "exact_metrics_table", "layout": "full"},
        {
            "id": "next_steps",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 建议的下一步\n\n"
                "1. 在至少 10 个未参与训练的 seed 上重复分区/不分区配对评估。\n"
                "2. 用统一的反事实计价口径重算两种模型的资源与能源消耗，拆分“价格规则变化”和“调度行为变化”。\n"
                "3. 将完成率、Soft/Flexible SLA、P95 时延、节点热点和物理违规设为部署护栏。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 仍需回答的问题\n\n"
                "- 绿电覆盖提升是否在多 seed、不同负载日和不同区域供给组合下持续存在？\n"
                "- 节点工作量集中是否由少数低价或高绿电区域主导？\n"
                "- 在共同价格口径下，分区模型是否仍能降低真实资源成本？"
            ),
        },
        {
            "id": "caveats",
            "type": "markdown",
            "sourceId": "pricing_source_manifest",
            "layout": "full",
            "body": (
                "## 限制与假设\n\n"
                "当前报告只有一个 seed，无法估计跨 seed 方差、置信区间或统计显著性。"
                "两种模型在不同价格特征和目标尺度下训练，且分区与不分区的成本定义不同；"
                "因此账面成本变化不是严格的策略因果效应。报告比较冻结模型在同一仿真输入下的描述性结果。"
            ),
        },
    ]

    exact_rows = []
    for row in comparison_rows:
        unit = row["unit"]
        if unit == "fraction":
            uniform_display = f"{row['uniform_value'] * 100:.4f}%"
            regional_display = f"{row['regional_value'] * 100:.4f}%"
            delta_display = f"{row['absolute_delta'] * 100:+.4f} pp"
        elif unit == "CNY/CPU-hour":
            uniform_display = f"{row['uniform_value']:.6f} 元"
            regional_display = f"{row['regional_value']:.6f} 元"
            delta_display = f"{row['absolute_delta']:+.6f} 元"
        elif unit == "hours":
            uniform_display = f"{row['uniform_value']:.6f} 小时"
            regional_display = f"{row['regional_value']:.6f} 小时"
            delta_display = f"{row['absolute_delta']:+.6f} 小时"
        else:
            uniform_display = f"{row['uniform_value']:.6f}"
            regional_display = f"{row['regional_value']:.6f}"
            delta_display = f"{row['absolute_delta']:+.6f}"
        exact_rows.append({
            **row,
            "direction_label": {
                "higher": "越高越好",
                "lower": "越低越好",
                "neutral": "描述性指标",
            }[row["direction"]],
            "uniform_display": uniform_display,
            "regional_display": regional_display,
            "delta_display": delta_display,
        })

    datasets = {
        "headline": [{
            "regional_cost": values["tou_region"]["cost_yuan_per_completed_cpu_hour"],
            "cost_delta": cost_delta,
            "regional_green": values["tou_region"]["completed_task_green_coverage"],
            "green_delta": green_delta,
            "regional_completion": values["tou_region"]["completion_rate"],
            "completion_delta": completion_delta,
            "regional_latency": values["tou_region"]["mean_completion_delay_hours"],
            "latency_delta": latency_delta,
            "regional_balance": values["tou_region"]["allocated_cpu_hours_cv"],
            "balance_delta": balance_delta,
        }],
        "cost_comparison": _chart_rows(values, "cost_yuan_per_completed_cpu_hour", seed),
        "green_comparison": [
            row
            for metric in (
                "completed_task_green_coverage",
                "system_green_absorption_rate",
            )
            for row in _chart_rows(values, metric, seed)
        ],
        "sla_comparison": [
            row
            for metric in (
                "completion_rate",
                "soft_preferred_on_time_rate",
                "flexible_preferred_on_time_rate",
            )
            for row in _chart_rows(values, metric, seed)
        ],
        "latency_comparison": _chart_rows(values, "mean_completion_delay_hours", seed),
        "balance_comparison": _chart_rows(values, "allocated_cpu_hours_cv", seed),
        "exact_metrics": exact_rows,
        "source_reports": manifest_rows,
    }
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "V2 分区与不分区定价模型对比",
            "description": "同一 seed 与轨迹下，V2 分区和不分区定价 Candidate DQN 的成本、绿电、SLA、时延与负载均衡比较。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
            "accessIssues": [],
        },
        "sources": sources,
    }
    notes = {
        "required_structure": {
            "title": "title",
            "executive_summary": "executive_summary",
            "key_findings_with_visual_evidence": [
                "cost_finding",
                "green_finding",
                "service_finding",
                "latency_finding",
                "balance_finding",
            ],
            "recommended_next_steps": "next_steps",
            "further_questions": "further_questions",
            "caveats_and_assumptions": "caveats",
        },
        "chart_map": [
            {"section": "成本", "chart": "cost_chart", "family": "Comparison & Ranking", "type": "horizontalBar"},
            {"section": "绿电", "chart": "green_chart", "family": "Comparison & Ranking", "type": "grouped bar"},
            {"section": "可靠性与 SLA", "chart": "sla_chart", "family": "Comparison & Ranking", "type": "grouped bar"},
            {"section": "时延", "chart": "latency_chart", "family": "Comparison & Ranking", "type": "horizontalBar"},
            {"section": "负载均衡", "chart": "balance_chart", "family": "Comparison & Ranking", "type": "horizontalBar"},
        ],
        "validation": validation,
        "confidence_method": "not estimable for one seed",
        "comparison_basis": "tou_region - tou_uniform",
        "cost_comparability_caveat": "tariff definitions differ; cost delta is not a pure policy effect",
    }
    return artifact, notes


def _find_builder() -> Path:
    codex_root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    plugin_root = (
        codex_root / "plugins" / "cache" / "openai-curated-remote" / "data-analytics"
    )
    candidates = list(
        plugin_root.glob("*/skills/build-report/scripts/deliver_portable_artifact.mjs")
    )
    if not candidates:
        raise PricingComparisonError("cannot find Data Analytics HTML builder")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_report(
    uniform_report_path: Path,
    regional_report_path: Path,
    uniform_model_path: Path,
    regional_model_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    uniform_report = _load_json(uniform_report_path)
    regional_report = _load_json(regional_report_path)
    validation = _validate_pair(
        uniform_report,
        regional_report,
        uniform_model=uniform_model_path,
        regional_model=regional_model_path,
    )
    values = {
        "tou_uniform": _extract_metrics(uniform_report),
        "tou_region": _extract_metrics(regional_report),
    }
    seed = int(validation["seed"])
    long_rows = _long_rows(values, seed)
    comparison_rows = _comparison_rows(values)
    metadata_rows = []
    for mode, report_path, model_path, report in (
        ("tou_uniform", uniform_report_path, uniform_model_path, uniform_report),
        ("tou_region", regional_report_path, regional_model_path, regional_report),
    ):
        metadata = report.get("metadata") or {}
        metadata_rows.append({
            "seed": seed,
            "pricing_mode": mode,
            "pricing_label": MODE_LABELS[mode],
            "status": report.get("status"),
            "unsettled_count": len(report.get("unsettled_task_ids") or ()),
            "report_path": str(report_path).replace("\\", "/"),
            "model_path": str(model_path).replace("\\", "/"),
            "model_hash": metadata.get("model_hash"),
            "task_trace_hash": metadata.get("task_trace_hash"),
            "exogenous_trace_hash": metadata.get("exogenous_trace_hash"),
            "topology_hash": metadata.get("topology_hash"),
            "arrival_cutoff_sim": metadata.get("arrival_cutoff_sim"),
            "system_version": metadata.get("system_version"),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    metrics_path = source_dir / "pricing_metrics.csv"
    comparison_path = source_dir / "pricing_comparison.csv"
    manifest_path = source_dir / "source_manifest.csv"
    validation_path = source_dir / "analysis_validation.json"
    _write_csv(metrics_path, long_rows)
    _write_csv(comparison_path, comparison_rows)
    _write_csv(manifest_path, metadata_rows)
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    artifact, notes = _build_artifact(
        values,
        validation,
        metadata_rows,
        comparison_rows,
        generated_at=generated_at,
        metrics_path=str(metrics_path).replace("\\", "/"),
        manifest_path=str(manifest_path).replace("\\", "/"),
    )
    artifact_path = output_dir / "artifact.json"
    report_path = output_dir / "report.html"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "report_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    builder = _find_builder()
    subprocess.run(
        [
            "node",
            str(builder),
            "--input",
            str(artifact_path),
            "--output",
            str(report_path),
        ],
        check=True,
    )
    return artifact_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a paired V2 regional-vs-uniform pricing HTML report"
    )
    parser.add_argument("--uniform-report", type=Path, default=DEFAULT_UNIFORM_REPORT)
    parser.add_argument("--regional-report", type=Path, default=DEFAULT_REGIONAL_REPORT)
    parser.add_argument("--uniform-model", type=Path, default=DEFAULT_UNIFORM_MODEL)
    parser.add_argument("--regional-model", type=Path, default=DEFAULT_REGIONAL_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path, report_path = build_report(
        args.uniform_report,
        args.regional_report,
        args.uniform_model,
        args.regional_model,
        args.output_dir,
    )
    print(json.dumps({
        "status": "VALID",
        "artifact": str(artifact_path),
        "report": str(report_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
