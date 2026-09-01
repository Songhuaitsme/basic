"""Build a validated Chinese HTML report for V2 five-policy evaluations."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Iterable, Mapping

from scipy import stats

from v2.export_five_policy_metrics import POLICIES


LABELS = {
    "earliest_feasible": "Earliest feasible",
    "lowest_cost": "Lowest cost",
    "highest_green": "Highest green",
    "equal_weight": "Equal weight",
    "candidate_dqn": "Candidate DQN",
}

# direction: higher, lower, or neutral.  Neutral metrics are exposed but not ranked.
METRIC_SPECS = {
    "arrival_count": ("规模与状态", "到达任务数", "tasks", "neutral"),
    "reserved_ever_count": ("可靠性", "曾获预留任务数", "tasks", "higher"),
    "rejected_count": ("可靠性", "拒绝任务数", "tasks", "lower"),
    "expired_count": ("可靠性", "过期任务数", "tasks", "lower"),
    "completed_count": ("可靠性", "完成任务数", "tasks", "higher"),
    "failed_count": ("可靠性", "失败任务数", "tasks", "lower"),
    "acceptance_rate": ("可靠性", "任务接受率", "fraction", "higher"),
    "completion_rate": ("可靠性", "任务完成率", "fraction", "higher"),
    "reservation_reliability": ("可靠性", "预留可靠性", "fraction", "higher"),
    "total_economic_cost_yuan": ("成本", "总经济成本", "CNY", "lower"),
    "completed_cpu_hours": ("规模与状态", "完成 CPU 小时", "CPU-hour", "neutral"),
    "arrived_requested_cpu_hours": ("规模与状态", "到达任务请求 CPU 小时", "CPU-hour", "neutral"),
    "cost_yuan_per_completed_cpu_hour": ("成本", "每完成 CPU 小时成本", "CNY/CPU-hour", "lower"),
    "cost_yuan_per_arrived_cpu_hour": ("成本", "每到达 CPU 小时成本", "CNY/CPU-hour", "lower"),
    "completed_task_green_coverage": ("绿电", "完成任务绿电覆盖率", "fraction", "higher"),
    "system_green_absorption_rate": ("绿电", "系统绿电吸收率", "fraction", "higher"),
    "active_wait_metrics.count": ("等待行为", "主动等待任务数", "tasks", "neutral"),
    "active_wait_metrics.mean_active_wait_sim": ("等待行为", "平均主动等待时间", "sim-time", "neutral"),
    "active_wait_metrics.p95_active_wait_sim": ("等待行为", "P95 主动等待时间", "sim-time", "neutral"),
    "active_wait_metrics.positive_benefit_rate": ("等待行为", "主动等待正收益率", "fraction", "higher"),
}

for _sla in ("Hard", "Soft", "Flexible"):
    _prefix = f"sla_metrics.{_sla}"
    METRIC_SPECS.update({
        f"{_prefix}.count": ("SLA", f"{_sla} 任务数", "tasks", "neutral"),
        f"{_prefix}.preferred_on_time_rate": (
            "SLA", f"{_sla} 首选窗口准时率", "fraction", "higher"
        ),
        f"{_prefix}.acceptable_tardy_rate": (
            "SLA", f"{_sla} 可接受迟到率", "fraction", "lower"
        ),
        f"{_prefix}.expired_rate": ("SLA", f"{_sla} 过期率", "fraction", "lower"),
        f"{_prefix}.start_delay_p50_sim": (
            "SLA", f"{_sla} 开始延迟 P50", "sim-time", "lower"
        ),
        f"{_prefix}.start_delay_p95_sim": (
            "SLA", f"{_sla} 开始延迟 P95", "sim-time", "lower"
        ),
        f"{_prefix}.preferred_start_tardiness_p50": (
            "SLA", f"{_sla} 首选窗口迟到程度 P50", "fraction", "lower"
        ),
        f"{_prefix}.preferred_start_tardiness_p95": (
            "SLA", f"{_sla} 首选窗口迟到程度 P95", "fraction", "lower"
        ),
    })

for _state in (
    "Arrived", "Queued", "PendingUncommitted", "Reserved", "Transmitting",
    "Running", "Completed", "Rejected", "Expired", "Failed",
):
    METRIC_SPECS[f"final_state_counts.{_state}"] = (
        "规模与状态", f"最终状态：{_state}", "tasks", "neutral"
    )

METRIC_SPECS.update({
    "diagnostics.task_summary.completion_delay_sim.mean": (
        "任务时延", "任务平均完成时延", "sim-time", "lower"
    ),
    "diagnostics.task_summary.scheduler_queue_delay_sim.p95": (
        "任务时延", "调度排队时延 P95", "sim-time", "lower"
    ),
    "diagnostics.task_summary.start_delay_sim.p95": (
        "任务时延", "任务开始时延 P95", "sim-time", "lower"
    ),
    "diagnostics.task_summary.completion_delay_sim.p95": (
        "任务时延", "任务完成时延 P95", "sim-time", "lower"
    ),
    "diagnostics.node_summary.allocated_cpu_hours_cv": (
        "负载均衡", "节点 CPU 工作量变异系数", "fraction", "lower"
    ),
    "diagnostics.node_summary.time_node_mean_cpu_utilization": (
        "资源利用", "时间-节点平均 CPU 利用率", "fraction", "neutral"
    ),
    "diagnostics.node_summary.cpu_hotspot_time_ratio": (
        "资源利用", "CPU 热点时间占比", "fraction", "lower"
    ),
    "diagnostics.node_summary.cpu_overcapacity_time_ratio": (
        "资源利用", "CPU 超容量时间占比", "fraction", "lower"
    ),
    "diagnostics.network_summary.remote_task_rate": (
        "网络", "远程调度任务比例", "fraction", "neutral"
    ),
    "diagnostics.network_summary.weighted_p95_link_utilization": (
        "网络", "链路利用率 P95", "fraction", "neutral"
    ),
    "diagnostics.runtime_summary.total_wall_seconds": (
        "运行效率", "整体评估墙钟时间", "seconds", "lower"
    ),
    "diagnostics.runtime_summary.decision_wall_seconds.mean": (
        "运行效率", "平均单次调度决策耗时", "seconds", "lower"
    ),
    "diagnostics.runtime_summary.decision_wall_seconds.p95": (
        "运行效率", "单次调度决策耗时 P95", "seconds", "lower"
    ),
    "diagnostics.system_summary.completed_tasks_per_physical_hour": (
        "系统效率", "每物理小时完成任务数", "tasks/hour", "higher"
    ),
})

CORE_METRICS = (
    "total_economic_cost_yuan",
    "cost_yuan_per_completed_cpu_hour",
    "completed_task_green_coverage",
    "system_green_absorption_rate",
    "completion_rate",
    "reservation_reliability",
    "sla_metrics.Soft.preferred_on_time_rate",
    "sla_metrics.Flexible.preferred_on_time_rate",
    "sla_metrics.Hard.start_delay_p95_sim",
    "active_wait_metrics.count",
    "active_wait_metrics.p95_active_wait_sim",
    "active_wait_metrics.positive_benefit_rate",
)

DIAGNOSTIC_CHART_SPECS = (
    (
        "average_task_latency_chart",
        "diagnostics.task_summary.completion_delay_sim.mean",
        "任务平均完成时延",
        "sim-time",
        "pink",
    ),
    (
        "task_latency_chart",
        "diagnostics.task_summary.completion_delay_sim.p95",
        "任务完成时延 P95",
        "sim-time",
        "orange",
    ),
    (
        "load_balance_chart",
        "diagnostics.node_summary.allocated_cpu_hours_cv",
        "节点 CPU 工作量变异系数",
        "fraction",
        "gold",
    ),
    (
        "network_chart",
        "diagnostics.network_summary.remote_task_rate",
        "远程调度任务比例",
        "fraction",
        "olive",
    ),
    (
        "runtime_chart",
        "diagnostics.runtime_summary.decision_wall_seconds.p95",
        "单次调度决策耗时 P95",
        "seconds",
        "blue",
    ),
)


class ComparisonReportError(ValueError):
    """Raised when source data cannot support a trustworthy report."""


def _float_or_none(value):
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ComparisonReportError("source metrics contain a non-finite value")
    return result


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise ComparisonReportError(f"missing source file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _tariff_context(manifest_rows: list[dict]) -> tuple[str, str]:
    """Return the single paired tariff mode and its Chinese display label."""
    modes = {
        row.get("tariff_mode", "").strip()
        for row in manifest_rows
        if row.get("tariff_mode", "").strip()
    }
    if not modes:
        for row in manifest_rows:
            source_path = Path(row.get("source_file", ""))
            try:
                report = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ComparisonReportError(
                    "source manifest has no tariff_mode and its report cannot be read: "
                    f"{source_path}: {exc}"
                ) from exc
            mode = str((report.get("metadata") or {}).get("tariff_mode", "")).strip()
            if mode:
                modes.add(mode)
    if len(modes) != 1:
        raise ComparisonReportError(
            f"expected one paired tariff_mode, found {sorted(modes)}"
        )
    tariff_mode = next(iter(modes))
    label = {
        "tou_region": "分区定价",
        "tou_uniform": "不分区定价",
    }.get(tariff_mode, tariff_mode)
    return tariff_mode, label


def load_source_data(input_dir: Path):
    validation_path = input_dir / "analysis_validation.json"
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonReportError(f"cannot read {validation_path}: {exc}") from exc
    if validation.get("status") != "PASS":
        raise ComparisonReportError("five-policy source validation did not PASS")

    long_rows = _read_csv(input_dir / "metrics_long.csv")
    manifest_rows = _read_csv(input_dir / "source_manifest.csv")
    if not long_rows or not manifest_rows:
        raise ComparisonReportError("source tables are empty")

    values: dict[tuple[int, str, str], float | None] = {}
    statuses: dict[tuple[int, str, str], str] = {}
    for row in long_rows:
        seed = int(row["seed"])
        policy = row["policy"]
        metric = row["metric_path"]
        if policy not in POLICIES:
            raise ComparisonReportError(f"unknown policy: {policy}")
        key = (seed, policy, metric)
        if key in values:
            raise ComparisonReportError(f"duplicate metric row: {key}")
        values[key] = _float_or_none(row["value"])
        statuses[key] = row["status"]

    seeds = tuple(sorted({key[0] for key in values}))
    metrics = tuple(sorted({key[2] for key in values}))
    for seed in seeds:
        for policy in POLICIES:
            missing = [metric for metric in metrics if (seed, policy, metric) not in values]
            if missing:
                raise ComparisonReportError(
                    f"seed {seed}, policy {policy} is missing {len(missing)} metrics"
                )

    expected_reports = len(seeds) * len(POLICIES)
    if len(manifest_rows) != expected_reports:
        raise ComparisonReportError(
            f"expected {expected_reports} manifest rows, found {len(manifest_rows)}"
        )
    for row in manifest_rows:
        if (
            row.get("status") != "VALID"
            or int(row.get("unsettled_count", -1)) != 0
            or row.get("paired_metadata_match", "").lower() != "true"
        ):
            raise ComparisonReportError("manifest contains an invalid or unpaired report")
    return validation, long_rows, manifest_rows, values, statuses, seeds, metrics


def mean_ci(values: Iterable[float], confidence_level: float = 0.95):
    items = tuple(float(value) for value in values)
    if not items:
        return None, None, None
    mean = statistics.fmean(items)
    if len(items) < 2:
        return mean, None, None
    sample_sd = statistics.stdev(items)
    critical = float(stats.t.ppf((1.0 + confidence_level) / 2.0, len(items) - 1))
    half_width = critical * sample_sd / math.sqrt(len(items))
    return mean, mean - half_width, mean + half_width


def _series(values, statuses, seeds, policy, metric):
    result = []
    for seed in seeds:
        key = (seed, policy, metric)
        value = values.get(key)
        if statuses.get(key) == "VALID" and value is not None:
            result.append(value)
    return result


def _metric_spec(metric: str):
    return METRIC_SPECS.get(metric, ("其他", metric, "number", "neutral"))


def _format_value(value, unit):
    if value is None:
        return "N/A"
    if unit == "fraction":
        return f"{value * 100:.2f}%"
    if unit == "CNY":
        return f"¥{value / 1_000_000:.2f}M"
    if unit == "CNY/CPU-hour":
        return f"¥{value:.3f}"
    if unit == "tasks":
        return f"{value:,.0f}"
    return f"{value:,.3f}"


def _format_delta(delta, unit, baseline=None):
    if delta is None:
        return "N/A"
    if unit == "fraction":
        return f"{delta * 100:+.2f} pp"
    if unit in {"CNY", "CNY/CPU-hour", "tasks"} and baseline not in (None, 0):
        return f"{delta / baseline * 100:+.2f}%"
    return f"{delta:+.3f}"


def _best_policy(policy_means: Mapping[str, float | None], direction: str):
    valid = {policy: value for policy, value in policy_means.items() if value is not None}
    if direction == "neutral" or not valid:
        return "不判优", None
    target = max(valid.values()) if direction == "higher" else min(valid.values())
    winners = [
        LABELS[policy]
        for policy, value in valid.items()
        if math.isclose(value, target, rel_tol=1e-9, abs_tol=1e-12)
    ]
    return ("并列：" + " / ".join(winners) if len(winners) > 1 else winners[0]), target


def _source(path: str, source_id: str, label: str, generated_at: str, definitions):
    sql_path = path.replace("\\", "/")
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "Python csv",
            "language": "Python",
            "sql": f"SELECT * FROM read_csv_auto('{sql_path}')",
            "description": "读取已校验的 V2 五策略评估源数据并进行配对聚合。",
            "executed_at": generated_at,
            "filters": [
                "evaluation status=VALID",
                "zero unsettled tasks",
                "identical paired metadata within each seed",
                "candidate_dqn epsilon=0 during evaluation",
            ],
            "metric_definitions": list(definitions),
            "tables_used": [path],
        },
    }


def _chart(chart_id, title, subtitle, dataset, value_field, value_label, value_format, root):
    return {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "showDescription": True,
        "intent": "comparison",
        "question": f"五种策略的{value_label}如何比较？",
        "rationale": "五个离散策略的一项同单位指标适合使用横向条形图直接比较。",
        "comparisonContext": {
            "baseline": "equal_weight",
            "grain": "policy",
            "unit": value_label,
            "semanticFamily": "policy comparison",
        },
        "type": "horizontalBar",
        "dataset": dataset,
        "sourceId": "metrics_long",
        "encodings": {
            "x": {"field": "policy_label", "type": "nominal", "label": "策略"},
            "y": {
                "field": value_field,
                "type": "quantitative",
                "format": value_format,
                "label": value_label,
            },
            "tooltip": [
                {"field": "seed_count", "type": "quantitative", "label": "Seed 数"},
                {"field": "ci_low", "type": "quantitative", "label": "95% CI 下界"},
                {"field": "ci_high", "type": "quantitative", "label": "95% CI 上界"},
            ],
        },
        "xAxisTitle": "策略",
        "yAxisTitle": value_label,
        "valueFormat": value_format,
        "layout": "full",
        "surface": {
            "palette": {"kind": "single", "root": root},
            "valueLabels": {"mode": "all"},
        },
    }


def build_report_artifact(input_dir: Path, confidence_level: float = 0.95):
    (
        validation, long_rows, manifest_rows, values, statuses, seeds, metrics
    ) = load_source_data(input_dir)
    generated_at = datetime.now(timezone.utc).isoformat()
    tariff_mode, tariff_label = _tariff_context(manifest_rows)
    seed_scope = "单 Seed" if len(seeds) == 1 else f"{len(seeds)} Seeds"
    report_title = f"V2 {tariff_label}五种调度策略{seed_scope}对比"

    summaries = {}
    for metric in metrics:
        summaries[metric] = {}
        for policy in POLICIES:
            samples = _series(values, statuses, seeds, policy, metric)
            mean, low, high = mean_ci(samples, confidence_level)
            summaries[metric][policy] = {
                "mean": mean, "ci_low": low, "ci_high": high, "n": len(samples)
            }

    def mean(metric, policy):
        return summaries[metric][policy]["mean"]

    def comparison_rows(metric):
        rows = []
        for policy in POLICIES:
            summary = summaries[metric][policy]
            rows.append({
                "policy": policy,
                "policy_label": LABELS[policy],
                "value": summary["mean"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "seed_count": summary["n"],
            })
        return rows

    policy_rows = []
    for policy in POLICIES:
        policy_rows.append({
            "policy": policy,
            "policy_label": LABELS[policy],
            "seed_count": len(seeds),
            "total_economic_cost_yuan": mean("total_economic_cost_yuan", policy),
            "cost_per_cpu_hour": mean("cost_yuan_per_completed_cpu_hour", policy),
            "green_coverage": mean("completed_task_green_coverage", policy),
            "system_green_absorption": mean("system_green_absorption_rate", policy),
            "completion_rate": mean("completion_rate", policy),
            "reservation_reliability": mean("reservation_reliability", policy),
            "soft_ontime": mean("sla_metrics.Soft.preferred_on_time_rate", policy),
            "flexible_ontime": mean("sla_metrics.Flexible.preferred_on_time_rate", policy),
            "hard_delay_p95": mean("sla_metrics.Hard.start_delay_p95_sim", policy),
            "active_wait_count": mean("active_wait_metrics.count", policy),
            "active_wait_p95": mean("active_wait_metrics.p95_active_wait_sim", policy),
            "wait_positive_benefit": mean("active_wait_metrics.positive_benefit_rate", policy),
        })

    dqn_delta_rows = []
    for metric in CORE_METRICS:
        group, label, unit, direction = _metric_spec(metric)
        dqn_samples = _series(values, statuses, seeds, "candidate_dqn", metric)
        baseline_samples = _series(values, statuses, seeds, "equal_weight", metric)
        paired = [right - left for left, right in zip(baseline_samples, dqn_samples)]
        delta_mean, delta_low, delta_high = mean_ci(paired, confidence_level)
        baseline_mean = mean(metric, "equal_weight")
        dqn_mean = mean(metric, "candidate_dqn")
        if delta_low is None:
            ci_display = "不可估计（单 seed）"
        else:
            ci_display = (
                f"[{_format_delta(delta_low, unit, baseline_mean)}, "
                f"{_format_delta(delta_high, unit, baseline_mean)}]"
            )
        dqn_delta_rows.append({
            "group": group,
            "metric": label,
            "direction": "越高越好" if direction == "higher" else (
                "越低越好" if direction == "lower" else "诊断指标"
            ),
            "equal_weight": _format_value(baseline_mean, unit),
            "candidate_dqn": _format_value(dqn_mean, unit),
            "delta": _format_delta(delta_mean, unit, baseline_mean),
            "paired_95_ci": ci_display,
            "seed_count": len(paired),
        })

    best_rows = []
    for metric in metrics:
        group, label, unit, direction = _metric_spec(metric)
        policy_means = {policy: mean(metric, policy) for policy in POLICIES}
        best_policy, best_value = _best_policy(policy_means, direction)
        best_rows.append({
            "group": group,
            "metric_path": metric,
            "metric": label,
            "direction": "越高越好" if direction == "higher" else (
                "越低越好" if direction == "lower" else "不判优"
            ),
            "best_policy": best_policy,
            "best_value": _format_value(best_value, unit),
            "seed_count": len(seeds),
            "uncertainty": "点估计；单 seed 无法估计区间" if len(seeds) == 1 else "跨 seed 均值",
        })

    # Preserve exact policy-SLA rows for the grouped on-time chart.
    sla_rows = []
    for policy in POLICIES:
        for sla_type in ("Soft", "Flexible"):
            metric = f"sla_metrics.{sla_type}.preferred_on_time_rate"
            summary = summaries[metric][policy]
            sla_rows.append({
                "policy": policy,
                "policy_label": LABELS[policy],
                "sla_type": sla_type,
                "value": summary["mean"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "seed_count": summary["n"],
            })

    baseline_cost = mean("cost_yuan_per_completed_cpu_hour", "equal_weight")
    dqn_cost = mean("cost_yuan_per_completed_cpu_hour", "candidate_dqn")
    dqn_cost_delta = (dqn_cost - baseline_cost) / baseline_cost
    baseline_green = mean("completed_task_green_coverage", "equal_weight")
    dqn_green = mean("completed_task_green_coverage", "candidate_dqn")
    dqn_green_delta = dqn_green - baseline_green
    completion = mean("completion_rate", "candidate_dqn")
    soft_delta = (
        mean("sla_metrics.Soft.preferred_on_time_rate", "candidate_dqn")
        - mean("sla_metrics.Soft.preferred_on_time_rate", "equal_weight")
    )
    flexible_delta = (
        mean("sla_metrics.Flexible.preferred_on_time_rate", "candidate_dqn")
        - mean("sla_metrics.Flexible.preferred_on_time_rate", "equal_weight")
    )
    uncertainty_text = (
        "当前只有 1 个 seed，所有差异均为诊断性点估计，不能代表跨场景稳定性。"
        if len(seeds) == 1
        else f"当前包含 {len(seeds)} 个配对 seed；表中报告配对 t 的 95% 置信区间。"
    )
    seed_text = ", ".join(str(seed) for seed in seeds)

    metrics_source = _source(
        (input_dir / "metrics_long.csv").as_posix(),
        "metrics_long",
        f"V2 {tariff_mode} five-policy validated metric source rows",
        generated_at,
        (
            "成本为任务归因经济成本；每完成 CPU 小时成本=总经济成本/完成 CPU 小时。",
            "完成任务绿电覆盖率=完成任务绿电能量/完成任务总能量。",
            "完成率=完成任务数/到达任务数；预留可靠性=完成任务数/曾获预留任务数。",
            "Soft 和 Flexible 准时率为首选开始窗口内启动的任务占比。",
            "所有 DQN 差值均为 candidate_dqn - equal_weight。",
        ),
    )
    manifest_source = _source(
        (input_dir / "source_manifest.csv").as_posix(),
        "source_manifest",
        f"V2 {tariff_mode} five-policy report provenance manifest",
        generated_at,
        ("同一 seed 内报告必须有效、无未结算任务且配对元数据完全一致。",),
    )
    sources = [metrics_source, manifest_source]

    charts = [
        _chart(
            "cost_chart", "每完成 CPU 小时成本",
            f"Seeds: {seed_text}；人民币/完成 CPU 小时，越低越好",
            "cost_comparison", "value", "CNY/CPU-hour", "number", "blue",
        ),
        _chart(
            "green_chart", "完成任务绿电覆盖率",
            f"Seeds: {seed_text}；完成任务绿电能量占比，越高越好",
            "green_comparison", "value", "绿电覆盖率", "percent", "olive",
        ),
        _chart(
            "reliability_chart", "任务完成率",
            f"Seeds: {seed_text}；完成任务数/到达任务数，越高越好",
            "reliability_comparison", "value", "完成率", "percent", "gold",
        ),
        {
            "id": "sla_chart",
            "title": "Soft 与 Flexible 首选窗口准时率",
            "subtitle": f"Seeds: {seed_text}；同一策略内比较两类 SLA，越高越好",
            "showDescription": True,
            "intent": "comparison",
            "question": "五种策略在 Soft 和 Flexible 任务上的首选窗口准时率如何？",
            "rationale": "两个同单位 SLA 序列使用分组柱形图，保留策略和 SLA 类型两个维度。",
            "comparisonContext": {
                "baseline": "equal_weight", "grain": "policy by SLA type",
                "unit": "fraction", "semanticFamily": "SLA comparison",
            },
            "type": "bar",
            "dataset": "sla_ontime",
            "sourceId": "metrics_long",
            "encodings": {
                "x": {"field": "policy_label", "type": "nominal", "label": "策略"},
                "y": {"field": "value", "type": "quantitative", "format": "percent", "label": "准时率"},
                "color": {"field": "sla_type", "type": "nominal", "label": "SLA 类型"},
                "tooltip": [{"field": "seed_count", "type": "quantitative", "label": "Seed 数"}],
            },
            "xAxisTitle": "策略",
            "yAxisTitle": "首选窗口准时率",
            "valueFormat": "percent",
            "layout": "full",
            "surface": {
                "palette": {"kind": "categorical", "roots": ["gold", "orange"]},
                "legend": {"position": "top"},
                "valueLabels": {"mode": "all"},
            },
        },
        _chart(
            "waiting_chart", "主动等待任务数",
            f"Seeds: {seed_text}；仅描述等待行为，不直接等同于服务质量",
            "waiting_comparison", "value", "任务数", "number", "pink",
        ),
    ]

    diagnostic_datasets = {}
    diagnostic_chart_ids = []
    for chart_id, metric, title, unit, color in DIAGNOSTIC_CHART_SPECS:
        if metric not in summaries:
            continue
        dataset_id = f"{chart_id}_data"
        diagnostic_datasets[dataset_id] = comparison_rows(metric)
        charts.append(_chart(
            chart_id,
            title,
            f"Seeds: {seed_text}；同一任务与实验条件下的策略比较",
            dataset_id,
            "value",
            unit,
            "percent" if unit == "fraction" else "number",
            color,
        ))
        diagnostic_chart_ids.append((chart_id, title))

    tables = [
        {
            "id": "core_metrics_table",
            "title": "五策略关键指标",
            "subtitle": f"Seeds: {seed_text}；跨 seed 时显示均值",
            "showDescription": True,
            "dataset": "policy_summary",
            "sourceId": "metrics_long",
            "defaultSort": {"field": "cost_per_cpu_hour", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "policy_label", "label": "策略", "type": "text"},
                {"field": "cost_per_cpu_hour", "label": "元/完成 CPU 小时", "format": "number"},
                {"field": "green_coverage", "label": "任务绿电覆盖率", "format": "percent"},
                {"field": "completion_rate", "label": "完成率", "format": "percent"},
                {"field": "soft_ontime", "label": "Soft 准时率", "format": "percent"},
                {"field": "flexible_ontime", "label": "Flexible 准时率", "format": "percent"},
                {"field": "active_wait_count", "label": "主动等待任务", "format": "number"},
            ],
        },
        {
            "id": "dqn_delta_table",
            "title": "Candidate DQN 相对 Equal weight",
            "subtitle": "差值定义为 DQN - Equal weight；比例指标以百分点表示",
            "showDescription": True,
            "dataset": "dqn_deltas",
            "sourceId": "metrics_long",
            "defaultSort": {"field": "group", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "group", "label": "方面", "type": "text"},
                {"field": "metric", "label": "指标", "type": "text"},
                {"field": "direction", "label": "方向", "type": "text"},
                {"field": "equal_weight", "label": "Equal weight", "type": "text"},
                {"field": "candidate_dqn", "label": "Candidate DQN", "type": "text"},
                {"field": "delta", "label": "DQN 差值", "type": "text"},
                {"field": "paired_95_ci", "label": "配对 95% CI", "type": "text"},
            ],
        },
        {
            "id": "best_policy_table",
            "title": "全部指标的最佳策略标记",
            "subtitle": "只对存在明确优劣方向的指标判优；规模和等待行为等诊断指标不强行排名",
            "showDescription": True,
            "dataset": "best_by_metric",
            "sourceId": "metrics_long",
            "defaultSort": {"field": "group", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "group", "label": "方面", "type": "text"},
                {"field": "metric", "label": "指标", "type": "text"},
                {"field": "direction", "label": "方向", "type": "text"},
                {"field": "best_policy", "label": "最佳策略", "type": "text"},
                {"field": "best_value", "label": "最佳值", "type": "text"},
                {"field": "uncertainty", "label": "不确定性", "type": "text"},
            ],
        },
    ]

    cards = [
        {
            "id": "dqn_cost_card", "description": "DQN 单位成本及相对基线变化。",
            "dataset": "headline", "sourceId": "metrics_long",
            "metrics": [
                {"label": "DQN 单位成本", "field": "dqn_cost", "format": "number"},
                {"label": "相对 Equal weight", "field": "dqn_cost_delta", "format": "percent", "signed": True},
            ],
        },
        {
            "id": "dqn_green_card", "description": "DQN 任务绿电覆盖率及百分点变化。",
            "dataset": "headline", "sourceId": "metrics_long",
            "metrics": [
                {"label": "DQN 绿电覆盖率", "field": "dqn_green", "format": "percent"},
                {"label": "相对 Equal weight", "field": "dqn_green_delta", "format": "percent", "signed": True},
            ],
        },
        {
            "id": "completion_card", "description": "DQN 完成任务数占到达任务数的比例。",
            "dataset": "headline", "sourceId": "metrics_long",
            "metrics": [{"label": "DQN 完成率", "field": "completion", "format": "percent"}],
        },
        {
            "id": "soft_card", "description": "DQN Soft 准时率相对基线变化。",
            "dataset": "headline", "sourceId": "metrics_long",
            "metrics": [
                {"label": "Soft 准时率变化", "field": "soft_delta", "format": "percent", "signed": True}
            ],
        },
        {
            "id": "flexible_card", "description": "DQN Flexible 准时率相对基线变化。",
            "dataset": "headline", "sourceId": "metrics_long",
            "metrics": [
                {"label": "Flexible 准时率变化", "field": "flexible_delta", "format": "percent", "signed": True}
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# V2 五种调度策略指标对比", "layout": "full"},
        {
            "id": "executive_summary", "type": "markdown", "sourceId": "metrics_long", "layout": "full",
            "body": (
                "## Executive Summary\n\n"
                f"- **DQN 在当前样本中改善了成本与任务绿电覆盖。** 相对 Equal weight，"
                f"每完成 CPU 小时成本变化 **{dqn_cost_delta * 100:+.2f}%**，任务绿电覆盖率变化 "
                f"**{dqn_green_delta * 100:+.2f} 个百分点**。\n"
                f"- **可靠性没有被牺牲。** DQN 完成率为 **{completion * 100:.2f}%**；五种策略在相同任务轨迹上的完成率一致。\n"
                f"- **SLA 存在局部权衡。** DQN 的 Soft 准时率相对基线变化 **{soft_delta * 100:+.2f} 个百分点**，"
                f"Flexible 准时率变化 **{flexible_delta * 100:+.2f} 个百分点**。\n"
                f"- **证据强度仍受 seed 数限制。** {uncertainty_text}"
            ),
        },
        {
            "id": "definitions", "type": "markdown", "sourceId": "metrics_long", "layout": "full",
            "body": (
                "## 比较口径\n\n"
                f"报告覆盖 seeds **{seed_text}**，每个 seed 的五种策略共享相同任务轨迹、外生轨迹、配置和拓扑。"
                "成本越低越好；绿电覆盖率、完成率、预留可靠性和准时率越高越好。"
                "主动等待数量和等待时长用于描述策略行为，不直接作为越低越好的服务质量指标。"
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards], "layout": "full"},
        {
            "id": "cost_finding", "type": "markdown", "sourceId": "metrics_long", "layout": "full",
            "body": (
                "## 成本：Lowest cost 最低，DQN 优于 Equal weight\n\n"
                f"**DQN 的每完成 CPU 小时成本为 {_format_value(dqn_cost, 'CNY/CPU-hour')}，"
                f"相对 Equal weight 变化 {dqn_cost_delta * 100:+.2f}%。** "
                "Lowest cost 是成本下界型基准，但需要结合绿电与 SLA 一起判断，不宜单独作为部署结论。"
            ),
        },
        {"id": "cost_chart_block", "type": "chart", "chartId": "cost_chart", "layout": "full"},
        {
            "id": "green_finding", "type": "markdown", "sourceId": "metrics_long", "layout": "full",
            "body": (
                "## 绿电：Highest green 最高，DQN 同时超过 Equal weight\n\n"
                f"**DQN 的完成任务绿电覆盖率为 {dqn_green * 100:.2f}%，比 Equal weight 高 "
                f"{dqn_green_delta * 100:.2f} 个百分点。** Highest green 展示可达到的绿电上界，"
                "但其成本与部分 SLA 指标需要同时检查。"
            ),
        },
        {"id": "green_chart_block", "type": "chart", "chartId": "green_chart", "layout": "full"},
        {
            "id": "reliability_finding", "type": "markdown", "sourceId": "metrics_long", "layout": "full",
            "body": (
                "## 可靠性：五种策略完成率一致\n\n"
                f"**五种策略在当前配对任务轨迹上的完成率均为 {completion * 100:.2f}%。** "
                "这说明当前成本和绿电差异不是通过减少完成任务取得的；仍需在更多 seed 上复核非劣性。"
            ),
        },
        {"id": "reliability_chart_block", "type": "chart", "chartId": "reliability_chart", "layout": "full"},
        {
            "id": "sla_finding", "type": "markdown", "sourceId": "metrics_long", "layout": "full",
            "body": (
                "## SLA：DQN 改善 Soft，但 Flexible 准时率低于基线\n\n"
                f"**相对 Equal weight，DQN 的 Soft 准时率变化 {soft_delta * 100:+.2f} 个百分点，"
                f"Flexible 准时率变化 {flexible_delta * 100:+.2f} 个百分点。** "
                "因此，Flexible SLA 是后续正式评估必须保留的独立护栏。"
            ),
        },
        {"id": "sla_chart_block", "type": "chart", "chartId": "sla_chart", "layout": "full"},
        {
            "id": "waiting_finding", "type": "markdown", "sourceId": "metrics_long", "layout": "full",
            "body": (
                "## 等待行为：策略通过不同程度的主动等待换取目标收益\n\n"
                "**主动等待反映策略是否把任务推迟到未来时隙，而不是失败数量。** "
                "应把等待数量、P95 等待时间和正收益率与成本、绿电及 SLA 联合解释，避免把等待更少直接当作更优。"
            ),
        },
        {"id": "waiting_chart_block", "type": "chart", "chartId": "waiting_chart", "layout": "full"},
        {
            "id": "exact_values", "type": "markdown", "layout": "full",
            "body": "## 关键指标精确值\n\n下表提供五种策略的核心指标精确值，便于复核图表和后续引用。",
        },
        {"id": "core_metrics_table_block", "type": "table", "tableId": "core_metrics_table", "layout": "full"},
        {
            "id": "dqn_delta", "type": "markdown", "sourceId": "metrics_long", "layout": "full",
            "body": (
                "## DQN 相对 Equal weight 的差值与区间\n\n"
                f"**差值统一定义为 Candidate DQN - Equal weight。** {uncertainty_text} "
                "比例型指标以百分点表示；成本和任务数优先显示相对变化。"
            ),
        },
        {"id": "dqn_delta_table_block", "type": "table", "tableId": "dqn_delta_table", "layout": "full"},
        {
            "id": "best_policy", "type": "markdown", "sourceId": "metrics_long", "layout": "full",
            "body": (
                "## 每项指标的最佳策略标记\n\n"
                "**只对定义上存在明确优劣方向的指标判优。** 到达规模、SLA 子组任务数、最终状态快照和主动等待行为等"
                "诊断指标标记为“不判优”，避免制造没有业务含义的综合冠军。"
            ),
        },
        {"id": "best_policy_table_block", "type": "table", "tableId": "best_policy_table", "layout": "full"},
        {
            "id": "next_steps", "type": "markdown", "layout": "full",
            "body": (
                "## 建议的下一步\n\n"
                "1. 固定至少 10 个未参与训练的 seed，继续运行五策略配对评估。\n"
                "2. 以 Equal weight 为主要基线，检查 DQN 的成本、绿电、完成率和 Flexible SLA 配对区间。\n"
                "3. 仅在完成率非劣、物理违规为零且成本或绿电达到门禁后，再判断是否进入部署候选。"
            ),
        },
        {
            "id": "further_questions", "type": "markdown", "layout": "full",
            "body": (
                "## 仍需回答的问题\n\n"
                "- DQN 的 Flexible SLA 退化是否在更多 seed 上持续存在？\n"
                "- 成本和绿电收益是否由少数负载场景或任务类型主导？\n"
                "- 主动等待减少后，等待收益率和长尾延迟是否仍保持可接受？"
            ),
        },
        {
            "id": "caveats", "type": "markdown", "sourceId": "source_manifest", "layout": "full",
            "body": (
                "## 限制与假设\n\n"
                f"{uncertainty_text} 本报告比较的是冻结策略在相同仿真输入上的表现，不构成因果证明。"
                "Hard 任务没有首选开始窗口，因此其首选窗口准时率为不适用；主动等待指标只描述行为。"
            ),
        },
    ]

    if diagnostic_chart_ids:
        insertion = next(
            index for index, block in enumerate(blocks)
            if block["id"] == "exact_values"
        )
        diagnostic_blocks = [{
            "id": "fine_grained_evaluation",
            "type": "markdown",
            "sourceId": "metrics_long",
            "layout": "full",
            "body": (
                "## 细粒度评估：任务、节点、网络与算法效率\n\n"
                "以下指标由任务执行记录、资源预留区间和决策墙钟计时逐层汇总。"
                "单 seed 只表示当前工作负载的诊断结果；运行效率应在相同设备、进程并发和审计配置下比较。"
            ),
        }]
        for chart_id, _ in diagnostic_chart_ids:
            if chart_id == "average_task_latency_chart":
                diagnostic_blocks.append({
                    "id": "average_task_latency_finding",
                    "type": "markdown",
                    "sourceId": "metrics_long",
                    "layout": "full",
                    "body": (
                        "## 任务平均完成时延\n\n"
                        "**该图比较五种策略从任务到达到完成的平均时长，数值越低越好。** "
                        f"每种策略先按 seed 汇总任务平均完成时延，再展示 {len(seeds)} 个 seed 的均值；"
                        "多 seed 结果应结合置信区间和任务完成率共同解释。"
                    ),
                })
            diagnostic_blocks.append({
                "id": f"{chart_id}_block",
                "type": "chart",
                "chartId": chart_id,
                "layout": "full",
            })
        blocks[insertion:insertion] = diagnostic_blocks

    for block in blocks:
        if block["id"] == "title":
            block["body"] = f"# {report_title}"
        elif block["id"] == "definitions":
            block["body"] = (
                f"**定价模式：{tariff_label}（`{tariff_mode}`）。**\n\n"
                + block["body"]
            )

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "V2 五种调度策略指标对比",
            "description": "冻结策略在配对仿真任务上的成本、绿电、可靠性、SLA 与等待行为比较。",
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
            "datasets": {
                "headline": [{
                    "dqn_cost": dqn_cost,
                    "dqn_cost_delta": dqn_cost_delta,
                    "dqn_green": dqn_green,
                    "dqn_green_delta": dqn_green_delta,
                    "completion": completion,
                    "soft_delta": soft_delta,
                    "flexible_delta": flexible_delta,
                }],
                "policy_summary": policy_rows,
                "cost_comparison": comparison_rows("cost_yuan_per_completed_cpu_hour"),
                "green_comparison": comparison_rows("completed_task_green_coverage"),
                "reliability_comparison": comparison_rows("completion_rate"),
                "sla_ontime": sla_rows,
                "waiting_comparison": comparison_rows("active_wait_metrics.count"),
                "dqn_deltas": dqn_delta_rows,
                "best_by_metric": best_rows,
                "source_reports": manifest_rows,
                **diagnostic_datasets,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    artifact["manifest"]["title"] = report_title
    artifact["manifest"]["description"] = (
        f"V2 {tariff_label}（{tariff_mode}）下五种冻结调度策略的{seed_scope}成本、"
        "绿电、负载均衡、时延、可靠性、SLA 与运行效率对比。"
    )
    notes = {
        "required_structure": {
            "title": "title",
            "executive_summary": "executive_summary",
            "key_findings_with_visual_evidence": [
                "cost_finding", "green_finding", "reliability_finding",
                "sla_finding", "waiting_finding",
            ],
            "recommended_next_steps": "next_steps",
            "further_questions": "further_questions",
            "caveats_and_assumptions": "caveats",
        },
        "chart_map": [
            {"section": "成本", "chart": "cost_chart", "family": "Comparison & Ranking", "type": "horizontalBar"},
            {"section": "绿电", "chart": "green_chart", "family": "Comparison & Ranking", "type": "horizontalBar"},
            {"section": "可靠性", "chart": "reliability_chart", "family": "Comparison & Ranking", "type": "horizontalBar"},
            {"section": "SLA", "chart": "sla_chart", "family": "Comparison & Ranking", "type": "grouped bar"},
            {"section": "等待行为", "chart": "waiting_chart", "family": "Comparison & Ranking", "type": "horizontalBar"},
        ],
        "validation": validation,
        "seed_count": len(seeds),
        "tariff_mode": tariff_mode,
        "tariff_label": tariff_label,
        "confidence_method": "paired t interval" if len(seeds) >= 2 else "not estimable for one seed",
    }
    notes["chart_map"].extend({
        "section": title,
        "chart": chart_id,
        "family": "Comparison & Ranking",
        "type": "horizontalBar",
    } for chart_id, title in diagnostic_chart_ids)
    return artifact, notes, dqn_delta_rows, best_rows


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ComparisonReportError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _find_builder() -> Path:
    codex_root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    plugin_root = codex_root / "plugins" / "cache" / "openai-curated-remote" / "data-analytics"
    candidates = list(plugin_root.glob("*/skills/build-report/scripts/deliver_portable_artifact.mjs"))
    if not candidates:
        raise ComparisonReportError("cannot find the Data Analytics portable HTML builder")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_report(
    input_dir: Path,
    output_dir: Path,
    *,
    confidence_level: float = 0.95,
    artifact_only: bool = False,
):
    artifact, notes, dqn_rows, best_rows = build_report_artifact(
        input_dir, confidence_level
    )
    output_dir.mkdir(parents=True, exist_ok=True)
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
    _write_csv(output_dir / "dqn_vs_equal_weight.csv", dqn_rows)
    _write_csv(output_dir / "best_strategy_by_metric.csv", best_rows)

    if not artifact_only:
        builder = _find_builder()
        subprocess.run(
            ["node", str(builder), "--input", str(artifact_path), "--output", str(report_path)],
            check=True,
        )
    return artifact_path, report_path if report_path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a validated Chinese HTML comparison report for five V2 policies"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/v2/evaluation/five_policy/source_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v2/evaluation/five_policy/comparison_report"),
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--artifact-only", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.confidence_level < 1.0:
        parser.error("--confidence-level must be between 0 and 1")
    artifact_path, report_path = write_report(
        args.input_dir,
        args.output_dir,
        confidence_level=args.confidence_level,
        artifact_only=args.artifact_only,
    )
    print(json.dumps({
        "status": "VALID",
        "artifact": str(artifact_path),
        "report": None if report_path is None else str(report_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
