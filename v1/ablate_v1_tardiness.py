# 文件功能：比较不同 SLA 迟到惩罚权重，汇总配对种子实验结果。
# 主要内容：_value、_parse_weights、_row、_mean、summarize、run_ablation、main。
# 模块上游（导入关系）：tests.v1.test_tardiness_ablation_v1。
# 模块下游（项目依赖）：v1.domain.models、v1.evaluate_v1。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Frozen-trace ablation for Soft/Flexible preferred-start penalties."""

import argparse
import csv
import json
from pathlib import Path
import statistics

from v1.domain.models import SlaType
from v1.evaluate_v1 import _jsonable, run_evaluation


# 作用：读取指标值并处理缺失。
# 上游（直接调用）：v1.ablate_v1_tardiness._row。
def _value(metric):
    return None if metric is None or metric.value is None else float(metric.value)


# 作用：解析 SLA 惩罚权重组合。
# 上游（直接调用）：tests.v1.test_tardiness_ablation_v1.TardinessAblationV1Test.test_weight_parser_requires_non_negative_pair。
# 下游调用：value.split。
def _parse_weights(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("weights must be SOFT,FLEXIBLE")
    try:
        soft, flexible = (float(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))
    if soft < 0.0 or flexible < 0.0:
        raise argparse.ArgumentTypeError("weights must be non-negative")
    return soft, flexible


# 作用：将单次评估或校准结果整理为表格行。
# 上游（直接调用）：v1.ablate_v1_tardiness.run_ablation。
# 下游调用：_value。
def _row(report, soft_weight, flexible_weight):
    metrics = report.metrics
    soft = metrics.sla_metrics[SlaType.SOFT]
    flexible = metrics.sla_metrics[SlaType.FLEXIBLE]
    return {
        "seed": report.metadata.seed,
        "soft_weight": soft_weight,
        "flexible_weight": flexible_weight,
        "arrival_count": metrics.arrival_count,
        "completion_rate": _value(metrics.completion_rate),
        "expired_count": metrics.expired_count,
        "failed_count": metrics.failed_count,
        "total_economic_cost_yuan": metrics.total_economic_cost_yuan,
        "cost_yuan_per_completed_cpu_hour": _value(
            metrics.cost_yuan_per_completed_cpu_hour
        ),
        "completed_task_green_coverage": _value(
            metrics.completed_task_green_coverage
        ),
        "system_green_absorption_rate": _value(
            metrics.system_green_absorption_rate
        ),
        "active_wait_count": metrics.active_wait_metrics.count,
        "soft_count": soft.count,
        "soft_preferred_on_time_rate": _value(soft.preferred_on_time_rate),
        "soft_acceptable_tardy_rate": _value(soft.acceptable_tardy_rate),
        "soft_tardiness_p95": _value(soft.preferred_start_tardiness_p95),
        "flexible_count": flexible.count,
        "flexible_preferred_on_time_rate": _value(
            flexible.preferred_on_time_rate
        ),
        "flexible_acceptable_tardy_rate": _value(
            flexible.acceptable_tardy_rate
        ),
        "flexible_tardiness_p95": _value(
            flexible.preferred_start_tardiness_p95
        ),
    }


# 作用：计算有效样本均值。
# 上游（直接调用）：v1.ablate_v1_tardiness.summarize。
def _mean(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return statistics.fmean(values) if values else None


# 作用：汇总样本或实验统计。
# 上游（直接调用）：v1.ablate_v1_tardiness.run_ablation。
# 下游调用：_mean。
def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(
            (row["soft_weight"], row["flexible_weight"]), []
        ).append(row)
    fields = (
        "completion_rate",
        "expired_count",
        "failed_count",
        "total_economic_cost_yuan",
        "cost_yuan_per_completed_cpu_hour",
        "completed_task_green_coverage",
        "system_green_absorption_rate",
        "active_wait_count",
        "soft_preferred_on_time_rate",
        "soft_acceptable_tardy_rate",
        "soft_tardiness_p95",
        "flexible_preferred_on_time_rate",
        "flexible_acceptable_tardy_rate",
        "flexible_tardiness_p95",
    )
    summaries = []
    for weights, items in sorted(grouped.items()):
        summary = {
            "soft_weight": weights[0],
            "flexible_weight": weights[1],
            "seed_count": len(items),
        }
        summary.update({field: _mean(items, field) for field in fields})
        on_time = [
            value
            for value in (
                summary["soft_preferred_on_time_rate"],
                summary["flexible_preferred_on_time_rate"],
            )
            if value is not None
        ]
        summary["mean_soft_flexible_preferred_on_time_rate"] = (
            statistics.fmean(on_time) if on_time else None
        )
        summaries.append(summary)

    baseline = next(
        (
            item for item in summaries
            if item["soft_weight"] == 0.0
            and item["flexible_weight"] == 0.0
        ),
        None,
    )
    valid = [
        item for item in summaries
        if baseline is None
        or (
            item["completion_rate"] >= baseline["completion_rate"] - 1e-12
            and item["expired_count"] <= baseline["expired_count"] + 1e-12
            and item["failed_count"] <= baseline["failed_count"] + 1e-12
        )
    ]
    recommended = max(
        valid,
        key=lambda item: (
            -1.0
            if item["mean_soft_flexible_preferred_on_time_rate"] is None
            else item["mean_soft_flexible_preferred_on_time_rate"],
            -item["total_economic_cost_yuan"],
            -(item["soft_weight"] + item["flexible_weight"]),
        ),
    )
    return summaries, {
        "soft_weight": recommended["soft_weight"],
        "flexible_weight": recommended["flexible_weight"],
        "selection_rule": (
            "Preserve baseline completion/expired/failed gates; maximize the "
            "mean Soft/Flexible preferred-on-time rate; then minimize cost and "
            "total penalty weight."
        ),
    }


# 作用：按权重和种子执行 SLA 迟到消融。
# 上游（直接调用）：v1.ablate_v1_tardiness.main。
# 下游调用：run_evaluation、_jsonable、_row、summarize、target.mkdir 等。
def run_ablation(seeds, weights, cutoff, safety_cap, output_dir):
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for soft_weight, flexible_weight in weights:
        weight_id = f"soft_{soft_weight:g}_flex_{flexible_weight:g}"
        for seed in seeds:
            report = run_evaluation(
                "equal_weight",
                cutoff,
                seed,
                safety_cap,
                soft_tardiness_weight=soft_weight,
                flexible_tardiness_weight=flexible_weight,
            )
            if report.status.value != "VALID":
                raise RuntimeError(
                    f"invalid ablation report for weights {weight_id}, seed {seed}"
                )
            report_path = target / f"{weight_id}_seed_{seed}.json"
            report_path.write_text(
                json.dumps(
                    _jsonable(report), ensure_ascii=False, indent=2, allow_nan=False
                ),
                encoding="utf-8",
            )
            row = _row(report, soft_weight, flexible_weight)
            rows.append(row)
            print(json.dumps({"ablation_progress": row}, ensure_ascii=False), flush=True)

    summaries, recommendation = summarize(rows)
    result = {
        "status": "VALID_TARDINESS_ABLATION",
        "seeds": list(seeds),
        "arrival_cutoff_sim": cutoff,
        "weights": [
            {"soft": soft, "flexible": flexible}
            for soft, flexible in weights
        ],
        "per_seed": rows,
        "summary": summaries,
        "recommendation": recommendation,
    }
    json_path = target / "tardiness_ablation.json"
    csv_path = target / "tardiness_ablation.csv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


# 作用：解析或转发命令行参数，组织本文件执行流程。
# 调用约定：本文件命令行入口或上层版本入口调用。
# 下游调用：run_ablation、parser.add_argument、parser.parse_args。
def main():
    parser = argparse.ArgumentParser(description="Ablate v1 tardiness weights")
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument(
        "--weights",
        type=_parse_weights,
        action="append",
        default=None,
        help="repeat SOFT,FLEXIBLE pairs",
    )
    parser.add_argument("--arrival-cutoff", type=float, default=1.0)
    parser.add_argument("--safety-cap", type=int, default=1000000)
    parser.add_argument(
        "--output-dir", default="artifacts/v1/ablation/tardiness"
    )
    args = parser.parse_args()
    weights = args.weights or [(0.0, 0.0), (0.5, 0.25), (1.0, 0.5)]
    result = run_ablation(
        tuple(args.seed), tuple(weights), args.arrival_cutoff,
        args.safety_cap, args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "recommendation": result["recommendation"],
        "output_dir": args.output_dir,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
