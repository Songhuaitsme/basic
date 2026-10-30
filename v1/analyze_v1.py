# 文件功能：读取评估报告，进行配对比较、置信区间估计和质量门槛判定。
# 主要内容：_load、_path_value、_paired_gate、_bootstrap_ci、compare_reports、_write_outputs、main。
# 使用入口：命令行执行，或由其他脚本导入后调用。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Paired effect analysis for formal v1.0 evaluation reports."""

import argparse
import csv
import json
from pathlib import Path
import random
import statistics


METRICS = {
    "acceptance_rate": ("acceptance_rate", "higher"),
    "completion_rate": ("completion_rate", "higher"),
    "reservation_reliability": ("reservation_reliability", "higher"),
    "total_economic_cost_yuan": ("total_economic_cost_yuan", "lower"),
    "cost_yuan_per_completed_cpu_hour": (
        "cost_yuan_per_completed_cpu_hour", "lower"
    ),
    "completed_task_green_coverage": ("completed_task_green_coverage", "higher"),
    "system_green_absorption_rate": ("system_green_absorption_rate", "higher"),
    "expired_count": ("expired_count", "lower"),
    "failed_count": ("failed_count", "lower"),
    "active_wait_count": ("active_wait_metrics.count", "diagnostic"),
}


# 作用：读取结构化评估数据。
# 上游（直接调用）：v1.analyze_v1.compare_reports。
# 下游调用：source.read_text。
def _load(path):
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("status") != "VALID" or data.get("metrics") is None:
        raise ValueError(f"evaluation report is not VALID: {source}")
    return source, data


# 作用：按字段路径读取嵌套指标。
# 上游（直接调用）：v1.analyze_v1.compare_reports。
# 下游调用：dotted.split。
def _path_value(mapping, dotted):
    value = mapping
    for part in dotted.split("."):
        value = value[part]
    if isinstance(value, dict) and "value" in value:
        if value.get("status") != "VALID":
            return None
        value = value["value"]
    return None if value is None else float(value)


# 作用：对配对评估应用质量判定。
# 上游（直接调用）：v1.analyze_v1.compare_reports。
# 下游调用：', '.join。
def _paired_gate(baseline, treatment):
    left = baseline["metadata"]
    right = treatment["metadata"]
    required_equal = (
        "requirements_version", "algorithm_version", "candidate_mode", "seed",
        "arrival_cutoff_sim", "task_trace_hash", "exogenous_trace_hash",
        "topology_hash", "config_hash", "dependency_lock_hash",
    )
    mismatches = [key for key in required_equal if left.get(key) != right.get(key)]
    if mismatches:
        raise ValueError(
            "reports are not a valid paired comparison; mismatched metadata: "
            + ", ".join(mismatches)
        )


# 作用：重采样估计置信区间。
# 上游（直接调用）：v1.analyze_v1.compare_reports。
# 下游调用：rng.choice、means.sort。
def _bootstrap_ci(values, seed=20260721, draws=10000):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    return means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]


# 作用：检查可比性并比较配对指标。
# 上游（直接调用）：v1.analyze_v1.main。
# 下游调用：_load、_paired_gate、_path_value、_bootstrap_ci。
def compare_reports(baseline_paths, treatment_paths):
    if len(baseline_paths) != len(treatment_paths) or not baseline_paths:
        raise ValueError("baseline and treatment must contain the same non-zero pair count")
    pairs = []
    for baseline_path, treatment_path in zip(baseline_paths, treatment_paths):
        left_path, left = _load(baseline_path)
        right_path, right = _load(treatment_path)
        _paired_gate(left, right)
        pairs.append((left_path, left, right_path, right))

    rows = []
    for name, (path, direction) in METRICS.items():
        observations = []
        for left_path, left, right_path, right in pairs:
            baseline = _path_value(left["metrics"], path)
            treatment = _path_value(right["metrics"], path)
            if baseline is None or treatment is None:
                continue
            observations.append((
                int(left["metadata"]["seed"]), baseline, treatment,
                treatment - baseline,
            ))
        deltas = [item[3] for item in observations]
        lower, upper = _bootstrap_ci(deltas)
        mean_delta = statistics.fmean(deltas) if deltas else None
        if mean_delta is None or direction == "diagnostic":
            interpretation = "NOT_APPLICABLE" if mean_delta is None else "DIAGNOSTIC"
        elif abs(mean_delta) <= 1e-12:
            interpretation = "NO_CHANGE"
        elif (direction == "higher" and mean_delta > 0) or (
            direction == "lower" and mean_delta < 0
        ):
            interpretation = "IMPROVED"
        else:
            interpretation = "DEGRADED"
        rows.append({
            "metric": name,
            "direction": direction,
            "paired_seed_count": len(observations),
            "baseline_mean": (
                statistics.fmean(item[1] for item in observations)
                if observations else None
            ),
            "treatment_mean": (
                statistics.fmean(item[2] for item in observations)
                if observations else None
            ),
            "mean_delta_treatment_minus_baseline": mean_delta,
            "bootstrap_95_ci_low": lower,
            "bootstrap_95_ci_high": upper,
            "interpretation": interpretation,
            "per_seed": [
                {"seed": seed, "baseline": base, "treatment": treat, "delta": delta}
                for seed, base, treat, delta in observations
            ],
        })
    return {
        "status": "VALID_PAIRED_COMPARISON",
        "pair_count": len(pairs),
        "seeds": [int(item[1]["metadata"]["seed"]) for item in pairs],
        "delta_definition": "treatment_minus_baseline",
        "ci_method": "paired_bootstrap_percentile",
        "ci_draws": 10000,
        "metrics": rows,
    }


# 作用：保存分析结果。
# 上游（直接调用）：v1.analyze_v1.main。
# 下游调用：show、prefix.parent.mkdir、prefix.with_suffix、json_path.write_text、csv_path.open 等。
def _write_outputs(result, output_prefix):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    fields = (
        "metric", "direction", "paired_seed_count", "baseline_mean",
        "treatment_mean", "mean_delta_treatment_minus_baseline",
        "bootstrap_95_ci_low", "bootstrap_95_ci_high", "interpretation",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["metrics"]:
            writer.writerow({key: row[key] for key in fields})
    lines = [
        "# v1.0模型效果配对分析",
        "",
        f"状态：{result['status']}；配对种子数：{result['pair_count']}。",
        "",
        "所有差值均为 treatment - baseline；置信区间为种子级配对Bootstrap。",
        "",
        "| 指标 | 基线均值 | 模型均值 | 平均差值 | 95% CI | 判断 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["metrics"]:
        # 作用：格式化分析结果的展示字段。
        # 上游（直接调用）：v1.analyze_v1._write_outputs。
        def show(value):
            return "N/A" if value is None else f"{value:.8g}"
        ci = f"[{show(row['bootstrap_95_ci_low'])}, {show(row['bootstrap_95_ci_high'])}]"
        lines.append(
            f"| {row['metric']} | {show(row['baseline_mean'])} | "
            f"{show(row['treatment_mean'])} | "
            f"{show(row['mean_delta_treatment_minus_baseline'])} | {ci} | "
            f"{row['interpretation']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path


# 作用：解析或转发命令行参数，组织本文件执行流程。
# 调用约定：本文件命令行入口或上层版本入口调用。
# 下游调用：compare_reports、_write_outputs、parser.add_argument、parser.parse_args。
def main():
    parser = argparse.ArgumentParser(description="Paired v1.0 model effect analysis")
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--treatment", action="append", required=True)
    parser.add_argument("--output-prefix", default="artifacts/v1/evaluation/model_effect")
    args = parser.parse_args()
    result = compare_reports(args.baseline, args.treatment)
    outputs = _write_outputs(result, args.output_prefix)
    print(json.dumps({
        "status": result["status"],
        "outputs": [str(item) for item in outputs],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
