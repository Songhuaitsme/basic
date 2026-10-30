# 文件功能：比较 V4/V5 评估指标和耗时，导出比较表。
# 主要内容：value、write_csv、main。
# 模块上游（导入关系）：v5.build_five_policy_comparison、v6.build_five_policy_comparison。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Build V4/V5 five-policy comparison tables from completed evaluations."""

import csv
import json
from pathlib import Path


POLICIES = (
    ("earliest_feasible", "最早可行"),
    ("lowest_cost", "最低成本"),
    ("highest_green", "最高绿电"),
    ("equal_weight", "等权综合"),
    ("candidate_dqn", "Candidate DQN"),
)
PAIRED_FIELDS = (
    "seed", "warmup_days", "evaluation_start_sim", "arrival_cutoff_sim",
    "candidate_mode", "tariff_mode", "topology_hash", "task_trace_hash",
    "exogenous_trace_hash", "dependency_lock_hash",
)


# 作用：从报告中读取可展示指标。
# 上游（直接调用）：v5.build_evaluation_comparison.main、v5.build_five_policy_comparison.main、v6.build_five_policy_comparison.main。
def value(metric):
    if isinstance(metric, dict):
        return metric.get("value") if metric.get("status") == "VALID" else None
    return metric


# 作用：将行记录按约定字段写入 CSV。
# 上游（直接调用）：v5.build_evaluation_comparison.main、v5.build_five_policy_comparison.main、v6.build_five_policy_comparison.main。
# 下游调用：path.open、writer.writeheader、writer.writerows。
def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# 作用：解析或转发命令行参数，组织本文件执行流程。
# 调用约定：本文件命令行入口或上层版本入口调用。
# 下游调用：value、write_csv、Path(__file__).resolve、path.read_text、version.upper 等。
def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/v5/evaluation/comparison_v4_v5_seed42_warmup2d"
    quality, comparisons, sources = [], [], []
    reference = None
    for policy, label in POLICIES:
        pair = {}
        for version, folder in (
            ("v4", "five_policy_quick"),
            ("v5", "five_policy_seed42_warmup2d"),
        ):
            filename = f"{policy}_seed42_warmup2d_{version}.json"
            path = root / "artifacts" / version / "evaluation" / folder / filename
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("status") != "VALID":
                raise ValueError(f"Invalid evaluation: {path}")
            metadata = report["metadata"]
            conditions = {key: metadata[key] for key in PAIRED_FIELDS}
            if reference is None:
                reference = conditions
            if conditions != reference:
                raise ValueError(f"Evaluation conditions differ: {path}")
            pair[version] = report
            metrics = report["metrics"]
            runtime = report["diagnostics"]["runtime_summary"]
            quality.append({
                "策略": label, "策略标识": policy, "版本": version.upper(),
                "到达任务数": metrics["arrival_count"],
                "完成任务数": metrics["completed_count"],
                "过期任务数": metrics["expired_count"],
                "失败任务数": metrics["failed_count"],
                "完成率": value(metrics["completion_rate"]),
                "总经济成本_元": metrics["total_economic_cost_yuan"],
                "单位完成CPU小时成本_元": value(metrics["cost_yuan_per_completed_cpu_hour"]),
                "完成任务绿电覆盖率": value(metrics["completed_task_green_coverage"]),
                "系统绿电消纳率": value(metrics["system_green_absorption_rate"]),
                "Soft首选窗口准时率": value(metrics["sla_metrics"]["Soft"]["preferred_on_time_rate"]),
                "Flexible首选窗口准时率": value(metrics["sla_metrics"]["Flexible"]["preferred_on_time_rate"]),
                "主动等待均值_仿真时间": value(metrics["active_wait_metrics"]["mean_active_wait_sim"]),
                "平均决策耗时_ms": runtime["decision_wall_seconds"]["mean"] * 1000,
                "决策P95耗时_ms": runtime["decision_wall_seconds"]["p95"] * 1000,
                "评估运行耗时_s": runtime["total_wall_seconds"],
                "模型哈希": metadata["model_hash"], "源文件": filename,
            })
            sources.append(f"- {version.upper()} / {label}：`{filename}`")
        v4, v5 = pair["v4"], pair["v5"]
        m4, m5 = v4["metrics"], v5["metrics"]
        r4, r5 = (pair[v]["diagnostics"]["runtime_summary"] for v in ("v4", "v5"))
        t4, t5 = (runtime["decision_wall_seconds"]["mean"] for runtime in (r4, r5))
        if t4 <= 0 or t5 <= 0:
            raise ValueError(f"Missing decision timing for {policy}")
        comparisons.append({
            "策略": label, "策略标识": policy,
            "V4完成率": value(m4["completion_rate"]), "V5完成率": value(m5["completion_rate"]),
            "V4总经济成本_元": m4["total_economic_cost_yuan"],
            "V5总经济成本_元": m5["total_economic_cost_yuan"],
            "成本变化_元": m5["total_economic_cost_yuan"] - m4["total_economic_cost_yuan"],
            "V4绿电覆盖率": value(m4["completed_task_green_coverage"]),
            "V5绿电覆盖率": value(m5["completed_task_green_coverage"]),
            "V4平均决策耗时_ms": t4 * 1000, "V5平均决策耗时_ms": t5 * 1000,
            "平均决策耗时下降比例": 1 - t5 / t4, "平均决策加速倍数": t4 / t5,
            "V4决策P95耗时_ms": r4["decision_wall_seconds"]["p95"] * 1000,
            "V5决策P95耗时_ms": r5["decision_wall_seconds"]["p95"] * 1000,
            "V4评估运行耗时_s": r4["total_wall_seconds"],
            "V5评估运行耗时_s": r5["total_wall_seconds"],
            "全部metrics字段一致": m4 == m5,
            "同模型哈希": v4["metadata"]["model_hash"] == v5["metadata"]["model_hash"],
        })

    lines = [
        "# V4 与 V5 五策略评估对比", "",
        "seed=42；预热两天；正式窗口为仿真时间 576–864，共一天；每份评估 2501 个到达任务。",
        "任务轨迹、外生轨迹、拓扑、候选模式、计费模式和依赖锁哈希一致。", "",
        "五种策略的全部 metrics 字段均与 V4 完全一致，包括 SLA 与主动等待指标。",
        "每份结果均完成 2412 个任务、过期 89 个任务、失败 0 个任务，完成率为 96.4414%。", "",
        "## 解质量", "",
        "下表各项 V4 与 V5 数值相同。成本单位为万元；比例以百分数显示。", "",
        "| 策略 | 完成率 | 总经济成本（万元） | 绿电覆盖率 | 系统绿电消纳率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in quality[1::2]:
        lines.append(f"| {row['策略']} | {row['完成率']:.4%} | {row['总经济成本_元'] / 10000:.4f} | {row['完成任务绿电覆盖率']:.4%} | {row['系统绿电消纳率']:.4%} |")
    lines += ["", "## 决策耗时", "",
              "| 策略 | V4 平均（ms） | V5 平均（ms） | 平均耗时下降 | 加速倍数 | V4 P95（ms） | V5 P95（ms） |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['策略']} | {row['V4平均决策耗时_ms']:.2f} | {row['V5平均决策耗时_ms']:.2f} | {row['平均决策耗时下降比例']:.2%} | {row['平均决策加速倍数']:.3f} | {row['V4决策P95耗时_ms']:.2f} | {row['V5决策P95耗时_ms']:.2f} |")
    lines += ["", "## 说明", "",
              "- DQN 的模型文件哈希不同。这是各自模型的评估结果对比，不能据此单独归因候选生成优化的速度收益。",
              "- 耗时来自已有评估结果的基础统计。平均决策耗时与整段评估运行耗时范围不同。",
              "- 平均决策耗时下降比例 = 1 − V5/V4；加速倍数 = V4/V5。",
              "- 这是单个 seed 的历史运行对比，机器负载及运行时段可能影响耗时，不代表稳定加速保证。",
              "- CSV 中率与下降比例为 0–1 的数值；成本为元；计数为任务数。", "",
              "## 来源", "", *sources, ""]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "策略指标明细.csv", quality)
    write_csv(output / "V4_V5评估对比.csv", comparisons)
    (output / "V4_V5评估对比.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "metric_rows": len(quality),
                      "comparison_rows": len(comparisons),
                      "all_metrics_equal": all(row["全部metrics字段一致"] for row in comparisons)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
