# 文件功能：汇总 V6 五策略评估指标，输出可阅读的比较结果。
# 主要内容：display、main。
# 使用入口：命令行执行，或由其他脚本导入后调用。
# 模块下游（项目依赖）：v5.build_evaluation_comparison。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Export the completed V6 five-policy evaluations as CSV and Markdown."""

import json
from pathlib import Path

from v5.build_evaluation_comparison import PAIRED_FIELDS, POLICIES, value, write_csv


# 作用：格式化策略比较指标。
# 上游（直接调用）：v6.build_five_policy_comparison.main。
def display(number, spec):
    return "不适用" if number is None else format(number, spec)


# 作用：解析或转发命令行参数，组织本文件执行流程。
# 调用约定：本文件命令行入口或上层版本入口调用。
# 下游调用：value、display、write_csv、Path(__file__).resolve、(directory / filename).read_text 等。
def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "artifacts/v6/evaluation/five_policy_seed42_warmup2d"
    rows, sources = [], []
    reference = None
    for policy, label in POLICIES:
        filename = f"{policy}_seed42_warmup2d_v6.json"
        report = json.loads((directory / filename).read_text(encoding="utf-8"))
        if report.get("status") != "VALID":
            raise ValueError(f"Invalid evaluation: {filename}")
        metadata = report["metadata"]
        conditions = {key: metadata[key] for key in PAIRED_FIELDS}
        if reference is None:
            reference = conditions
        if conditions != reference:
            raise ValueError(f"Evaluation conditions differ: {filename}")
        metrics = report["metrics"]
        runtime = report["diagnostics"]["runtime_summary"]
        rows.append({
            "策略": label, "策略标识": policy,
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
            "主动等待P95_仿真时间": value(metrics["active_wait_metrics"]["p95_active_wait_sim"]),
            "平均决策耗时_ms": runtime["decision_wall_seconds"]["mean"] * 1000,
            "决策P95耗时_ms": runtime["decision_wall_seconds"]["p95"] * 1000,
            "评估运行耗时_s": runtime["total_wall_seconds"],
            "源文件": filename,
        })
        sources.append(f"- {label}：`{filename}`")
    lines = [
        "# V6 五策略评估对比", "",
        f"seed={reference['seed']}；预热 {reference['warmup_days']:g} 天；"
        f"正式窗口为仿真时间 {reference['evaluation_start_sim']:g}–{reference['arrival_cutoff_sim']:g}。",
        "五份结果的任务轨迹、外生轨迹、拓扑、候选模式和计费模式一致。", "",
        "## 核心指标", "",
        "| 策略 | 完成率 | 总经济成本（万元） | 绿电覆盖率 | 系统绿电消纳率 | 平均决策耗时（ms） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['策略']} | {display(row['完成率'], '.4%')} | {row['总经济成本_元'] / 10000:.4f} | {display(row['完成任务绿电覆盖率'], '.4%')} | {display(row['系统绿电消纳率'], '.4%')} | {row['平均决策耗时_ms']:.2f} |")
    lines += ["", "## SLA 与等待", "",
              "| 策略 | Soft 首选窗口准时率 | Flexible 首选窗口准时率 | 主动等待均值（仿真时间） | 决策 P95（ms） |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['策略']} | {display(row['Soft首选窗口准时率'], '.4%')} | {display(row['Flexible首选窗口准时率'], '.4%')} | {display(row['主动等待均值_仿真时间'], '.4f')} | {row['决策P95耗时_ms']:.2f} |")
    lines += ["", "## 任务结果", "",
              "| 策略 | 到达 | 完成 | 过期 | 失败 |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['策略']} | {row['到达任务数']} | {row['完成任务数']} | {row['过期任务数']} | {row['失败任务数']} |")
    lines += ["", "## 指标说明", "",
              "- 绿电覆盖率为完成任务所用能源中的绿电比例；系统消纳率为评估窗口绿电供应的消纳比例。",
              "- CSV 中比例保留 0–1 数值，成本单位为元；聊天与本表将成本展示为万元。",
              "- 首选窗口准时率遵循原评估定义，分母为相应 SLA 类型的到达任务数。",
              "- 主动等待均值以主动等待任务为统计对象。最早可行策略没有主动等待样本，记为不适用。",
              "- 耗时来自已有评估统计；此次未重跑评估。单个 seed 的结果不代表跨任务轨迹的稳定排名。",
              "", "## 来源", "", *sources, ""]
    csv_path = directory / "V6五策略对比.csv"
    markdown_path = directory / "V6五策略对比.md"
    write_csv(csv_path, rows)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(markdown_path),
                      "policy_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
