#!/usr/bin/env python3
"""Build a reproducible Pilot-effect report artifact from saved v1 outputs."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "artifacts" / "v1" / "pilot" / "diagnostic"
TRAINING_CSV = ROOT / "artifacts" / "v1" / "pilot" / "candidate_dqn_pilot_seed7_b1.training.csv"
EFFECT_JSON = (
    ROOT
    / "artifacts"
    / "v1"
    / "evaluation"
    / "v1"
    / "pilot_seed7_unseen4"
    / "pilot_seed7_unseen4_effect.json"
)
MODEL_PATH = ROOT / "artifacts" / "v1" / "pilot" / "candidate_dqn_pilot_seed7_b1.pt"


def number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_training() -> tuple[list[dict[str, object]], dict[str, float]]:
    with TRAINING_CSV.open("r", encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))

    # Resume attempts repeat cycles 1100-1300. The final occurrence is the
    # authoritative curve point; all raw intervals still count as compute spent.
    by_cycle: dict[int, dict[str, str]] = {}
    for row in raw:
        by_cycle[int(row["cycle"])] = row

    curve: list[dict[str, object]] = []
    for cycle in sorted(by_cycle):
        row = by_cycle[cycle]
        transitions = int(row["transitions"])
        candidates = int(row["candidates"])
        curve.append(
            {
                "cycle": cycle,
                "simulated_time": float(row["time_sim"]),
                "tasks": int(row["tasks"]),
                "transitions": transitions,
                "updates": int(row["updates"]),
                "replay_size": int(row["replay_size"]),
                "epsilon": float(row["epsilon"]),
                "mean_loss": number(row["mean_loss"]),
                "selection_candidates_total": candidates,
                "selection_candidates_since_log": int(row["candidates_since_log"]),
                "wall_seconds_since_log": float(row["wall_seconds_since_log"]),
                "device": row["device"],
                "selection_candidates_per_transition": (
                    candidates / transitions if transitions else None
                ),
            }
        )

    final = curve[-1]
    raw_wall_seconds = sum(float(row["wall_seconds_since_log"]) for row in raw)
    summary = {
        "raw_log_rows": float(len(raw)),
        "unique_cycle_points": float(len(curve)),
        "duplicate_cycle_rows": float(len(raw) - len(curve)),
        "logged_wall_hours_including_rework": raw_wall_seconds / 3600.0,
        "final_transitions": float(final["transitions"]),
        "final_updates": float(final["updates"]),
        "final_epsilon": float(final["epsilon"]),
        "final_selection_candidates": float(final["selection_candidates_total"]),
        "selection_candidates_per_transition": (
            float(final["selection_candidates_total"]) / float(final["transitions"])
        ),
        "logged_wall_seconds_per_transition": (
            raw_wall_seconds / float(final["transitions"])
        ),
    }
    return curve, summary


def metric_map(effect: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(metric["metric"]): metric
        for metric in effect["metrics"]  # type: ignore[index]
    }


def paired_rows(metrics: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    names = [
        "total_economic_cost_yuan",
        "cost_yuan_per_completed_cpu_hour",
        "completed_task_green_coverage",
        "system_green_absorption_rate",
        "active_wait_count",
    ]
    indexed: dict[int, dict[str, object]] = {}
    for name in names:
        for row in metrics[name]["per_seed"]:  # type: ignore[index]
            seed = int(row["seed"])
            indexed.setdefault(seed, {"seed": seed, "seed_label": f"Seed {seed}"})
            indexed[seed][f"{name}_baseline"] = row["baseline"]
            indexed[seed][f"{name}_treatment"] = row["treatment"]
            indexed[seed][f"{name}_delta"] = row["delta"]
    return [indexed[seed] for seed in sorted(indexed)]


def build_sqlite_snapshot(
    pairs: list[dict[str, object]],
    curve: list[dict[str, object]],
    training: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    db_path = OUT_DIR / "pilot_diagnostic.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP TABLE IF EXISTS paired_seed_effects")
        connection.execute("DROP TABLE IF EXISTS training_curve")
        connection.execute("DROP TABLE IF EXISTS training_summary")

        pair_columns = list(pairs[0])
        pair_types = [
            "INTEGER" if column == "seed" else "TEXT" if column == "seed_label" else "REAL"
            for column in pair_columns
        ]
        connection.execute(
            "CREATE TABLE paired_seed_effects ("
            + ", ".join(
                f'"{column}" {column_type}'
                for column, column_type in zip(pair_columns, pair_types)
            )
            + ")"
        )
        connection.executemany(
            "INSERT INTO paired_seed_effects VALUES ("
            + ", ".join("?" for _ in pair_columns)
            + ")",
            [[row[column] for column in pair_columns] for row in pairs],
        )

        curve_columns = list(curve[0])
        curve_types = [
            "TEXT" if column == "device" else "INTEGER"
            if column in {
                "cycle",
                "tasks",
                "transitions",
                "updates",
                "replay_size",
                "selection_candidates_total",
                "selection_candidates_since_log",
            }
            else "REAL"
            for column in curve_columns
        ]
        connection.execute(
            "CREATE TABLE training_curve ("
            + ", ".join(
                f'"{column}" {column_type}'
                for column, column_type in zip(curve_columns, curve_types)
            )
            + ")"
        )
        connection.executemany(
            "INSERT INTO training_curve VALUES ("
            + ", ".join("?" for _ in curve_columns)
            + ")",
            [[row[column] for column in curve_columns] for row in curve],
        )

        summary_columns = list(training)
        connection.execute(
            "CREATE TABLE training_summary ("
            + ", ".join(f'"{column}" REAL' for column in summary_columns)
            + ")"
        )
        connection.execute(
            "INSERT INTO training_summary VALUES ("
            + ", ".join("?" for _ in summary_columns)
            + ")",
            [training[column] for column in summary_columns],
        )
        connection.commit()

        connection.row_factory = sqlite3.Row
        queried_pairs = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM paired_seed_effects ORDER BY seed"
            ).fetchall()
        ]
        queried_curve = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM training_curve ORDER BY cycle"
            ).fetchall()
        ]
    finally:
        connection.close()
    return queried_pairs, queried_curve


def source_specs(generated_at: str) -> list[dict[str, object]]:
    return [
        {
            "id": "paired_effect",
            "label": "4-seed paired Pilot evaluation (treatment minus baseline; 10,000 paired bootstrap draws)",
            "path": "artifacts/v1/pilot/diagnostic/pilot_diagnostic.sqlite",
            "query": {
                "engine": "SQLite",
                "language": "SQL",
                "sql": "SELECT * FROM paired_seed_effects ORDER BY seed;",
                "description": "Read all reviewed paired seed effects used by the report.",
                "executed_at": generated_at,
                "filters": [
                    "cutoff_time=1.0",
                    "candidate_dqn evaluation epsilon=0",
                    "paired seeds=1,20,31,34",
                    "raw input=artifacts/v1/evaluation/v1/pilot_seed7_unseen4/pilot_seed7_unseen4_effect.json",
                ],
                "metric_definitions": [
                    "Every delta is treatment minus baseline at seed grain.",
                    "Economic cost is lower-is-better; green coverage and absorption are higher-is-better.",
                    "Intervals in the raw effect file are paired seed-level bootstrap percentile intervals with 10,000 draws.",
                ],
                "tables_used": ["pilot_diagnostic.paired_seed_effects"],
            },
        },
        {
            "id": "training_log",
            "label": "Pilot training curve (last resumed row retained per cycle)",
            "path": "artifacts/v1/pilot/diagnostic/pilot_diagnostic.sqlite",
            "query": {
                "engine": "SQLite",
                "language": "SQL",
                "sql": "SELECT * FROM training_curve ORDER BY cycle;",
                "description": "Read the reviewed training curve after keeping the last resumed row for each cycle.",
                "executed_at": generated_at,
                "filters": [
                    "cycles=100..3500",
                    "seed=7",
                    "batch_size=1",
                    "updates_per_transition=1",
                    "device=cpu",
                    "raw input=artifacts/v1/pilot/candidate_dqn_pilot_seed7_b1.training.csv",
                ],
                "metric_definitions": [
                    "Selection candidates are the cumulative candidate counter written by the trainer.",
                    "Logged wall hours include repeated work recorded across resume attempts.",
                    "Curve points keep the final occurrence of each cycle after resume duplication.",
                ],
                "tables_used": ["pilot_diagnostic.training_curve"],
            },
        },
        {
            "id": "pilot_model",
            "label": "Pilot candidate-DQN model",
            "path": "artifacts/v1/pilot/candidate_dqn_pilot_seed7_b1.pt",
        },
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    effect = json.loads(EFFECT_JSON.read_text(encoding="utf-8"))
    metrics = metric_map(effect)
    pairs = paired_rows(metrics)
    curve, training = load_training()
    pairs, curve = build_sqlite_snapshot(pairs, curve, training)

    cost = metrics["total_economic_cost_yuan"]
    cost_rate = metrics["cost_yuan_per_completed_cpu_hour"]
    green = metrics["completed_task_green_coverage"]
    absorption = metrics["system_green_absorption_rate"]
    waiting = metrics["active_wait_count"]

    cst_delta = float(cost["mean_delta_treatment_minus_baseline"])
    cst_low = float(cost["bootstrap_95_ci_low"])
    cst_high = float(cost["bootstrap_95_ci_high"])
    rate_delta = float(cost_rate["mean_delta_treatment_minus_baseline"])
    green_delta = float(green["mean_delta_treatment_minus_baseline"])
    absorption_delta = float(absorption["mean_delta_treatment_minus_baseline"])
    waiting_delta = float(waiting["mean_delta_treatment_minus_baseline"])
    waiting_low = float(waiting["bootstrap_95_ci_low"])
    waiting_high = float(waiting["bootstrap_95_ci_high"])

    generated_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    sources = source_specs(generated_at)

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Pilot 模型效果诊断",
            "description": "117 次 candidate-DQN 更新后的训练吞吐与 4 个未见种子配对效果。",
            "generatedAt": generated_at,
            "sources": sources,
            "charts": [
                {
                    "id": "cost_delta_by_seed",
                    "title": "各未见种子的经济成本差值",
                    "subtitle": "模型减基线，人民币；负值表示模型更好，n=4",
                    "showDescription": True,
                    "intent": "comparison",
                    "question": "经济成本变化是普遍改善，还是由种子差异主导？",
                    "rationale": "四个离散配对种子适合用横向条形图直接比较正负差值。",
                    "comparisonContext": {
                        "baseline": "equal_weight",
                        "grain": "paired seed",
                        "unit": "CNY",
                        "semanticFamily": "treatment minus baseline",
                    },
                    "type": "horizontalBar",
                    "dataset": "paired_seed_effects",
                    "sourceId": "paired_effect",
                    "encodings": {
                        "x": {"field": "seed_label", "type": "nominal", "label": "未见种子"},
                        "y": {
                            "field": "total_economic_cost_yuan_delta",
                            "type": "quantitative",
                            "format": "currency",
                            "label": "经济成本差值",
                            "unit": "CNY",
                        },
                    },
                    "xAxisTitle": "未见种子",
                    "yAxisTitle": "模型 - 基线（元）",
                    "valueFormat": "currency",
                    "unit": "CNY",
                    "layout": "full",
                    "surface": {
                        "palette": {"kind": "diverging", "root": "blue"},
                        "valueLabels": {"mode": "all"},
                    },
                },
                {
                    "id": "waiting_delta_by_seed",
                    "title": "各未见种子的活跃等待任务差值",
                    "subtitle": "模型减基线，任务数；正值表示等待任务更多，n=4",
                    "showDescription": True,
                    "intent": "comparison",
                    "question": "等待任务增加是否只由单个异常种子造成？",
                    "rationale": "同一单位的四个配对差值用横向条形图可直接检查方向一致性。",
                    "comparisonContext": {
                        "baseline": "equal_weight",
                        "grain": "paired seed",
                        "unit": "tasks",
                        "semanticFamily": "treatment minus baseline",
                    },
                    "type": "horizontalBar",
                    "dataset": "paired_seed_effects",
                    "sourceId": "paired_effect",
                    "encodings": {
                        "x": {"field": "seed_label", "type": "nominal", "label": "未见种子"},
                        "y": {
                            "field": "active_wait_count_delta",
                            "type": "quantitative",
                            "format": "number",
                            "label": "活跃等待任务差值",
                            "unit": "tasks",
                        },
                    },
                    "xAxisTitle": "未见种子",
                    "yAxisTitle": "模型 - 基线（任务数）",
                    "valueFormat": "number",
                    "unit": "tasks",
                    "layout": "full",
                    "surface": {
                        "palette": {"kind": "single", "root": "orange"},
                        "valueLabels": {"mode": "all"},
                    },
                },
                {
                    "id": "training_transitions",
                    "title": "Pilot 训练 transition 累积曲线",
                    "subtitle": "Seed 7，3500 cycle；恢复重复 cycle 按最后一次记录去重",
                    "showDescription": True,
                    "intent": "trend",
                    "question": "当前仿真 cycle 能以多快速度产生有效训练 transition？",
                    "rationale": "35 个有序检查点足以显示 transition 随 cycle 的稀疏累积。",
                    "comparisonContext": {
                        "grain": "100 simulation cycles",
                        "unit": "transitions",
                        "semanticFamily": "cumulative training progress",
                    },
                    "type": "line",
                    "dataset": "training_curve",
                    "sourceId": "training_log",
                    "encodings": {
                        "x": {"field": "cycle", "type": "quantitative", "label": "仿真 cycle"},
                        "y": {
                            "field": "transitions",
                            "type": "quantitative",
                            "format": "number",
                            "label": "累计 transition",
                        },
                    },
                    "xAxisTitle": "仿真 cycle",
                    "yAxisTitle": "累计 transition",
                    "valueFormat": "number",
                    "layout": "full",
                    "surface": {
                        "palette": {"kind": "single", "root": "blue"},
                        "points": "always",
                        "valueLabels": {"mode": "endpoints"},
                    },
                },
            ],
            "tables": [
                {
                    "id": "paired_seed_audit",
                    "title": "未见种子配对明细",
                    "subtitle": "模型减基线；经济成本为负更好，等待任务为正更差",
                    "showDescription": True,
                    "dataset": "paired_seed_effects",
                    "defaultSort": {"field": "seed", "direction": "asc"},
                    "density": "spacious",
                    "sourceId": "paired_effect",
                    "layout": "full",
                    "columns": [
                        {"field": "seed", "label": "Seed", "type": "number"},
                        {
                            "field": "total_economic_cost_yuan_baseline",
                            "label": "基线成本（元）",
                            "format": "currency",
                        },
                        {
                            "field": "total_economic_cost_yuan_treatment",
                            "label": "模型成本（元）",
                            "format": "currency",
                        },
                        {
                            "field": "total_economic_cost_yuan_delta",
                            "label": "成本差值（元）",
                            "format": "currency",
                            "movement": True,
                        },
                        {
                            "field": "completed_task_green_coverage_delta",
                            "label": "绿色覆盖差值",
                            "format": "percent",
                            "movement": True,
                        },
                        {
                            "field": "active_wait_count_delta",
                            "label": "等待任务差值",
                            "format": "number",
                            "movement": True,
                        },
                    ],
                }
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# Pilot 模型效果诊断",
                    "layout": "full",
                },
                {
                    "id": "executive_summary",
                    "type": "markdown",
                    "body": (
                        "## Executive Summary\n\n"
                        f"- **还不能进入正式大规模训练。** 117 次更新后的冻结策略在 4 个未见种子上，"
                        f"经济成本平均增加 **{cst_delta:.0f} 元**；95% 配对 bootstrap 区间为 "
                        f"**[{cst_low:.0f}, {cst_high:.0f}] 元**，没有证明改善。\n"
                        f"- **出现了目标间权衡。** 完成任务绿色覆盖平均提高 **{green_delta:.1%}**，"
                        f"系统绿色吸收率提高 **{absorption_delta:.4%}**，但成本/完成 CPU 小时增加 "
                        f"**{rate_delta:.3f} 元**，活跃等待任务平均增加 **{waiting_delta:.2f} 个**。\n"
                        f"- **训练吞吐是当前硬约束。** 3500 cycle 只产生 **{int(training['final_transitions'])}** "
                        f"条 transition，却累计扫描 **{training['final_selection_candidates']/1e6:.1f}M** "
                        f"个选择候选，约 **{training['selection_candidates_per_transition']/1e6:.2f}M 候选/transition**。\n"
                        "- **下一步门槛明确。** 先完成同一负载的 CUDA/CPU 对照；随后必须提高每次候选枚举产生的"
                        "训练样本效率，再进行至少 10 个配对种子的正式评估。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "definitions",
                    "type": "markdown",
                    "body": (
                        "## 比较口径与成功标准\n\n"
                        "处理组是 seed 7 Pilot 训练后的 candidate-DQN 冻结策略，评估时 epsilon=0；"
                        "基线是 equal-weight。比较粒度为未参与训练的 seeds 1、20、31、34，"
                        "所有差值均定义为“模型减基线”。主成功标准是降低经济成本，同时不牺牲完成率、"
                        "可靠性或等待队列；绿色指标属于共同目标，但不能单独替代经济与服务约束。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "cost_finding",
                    "type": "markdown",
                    "sourceId": "paired_effect",
                    "body": (
                        "## 经济目标没有改善，种子差异主导结果\n\n"
                        f"平均成本变化为 **+{cst_delta:.0f} 元**，但区间跨过 0，证据不足以判断真实平均效果。"
                        "四个种子中，seed 1 和 20 成本下降，seed 31 基本不变，seed 34 增加约 2757 元。"
                        "这说明小样本均值易被场景差异左右，不能据此宣称模型优于基线。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "cost_chart",
                    "type": "chart",
                    "chartId": "cost_delta_by_seed",
                    "layout": "full",
                },
                {
                    "id": "waiting_finding",
                    "type": "markdown",
                    "sourceId": "paired_effect",
                    "body": (
                        "## 等待任务增加是跨种子一致现象\n\n"
                        f"活跃等待任务平均增加 **{waiting_delta:.2f} 个**，95% 区间为 "
                        f"**[{waiting_low:.2f}, {waiting_high:.2f}]**。四个种子的差值均为正，"
                        "范围从 +3 到 +7。与成本的不确定结果不同，这个方向一致的服务侧退化需要在下一轮"
                        "奖励设计或状态/动作表达中直接处理。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "waiting_chart",
                    "type": "chart",
                    "chartId": "waiting_delta_by_seed",
                    "layout": "full",
                },
                {
                    "id": "training_finding",
                    "type": "markdown",
                    "sourceId": "training_log",
                    "body": (
                        "## 训练样本产出远慢于候选枚举\n\n"
                        f"最终只有 **{int(training['final_updates'])} 次更新**，epsilon 仍为 "
                        f"**{training['final_epsilon']:.3f}**。原始日志记录了约 "
                        f"**{training['logged_wall_hours_including_rework']:.1f} 小时**计算（含恢复重算），"
                        f"相当于约 **{training['logged_wall_seconds_per_transition']:.0f} 秒/transition**。"
                        "loss 能下降只说明数值参数发生学习，不能抵消样本稀少和策略效果未改善的证据。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "training_chart",
                    "type": "chart",
                    "chartId": "training_transitions",
                    "layout": "full",
                },
                {
                    "id": "audit_table_intro",
                    "type": "markdown",
                    "sourceId": "paired_effect",
                    "body": (
                        "## 配对明细保留了均值背后的差异\n\n"
                        "下表给出每个种子的基线、模型和差值，便于核对成本权衡是否由单一场景主导。"
                        "绿色覆盖差值按比例展示；等待任务差值为期末活跃等待计数之差。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "audit_table",
                    "type": "table",
                    "tableId": "paired_seed_audit",
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 建议的下一步\n\n"
                        "1. 完成相同候选评分负载的 CPU/GPU 对照，分别记录神经网络推理时间与端到端时间。\n"
                        "2. 正式训练前提高样本效率：减少无决策 cycle 的重复枚举、缓存可复用候选特征，"
                        "或让一次昂贵枚举支持更多 replay 更新，但必须保持完整候选语义。\n"
                        "3. 只有当小规模门槛同时满足“成本不劣化、等待不增加、绿色指标不倒退”时，"
                        "才扩大训练；最终用至少 10 个固定配对种子报告区间，而不是只看均值。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 仍需回答的问题\n\n"
                        "- 等待任务增加来自模型偏好更晚的槽位、奖励延迟，还是状态中缺少队列压力？\n"
                        "- CUDA 是否只加速约占少数的网络评分，端到端收益是否不足以改变训练可行性？\n"
                        "- 提高 replay 更新比后，样本相关性和离策略偏差是否仍可接受？"
                    ),
                    "layout": "full",
                },
                {
                    "id": "caveats",
                    "type": "markdown",
                    "body": (
                        "## 限制与假设\n\n"
                        "效果评估只有 4 个未见种子，区间较宽，不能作为最终统计结论。训练日志在恢复时重复了"
                        "1100–1300 cycle：曲线按 cycle 保留最后一条，计算成本则保留全部原始区间以反映实际投入。"
                        "候选计数是训练器记录的选择候选累计值，不等同于包含 replay 重评分在内的全部模型输入行数。"
                        "完成率、接受率和预约可靠性在本样本中均为 1，可能存在天花板效应。"
                    ),
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "paired_seed_effects": pairs,
                "training_curve": curve,
                "training_summary": [training],
            },
        },
        "sources": sources,
    }

    artifact_path = OUT_DIR / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (OUT_DIR / "training_curve_deduplicated.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)

    (OUT_DIR / "analysis_summary.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "training": training,
                "paired_effect": {
                    "pair_count": effect["pair_count"],
                    "seeds": effect["seeds"],
                    "cost_delta_mean": cst_delta,
                    "cost_delta_ci95": [cst_low, cst_high],
                    "cost_rate_delta_mean": rate_delta,
                    "green_coverage_delta_mean": green_delta,
                    "green_absorption_delta_mean": absorption_delta,
                    "active_wait_delta_mean": waiting_delta,
                    "active_wait_delta_ci95": [waiting_low, waiting_high],
                },
                "model_bytes": MODEL_PATH.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(artifact_path)


if __name__ == "__main__":
    main()
