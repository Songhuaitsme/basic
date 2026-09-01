"""Compare the uniform-tariff V1 and V2 DQN checkpoints on one seed.

The command can either consume existing formal reports or run fresh V1/V2
evaluations before producing static comparison figures and machine-readable
evidence files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import torch


UNIFORM_TARIFF_MODE = "tou_uniform"
DEFAULT_V1_MODEL = Path(
    "artifacts/v1/formal/candidate_dqn_seed7_layered_pool_600000_v2.pt"
)
DEFAULT_V2_MODEL = Path(
    "artifacts/v2/formal/"
    "candidate_dqn_seed7_layered_pool_600000_from_scratch.pt"
)
DEFAULT_V1_REPORT = Path("artifacts/v1/evaluation/candidate_dqn_seed42.json")
DEFAULT_V2_REPORT = Path(
    "artifacts/v2/evaluation/five_policy/seed_42/candidate_dqn_seed42.json"
)
DEFAULT_V1_PROFILE = Path(
    "artifacts/v1/formal/candidate_dqn_seed7_layered_pool_50000_v1.profile.json"
)
DEFAULT_V2_PROFILE = Path(
    "artifacts/v2/formal/"
    "candidate_dqn_seed7_layered_pool_600000_from_scratch.profile.json"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_summary(path: Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    run_config = checkpoint.get("run_config") or {}
    tariff_mode = run_config.get("V1_TARIFF_MODE")
    if tariff_mode != UNIFORM_TARIFF_MODE:
        raise ValueError(
            f"{path} uses {tariff_mode!r}; expected unpartitioned pricing "
            f"{UNIFORM_TARIFF_MODE!r}"
        )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"{path} does not contain model_state_dict")
    summary = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tariff_mode": tariff_mode,
        "training_steps": checkpoint.get("training_steps"),
        "training_seed": checkpoint.get("seed"),
        "system_version": run_config.get("SYSTEM_VERSION", "1.0"),
    }
    return summary, state_dict


def _compare_parameters(v1_state: dict, v2_state: dict) -> tuple[bool, float]:
    if set(v1_state) != set(v2_state):
        return False, math.inf
    maximum = 0.0
    identical = True
    for key in sorted(v1_state):
        left = v1_state[key]
        right = v2_state[key]
        if left.shape != right.shape:
            return False, math.inf
        if not torch.equal(left, right):
            identical = False
        maximum = max(maximum, float((left - right).abs().max().item()))
    return identical, maximum


def _metric_value(value):
    if isinstance(value, dict):
        if value.get("status") not in (None, "VALID"):
            raise ValueError(f"metric is not valid: {value}")
        value = value.get("value")
    if value is None:
        raise ValueError("required metric has no value")
    return float(value)


def _completed_tasks(report: dict) -> list[dict]:
    return [
        item
        for item in report.get("task_records") or ()
        if item.get("final_state") == "Completed"
    ]


def extract_quality_metrics(report: dict) -> dict[str, float]:
    if report.get("status") != "VALID":
        raise ValueError(f"evaluation report is not VALID: {report.get('status')}")
    metrics = report.get("metrics") or {}
    completed = _completed_tasks(report)
    if not completed:
        raise ValueError("evaluation report contains no completed task records")

    cpu_by_node: dict[str, float] = {}
    completion_delays = []
    for item in completed:
        node = item.get("target_node")
        if node:
            cpu_by_node[node] = cpu_by_node.get(node, 0.0) + float(
                item.get("cpu_work_cpu_hours") or 0.0
            )
        delay = item.get("completion_delay_sim")
        if delay is not None:
            completion_delays.append(float(delay))
    allocations = tuple(cpu_by_node.values())
    allocation_mean = statistics.fmean(allocations) if allocations else 0.0
    load_cv = (
        statistics.pstdev(allocations) / allocation_mean
        if allocations and allocation_mean > 0.0
        else 0.0
    )
    seconds_per_sim_unit = 300.0
    return {
        "cost_yuan_per_completed_cpu_hour": _metric_value(
            metrics.get("cost_yuan_per_completed_cpu_hour")
        ),
        "completed_task_green_coverage": _metric_value(
            metrics.get("completed_task_green_coverage")
        ),
        "allocated_cpu_hours_cv": load_cv,
        "mean_completion_delay_hours": (
            statistics.fmean(completion_delays) * seconds_per_sim_unit / 3600.0
        ),
    }


def _runtime_seconds(report: dict) -> float | None:
    diagnostics = report.get("diagnostics") or {}
    value = (
        diagnostics.get("runtime_summary", {}).get("total_wall_seconds")
    )
    return None if value is None else float(value)


def _validate_pair(v1_report: dict, v2_report: dict, seed: int) -> dict:
    v1_metadata = v1_report.get("metadata") or {}
    v2_metadata = v2_report.get("metadata") or {}
    for label, metadata in (("V1", v1_metadata), ("V2", v2_metadata)):
        if int(metadata.get("seed", -1)) != seed:
            raise ValueError(f"{label} report seed does not equal {seed}")
        if metadata.get("tariff_mode") != UNIFORM_TARIFF_MODE:
            raise ValueError(
                f"{label} report does not use {UNIFORM_TARIFF_MODE} pricing"
            )
    keys = ("task_trace_hash", "exogenous_trace_hash", "topology_hash")
    mismatched = [
        key for key in keys if v1_metadata.get(key) != v2_metadata.get(key)
    ]
    if mismatched:
        raise ValueError(
            "V1/V2 reports are not paired; mismatched metadata: "
            + ", ".join(mismatched)
        )
    return {key: v1_metadata.get(key) for key in keys}


def _run_evaluation(
    module: str,
    *,
    model: Path,
    seed: int,
    cutoff: float,
    device: str,
    output: Path,
    audit: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        module,
        "--policy",
        "candidate_dqn",
        "--seed",
        str(seed),
        "--arrival-cutoff",
        str(cutoff),
        "--model-path",
        str(model),
        "--device",
        device,
        "--audit",
        audit,
        "--report-mode",
        "compact",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    return command


def _configure_font() -> None:
    available = {item.name for item in font_manager.fontManager.ttflist}
    for family in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial"):
        if family in available:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False


def _annotate_bar(axis, bar, value: float, formatter) -> None:
    axis.annotate(
        formatter(value),
        (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#20242A",
    )


def _quality_figure(
    values: dict[str, dict[str, float]],
    output: Path,
    *,
    seed: int,
    parameter_identical: bool,
) -> None:
    _configure_font()
    specs = (
        (
            "cost_yuan_per_completed_cpu_hour",
            "单位完成成本",
            "元 / 完成 CPU 小时；越低越好",
            lambda value: f"{value:,.4f}",
        ),
        (
            "completed_task_green_coverage",
            "完成任务绿电覆盖率",
            "占比；越高越好",
            lambda value: f"{value * 100:.4f}%",
        ),
        (
            "allocated_cpu_hours_cv",
            "节点 CPU 工作量变异系数",
            "CV；越低表示越均衡",
            lambda value: f"{value:.6f}",
        ),
        (
            "mean_completion_delay_hours",
            "平均任务完成时延",
            "小时；仅统计已完成任务，越低越好",
            lambda value: f"{value:.4f}",
        ),
    )
    colors = ("#2F6B9A", "#D98C30")
    hatches = ("///", "...")
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    for axis, (key, title, subtitle, formatter) in zip(axes.flat, specs):
        plotted = [values["V1"][key], values["V2"][key]]
        bars = axis.bar(
            ("V1", "V2"),
            plotted,
            color=colors,
            edgecolor="#30343B",
            linewidth=0.8,
        )
        for bar, hatch, value in zip(bars, hatches, plotted):
            bar.set_hatch(hatch)
            _annotate_bar(axis, bar, value, formatter)
        upper = max(plotted)
        axis.set_ylim(0.0, upper * 1.22 if upper > 0.0 else 1.0)
        axis.set_title(
            title,
            loc="left",
            y=1.08,
            fontsize=12,
            fontweight="bold",
        )
        axis.text(
            0.0,
            1.015,
            subtitle,
            transform=axis.transAxes,
            fontsize=9,
            color="#59616B",
        )
        axis.grid(axis="y", color="#D9DEE5", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    identity = "完全一致" if parameter_identical else "存在差异"
    figure.suptitle(
        f"Seed {seed}：V1 与 V2 未分区定价 DQN 效果对比",
        x=0.06,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#20242A",
    )
    figure.text(
        0.06,
        0.925,
        f"相同任务、拓扑和外生轨迹；checkpoint 网络参数：{identity}",
        fontsize=10,
        color="#59616B",
    )
    figure.tight_layout(rect=(0.04, 0.04, 0.98, 0.89), h_pad=4.2, w_pad=1.8)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _training_speed_figure(
    v1_profile: Path,
    v2_profile: Path,
    output: Path,
    *,
    v1_steps: int,
    v2_steps: int,
) -> dict:
    profiles = (_load_json(v1_profile), _load_json(v2_profile))
    seconds_per_1000 = (
        profiles[0]["total_wall_seconds"] / v1_steps * 1000.0,
        profiles[1]["total_wall_seconds"] / v2_steps * 1000.0,
    )
    speedup = seconds_per_1000[0] / seconds_per_1000[1]
    _configure_font()
    figure, axis = plt.subplots(figsize=(8.6, 5.4))
    bars = axis.bar(
        ("V1", "V2"),
        seconds_per_1000,
        color=("#2F6B9A", "#D98C30"),
        edgecolor="#30343B",
        linewidth=0.8,
    )
    for bar, hatch, value in zip(bars, ("///", "..."), seconds_per_1000):
        bar.set_hatch(hatch)
        _annotate_bar(axis, bar, value, lambda item: f"{item:.2f} 秒")
    axis.set_ylim(0.0, max(seconds_per_1000) * 1.25)
    axis.set_title(
        "训练墙钟时间（按虚拟周期归一化）",
        loc="left",
        y=1.07,
        fontsize=14,
        fontweight="bold",
    )
    axis.text(
        0.0,
        1.015,
        "每 1,000 个虚拟训练周期；越低越好",
        transform=axis.transAxes,
        fontsize=10,
        color="#59616B",
    )
    axis.text(
        0.98,
        0.92,
        f"V2 约快 {speedup:.2f}×",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#20242A",
    )
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(pad=1.8)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "v1_seconds_per_1000_virtual_cycles": seconds_per_1000[0],
        "v2_seconds_per_1000_virtual_cycles": seconds_per_1000[1],
        "v2_speedup": speedup,
        "v1_profile": str(v1_profile),
        "v2_profile": str(v2_profile),
        "v1_profile_steps": v1_steps,
        "v2_profile_steps": v2_steps,
    }


def _evaluation_speed_figure(values: dict[str, float], output: Path) -> dict:
    speedup = values["V1"] / values["V2"]
    _configure_font()
    figure, axis = plt.subplots(figsize=(8.6, 5.4))
    bars = axis.bar(
        ("V1 默认审计", "V2 默认审计"),
        (values["V1"], values["V2"]),
        color=("#2F6B9A", "#D98C30"),
        edgecolor="#30343B",
        linewidth=0.8,
    )
    for bar, hatch, value in zip(bars, ("///", "..."), values.values()):
        bar.set_hatch(hatch)
        _annotate_bar(axis, bar, value, lambda item: f"{item:.2f} 秒")
    axis.set_ylim(0.0, max(values.values()) * 1.25)
    axis.set_title(
        "Seed 评估墙钟时间",
        loc="left",
        y=1.07,
        fontsize=14,
        fontweight="bold",
    )
    axis.text(
        0.0,
        1.015,
        "同 seed、同设备；V1 full audit，V2 periodic audit；越低越好",
        transform=axis.transAxes,
        fontsize=10,
        color="#59616B",
    )
    axis.text(
        0.98,
        0.92,
        f"V2 约快 {speedup:.2f}×",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#20242A",
    )
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(pad=1.8)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {"v1_seconds": values["V1"], "v2_seconds": values["V2"], "speedup": speedup}


def _write_metrics_csv(path: Path, values: dict[str, dict[str, float]]) -> None:
    fields = ("metric", "v1", "v2", "absolute_delta", "relative_delta")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in values["V1"]:
            v1_value = values["V1"][metric]
            v2_value = values["V2"][metric]
            writer.writerow(
                {
                    "metric": metric,
                    "v1": v1_value,
                    "v2": v2_value,
                    "absolute_delta": v2_value - v1_value,
                    "relative_delta": (
                        (v2_value - v1_value) / v1_value if v1_value else None
                    ),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare uniform-tariff V1/V2 DQN checkpoints and generate "
            "quality and speed figures."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-cutoff", type=float, default=288.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--v1-model", type=Path, default=DEFAULT_V1_MODEL)
    parser.add_argument("--v2-model", type=Path, default=DEFAULT_V2_MODEL)
    parser.add_argument("--v1-report", type=Path, default=DEFAULT_V1_REPORT)
    parser.add_argument("--v2-report", type=Path, default=DEFAULT_V2_REPORT)
    parser.add_argument("--v1-profile", type=Path, default=DEFAULT_V1_PROFILE)
    parser.add_argument("--v2-profile", type=Path, default=DEFAULT_V2_PROFILE)
    parser.add_argument("--v1-profile-steps", type=int, default=50000)
    parser.add_argument("--v2-profile-steps", type=int, default=600000)
    parser.add_argument(
        "--run-evaluations",
        action="store_true",
        help="run fresh V1 and V2 evaluations before plotting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v2/evaluation/v1_v2_seed42"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing comparison files in the output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    owned_outputs = (
        args.output_dir / "quality_comparison.png",
        args.output_dir / "training_speed_comparison.png",
        args.output_dir / "evaluation_speed_comparison.png",
        args.output_dir / "metrics.csv",
        args.output_dir / "comparison.json",
    )
    existing = [path for path in owned_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "comparison outputs already exist; rerun with --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    v1_model, v1_state = _checkpoint_summary(args.v1_model)
    v2_model, v2_state = _checkpoint_summary(args.v2_model)
    parameter_identical, maximum_parameter_delta = _compare_parameters(
        v1_state, v2_state
    )

    commands = []
    v1_report_path = args.v1_report
    v2_report_path = args.v2_report
    if args.run_evaluations:
        v1_report_path = args.output_dir / f"v1_candidate_dqn_seed{args.seed}.json"
        v2_report_path = args.output_dir / f"v2_candidate_dqn_seed{args.seed}.json"
        commands.append(_run_evaluation(
            "v1.evaluate_v1",
            model=args.v1_model,
            seed=args.seed,
            cutoff=args.arrival_cutoff,
            device=args.device,
            output=v1_report_path,
            audit="full",
        ))
        commands.append(_run_evaluation(
            "v2.evaluate_v2",
            model=args.v2_model,
            seed=args.seed,
            cutoff=args.arrival_cutoff,
            device=args.device,
            output=v2_report_path,
            audit="periodic",
        ))

    v1_report = _load_json(v1_report_path)
    v2_report = _load_json(v2_report_path)
    paired_hashes = _validate_pair(v1_report, v2_report, args.seed)
    values = {
        "V1": extract_quality_metrics(v1_report),
        "V2": extract_quality_metrics(v2_report),
    }
    _quality_figure(
        values,
        args.output_dir / "quality_comparison.png",
        seed=args.seed,
        parameter_identical=parameter_identical,
    )
    _write_metrics_csv(args.output_dir / "metrics.csv", values)

    training_speed = _training_speed_figure(
        args.v1_profile,
        args.v2_profile,
        args.output_dir / "training_speed_comparison.png",
        v1_steps=args.v1_profile_steps,
        v2_steps=args.v2_profile_steps,
    )
    evaluation_times = {
        "V1": _runtime_seconds(v1_report),
        "V2": _runtime_seconds(v2_report),
    }
    evaluation_speed = None
    if all(value is not None and value > 0.0 for value in evaluation_times.values()):
        evaluation_speed = _evaluation_speed_figure(
            evaluation_times,
            args.output_dir / "evaluation_speed_comparison.png",
        )

    payload = {
        "seed": args.seed,
        "arrival_cutoff_sim": args.arrival_cutoff,
        "pricing_mode": UNIFORM_TARIFF_MODE,
        "models": {"V1": v1_model, "V2": v2_model},
        "model_parameters_identical": parameter_identical,
        "maximum_parameter_absolute_delta": maximum_parameter_delta,
        "paired_trace_hashes": paired_hashes,
        "reports": {"V1": str(v1_report_path), "V2": str(v2_report_path)},
        "quality_metrics": values,
        "training_speed": training_speed,
        "evaluation_speed": evaluation_speed,
        "commands": commands,
        "notes": [
            "Quality bars use zero baselines and exact labels.",
            "Training speed is normalized by virtual training cycles from the supplied profiles.",
            "The bundled V1 profile covers 50,000 cycles; the V2 profile covers 600,000 cycles.",
            "Fresh evaluation speed compares the entry points' intended defaults: V1 full audit and V2 periodic audit.",
        ],
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "quality_figure": str(args.output_dir / "quality_comparison.png"),
        "training_speed_figure": str(
            args.output_dir / "training_speed_comparison.png"
        ),
        "evaluation_speed_figure": (
            str(args.output_dir / "evaluation_speed_comparison.png")
            if evaluation_speed is not None else None
        ),
        "metrics": str(args.output_dir / "metrics.csv"),
        "comparison": str(args.output_dir / "comparison.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
