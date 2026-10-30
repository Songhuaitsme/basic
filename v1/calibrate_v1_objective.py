# 文件功能：从候选流采样，校准目标函数的成本和绿电尺度。
# 主要内容：ReservoirCalibrationPolicy、_distribution、run_multi_seed_calibration、run_calibration、main。
# 模块上游（导入关系）：tests.v1.test_calibration_v1。
# 模块下游（项目依赖）：shared.config、v1.evaluate_v1、v1.evaluation_v1.runner、v1.scheduler.policies、v1.v1_runtime。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Pilot calibration for fixed v1.0 candidate feature/objective scales."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from shared import config
from v1.evaluate_v1 import _generate_trace
from v1.evaluation_v1 import EvaluationRunner
from v1.scheduler import CandidateStreamSelection
from v1.v1_runtime import create_v1_runtime


class ReservoirCalibrationPolicy:
    name = "calibration_earliest_feasible"

    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.calibrate_v1_objective.run_multi_seed_calibration。
    # 调用约定：构造实例时由 Python 调用。
    def __init__(self, capacity, seed):
        if capacity <= 0:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.random = random.Random(seed)
        self.samples = []
        self.seen = 0

    # 作用：构造稳定采样或索引键。
    # 上游（直接调用）：v1.calibrate_v1_objective.ReservoirCalibrationPolicy.select_stream。
    @staticmethod
    def _key(candidate):
        return (
            candidate.compute_start_sim,
            candidate.target_node,
            candidate.path.path_id,
            candidate.candidate_id,
        )

    # 作用：将候选加入校准采样统计。
    # 上游（直接调用）：v1.calibrate_v1_objective.ReservoirCalibrationPolicy.select_stream。
    # 下游调用：self.random.randrange。
    def _observe(self, candidate, task=None):
        self.seen += 1
        sample = {
            "cost_yuan": candidate.estimated_candidate_marginal_system_cost_yuan,
            "green_absorption_delta": candidate.estimated_green_absorption_delta,
            "green_coverage": candidate.estimated_green_coverage,
            "capacity_margin": candidate.capacity_margin,
            "preferred_start_tardiness_ratio": (
                candidate.preferred_start_tardiness_ratio
            ),
            "tardiness_applicable": candidate.preferred_start_tardiness_applicable,
            "sla_type": None if task is None else task.sla_type.value,
        }
        if len(self.samples) < self.capacity:
            self.samples.append(sample)
            return
        index = self.random.randrange(self.seen)
        if index < self.capacity:
            self.samples[index] = sample

    # 作用：遍历候选流，汇总选中与最早候选及决策信息。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    # 下游调用：self._observe、self._key、candidate.candidate_id.encode、CandidateStreamSelection、digest.hexdigest。
    def select_stream(self, candidates, task=None, context=None):
        selected = None
        count = 0
        digest = hashlib.sha256()
        for candidate in candidates:
            count += 1
            digest.update(candidate.candidate_id.encode("utf-8"))
            digest.update(b"\0")
            self._observe(candidate, task)
            if selected is None or self._key(candidate) < self._key(selected):
                selected = candidate
        if selected is None:
            raise ValueError("calibration received an empty candidate stream")
        return CandidateStreamSelection(
            selected, selected, count, digest.hexdigest(), context
        )


# 作用：计算数量、分位数及范围等分布统计。
# 上游（直接调用）：v1.calibrate_v1_objective.run_multi_seed_calibration。
def _distribution(samples, field, *, applicable_only=False, sla_type=None):
    values = [
        float(item[field])
        for item in samples
        if not applicable_only or item["tardiness_applicable"]
        if sla_type is None or item.get("sla_type") == sla_type
    ]
    if not values:
        return None
    percentiles = np.percentile(values, (0, 10, 25, 50, 75, 90, 100))
    return {
        "count": len(values),
        "min": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p90": float(percentiles[5]),
        "max": float(percentiles[6]),
    }


# 作用：跨种子采样候选并汇总尺度校准。
# 上游（直接调用）：v1.calibrate_v1_objective.main、v1.calibrate_v1_objective.run_calibration。
# 下游调用：ReservoirCalibrationPolicy、create_v1_runtime、_generate_trace、EvaluationRunner、_distribution 等。
def run_multi_seed_calibration(
    cutoff,
    seeds,
    reservoir_size,
    safety_cap,
    *,
    reservoir_seed=20260722,
    candidate_mode="complete",
):
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("calibration seeds cannot be empty")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("calibration seeds must be unique")
    horizon = cutoff + config.V1_MAX_FORECAST_LOOKAHEAD_SIM
    collector = ReservoirCalibrationPolicy(reservoir_size, reservoir_seed)
    per_seed = []
    total_tasks = 0
    for seed in seed_values:
        configured_candidate_mode = config.V1_CANDIDATE_MODE
        try:
            config.V1_CANDIDATE_MODE = candidate_mode
            runtime = create_v1_runtime(forecast_end_sim=horizon, random_seed=seed)
        finally:
            config.V1_CANDIDATE_MODE = configured_candidate_mode
        runtime.scheduler.policy = collector
        trace = _generate_trace(runtime, cutoff, seed)
        seen_before = collector.seen
        report = EvaluationRunner(
            runtime.scheduler, runtime.time_converter, safety_cap
        ).run_frozen_policy(trace, arrival_cutoff_sim=cutoff, seed=seed)
        if report.status.value != "VALID":
            raise RuntimeError(
                f"calibration evaluation failed for seed {seed}: "
                f"{report.status.value}"
            )
        sla_counts = {}
        for task in trace:
            sla_counts[task.sla_type.value] = (
                sla_counts.get(task.sla_type.value, 0) + 1
            )
        total_tasks += len(trace)
        seed_summary = {
            "seed": seed,
            "task_count": len(trace),
            "sla_counts": sla_counts,
            "candidate_count_seen": collector.seen - seen_before,
        }
        per_seed.append(seed_summary)
        print(json.dumps({"calibration_progress": seed_summary}, ensure_ascii=False), flush=True)
    cost = _distribution(collector.samples, "cost_yuan")
    absorption = _distribution(collector.samples, "green_absorption_delta")
    tardiness = _distribution(
        collector.samples,
        "preferred_start_tardiness_ratio",
        applicable_only=True,
    )
    soft_tardiness = _distribution(
        collector.samples,
        "preferred_start_tardiness_ratio",
        applicable_only=True,
        sla_type="Soft",
    )
    flexible_tardiness = _distribution(
        collector.samples,
        "preferred_start_tardiness_ratio",
        applicable_only=True,
        sla_type="Flexible",
    )
    cost_scale = None if cost is None else max(cost["p90"] - cost["p10"], 1e-8)
    absorption_scale = (
        None if absorption is None else max(abs(absorption["p90"]), 1e-8)
    )
    return {
        "status": (
            "PILOT_SCALE_ESTIMATE"
            if len(seed_values) == 1 else "MULTI_SEED_SCALE_ESTIMATE"
        ),
        "seed": seed_values[0] if len(seed_values) == 1 else None,
        "seeds": list(seed_values),
        "reservoir_seed": int(reservoir_seed),
        "arrival_cutoff_sim": cutoff,
        "task_count": total_tasks,
        "candidate_count_seen": collector.seen,
        "reservoir_size": len(collector.samples),
        "per_seed": per_seed,
        "candidate_mode": candidate_mode,
        "distributions": {
            "cost_yuan": cost,
            "green_absorption_delta": absorption,
            "preferred_start_tardiness_ratio": tardiness,
            "soft_tardiness_ratio": soft_tardiness,
            "flexible_tardiness_ratio": flexible_tardiness,
        },
        "suggested_config": {
            "V1_COST_REFERENCE_YUAN": None if cost is None else cost["p50"],
            "V1_COST_SCALE_YUAN": cost_scale,
            "V1_GREEN_ABSORPTION_DELTA_SCALE": absorption_scale,
            "V1_SOFT_TARDINESS_WEIGHT": "REQUIRES_PILOT_ABLATION_DECISION",
            "V1_FLEXIBLE_TARDINESS_WEIGHT": "REQUIRES_PILOT_ABLATION_DECISION",
        },
        "note": (
            "Scale estimates are descriptive, not business thresholds. Freeze "
            "tardiness weights through a documented pilot/ablation before full training."
        ),
    }


# 作用：执行单次候选采样和尺度校准。
# 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
# 下游调用：run_multi_seed_calibration。
def run_calibration(cutoff, seed, reservoir_size, safety_cap):
    return run_multi_seed_calibration(
        cutoff,
        (seed,),
        reservoir_size,
        safety_cap,
        reservoir_seed=seed,
    )


# 作用：解析或转发命令行参数，组织本文件执行流程。
# 调用约定：本文件命令行入口或上层版本入口调用。
# 下游调用：run_multi_seed_calibration、parser.add_argument、parser.parse_args、output.parent.mkdir、output.write_text。
def main():
    parser = argparse.ArgumentParser(description="Calibrate fixed v1.0 objective scales")
    parser.add_argument("--arrival-cutoff", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="repeat for a global candidate-weighted multi-seed reservoir",
    )
    parser.add_argument("--reservoir-seed", type=int, default=20260722)
    parser.add_argument("--reservoir-size", type=int, default=100000)
    parser.add_argument("--safety-cap", type=int, default=1000000)
    parser.add_argument(
        "--candidate-mode",
        choices=("complete", "layered_pool"),
        default="complete",
    )
    parser.add_argument("--output", default="artifacts/v1/logs/objective_calibration.json")
    args = parser.parse_args()
    seeds = args.seed or [42]
    result = run_multi_seed_calibration(
        args.arrival_cutoff,
        seeds,
        args.reservoir_size,
        args.safety_cap,
        reservoir_seed=args.reservoir_seed,
        candidate_mode=args.candidate_mode,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
