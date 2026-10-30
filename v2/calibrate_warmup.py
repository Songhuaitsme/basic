# 文件功能：比较不同预热时长的指标，为正式测量窗口提供校准依据。
# 主要内容：_row、main。
# 使用入口：命令行执行，或由其他脚本导入后调用。
# 模块下游（项目依赖）：shared.config、v1.evaluate_v1、v1.v1_runtime、v2.evaluate_five_policies、v2.export_five_policy_metrics。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Compare steady-state boundary/load indicators across warm-up lengths."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from shared import config
from v1.evaluate_v1 import _canonical_hash, _generate_trace, run_evaluation
from v1.v1_runtime import create_v1_runtime
from v2.evaluate_five_policies import DEFAULT_MODEL_PATH
from v2.export_five_policy_metrics import POLICIES


CALIBRATION_FIELDS = (
    "mean_node_cpu_utilization",
    "queued_or_unadmitted_task_count",
    "active_task_count",
    "future_cpu_reserved_cpu_sim",
    "mean_link_utilization",
)


# 作用：将单次评估或校准结果整理为表格行。
# 上游（直接调用）：v2.calibrate_warmup.main。
def _row(report, *, policy, warmup_days):
    boundary = report.warmup_snapshot or {}
    diagnostics = report.diagnostics or {}
    time_summary = diagnostics.get("time_summary") or {}
    return {
        "policy": policy,
        "warmup_days": warmup_days,
        "measurement_start_sim": report.metadata.evaluation_start_sim,
        "measurement_end_sim": report.metadata.arrival_cutoff_sim,
        "final_settlement_time_sim": report.metadata.final_settlement_time_sim,
        "measured_task_count": len(report.measured_task_ids),
        "task_trace_hash": report.metadata.task_trace_hash,
        "exogenous_trace_hash": report.metadata.exogenous_trace_hash,
        **{field: boundary.get(field) for field in CALIBRATION_FIELDS},
        "measurement_cpu_utilization": time_summary.get(
            "time_node_mean_cpu_utilization"
        ),
        "measurement_link_utilization": time_summary.get(
            "time_link_mean_utilization"
        ),
    }


# 作用：解析或转发命令行参数，组织本文件执行流程。
# 调用约定：本文件命令行入口或上层版本入口调用。
# 下游调用：create_v1_runtime、_generate_trace、_canonical_hash、run_evaluation、_row 等。
def main():
    parser = argparse.ArgumentParser(
        description="Run the 0.5/1/2-day warm-up steady-state calibration"
    )
    parser.add_argument(
        "--warmup-days", type=float, nargs="+", default=(0.5, 1.0, 2.0)
    )
    parser.add_argument(
        "--measurement-duration",
        type=float,
        default=config.TRAFFIC_DAY_DURATION_IN_SIM,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--safety-cap", type=int, default=1_000_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--candidate-chunk-size", type=int, default=65_536)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v2/evaluation/warmup_calibration.json"),
    )
    args = parser.parse_args()

    if any(days < 0.0 for days in args.warmup_days):
        parser.error("--warmup-days values must be non-negative")
    if args.measurement_duration <= 0.0:
        parser.error("--measurement-duration must be positive")
    if "candidate_dqn" in args.policies and not args.model_path.is_file():
        parser.error("candidate_dqn calibration requires an existing --model-path")

    rows = []
    trace_manifest = []
    for days in args.warmup_days:
        trace_end = (
            days * config.TRAFFIC_DAY_DURATION_IN_SIM
            + args.measurement_duration
        )
        trace_runtime = create_v1_runtime(
            policy_name="earliest_feasible",
            forecast_end_sim=trace_end + config.V1_MAX_FORECAST_LOOKAHEAD_SIM,
            random_seed=args.seed,
            device="cpu",
            candidate_chunk_size=args.candidate_chunk_size,
        )
        trace = _generate_trace(trace_runtime, trace_end, args.seed)
        trace_manifest.append({
            "warmup_days": days,
            "task_count": len(trace),
            "task_trace_hash": _canonical_hash(
                [asdict(task) for task in trace]
            ),
        })
        for policy in args.policies:
            report = run_evaluation(
                policy,
                args.measurement_duration,
                args.seed,
                args.safety_cap,
                args.model_path if policy == "candidate_dqn" else None,
                device=args.device,
                candidate_chunk_size=args.candidate_chunk_size,
                system_version="2.0",
                audit_mode="periodic",
                task_trace=trace,
                warmup_days=days,
            )
            if report.status.value != "VALID":
                raise RuntimeError(
                    f"{policy} warmup={days} is {report.status.value}: "
                    f"{report.unsettled_task_ids}"
                )
            rows.append(_row(report, policy=policy, warmup_days=days))

    payload = {
        "schema_version": "1.0",
        "seed": args.seed,
        "warmup_days": list(args.warmup_days),
        "measurement_duration_sim": args.measurement_duration,
        "policies": list(args.policies),
        "calibration_fields": list(CALIBRATION_FIELDS),
        "rows": rows,
        "trace_manifest": trace_manifest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "VALID",
        "output": str(args.output),
        "row_count": len(rows),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
