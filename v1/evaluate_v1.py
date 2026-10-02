"""Executable formal v1.0 frozen-policy evaluation entry point."""

import argparse
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
import random
import time

import networkx as nx
import numpy as np
import torch

from shared import config
from v1.ablation_settings import apply_ablation_variant, variant_names
from v1.domain.models import TaskSpec
from v1.evaluation_v1 import EvaluationRunner
from v1.evaluation_v1.diagnostics import build_evaluation_diagnostics
from v1.learning import validate_checkpoint_metadata
from v1.profiling import TrainingPerformanceProfiler
from v1.scheduler import ObjectiveConfig
from v1.v1_runtime import (
    create_v1_runtime,
    ensure_v1_runtime_forecasts_for_tasks,
)


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key.value if isinstance(key, Enum) else key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("formal evaluation output contains NaN or Infinity")
    return value


def _generate_trace(runtime, cutoff, seed):
    random.seed(seed)
    np.random.seed(seed)
    tasks = []
    time_sim = 0.0
    cycle = 0
    total_capacity = sum(
        runtime.calendar.node_capacity(node)
        for node in runtime.infrastructure.compute_nodes
    )
    while time_sim < cutoff - 1e-12:
        time_sim += config.SCHEDULING_CYCLE
        lam, _ = runtime.task_manager.get_dynamic_task_rate(time_sim)
        tasks.extend(runtime.task_manager.generate_task_specs(
            np.random.poisson(lam),
            time_sim,
            cycle,
            cpu_budget=(
                total_capacity
                * config.SCHEDULING_CYCLE
                * config.TASK_PEAK_LOAD_MULTIPLIER
            ),
        ))
        cycle += 1
    return tuple(task for task in tasks if task.arrival_time_sim < cutoff)


def run_evaluation(
    policy,
    cutoff,
    seed,
    safety_cap,
    model_path=None,
    *,
    device="cpu",
    candidate_chunk_size=None,
    soft_tardiness_weight=None,
    flexible_tardiness_weight=None,
    ablation_variant=None,
    system_version=None,
    audit_mode="full",
    audit_interval=500,
    task_trace=None,
    warmup_days=None,
):
    system_version = str(
        config.SYSTEM_VERSION if system_version is None else system_version
    )
    if ablation_variant is not None:
        with apply_ablation_variant(ablation_variant):
            return run_evaluation(
                policy,
                cutoff,
                seed,
                safety_cap,
                model_path,
                device=device,
                candidate_chunk_size=candidate_chunk_size,
                soft_tardiness_weight=soft_tardiness_weight,
                flexible_tardiness_weight=flexible_tardiness_weight,
                system_version=system_version,
                audit_mode=audit_mode,
                audit_interval=audit_interval,
                task_trace=task_trace,
                warmup_days=warmup_days,
            )
    warmup_days = (
        config.WARMUP_DAYS
        if warmup_days is None else float(warmup_days)
    )
    if not np.isfinite(warmup_days) or warmup_days < 0.0:
        raise ValueError("warmup_days must be a finite non-negative number")
    measurement_duration = float(cutoff)
    if not np.isfinite(measurement_duration) or measurement_duration < 0.0:
        raise ValueError("cutoff must be a finite non-negative duration")
    measurement_start = (
        warmup_days * config.TRAFFIC_DAY_DURATION_IN_SIM
    )
    measurement_end = measurement_start + measurement_duration
    if (
        (soft_tardiness_weight is not None or flexible_tardiness_weight is not None)
        and policy != "equal_weight"
    ):
        raise ValueError("tardiness-weight overrides require equal_weight policy")
    soft_weight = (
        config.V1_SOFT_TARDINESS_WEIGHT
        if soft_tardiness_weight is None else float(soft_tardiness_weight)
    )
    flexible_weight = (
        config.V1_FLEXIBLE_TARDINESS_WEIGHT
        if flexible_tardiness_weight is None else float(flexible_tardiness_weight)
    )
    objective = ObjectiveConfig(
        config.V1_COST_REFERENCE_YUAN,
        config.V1_COST_SCALE_YUAN,
        config.V1_GREEN_ABSORPTION_DELTA_SCALE,
        config.V1_OBJECTIVE_COST_WEIGHT,
        config.V1_OBJECTIVE_GREEN_WEIGHT,
        config.V1_OBJECTIVE_BALANCE_WEIGHT,
        soft_weight,
        flexible_weight,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    horizon = measurement_end + config.V1_MAX_FORECAST_LOOKAHEAD_SIM
    runtime = create_v1_runtime(
        policy_name=policy,
        forecast_end_sim=horizon,
        random_seed=seed,
        device=device,
        candidate_chunk_size=candidate_chunk_size,
        objective_config=objective,
    )
    model_hash = _canonical_hash({"policy": policy})
    if policy == "candidate_dqn":
        if not model_path:
            raise ValueError("candidate_dqn evaluation requires --model-path")
        checkpoint_path = Path(model_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"model checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint.get("model_state_dict"), dict):
            raise ValueError("model checkpoint does not contain model_state_dict")
        metadata = checkpoint.get("metadata", {})
        architecture = "shared_candidate_q_v1"
        if not config.V1_DQN_USE_GLOBAL_STATE or not config.V1_DQN_DOUBLE_DQN:
            architecture += (
                f":global={int(config.V1_DQN_USE_GLOBAL_STATE)}"
                f":double={int(config.V1_DQN_DOUBLE_DQN)}"
            )
        validate_checkpoint_metadata(
            metadata,
            runtime.candidate_feature_encoder.feature_schema_hash,
            architecture,
        )
        runtime.candidate_q_network.load_state_dict(
            checkpoint["model_state_dict"]
        )
        runtime.candidate_q_network.eval()
        runtime.scheduler.policy.epsilon = 0.0
        model_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    trace = (
        _generate_trace(runtime, measurement_end, seed)
        if task_trace is None
        else tuple(task_trace)
    )
    if any(task.arrival_time_sim >= measurement_end for task in trace):
        raise ValueError("task trace contains arrivals outside the evaluation cutoff")
    # The configured lookahead is only an initial allocation.  A generated
    # task can legally start near the end of its SLA window and then execute
    # beyond ``cutoff + V1_MAX_FORECAST_LOOKAHEAD_SIM``.  Extend both physical
    # forecasts from the realized trace before any candidate is evaluated.
    ensure_v1_runtime_forecasts_for_tasks(runtime, trace)
    graph_data = nx.node_link_data(runtime.infrastructure.topo_manager.graph)
    config_view = {
        name: getattr(config, name)
        for name in dir(config)
        if name.isupper()
        and isinstance(
            getattr(config, name),
            (str, int, float, bool, tuple, list, dict, type(None)),
        )
    }
    config_view["V1_SOFT_TARDINESS_WEIGHT"] = soft_weight
    config_view["V1_FLEXIBLE_TARDINESS_WEIGHT"] = flexible_weight
    config_view["SYSTEM_VERSION"] = system_version
    config_view.update(getattr(runtime, "config_view_overrides", {}))
    package_root = Path(__file__).resolve().parent
    code_files = (
        package_root / "v1_runtime.py",
        package_root / "evaluate_v1.py",
        package_root / "scheduler" / "v1_scheduler.py",
        package_root / "scheduler" / "candidate_generator.py",
        package_root / "scheduler" / "queue_manager.py",
        package_root / "scheduler" / "resource_calendar.py",
        package_root / "accounting" / "energy.py",
        package_root / "evaluation_v1" / "runner.py",
        package_root / "evaluation_v1" / "diagnostics.py",
        package_root / "profiling.py",
        package_root / "learning" / "candidate_dqn.py",
        package_root / "simulation" / "state_machine.py",
    )
    code_files += tuple(getattr(runtime, "evaluation_code_files", ()))
    code_hash = hashlib.sha256(
        b"".join(path.read_bytes() for path in code_files)
    ).hexdigest()
    forecast_versions = {
        node: (
            runtime.accounting.tariff_by_node[node].version,
            runtime.accounting.green_by_node[node].version,
        )
        for node in runtime.infrastructure.compute_nodes
    }
    context = {
        "code_hash": code_hash,
        "model_hash": model_hash,
        "config_hash": _canonical_hash(config_view),
        "topology_hash": _canonical_hash(graph_data),
        "task_trace_hash": _canonical_hash([asdict(task) for task in trace]),
        "exogenous_trace_hash": _canonical_hash(forecast_versions),
        "dependency_lock_hash": _canonical_hash({
            "numpy": np.__version__,
            "torch": torch.__version__,
        }),
        "tariff_mode": config.V1_TARIFF_MODE,
        "gamma_per_second": config.V1_GAMMA_PER_SECOND,
        "warmup_days": warmup_days,
    }
    runner = EvaluationRunner(
        runtime.scheduler,
        runtime.time_converter,
        safety_cap,
        metadata_context=context,
        system_version=system_version,
        audit_mode=audit_mode,
        audit_interval=audit_interval,
    )
    profiler = TrainingPerformanceProfiler()
    runtime.scheduler.profiler = profiler
    runtime.scheduler.candidate_generator.profiler = profiler
    if hasattr(runtime.scheduler.policy, "profiler"):
        runtime.scheduler.policy.profiler = profiler
    evaluation_started = time.perf_counter()
    report = runner.run_frozen_policy(
        trace,
        arrival_cutoff_sim=measurement_end,
        evaluation_start_sim=measurement_start,
        seed=seed,
    )
    evaluation_wall_seconds = time.perf_counter() - evaluation_started
    node_capacities = {
        node: runtime.calendar.node_capacity(node)
        for node in runtime.infrastructure.compute_nodes
    }
    graph = runtime.infrastructure.topo_manager.graph
    link_capacities = {
        tuple(edge): runtime.calendar.link_capacity(tuple(edge))
        for edge in graph.edges
        if runtime.calendar.link_capacity(tuple(edge)) is not None
    }
    diagnostics = build_evaluation_diagnostics(
        report,
        reservations=runtime.calendar.reservations(),
        node_capacities=node_capacities,
        link_capacities=link_capacities,
        time_converter=runtime.time_converter,
        profiler_summary=profiler.summary(evaluation_wall_seconds),
        energy_accounting=runtime.accounting,
    )
    return replace(report, diagnostics=diagnostics)


def main(
    *,
    system_version=None,
    default_output="artifacts/v1/evaluation/report.json",
    default_audit_mode="full",
    default_report_mode="full",
    default_candidate_chunk_size=None,
):
    effective_system_version = str(
        config.SYSTEM_VERSION if system_version is None else system_version
    )
    parser = argparse.ArgumentParser(
        description=(
            f"Formal frozen-policy evaluation with system "
            f"V{effective_system_version}"
        )
    )
    parser.add_argument(
        "--policy",
        choices=(
            "earliest_feasible",
            "lowest_cost",
            "highest_green",
            "equal_weight",
            "candidate_dqn",
        ),
        default="earliest_feasible",
    )
    parser.add_argument(
        "--arrival-cutoff",
        type=float,
        default=config.TRAFFIC_DAY_DURATION_IN_SIM,
        help="Measurement duration in simulation-time units",
    )
    parser.add_argument(
        "--warmup-days",
        type=float,
        default=config.WARMUP_DAYS,
        help="Warm-up duration in simulated traffic days (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safety-cap", type=int, default=1000000)
    parser.add_argument("--model-path")
    parser.add_argument(
        "--task-trace-input",
        type=Path,
        help="JSON task trace reused verbatim instead of generating new tasks",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=(
            config.V1_CANDIDATE_CHUNK_SIZE
            if default_candidate_chunk_size is None
            else default_candidate_chunk_size
        ),
    )
    parser.add_argument("--output", default=default_output)
    parser.add_argument(
        "--audit",
        choices=("full", "periodic", "final"),
        default=default_audit_mode,
    )
    parser.add_argument("--audit-interval", type=int, default=500)
    parser.add_argument(
        "--report-mode",
        choices=("compact", "full"),
        default=default_report_mode,
    )
    parser.add_argument("--soft-tardiness-weight", type=float)
    parser.add_argument("--flexible-tardiness-weight", type=float)
    parser.add_argument(
        "--ablation-variant", choices=variant_names()
    )
    args = parser.parse_args()
    task_trace = None
    if args.task_trace_input is not None:
        trace_payload = json.loads(
            args.task_trace_input.read_text(encoding="utf-8")
        )
        trace_rows = (
            trace_payload.get("tasks")
            if isinstance(trace_payload, dict)
            else trace_payload
        )
        if not isinstance(trace_rows, list):
            parser.error("--task-trace-input must contain a JSON task list")
        task_trace = tuple(TaskSpec.from_mapping(row) for row in trace_rows)
    report = run_evaluation(
        args.policy,
        args.arrival_cutoff,
        args.seed,
        args.safety_cap,
        args.model_path,
        device=args.device,
        candidate_chunk_size=args.candidate_chunk_size,
        soft_tardiness_weight=args.soft_tardiness_weight,
        flexible_tardiness_weight=args.flexible_tardiness_weight,
        ablation_variant=args.ablation_variant,
        system_version=effective_system_version,
        audit_mode=args.audit,
        audit_interval=args.audit_interval,
        task_trace=task_trace,
        warmup_days=args.warmup_days,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(report)
    if args.report_mode == "compact":
        payload["cycle_result_count"] = len(payload["cycle_results"])
        payload["cycle_results"] = []
    output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(
        {"status": report.status.value, "output": str(output)},
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
