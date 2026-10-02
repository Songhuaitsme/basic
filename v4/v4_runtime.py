"""V4 runtime: V3 candidate/accounting engine plus net-benefit wait gate."""

from __future__ import annotations

from pathlib import Path

from shared import config
from v1.scheduler.objectives import ObjectiveConfig
from v3.v3_runtime import create_v3_runtime

from .gating import NetBenefitWaitGate, WaitGateConfig, default_objective_config
from .policy import NetBenefitGatePolicy
from .scheduler import V4Scheduler


def create_v4_runtime(*, wait_gate_config=None, **kwargs):
    objective = kwargs.get("objective_config") or default_objective_config()
    if not isinstance(objective, ObjectiveConfig):
        raise TypeError("objective_config must be ObjectiveConfig")
    gate_config = wait_gate_config or WaitGateConfig()

    runtime = create_v3_runtime(**kwargs)
    old_scheduler = runtime.scheduler
    gate = NetBenefitWaitGate(objective, gate_config)
    policy = NetBenefitGatePolicy(old_scheduler.policy, gate)
    scheduler = V4Scheduler(
        runtime.calendar,
        old_scheduler.candidate_generator,
        config.MAX_QUEUE_LENGTH,
        config.MAX_TASKS_PER_CYCLE,
        config.MAX_COMMIT_ATTEMPTS_PER_DECISION,
        policy=policy,
        metrics_ledger=runtime.metrics_ledger,
    )
    runtime.scheduler = scheduler
    runtime.v4_wait_gate_config = gate_config
    runtime.config_view_overrides = {
        "V4_WAIT_GATE_ENABLED": True,
        "V4_WAIT_GATE_TIME_SCALE_SIM": gate_config.time_scale_sim,
        "V4_WAIT_GATE_PENALTY_WEIGHT_BY_SLA": {
            key.value: float(value)
            for key, value in gate_config.penalty_weight_by_sla.items()
        },
        "V4_WAIT_GATE_MIN_GAIN_BY_SLA": {
            key.value: float(value)
            for key, value in gate_config.min_gain_by_sla.items()
        },
        "V4_WAIT_GATE_TOLERANCE": gate_config.tolerance,
    }
    runtime.evaluation_code_files = (
        Path(__file__),
        Path(__file__).with_name("gating.py"),
        Path(__file__).with_name("policy.py"),
        Path(__file__).with_name("scheduler.py"),
    )
    return runtime
