# 文件功能：在 V3 上装配等待净收益门控、策略包装器和带诊断的 V4 调度器。
# 主要内容：create_v4_runtime。
# 模块上游（导入关系）：v4、v4._entrypoint、v5.v5_runtime、v4.tests.test_wait_gate。
# 模块下游（项目依赖）：shared.config、v1.scheduler.objectives、v3.v3_runtime、v4.gating、v4.policy、v4.scheduler。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V4 runtime: V3 candidate/accounting engine plus net-benefit wait gate."""

from __future__ import annotations

from pathlib import Path

from shared import config
from v1.scheduler.objectives import ObjectiveConfig
from v3.v3_runtime import create_v3_runtime

from .gating import NetBenefitWaitGate, WaitGateConfig, default_objective_config
from .policy import NetBenefitGatePolicy
from .scheduler import V4Scheduler


# 作用：构建 V3 后装配等待净收益门控和带诊断的调度器。
# 上游（直接调用）：v5.v5_runtime.create_v5_runtime。
# 调用上下文：由 V4 适配上下文注入 V1 入口；V5 工厂继续调用本函数叠加候选池优化。
# 下游调用：default_objective_config、create_v3_runtime、NetBenefitWaitGate、NetBenefitGatePolicy、WaitGateConfig 等。
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
