# 文件功能：构建 V1 运行时后替换 V3 候选生成器和扫描线能源核算器，并重接账本。
# 主要内容：create_v3_runtime。
# 模块上游（导入关系）：v3、v3._entrypoint、v4.v4_runtime。
# 模块下游（项目依赖）：v1.accounting.ledger、v1.v1_runtime、v3.accounting.energy、v3.scheduler.candidate_generator。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Composition adapter for the isolated V3 candidate-generation runtime."""

from v1.accounting import MetricsLedger
from v1.v1_runtime import V1Runtime, create_v1_runtime

from .accounting import SweepLineEnergyAccounting
from .scheduler import V3CandidateGenerator


# 作用：构建 V1 后替换扫描线核算器、V3 候选生成器及关联账本。
# 上游（直接调用）：v4.v4_runtime.create_v4_runtime。
# 调用上下文：由 V3 适配上下文注入 V1 入口；V4 工厂继续调用本函数叠加门控。
# 下游调用：create_v1_runtime、SweepLineEnergyAccounting、MetricsLedger、V3CandidateGenerator。
def create_v3_runtime(**kwargs) -> V1Runtime:
    """Build a V1-compatible runtime with V3 candidate internals.

    Constructing the frozen runtime first keeps every public domain, policy,
    feature, and scheduler contract unchanged.  The untouched initial
    accounting/generator objects are then replaced before any scheduling work
    can occur.
    """

    runtime = create_v1_runtime(**kwargs)
    old_accounting = runtime.accounting
    accounting = SweepLineEnergyAccounting(
        runtime.time_converter,
        old_accounting.power_model,
        old_accounting.tariff_by_node,
        old_accounting.green_by_node,
        node_bill_rate_model=old_accounting.node_bill_rate_model,
    )

    old_generator = runtime.scheduler.candidate_generator
    generator = V3CandidateGenerator(
        old_generator.compute_nodes,
        old_generator.scheduling_cycle_sim,
        old_generator.path_provider,
        old_generator.transmission_model,
        old_generator.calendar,
        old_generator.time_tolerance,
        candidate_mode=old_generator.candidate_mode.value,
        active_wait_enabled=old_generator.active_wait_enabled,
        pool_max_by_sla=old_generator.pool_max_by_sla,
        pool_node_limit_by_sla=old_generator.pool_node_limit_by_sla,
        pool_time_samples_by_sla=old_generator.pool_time_samples_by_sla,
    )

    ledger = MetricsLedger(accounting)
    runtime.accounting = accounting
    runtime.metrics_ledger = ledger
    runtime.scheduler.candidate_generator = generator
    runtime.scheduler.metrics_ledger = ledger
    return runtime
