# 文件功能：在 V5 上替换能源核算器并重接账本，登记 V6 配置及代码指纹文件。
# 主要内容：accounting_config_view、create_v6_runtime。
# 模块上游（导入关系）：v6、v6._entrypoint、v6.tests.test_energy。
# 模块下游（项目依赖）：v1.accounting.ledger、v5.v5_runtime、v6.accounting.energy。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V6 runtime: V5 candidate pool with faster exact energy queries."""

from pathlib import Path

from v1.accounting import MetricsLedger
from v5.v5_runtime import create_v5_runtime

from .accounting import V6EnergyAccounting


# 作用：返回 V6 能源缓存和只读积分数组配置。
# 上游（直接调用）：v6._entrypoint.use_v6_runtime.v6_training_config_view、v6.v6_runtime.create_v6_runtime。
def accounting_config_view():
    return {
        "V6_ACCOUNTING_OPTIMIZATION_VERSION": V6EnergyAccounting.optimization_version,
        "V6_IDENTITY_KEYED_ENERGY_CACHE": True,
        "V6_READONLY_INTEGRAL_ARRAYS": True,
    }


# 作用：构建 V5 后替换能源核算器，同步运行时和调度器账本引用。
# 上游（直接调用）：v6.tests.test_energy.EnergyTest.test_runtime_and_nonlinear_billing_keep_existing_semantics。
# 调用上下文：由 use_v6_runtime 赋给 V1 训练与评估模块的工厂引用，随后随主流程构建运行时。
# 下游调用：create_v5_runtime、MetricsLedger、accounting_config_view、V6EnergyAccounting。
def create_v6_runtime(**kwargs):
    runtime = create_v5_runtime(**kwargs)
    old = runtime.accounting
    accounting = V6EnergyAccounting(
        runtime.time_converter, old.power_model, old.tariff_by_node,
        old.green_by_node, node_bill_rate_model=old.node_bill_rate_model,
    )
    ledger = MetricsLedger(accounting)
    runtime.accounting = accounting
    runtime.metrics_ledger = ledger
    runtime.scheduler.metrics_ledger = ledger
    runtime.config_view_overrides.update(accounting_config_view())
    runtime.evaluation_code_files += (
        Path(__file__), Path(__file__).parent / "accounting" / "energy.py",
    )
    return runtime
