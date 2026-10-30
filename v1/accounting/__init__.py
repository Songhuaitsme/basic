# 文件功能：声明 v1.accounting 包，汇集对外导出的类型与组件。
# 主要内容：导出或引用 AccountingReport、CandidateAccountingMetrics、ExogenousEnergyAccounting、LinearPowerModel、TaskAccountingRecord、ForecastCoverageError、ForecastSegment 等。
# 使用入口：包导入或外部调用方使用。
# 模块下游（项目依赖）：v1.accounting.energy、v1.accounting.forecast、v1.accounting.ledger。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""v1.0 physical energy, cost, green attribution, and ledger contracts."""

from .energy import (
    AccountingReport,
    CandidateAccountingMetrics,
    ExogenousEnergyAccounting,
    LinearPowerModel,
    TaskAccountingRecord,
)
from .forecast import (
    ForecastCoverageError,
    ForecastSegment,
    PiecewiseConstantForecast,
)
from .ledger import MetricsLedger

__all__ = [
    "AccountingReport",
    "CandidateAccountingMetrics",
    "ExogenousEnergyAccounting",
    "ForecastCoverageError",
    "ForecastSegment",
    "LinearPowerModel",
    "MetricsLedger",
    "PiecewiseConstantForecast",
    "TaskAccountingRecord",
]
