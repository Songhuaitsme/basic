# 文件功能：声明 v1.evaluation_v1 包，汇集对外导出的类型与组件。
# 主要内容：导出或引用 ActiveWaitMetrics、SeedMetrics、SlaMetrics、TaskOutcome、build_seed_metrics、linear_percentile、ratio_metric 等。
# 使用入口：包导入或外部调用方使用。
# 模块下游（项目依赖）：v1.evaluation_v1.metrics、v1.evaluation_v1.runner、v1.evaluation_v1.diagnostics、v1.evaluation_v1.statistics、v1.evaluation_v1.schema。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""v1.0 formal evaluation and metric aggregation."""

from .metrics import (
    ActiveWaitMetrics,
    SeedMetrics,
    SlaMetrics,
    TaskOutcome,
    build_seed_metrics,
    linear_percentile,
    ratio_metric,
    summarize_active_wait,
    summarize_sla,
)
from .runner import (
    EvaluationMetadata,
    EvaluationReport,
    EvaluationRunner,
    EvaluationStatus,
    TaskEvaluationRecord,
)
from .diagnostics import (
    DIAGNOSTIC_SCHEMA_VERSION,
    METRIC_DEFINITIONS,
    build_evaluation_diagnostics,
)
from .statistics import (
    BootstrapSummary,
    LoadMetrics,
    PairedStatus,
    PairedSummary,
    UtilizationInterval,
    paired_bootstrap,
    paired_sample_size,
    paired_t_summary,
    relative_change,
    summarize_load,
)
from .schema import FormalSchemaError, to_canonical_json, to_primitive

__all__ = [
    "ActiveWaitMetrics",
    "EvaluationMetadata",
    "EvaluationReport",
    "EvaluationRunner",
    "EvaluationStatus",
    "FormalSchemaError",
    "BootstrapSummary",
    "LoadMetrics",
    "PairedStatus",
    "PairedSummary",
    "SeedMetrics",
    "SlaMetrics",
    "TaskOutcome",
    "TaskEvaluationRecord",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "METRIC_DEFINITIONS",
    "UtilizationInterval",
    "build_seed_metrics",
    "build_evaluation_diagnostics",
    "linear_percentile",
    "ratio_metric",
    "summarize_active_wait",
    "paired_bootstrap",
    "paired_sample_size",
    "paired_t_summary",
    "relative_change",
    "summarize_load",
    "to_canonical_json",
    "to_primitive",
    "summarize_sla",
]
