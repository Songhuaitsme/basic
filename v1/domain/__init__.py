# 文件功能：声明 v1.domain 包，汇集对外导出的类型与组件。
# 主要内容：导出或引用 FLEXIBLE_LATEST_START_MULTIPLIER、SOFT_LATEST_START_MULTIPLIER、TASK_SCHEMA_VERSION、MetricStatus、MetricValue、SlaType、TaskRuntime 等。
# 使用入口：包导入或外部调用方使用。
# 模块下游（项目依赖）：v1.domain.models、v1.domain.sla、v1.domain.reservations、v1.domain.candidates、v1.domain.units。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Public v1.0 domain foundation API."""

from .models import (
    FLEXIBLE_LATEST_START_MULTIPLIER,
    SOFT_LATEST_START_MULTIPLIER,
    TASK_SCHEMA_VERSION,
    MetricStatus,
    MetricValue,
    SlaType,
    TaskRuntime,
    TaskSpec,
    TaskValidationResult,
    TaskState,
    TaskValidationError,
    migrate_legacy_task,
    validate_task_mapping,
    to_legacy_task_dict,
)
from .sla import SlaLimits, SlaPolicy
from .reservations import (
    CommitStatus,
    PathSpec,
    ReleaseStatus,
    Reservation,
    ReservationRequest,
    ReservationValidationError,
    TimeInterval,
    canonical_edge,
)
from .candidates import (
    CANDIDATE_SCHEMA_VERSION,
    Candidate,
    CandidateGenerationStatus,
    CandidateMode,
    CandidateSetResult,
)
from .units import (
    DataUnitConverter,
    TariffConverter,
    TimeConverter,
    UnitValidationError,
    cpu_work_cpu_hours,
    cpu_work_sim_units,
    validate_scheduling_grid,
)


__all__ = [
    "DataUnitConverter",
    "CommitStatus",
    "CANDIDATE_SCHEMA_VERSION",
    "Candidate",
    "CandidateGenerationStatus",
    "CandidateMode",
    "CandidateSetResult",
    "FLEXIBLE_LATEST_START_MULTIPLIER",
    "MetricStatus",
    "MetricValue",
    "PathSpec",
    "ReleaseStatus",
    "Reservation",
    "ReservationRequest",
    "ReservationValidationError",
    "SOFT_LATEST_START_MULTIPLIER",
    "SlaLimits",
    "SlaPolicy",
    "SlaType",
    "TASK_SCHEMA_VERSION",
    "TariffConverter",
    "TaskRuntime",
    "TaskSpec",
    "TaskValidationResult",
    "TaskState",
    "TaskValidationError",
    "TimeConverter",
    "TimeInterval",
    "UnitValidationError",
    "cpu_work_cpu_hours",
    "cpu_work_sim_units",
    "migrate_legacy_task",
    "validate_task_mapping",
    "to_legacy_task_dict",
    "validate_scheduling_grid",
    "canonical_edge",
]
