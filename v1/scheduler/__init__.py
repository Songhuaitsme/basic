# 文件功能：声明 v1.scheduler 包，汇集对外导出的类型与组件。
# 主要内容：导出或引用 JitSchedule、TransmissionDuration、TransmissionModel、build_path_spec、CalendarAllocation、CommitResult、FeasibilityResult 等。
# 使用入口：包导入或外部调用方使用。
# 模块下游（项目依赖）：v1.scheduler.transmission、v1.scheduler.resource_calendar、v1.scheduler.path_provider、v1.scheduler.candidate_generator、v1.scheduler.queue_manager、v1.scheduler.reservation_manager 等。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""v1.0 scheduler infrastructure."""

from .transmission import (
    JitSchedule,
    TransmissionDuration,
    TransmissionModel,
    build_path_spec,
)
from .resource_calendar import (
    CalendarAllocation,
    CommitResult,
    FeasibilityResult,
    ReleaseResult,
    ReservationCalendar,
    ReservationSnapshot,
)
from .path_provider import StaticPathProvider
from .candidate_generator import CandidateGenerator, complete_time_grid
from .queue_manager import TaskQueueManager, queue_order_key
from .reservation_manager import (
    CommitDecisionResult,
    CommitDecisionStatus,
    ReservationManager,
)
from .policies import (
    CandidateStreamSelection,
    EarliestFeasiblePolicy,
    EqualWeightPolicy,
    HighestGreenPolicy,
    LowestCostPolicy,
)
from .v1_scheduler import SchedulingCycleResult, SchedulingDecision, V1Scheduler
from .objectives import (
    ObjectiveBreakdown,
    ObjectiveConfig,
    ObjectiveScorer,
    pareto_frontier,
)
from .approximate import CandidateCompressionResult, compress_candidates


__all__ = [
    "JitSchedule",
    "CalendarAllocation",
    "CandidateGenerator",
    "CandidateStreamSelection",
    "CandidateCompressionResult",
    "CommitDecisionResult",
    "CommitDecisionStatus",
    "CommitResult",
    "EarliestFeasiblePolicy",
    "EqualWeightPolicy",
    "FeasibilityResult",
    "HighestGreenPolicy",
    "LowestCostPolicy",
    "ObjectiveBreakdown",
    "ObjectiveConfig",
    "ObjectiveScorer",
    "ReleaseResult",
    "ReservationCalendar",
    "ReservationManager",
    "ReservationSnapshot",
    "SchedulingCycleResult",
    "SchedulingDecision",
    "StaticPathProvider",
    "TaskQueueManager",
    "V1Scheduler",
    "complete_time_grid",
    "compress_candidates",
    "queue_order_key",
    "pareto_frontier",
    "TransmissionDuration",
    "TransmissionModel",
    "build_path_spec",
]
