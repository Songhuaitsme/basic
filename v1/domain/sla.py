# 文件功能：推导 SLA 开始时间约束，计算可行性、偏好迟到及迟到特征。
# 主要内容：SlaLimits、SlaPolicy。
# 模块上游（导入关系）：v1.domain、tests.v1.test_domain_foundations、tests.v1.test_v1_acceptance。
# 模块下游（项目依赖）：v1.domain.models、v1.domain.units。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Pure v1.0 SLA derivation, feasibility, and tardiness evaluation."""

from dataclasses import dataclass
from typing import Optional, Tuple

from .models import MetricStatus, MetricValue, SlaType, TaskSpec
from .units import UnitValidationError, finite_number


@dataclass(frozen=True)
class SlaLimits:
    preferred_start_limit_sim: Optional[float]
    latest_start_limit_sim: float
    absolute_preferred_start_sim: Optional[float]
    absolute_latest_start_sim: float


class SlaPolicy:
    """SLA rules independent from candidate scores and reward weights."""

    # 作用：按 SLA 及任务约束推导偏好和最晚开始时限。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：SlaLimits。
    @staticmethod
    def derive_limits(task_spec: TaskSpec) -> SlaLimits:
        return SlaLimits(
            preferred_start_limit_sim=task_spec.preferred_start_limit_sim,
            latest_start_limit_sim=task_spec.latest_start_limit_sim,
            absolute_preferred_start_sim=task_spec.absolute_preferred_start_sim,
            absolute_latest_start_sim=task_spec.absolute_latest_start_sim,
        )

    # 作用：取得输入中的计算开始时刻。
    # 上游（直接调用）：v1.domain.sla.SlaPolicy.is_start_feasible、v1.domain.sla.SlaPolicy.preferred_start_tardiness。
    # 下游调用：finite_number。
    @staticmethod
    def _compute_start(compute_start_sim: float) -> float:
        try:
            return finite_number("compute_start_sim", compute_start_sim)
        except UnitValidationError as exc:
            raise ValueError(str(exc))

    # 作用：检查开始时刻是否满足 SLA 最晚时限。
    # 上游（直接调用）：tests.v1.test_domain_foundations.DomainFoundationTest.test_sla_001_hard_boundary_is_feasible、tests.v1.test_domain_foundations.DomainFoundationTest.test_sla_002_hard_after_boundary_is_infeasible、tests.v1.test_domain_foundations.DomainFoundationTest.test_sla_005_soft_latest_boundary_has_full_tardiness 等。
    # 下游调用：cls._compute_start。
    @classmethod
    def is_start_feasible(cls, task_spec: TaskSpec, compute_start_sim: float) -> bool:
        compute_start = cls._compute_start(compute_start_sim)
        return (
            compute_start >= task_spec.arrival_time_sim
            and compute_start <= task_spec.absolute_latest_start_sim
        )

    # 作用：计算超过偏好开始时刻的迟到量。
    # 上游（直接调用）：v1.domain.sla.SlaPolicy.tardiness_model_feature。
    # 下游调用：cls._compute_start、MetricValue.invalid、MetricValue.not_applicable、MetricValue.valid。
    @classmethod
    def preferred_start_tardiness(
        cls,
        task_spec: TaskSpec,
        compute_start_sim: float,
    ) -> MetricValue:
        compute_start = cls._compute_start(compute_start_sim)
        if compute_start < task_spec.arrival_time_sim:
            return MetricValue.invalid("compute_start_sim is earlier than arrival_time_sim")
        if task_spec.sla_type is SlaType.HARD:
            return MetricValue.not_applicable(
                "Hard tasks do not have a preferred start limit"
            )

        start_delay = compute_start - task_spec.arrival_time_sim
        preferred = task_spec.preferred_start_limit_sim
        latest = task_spec.latest_start_limit_sim
        denominator = latest - preferred
        ratio = min(1.0, max(0.0, (start_delay - preferred) / denominator))
        return MetricValue.valid(
            ratio,
            numerator=start_delay - preferred,
            denominator=denominator,
        )

    # 作用：把迟到量转换为归一化模型特征。
    # 上游（直接调用）：tests.v1.test_domain_foundations.DomainFoundationTest.test_hard_tardiness_is_not_applicable_and_model_feature_is_flagged。
    # 下游调用：cls.preferred_start_tardiness。
    @classmethod
    def tardiness_model_feature(
        cls,
        task_spec: TaskSpec,
        compute_start_sim: float,
    ) -> Tuple[float, bool]:
        metric = cls.preferred_start_tardiness(task_spec, compute_start_sim)
        if metric.status is MetricStatus.NOT_APPLICABLE:
            return 0.0, False
        if metric.value is None:
            raise ValueError(metric.reason or "invalid tardiness metric")
        return metric.value, True
