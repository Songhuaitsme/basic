# 文件功能：定义目标权重与尺度，计算成本、绿电和 SLA 评分及 Pareto 前沿。
# 主要内容：ObjectiveBreakdown、ObjectiveConfig、ObjectiveScorer、_clip、pareto_frontier。
# 模块上游（导入关系）：v1.evaluate_v1、v1.scheduler、v1.scheduler.approximate、v1.scheduler.policies、v1.train_v1 等。
# 模块下游（项目依赖）：v1.domain.candidates、v1.domain.models、v1.domain.units。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Fixed-scale v1.0 multi-objective scoring and Pareto diagnostics."""

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Tuple

from v1.domain.candidates import Candidate
from v1.domain.models import SlaType
from v1.domain.units import finite_number, non_negative_finite, positive_finite


# 作用：将输入限制在指定边界内。
# 上游（直接调用）：v1.scheduler.objectives.ObjectiveScorer.score。
def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class ObjectiveBreakdown:
    cost_score: float
    green_coverage_score: float
    green_absorption_score: float
    green_score: float
    balance_score: float
    preferred_start_tardiness_penalty: float
    total_score: float


@dataclass(frozen=True)
class ObjectiveConfig:
    reference_marginal_cost_yuan: float
    cost_scale_yuan: float
    absorption_delta_scale: float
    cost_weight: float = 0.5
    green_weight: float = 0.5
    balance_weight: float = 0.1
    soft_tardiness_weight: float = 0.0
    flexible_tardiness_weight: float = 0.0

    # 作用：在数据类构造后校验字段或建立派生数据。
    # 调用约定：由 dataclass 生成的构造流程调用。
    # 下游调用：finite_number、positive_finite、non_negative_finite。
    def __post_init__(self):
        object.__setattr__(
            self,
            "reference_marginal_cost_yuan",
            finite_number(
                "reference_marginal_cost_yuan",
                self.reference_marginal_cost_yuan,
            ),
        )
        object.__setattr__(
            self,
            "cost_scale_yuan",
            positive_finite("cost_scale_yuan", self.cost_scale_yuan),
        )
        object.__setattr__(
            self,
            "absorption_delta_scale",
            positive_finite(
                "absorption_delta_scale",
                self.absorption_delta_scale,
            ),
        )
        for field in (
            "cost_weight",
            "green_weight",
            "balance_weight",
            "soft_tardiness_weight",
            "flexible_tardiness_weight",
        ):
            object.__setattr__(
                self,
                field,
                non_negative_finite(field, getattr(self, field)),
            )
        if self.cost_weight + self.green_weight <= 0.0:
            raise ValueError("cost_weight + green_weight must be positive")

    # 作用：生成包含目标配置身份的策略标识。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    @property
    def policy_id(self) -> str:
        payload = json.dumps(
            self.__dict__,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "objective-" + hashlib.sha256(payload).hexdigest()[:16]


class ObjectiveScorer:
    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.scheduler.policies.EqualWeightPolicy.__init__、v1.train_v1.V1TrainingLoop.__init__、v4.gating.NetBenefitWaitGate.__init__。
    # 调用约定：构造实例时由 Python 调用。
    def __init__(self, config: ObjectiveConfig):
        self.config = config

    # 作用：计算当前策略或目标配置下的评分。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：SlaType.parse、_clip、ObjectiveBreakdown。
    def score(self, candidate: Candidate, sla_type: SlaType) -> ObjectiveBreakdown:
        if not isinstance(sla_type, SlaType):
            sla_type = SlaType.parse(sla_type)
        cost_score = _clip(
            (
                self.config.reference_marginal_cost_yuan
                - candidate.estimated_candidate_marginal_system_cost_yuan
            ) / self.config.cost_scale_yuan,
            -1.0,
            1.0,
        )
        coverage = _clip(candidate.estimated_green_coverage, 0.0, 1.0)
        absorption = (
            _clip(
                candidate.estimated_green_absorption_delta
                / self.config.absorption_delta_scale,
                0.0,
                1.0,
            )
            if candidate.estimated_green_opportunity
            else 0.0
        )
        green_score = 0.5 * coverage + 0.5 * absorption
        balance = _clip(candidate.capacity_margin, 0.0, 1.0)
        tardiness_weight = {
            SlaType.HARD: 0.0,
            SlaType.SOFT: self.config.soft_tardiness_weight,
            SlaType.FLEXIBLE: self.config.flexible_tardiness_weight,
        }[sla_type]
        tardiness_penalty = (
            tardiness_weight * candidate.preferred_start_tardiness_ratio
            if candidate.preferred_start_tardiness_applicable
            else 0.0
        )
        total = (
            self.config.cost_weight * cost_score
            + self.config.green_weight * green_score
            + self.config.balance_weight * balance
            - tardiness_penalty
        )
        return ObjectiveBreakdown(
            cost_score,
            coverage,
            absorption,
            green_score,
            balance,
            tardiness_penalty,
            total,
        )


# 作用：按目标维度筛选未被支配的候选。
# 上游（直接调用）：v1.scheduler.approximate.compress_candidates。
def pareto_frontier(candidates: Iterable[Candidate]) -> Tuple[Candidate, ...]:
    """Cost-minimizing, green-score-maximizing non-dominated candidates."""

    items = tuple(candidates)
    frontier = []
    for candidate in items:
        candidate_green = (
            candidate.estimated_green_coverage
            + candidate.estimated_green_absorption_delta
        )
        dominated = False
        for other in items:
            if other is candidate:
                continue
            other_green = (
                other.estimated_green_coverage
                + other.estimated_green_absorption_delta
            )
            no_worse = (
                other.estimated_candidate_marginal_system_cost_yuan
                <= candidate.estimated_candidate_marginal_system_cost_yuan
                and other_green >= candidate_green
            )
            strictly_better = (
                other.estimated_candidate_marginal_system_cost_yuan
                < candidate.estimated_candidate_marginal_system_cost_yuan
                or other_green > candidate_green
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda item: item.candidate_id))
