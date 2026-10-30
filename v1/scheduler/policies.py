# 文件功能：实现最早可行、最低成本、最高绿电及等权基线策略和流式选择。
# 主要内容：CandidateStreamSelection、EarliestFeasiblePolicy、LowestCostPolicy、HighestGreenPolicy、EqualWeightPolicy、_require_candidates、_stable_tail、_select_stream。
# 模块上游（导入关系）：v1.calibrate_v1_objective、v1.learning.candidate_dqn、v1.scheduler、v1.scheduler.v1_scheduler、v1.v1_runtime 等。
# 模块下游（项目依赖）：v1.domain.candidates、v1.domain.models、v1.scheduler.objectives。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Deterministic v1.0 candidate policies; non-empty input always selects."""

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

from v1.domain.candidates import Candidate
from v1.domain.models import SlaType

from .objectives import ObjectiveConfig, ObjectiveScorer


# 作用：检查候选非空并返回集合。
# 上游（直接调用）：v1.scheduler.policies.EarliestFeasiblePolicy.select、v1.scheduler.policies.EqualWeightPolicy.select、v1.scheduler.policies.HighestGreenPolicy.select 等。
def _require_candidates(candidates: Iterable[Candidate]) -> tuple:
    items = tuple(candidates)
    if not items:
        raise ValueError("candidate policy requires a non-empty candidate set")
    return items


# 作用：提取用于稳定打破平局的候选身份字段。
# 上游（直接调用）：v1.scheduler.policies.EqualWeightPolicy.select、v1.scheduler.policies.EqualWeightPolicy.select_stream、v1.scheduler.policies.HighestGreenPolicy.select 等。
def _stable_tail(candidate: Candidate):
    return (
        candidate.compute_start_sim,
        candidate.target_node,
        candidate.path.path_id,
        candidate.candidate_id,
    )


@dataclass(frozen=True)
class CandidateStreamSelection:
    selected_candidate: Candidate
    earliest_candidate: Candidate
    candidate_count: int
    candidate_set_hash: str
    candidate_context: object = None


# 作用：按评分和稳定次序完成流式选择。
# 上游（直接调用）：v1.scheduler.policies.EarliestFeasiblePolicy.select_stream、v1.scheduler.policies.EqualWeightPolicy.select_stream、v1.scheduler.policies.HighestGreenPolicy.select_stream 等。
# 下游调用：_stable_tail、candidate.candidate_id.encode、CandidateStreamSelection、digest.hexdigest。
def _select_stream(candidates, key, context=None):
    selected = None
    earliest = None
    count = 0
    digest = hashlib.sha256()
    for candidate in candidates:
        count += 1
        digest.update(candidate.candidate_id.encode("utf-8"))
        digest.update(b"\0")
        if earliest is None or _stable_tail(candidate) < _stable_tail(earliest):
            earliest = candidate
        if selected is None or key(candidate) < key(selected):
            selected = candidate
    if selected is None:
        raise ValueError("candidate policy requires a non-empty candidate set")
    return CandidateStreamSelection(
        selected,
        earliest,
        count,
        digest.hexdigest(),
        context,
    )


class EarliestFeasiblePolicy:
    name = "earliest_feasible"

    # 作用：按当前策略从候选集合返回最终候选。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    # 下游调用：_require_candidates。
    def select(self, candidates: Sequence[Candidate], task=None) -> Candidate:
        return min(_require_candidates(candidates), key=_stable_tail)

    # 作用：遍历候选流，汇总选中与最早候选及决策信息。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    # 下游调用：_select_stream。
    def select_stream(self, candidates, task=None, context=None):
        return _select_stream(candidates, _stable_tail, context)


class LowestCostPolicy:
    name = "lowest_cost"

    # 作用：按当前策略从候选集合返回最终候选。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    # 下游调用：_require_candidates、_stable_tail。
    def select(self, candidates: Sequence[Candidate], task=None) -> Candidate:
        return min(
            _require_candidates(candidates),
            key=lambda item: (
                item.estimated_candidate_marginal_system_cost_yuan,
                *_stable_tail(item),
            ),
        )

    # 作用：遍历候选流，汇总选中与最早候选及决策信息。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    # 下游调用：_select_stream、_stable_tail。
    def select_stream(self, candidates, task=None, context=None):
        return _select_stream(
            candidates,
            lambda item: (
                item.estimated_candidate_marginal_system_cost_yuan,
                *_stable_tail(item),
            ),
            context,
        )


class HighestGreenPolicy:
    name = "highest_green"

    # 作用：按当前策略从候选集合返回最终候选。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    # 下游调用：_require_candidates、_stable_tail。
    def select(self, candidates: Sequence[Candidate], task=None) -> Candidate:
        return min(
            _require_candidates(candidates),
            key=lambda item: (
                -item.estimated_green_coverage,
                -item.estimated_green_absorption_delta,
                *_stable_tail(item),
            ),
        )

    # 作用：遍历候选流，汇总选中与最早候选及决策信息。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    # 下游调用：_select_stream、_stable_tail。
    def select_stream(self, candidates, task=None, context=None):
        return _select_stream(
            candidates,
            lambda item: (
                -item.estimated_green_coverage,
                -item.estimated_green_absorption_delta,
                *_stable_tail(item),
            ),
            context,
        )


class EqualWeightPolicy:
    """Fixed-scale cost/green objective; never derives scales from test candidates."""

    name = "equal_weight"

    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.v1_runtime.create_v1_runtime。
    # 调用约定：构造实例时由 Python 调用。
    # 下游调用：ObjectiveScorer、ObjectiveConfig。
    def __init__(self, objective_config=None):
        self.objective_config = objective_config or ObjectiveConfig(
            reference_marginal_cost_yuan=0.0,
            cost_scale_yuan=1.0,
            absorption_delta_scale=1.0,
        )
        self.scorer = ObjectiveScorer(self.objective_config)
        self.policy_id = "equal-weight-" + self.objective_config.policy_id

    # 作用：按当前策略从候选集合返回最终候选。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    # 下游调用：_require_candidates、_stable_tail、self.scorer.score。
    def select(self, candidates: Sequence[Candidate], task=None) -> Candidate:
        items = _require_candidates(candidates)
        sla_type = SlaType.HARD if task is None else task.sla_type
        return min(
            items,
            key=lambda item: (
                -self.scorer.score(item, sla_type).total_score,
                *_stable_tail(item),
            ),
        )

    # 作用：遍历候选流，汇总选中与最早候选及决策信息。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    # 下游调用：_select_stream、_stable_tail、self.scorer.score。
    def select_stream(self, candidates, task=None, context=None):
        sla_type = SlaType.HARD if task is None else task.sla_type
        return _select_stream(
            candidates,
            lambda item: (
                -self.scorer.score(item, sla_type).total_score,
                *_stable_tail(item),
            ),
            context,
        )
