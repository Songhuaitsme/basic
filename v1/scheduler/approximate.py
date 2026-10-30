# 文件功能：压缩候选集合，为显式启用的近似搜索提供候选子集。
# 主要内容：CandidateCompressionResult、compress_candidates。
# 模块上游（导入关系）：v1.scheduler、v1.scheduler.candidate_generator、tests.v1.test_v1_acceptance。
# 模块下游（项目依赖）：v1.domain.candidates、v1.scheduler.objectives。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Explicit experimental candidate compression, isolated from formal complete mode."""

from dataclasses import dataclass, replace
import hashlib
from typing import Callable, Iterable, Optional, Tuple

from v1.domain.candidates import Candidate, CandidateMode

from .objectives import pareto_frontier


@dataclass(frozen=True)
class CandidateCompressionResult:
    candidates: Tuple[Candidate, ...]
    original_count: int
    retained_count: int
    omitted_ratio: float
    utility_regret: Optional[float]


# 作用：按近似搜索限制保留代表性候选。
# 上游（直接调用）：v1.scheduler.candidate_generator.CandidateGenerator.generate_approximate。
# 下游调用：pareto_frontier、CandidateCompressionResult、(item.candidate_id + '|approximate-v1').encode。
def compress_candidates(
    candidates: Iterable[Candidate],
    max_candidates: int,
    utility_evaluator: Optional[Callable[[Candidate], float]] = None,
) -> CandidateCompressionResult:
    items = tuple(candidates)
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer")
    if not items:
        return CandidateCompressionResult((), 0, 0, 0.0, None)
    protected = {
        min(items, key=lambda item: (item.compute_start_sim, item.candidate_id)),
        min(items, key=lambda item: (
            item.estimated_candidate_marginal_system_cost_yuan,
            item.candidate_id,
        )),
        max(items, key=lambda item: (
            item.estimated_green_coverage + item.estimated_green_absorption_delta,
            item.candidate_id,
        )),
        max(items, key=lambda item: (item.capacity_margin, item.candidate_id)),
        *pareto_frontier(items),
    }
    target = max(max_candidates, len(protected))
    selected = list(sorted(protected, key=lambda item: item.candidate_id))
    remaining = [item for item in items if item not in protected]
    if remaining and len(selected) < target:
        slots = target - len(selected)
        if slots >= len(remaining):
            selected.extend(remaining)
        else:
            indices = {
                round(index * (len(remaining) - 1) / max(1, slots - 1))
                for index in range(slots)
            }
            selected.extend(remaining[index] for index in sorted(indices))
    selected = selected[:target]
    converted = tuple(sorted((
        replace(
            item,
            candidate_id="candidate-" + hashlib.sha256(
                (item.candidate_id + "|approximate-v1").encode("utf-8")
            ).hexdigest(),
            candidate_mode=CandidateMode.APPROXIMATE,
        )
        for item in selected
    ), key=lambda item: (
        item.compute_start_sim,
        item.target_node,
        item.path.path_id,
        item.candidate_id,
    )))
    regret = None
    if utility_evaluator is not None:
        best_full = max(utility_evaluator(item) for item in items)
        best_retained = max(utility_evaluator(item) for item in selected)
        regret = float(best_full - best_retained)
    return CandidateCompressionResult(
        converted,
        len(items),
        len(converted),
        1.0 - len(converted) / len(items),
        regret,
    )
