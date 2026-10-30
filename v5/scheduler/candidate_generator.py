# 文件功能：复用 V1 候选规则，通过决策内缓存、静态容量过滤及向量化 Pareto 加速候选池。
# 主要内容：V5CandidateGenerator。
# 模块上游（导入关系）：v5.scheduler、v5.v5_runtime、v5.tests.test_candidate_generator、v6.tests.test_energy。
# 模块下游（项目依赖）：v3.scheduler.candidate_generator。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Preserve the V4 heuristic pool while avoiding redundant candidate work."""

import hashlib

import numpy as np

from v3.scheduler.candidate_generator import V3CandidateGenerator
from v1.candidate_profiling import candidate_count, candidate_scope, candidate_stage


class V5CandidateGenerator(V3CandidateGenerator):
    optimization_version = "decision-cache-exact-pool-v1"

    # 作用：先按静态 CPU 容量过滤节点，再按上层规则构造分层候选池。
    # 调用上下文：由继承的 prepare_stream 按候选模式分派，V5/V6 工厂将此实现装入调度器。
    # 下游调用：super().prepare_layered_pool_stream。
    @candidate_stage("candidate_prepare")
    def prepare_layered_pool_stream(self, *args, **kwargs):
        # Both passes use the same context object. Never retain results across
        # decisions, resource snapshots, forecast extensions, or exceptions.
        previous = getattr(self, "_generation_cache", None)
        cache = {
            "context": None,
            "records": {},
            "candidates": {},
            "capacities": {},
            "stats": {
                "record_cache_hits": 0,
                "candidate_cache_hits": 0,
                "static_cpu_rejections": 0,
            },
        }
        self._generation_cache = cache
        try:
            return super().prepare_layered_pool_stream(*args, **kwargs)
        finally:
            self.last_generation_stats = dict(cache["stats"])
            self._generation_cache = previous

    # 作用：取得当前决策上下文对应的缓存。
    # 上游（直接调用）：v5.scheduler.candidate_generator.V5CandidateGenerator._record_at_tick、v5.scheduler.candidate_generator.V5CandidateGenerator.materialize_context_candidates。
    def _cache_for_context(self, context):
        cache = getattr(self, "_generation_cache", None)
        if cache is not None and cache["context"] is None:
            cache["context"] = context
        return cache if cache is not None and cache["context"] is context else None

    # 作用：复用决策内缓存，避免同刻度候选重复计算。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self._cache_for_context、super()._record_at_tick、self.calendar.node_capacity。
    @candidate_stage("record_cache_and_assembly")
    def _record_at_tick(self, context, target_node, path, duration, tick):
        cache = self._cache_for_context(context)
        if cache is None:
            return super()._record_at_tick(context, target_node, path, duration, tick)
        key = (target_node, path.path_id, tick)
        if key in cache["records"]:
            cache["stats"]["record_cache_hits"] += 1
            candidate_count(self.profiler, "record_cache_hits")
            return cache["records"][key]
        candidate_count(self.profiler, "record_cache_misses")
        if target_node not in cache["capacities"]:
            cache["capacities"][target_node] = self.calendar.node_capacity(target_node)
        capacity = cache["capacities"][target_node]
        # The empty-resource fast path and general CPU check use different
        # tolerances. Reject only when neither can admit this task.
        if capacity is None or context.task.cpu_demand > capacity + max(
            self.time_tolerance, 1e-12
        ):
            record = None
            cache["stats"]["static_cpu_rejections"] += 1
            candidate_count(self.profiler, "static_cpu_rejections")
        else:
            record = super()._record_at_tick(
                context, target_node, path, duration, tick
            )
        cache["records"][key] = record
        return record

    # 作用：批量创建完整候选并附加评估指标。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self._cache_for_context、super().materialize_context_candidates。
    @candidate_stage("materialization_prepare")
    def materialize_context_candidates(self, context, records, metric_evaluator=None):
        cache = self._cache_for_context(context)
        if cache is None:
            return super().materialize_context_candidates(
                context, records, metric_evaluator
            )
        items = tuple(records)
        candidate_count(self.profiler, "materialization_requested", len(items))
        # Include evaluator identity: reusable objects must have identical
        # accounting as well as identical task, time, and reservation context.
        candidates = cache["candidates"]
        evaluator_key = id(metric_evaluator)
        missing = {}
        for record in items:
            key = (evaluator_key, record.candidate_id)
            if key in candidates:
                cache["stats"]["candidate_cache_hits"] += 1
                candidate_count(self.profiler, "candidate_cache_hits")
            else:
                missing[key] = record
        candidate_count(self.profiler, "materialization_missing", len(missing))
        realized = super().materialize_context_candidates(
            context, tuple(missing.values()), metric_evaluator
        )
        candidates.update(zip(missing, realized))
        return tuple(candidates[(evaluator_key, item.candidate_id)] for item in items)

    # 作用：向量化筛选 Pareto 候选，保留基线排序语义。
    # 上游（直接调用）：v5.scheduler.candidate_generator.V5CandidateGenerator._select_layered_pool。
    @staticmethod
    def _pareto_candidates(candidates):
        items = tuple(candidates)
        if not items:
            return ()
        values = np.asarray([
            (
                item.estimated_candidate_marginal_system_cost_yuan,
                -(item.estimated_green_coverage + item.estimated_green_absorption_delta),
                item.compute_end_sim,
                item.preferred_start_tardiness_ratio,
                max(item.projected_node_utilization, item.projected_path_peak_utilization),
            )
            for item in items
        ], dtype=np.float64)
        retained = []
        # Bound temporary memory while preserving the exact <= / < dominance
        # rules and input order. Equal vectors never dominate one another.
        for first in range(0, len(items), 64):
            current = values[first:first + 64]
            no_worse = np.all(values[:, None, :] <= current[None, :, :], axis=2)
            better = np.any(values[:, None, :] < current[None, :, :], axis=2)
            dominated = np.any(no_worse & better, axis=0)
            retained.extend(
                items[first + offset]
                for offset, rejected in enumerate(dominated)
                if not rejected
            )
        return tuple(retained)

    # 作用：综合时间、目标及节点覆盖选出受限候选池。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：normalized、self._pareto_candidates、item.candidate_id.encode。
    @candidate_stage("layered_selection", phase="selection")
    def _select_layered_pool(self, candidates, limit):
        items = tuple(candidates)
        candidate_count(getattr(self, "profiler", None), "selection_input", len(items))
        if len(items) <= limit:
            selected = items
        else:
            costs = [item.estimated_candidate_marginal_system_cost_yuan for item in items]
            greens = [
                item.estimated_green_coverage + item.estimated_green_absorption_delta
                for item in items
            ]
            loads = [
                max(item.projected_node_utilization, item.projected_path_peak_utilization)
                for item in items
            ]
            bounds = tuple((min(values), max(values)) for values in (costs, greens, loads))

            # 作用：将当前比较维度归一化。
            # 上游（直接调用）：v5.scheduler.candidate_generator.V5CandidateGenerator._select_layered_pool。
            def normalized(value, bound):
                low, high = bound
                return 0.0 if high <= low else (value - low) / (high - low)

            balanced = {
                item.candidate_id: (
                    normalized(costs[index], bounds[0])
                    - normalized(greens[index], bounds[1])
                    + item.preferred_start_tardiness_ratio
                    + normalized(loads[index], bounds[2])
                )
                for index, item in enumerate(items)
            }
            stable = lambda item: (
                item.compute_start_sim, item.target_node, item.path.path_id, item.candidate_id
            )
            layers = (
                (0.20, lambda item: (item.compute_end_sim, *stable(item))),
                (0.20, lambda item: (item.estimated_candidate_marginal_system_cost_yuan, *stable(item))),
                (0.20, lambda item: (-(item.estimated_green_coverage + item.estimated_green_absorption_delta), *stable(item))),
                (0.10, lambda item: (item.preferred_start_tardiness_ratio, *stable(item))),
                (0.10, lambda item: (max(item.projected_node_utilization, item.projected_path_peak_utilization), *stable(item))),
                (0.15, lambda item: (balanced[item.candidate_id], *stable(item))),
                (0.05, lambda item: hashlib.sha256(item.candidate_id.encode("utf-8")).digest()),
            )
            selected_by_id = {}
            for ratio, key in layers:
                quota = max(1, int(round(limit * ratio)))
                for item in sorted(items, key=key)[:quota]:
                    selected_by_id[item.candidate_id] = item
            for item in sorted(items, key=lambda value: (balanced[value.candidate_id], *stable(value))):
                if len(selected_by_id) >= limit:
                    break
                selected_by_id[item.candidate_id] = item
            selected = tuple(selected_by_id.values())
        candidate_count(getattr(self, "profiler", None), "pareto_input", len(selected))
        with candidate_scope(getattr(self, "profiler", None), "pareto"):
            frontier = self._pareto_candidates(selected)
        result = frontier or (min(items, key=lambda item: item.compute_end_sim),)
        candidate_count(getattr(self, "profiler", None), "selection_output", len(result))
        return result
