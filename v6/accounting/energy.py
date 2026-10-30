# 文件功能：以快照和预测对象身份缓存积分索引，复用只读数组加速精确批量积分。
# 主要内容：V6CandidateIntegralIndex、V6EnergyAccounting。
# 模块上游（导入关系）：v6.accounting、v6.v6_runtime、v6.tests.test_energy。
# 模块下游（项目依赖）：v1.accounting.energy、v1.domain.models、v1.domain.units、v3.accounting.energy。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Identity-keyed accounting caches and reusable integral array views."""

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from v1.accounting.energy import CandidateAccountingMetrics, _CandidateIntegralIndex
from v1.domain.models import MetricValue
from v1.domain.units import finite_number
from v3.accounting import SweepLineEnergyAccounting


@dataclass(frozen=True)
class V6CandidateIntegralIndex(_CandidateIntegralIndex):
    _boundaries_array: object = field(init=False, repr=False, compare=False)
    _rates_array: object = field(init=False, repr=False, compare=False)
    _prefixes_array: object = field(init=False, repr=False, compare=False)

    # 作用：把积分边界、速率和前缀预转为只读 float64 数组，供批量查询复用。
    # 调用约定：由 dataclass 生成的构造流程调用。
    # 下游调用：array.setflags。
    def __post_init__(self):
        for name, values in (
            ("_boundaries_array", self.boundaries),
            ("_rates_array", self.rates),
            ("_prefixes_array", self.prefixes),
        ):
            array = np.asarray(values, dtype=np.float64)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    # 作用：复用只读数组批量积分，沿用既有边界容差和浮点运算次序。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def _cumulative_many(self, times_sim):
        # Identical arithmetic and boundary rules to the frozen implementation;
        # only tuple-to-array conversions move out of this repeated operation.
        times = np.asarray(times_sim, dtype=np.float64)
        if times.ndim != 1:
            raise ValueError("candidate times must be one-dimensional")
        result = np.empty((times.size, len(self.prefixes)), dtype=np.float64)
        if times.size == 0:
            return result
        boundaries = self._boundaries_array
        rates = self._rates_array
        prefixes = self._prefixes_array
        low = times <= boundaries[0] + 1e-12
        high = times >= boundaries[-1] - 1e-12
        middle = ~(low | high)
        result[low] = 0.0
        result[high] = prefixes[:, -1]
        if np.any(middle):
            middle_times = times[middle]
            indices = np.searchsorted(boundaries, middle_times, side="right") - 1
            elapsed_hours = (middle_times - boundaries[indices]) * self.hours_per_sim
            result[middle] = (
                prefixes[:, indices].T
                + rates[:, indices].T * elapsed_hours[:, None]
            )
        return result


class V6EnergyAccounting(SweepLineEnergyAccounting):
    optimization_version = "snapshot-identity-readonly-integrals-v1"

    # 作用：沿用 V3 扫描线构建次序，封装为带只读数组的 V6 积分索引。
    # 上游（直接调用）：v6.accounting.energy.V6EnergyAccounting._candidate_integral_index。
    # 下游调用：super()._build_candidate_integral_index、V6CandidateIntegralIndex。
    def _build_candidate_integral_index(self, *args, **kwargs):
        # Keep the original sweep-line and prefix accumulation order unchanged.
        index = super()._build_candidate_integral_index(*args, **kwargs)
        return V6CandidateIntegralIndex(
            index.boundaries, index.rates, index.prefixes, index.hours_per_sim
        )

    # 作用：以快照及预测对象身份查询有界缓存，持有源对象避免对象 ID 复用。
    # 上游（直接调用）：v6.accounting.energy.V6EnergyAccounting.candidate_metric_evaluator.lookup。
    # 下游调用：finite_number、self._build_candidate_integral_index、self._forecasts、self.power_model.task_power_mw。
    def _candidate_integral_index(
        self, task, target_node, reservation_snapshot, first_start_sim,
        required_end_sim=None,
    ):
        tariff, green = self._forecasts(target_node)
        forecast_start = max(
            tariff.segments[0].interval_sim.start_sim,
            green.segments[0].interval_sim.start_sim,
        )
        forecast_end = min(
            tariff.segments[-1].interval_sim.end_sim,
            green.segments[-1].interval_sim.end_sim,
        )
        local_start = finite_number("first_start_sim", first_start_sim)
        local_end = (
            task.absolute_latest_start_sim + task.execution_duration_sim
            if required_end_sim is None
            else finite_number("required_end_sim", required_end_sim)
        )
        domain_start = max(forecast_start, task.arrival_time_sim)
        domain_end = min(
            forecast_end, task.absolute_latest_start_sim + task.execution_duration_sim
        )
        if local_start < domain_start - 1e-12 or local_end > domain_end + 1e-12:
            domain_start = max(forecast_start, min(domain_start, local_start))
            domain_end = min(forecast_end, max(domain_end, local_end))
        task_power = self.power_model.task_power_mw(task.cpu_demand)
        key = (
            id(reservation_snapshot), target_node, id(tariff), id(green),
            task_power, domain_start, domain_end,
        )
        cached = self._candidate_index_cache.get(key)
        if cached is not None:
            self._candidate_index_cache.move_to_end(key)
            return cached[3]
        index = self._build_candidate_integral_index(
            task, target_node, reservation_snapshot, domain_start, domain_end
        )
        # Retain the immutable source objects with the index so Python cannot
        # reuse an id while that entry is live. Different forecasts/snapshots
        # never share an index, even when their version labels coincide.
        self._candidate_index_cache[key] = (reservation_snapshot, tariff, green, index)
        self._candidate_index_cache.move_to_end(key)
        while len(self._candidate_index_cache) > self._candidate_index_cache_capacity:
            self._candidate_index_cache.popitem(last=False)
        return index

    # 作用：创建绑定快照的指标回调，并附加批量评估入口。
    # 调用上下文：V6 工厂替换核算器后由调度器使用；非线性计费仍委托上层回调，其他情况用局部缓存闭包。
    # 下游调用：super().candidate_metric_evaluator。
    def candidate_metric_evaluator(self, reservation_snapshot):
        if self.node_bill_rate_model is not None:
            # Nonlinear subsidies/taxes keep their existing scalar semantics.
            return super().candidate_metric_evaluator(reservation_snapshot)
        indices = OrderedDict()

        # 作用：查询局部积分缓存，未命中时向核算器申请索引。
        # 上游（直接调用）：v6.accounting.energy.V6EnergyAccounting.candidate_metric_evaluator.evaluate、v6.accounting.energy.V6EnergyAccounting.candidate_metric_evaluator.evaluate_batch。
        # 下游调用：self._candidate_integral_index、self._forecasts、cached[3].covers。
        def lookup(task, node, snapshot, start, end):
            tariff, green = self._forecasts(node)
            key = (task, node, id(snapshot), id(tariff), id(green))
            cached = indices.get(key)
            if cached is not None and cached[3].covers(start, end):
                indices.move_to_end(key)
                return cached[3]
            index = self._candidate_integral_index(task, node, snapshot, start, end)
            indices[key] = (snapshot, tariff, green, index)
            indices.move_to_end(key)
            while len(indices) > self._candidate_index_cache_capacity:
                indices.popitem(last=False)
            return index

        # 作用：用局部积分缓存评估单候选，索引未覆盖时回退逐候选核算。
        # 调用上下文：作为 candidate_metric_evaluator 的返回值被候选生成器调用。
        # 下游调用：lookup、MetricValue.valid、MetricValue.not_applicable、index.covers、self.evaluate_candidate 等。
        def evaluate(**kwargs):
            task = kwargs["task"]
            node = kwargs.get("target_node", kwargs["path"].target_node)
            snapshot = kwargs.get("reservation_snapshot", reservation_snapshot)
            start, end = kwargs["compute_start_sim"], kwargs["compute_end_sim"]
            index = lookup(task, node, snapshot, start, end)
            if not index.covers(start, end):
                return self.evaluate_candidate(
                    task=task, target_node=node, compute_start_sim=start,
                    compute_end_sim=end, reservation_snapshot=snapshot,
                ).as_candidate_metrics()
            (
                task_energy, direct_cost, marginal_cost, attributed_green,
                marginal_green, green_supply_energy,
            ) = index.integrate(start, end)
            if green_supply_energy > 0.0:
                absorption = MetricValue.valid(
                    marginal_green / green_supply_energy,
                    numerator=marginal_green, denominator=green_supply_energy,
                )
                opportunity = True
            else:
                absorption = MetricValue.not_applicable("zero green supply over interval")
                opportunity = False
            return CandidateAccountingMetrics(
                task_energy, direct_cost, marginal_cost, attributed_green,
                attributed_green / task_energy, marginal_green, absorption, opportunity,
            ).as_candidate_metrics()

        # 作用：批量评估区间，返回对齐的成本和绿电指标数组。
        # 调用约定：作为指标回调的 evaluate_batch 属性供候选批处理调用。
        # 调用上下文：绑定为返回回调的 evaluate_batch 属性，供批量候选指标计算调用。
        # 下游调用：lookup、index.covers、self.evaluate_candidate、index.integrate_many。
        def evaluate_batch(**kwargs):
            starts = np.asarray(kwargs["compute_start_sim"], dtype=np.float64)
            ends = np.asarray(kwargs["compute_end_sim"], dtype=np.float64)
            if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape:
                raise ValueError("candidate interval arrays must be one-dimensional and aligned")
            task = kwargs["task"]
            node = kwargs.get("target_node", kwargs["path"].target_node)
            snapshot = kwargs.get("reservation_snapshot", reservation_snapshot)
            if starts.size == 0:
                return {
                    "system_cost_yuan": np.empty(0, dtype=np.float64),
                    "green_coverage": np.empty(0, dtype=np.float64),
                    "marginal_green_energy_mwh": np.empty(0, dtype=np.float64),
                    "green_absorption_delta": np.empty(0, dtype=np.float64),
                    "green_opportunity": np.empty(0, dtype=np.bool_),
                }
            first, last = float(np.min(starts)), float(np.max(ends))
            index = lookup(task, node, snapshot, first, last)
            if not index.covers(first, last):
                rows = [self.evaluate_candidate(
                    task=task, target_node=node, compute_start_sim=float(start),
                    compute_end_sim=float(end), reservation_snapshot=snapshot,
                ).as_candidate_metrics() for start, end in zip(starts, ends)]
                return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            values = index.integrate_many(starts, ends)
            task_energy = values[:, 0]
            marginal_cost = values[:, 2]
            attributed_green = values[:, 3]
            marginal_green = values[:, 4]
            green_supply_energy = values[:, 5]
            opportunity = green_supply_energy > 0.0
            absorption = np.divide(
                marginal_green, green_supply_energy,
                out=np.zeros_like(marginal_green), where=opportunity,
            )
            return {
                "system_cost_yuan": marginal_cost,
                "green_coverage": attributed_green / task_energy,
                "marginal_green_energy_mwh": marginal_green,
                "green_absorption_delta": absorption,
                "green_opportunity": opportunity,
            }

        evaluate.evaluate_batch = evaluate_batch
        return evaluate
