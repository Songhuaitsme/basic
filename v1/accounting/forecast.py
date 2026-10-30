# 文件功能：定义带版本的分段电价和绿电预测，校验覆盖范围并查询分段边界。
# 主要内容：ForecastCoverageError、ForecastSegment、PiecewiseConstantForecast。
# 模块上游（导入关系）：v1.accounting、v1.accounting.energy、v1.v1_runtime、tests.v1.test_accounting_v1、tests.v1.test_evaluation_v1 等。
# 模块下游（项目依赖）：v1.domain.reservations、v1.domain.units。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Strict piecewise-constant physical forecasts for v1.0 accounting."""

from dataclasses import dataclass
from bisect import bisect_right
from typing import Iterable, Tuple

from v1.domain.reservations import TimeInterval
from v1.domain.units import finite_number, non_negative_finite


class ForecastCoverageError(ValueError):
    pass


@dataclass(frozen=True)
class ForecastSegment:
    interval_sim: TimeInterval
    value: float


class PiecewiseConstantForecast:
    # 作用：控制属性写入，维护不可变性或包装器转发约定。
    # 调用约定：由 Python 属性赋值协议调用。
    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("physical forecast is immutable")
        object.__setattr__(self, name, value)

    # 作用：保存依赖与配置，建立实例初始状态。
    # 调用约定：构造实例时由 Python 调用。
    # 下游调用：non_negative_finite、finite_number、ForecastSegment、normalized.sort。
    def __init__(
        self,
        segments: Iterable[ForecastSegment],
        *,
        value_name: str,
        non_negative: bool,
        version: str = "forecast-1.0",
    ):
        if not isinstance(version, str) or not version:
            raise ValueError("forecast version must be non-empty")
        self.value_name = str(value_name)
        self.version = version
        normalized = []
        for segment in segments:
            value = (
                non_negative_finite(self.value_name, segment.value)
                if non_negative
                else finite_number(self.value_name, segment.value)
            )
            normalized.append(ForecastSegment(segment.interval_sim, value))
        normalized.sort(key=lambda item: item.interval_sim.start_sim)
        if not normalized:
            raise ValueError("forecast must contain at least one segment")
        for previous, current in zip(normalized[:-1], normalized[1:]):
            if current.interval_sim.start_sim < previous.interval_sim.end_sim:
                raise ValueError("forecast segments must not overlap")
        self._segments: Tuple[ForecastSegment, ...] = tuple(normalized)
        self._starts: Tuple[float, ...] = tuple(
            segment.interval_sim.start_sim for segment in normalized
        )
        self._frozen = True

    # 作用：构造以元每兆瓦时表示的分段电价预测。
    # 上游（直接调用）：v1.v1_runtime.create_v1_runtime、v1.v1_runtime.extend_v1_runtime_forecasts。
    @classmethod
    def tariff_yuan_per_mwh(cls, segments, version="tariff-1.0"):
        return cls(
            segments,
            value_name="tariff_yuan_per_mwh",
            non_negative=False,
            version=version,
        )

    # 作用：构造以兆瓦表示的分段绿电预测。
    # 上游（直接调用）：v1.v1_runtime.create_v1_runtime、v1.v1_runtime.extend_v1_runtime_forecasts。
    @classmethod
    def green_power_mw(cls, segments, version="green-1.0"):
        return cls(
            segments,
            value_name="green_power_mw",
            non_negative=True,
            version=version,
        )

    # 作用：返回不可变预测分段序列。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    @property
    def segments(self) -> Tuple[ForecastSegment, ...]:
        return self._segments

    # 作用：查找指定时刻的预测值。
    # 上游（直接调用）：v1.accounting.forecast.PiecewiseConstantForecast.boundaries。
    # 下游调用：finite_number、segment.interval_sim.contains、ForecastCoverageError。
    def value_at(self, time_sim: float) -> float:
        time_value = finite_number("time_sim", time_sim)
        index = bisect_right(self._starts, time_value) - 1
        if index >= 0:
            segment = self._segments[index]
            if segment.interval_sim.contains(time_value):
                return segment.value
        raise ForecastCoverageError(
            f"{self.value_name} does not cover simulation time {time_value}"
        )

    # 作用：返回区间内部的预测变化边界。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self.value_at、segment.interval_sim.overlaps。
    def boundaries(self, interval: TimeInterval) -> Tuple[float, ...]:
        boundaries = {interval.start_sim, interval.end_sim}
        index = max(0, bisect_right(self._starts, interval.start_sim) - 1)
        for segment in self._segments[index:]:
            if segment.interval_sim.start_sim >= interval.end_sim:
                break
            if segment.interval_sim.overlaps(interval):
                boundaries.add(max(interval.start_sim, segment.interval_sim.start_sim))
                boundaries.add(min(interval.end_sim, segment.interval_sim.end_sim))
        ordered = tuple(sorted(boundaries))
        for left, right in zip(ordered[:-1], ordered[1:]):
            if right > left:
                self.value_at(left + (right - left) / 2.0)
        return ordered
