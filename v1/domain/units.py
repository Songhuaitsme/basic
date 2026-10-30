# 文件功能：统一数值校验，以及工作量、电价、数据量和仿真时间的单位换算。
# 主要内容：UnitValidationError、TimeConverter、DataUnitConverter、TariffConverter、finite_number、positive_finite、non_negative_finite、validate_scheduling_grid 等。
# 模块上游（导入关系）：shared.pricing_manager、shared.task_manager、shared.time_conversion_calculator、shared.topology_manager、v1.accounting.energy 等。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Explicit unit conversion primitives for the v1.0 domain model.

The legacy scheduler stores most physical values as bare floats.  This module
is the single conversion boundary used by new v1.0 code so invalid values are
rejected instead of being silently replaced with epsilon defaults.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional


SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 24.0 * SECONDS_PER_HOUR
MEGABITS_PER_DECIMAL_MEGABYTE = 8.0
YUAN_PER_MWH_PER_YUAN_PER_KWH = 1000.0


class UnitValidationError(ValueError):
    """Raised when a conversion receives an invalid physical value."""

    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.domain.units.TimeConverter.from_traffic_day_duration、v1.domain.units.finite_number、v1.domain.units.non_negative_finite 等。
    # 调用约定：构造实例时由 Python 调用。
    # 下游调用：super().__init__。
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


# 作用：验证输入为有限数值并转为浮点值。
# 上游（直接调用）：v1.accounting.energy.ExogenousEnergyAccounting._bill_rate、v1.accounting.energy.ExogenousEnergyAccounting._candidate_integral_index、v1.accounting.forecast.PiecewiseConstantForecast.__init__ 等。
# 下游调用：UnitValidationError。
def finite_number(field: str, value: Real) -> float:
    """Return *value* as float after strict finite-number validation."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise UnitValidationError(field, "must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise UnitValidationError(field, "must be finite")
    return converted


# 作用：验证输入为严格正的有限数值。
# 上游（直接调用）：shared.time_conversion_calculator.calculate_time_conversion、shared.time_conversion_calculator.sim_hour_at_cycle、shared.topology_manager.TopologyManager.calculate_transmission_delay 等。
# 下游调用：finite_number、UnitValidationError。
def positive_finite(field: str, value: Real) -> float:
    converted = finite_number(field, value)
    if converted <= 0.0:
        raise UnitValidationError(field, "must be greater than 0")
    return converted


# 作用：验证输入为非负有限数值。
# 上游（直接调用）：shared.time_conversion_calculator.calculate_time_conversion、v1.accounting.forecast.PiecewiseConstantForecast.__init__、v1.domain.models._field_number 等。
# 下游调用：finite_number、UnitValidationError。
def non_negative_finite(field: str, value: Real) -> float:
    converted = finite_number(field, value)
    if converted < 0.0:
        raise UnitValidationError(field, "must be greater than or equal to 0")
    return converted


@dataclass(frozen=True)
class TimeConverter:
    """Convert simulation time to physical seconds and hours."""

    seconds_per_sim_unit: float

    # 作用：在数据类构造后校验字段或建立派生数据。
    # 调用约定：由 dataclass 生成的构造流程调用。
    # 下游调用：positive_finite。
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seconds_per_sim_unit",
            positive_finite("seconds_per_sim_unit", self.seconds_per_sim_unit),
        )

    # 作用：按一个流量日的仿真长度建立时间换算器。
    # 上游（直接调用）：shared.task_manager.TaskManager.__init__、shared.time_conversion_calculator.calculate_time_conversion、shared.topology_manager.TopologyManager.__init__ 等。
    # 下游调用：positive_finite、UnitValidationError。
    @classmethod
    def from_traffic_day_duration(
        cls,
        traffic_day_duration_in_sim: Real,
        declared_seconds_per_sim_unit: Optional[Real] = None,
    ) -> "TimeConverter":
        day_duration = positive_finite(
            "traffic_day_duration_in_sim",
            traffic_day_duration_in_sim,
        )
        derived = SECONDS_PER_DAY / day_duration
        if declared_seconds_per_sim_unit is not None:
            declared = positive_finite(
                "declared_seconds_per_sim_unit",
                declared_seconds_per_sim_unit,
            )
            if not math.isclose(declared, derived, rel_tol=1e-12, abs_tol=1e-12):
                raise UnitValidationError(
                    "declared_seconds_per_sim_unit",
                    f"must equal 86400 / traffic_day_duration_in_sim ({derived})",
                )
        return cls(derived)

    # 作用：把物理秒换算为仿真时间。
    # 上游（直接调用）：v1.domain.units.TimeConverter.hours_to_sim。
    # 下游调用：finite_number。
    def seconds_to_sim(self, seconds: Real) -> float:
        return finite_number("seconds", seconds) / self.seconds_per_sim_unit

    # 作用：把仿真时间换算为物理秒。
    # 上游（直接调用）：v1.domain.units.TimeConverter.scheduling_cycle_seconds、v1.domain.units.TimeConverter.sim_to_hours。
    # 下游调用：finite_number。
    def sim_to_seconds(self, sim_time: Real) -> float:
        return finite_number("sim_time", sim_time) * self.seconds_per_sim_unit

    # 作用：把仿真时间换算为物理小时。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self.sim_to_seconds。
    def sim_to_hours(self, sim_time: Real) -> float:
        return self.sim_to_seconds(sim_time) / SECONDS_PER_HOUR

    # 作用：把物理小时换算为仿真时间。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self.seconds_to_sim、finite_number。
    def hours_to_sim(self, hours: Real) -> float:
        return self.seconds_to_sim(finite_number("hours", hours) * SECONDS_PER_HOUR)

    # 作用：返回调度周期对应的物理秒数。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：positive_finite、self.sim_to_seconds。
    def scheduling_cycle_seconds(self, scheduling_cycle_sim: Real) -> float:
        cycle = positive_finite("scheduling_cycle", scheduling_cycle_sim)
        return self.sim_to_seconds(cycle)


# 作用：验证调度周期与时间网格约束。
# 上游（直接调用）：shared.task_manager.TaskManager.__init__、v1.v1_runtime.create_v1_runtime。
# 下游调用：positive_finite、UnitValidationError。
def validate_scheduling_grid(
    scheduling_cycle: Real,
    global_time_step_duration: Optional[Real] = None,
) -> float:
    """Validate the single scheduling-grid source required by v1.0."""

    cycle = positive_finite("scheduling_cycle", scheduling_cycle)
    if global_time_step_duration is not None:
        global_step = positive_finite(
            "global_time_step_duration",
            global_time_step_duration,
        )
        if not math.isclose(cycle, global_step, rel_tol=1e-12, abs_tol=1e-12):
            raise UnitValidationError(
                "global_time_step_duration",
                "must equal scheduling_cycle",
            )
    return cycle


class DataUnitConverter:
    """Decimal data-unit conversions used by the transmission contract."""

    # 作用：把十进制 MB 换算为 Mbit。
    # 上游（直接调用）：shared.topology_manager.TopologyManager.calculate_transmission_delay、v1.scheduler.transmission.TransmissionModel.duration。
    # 下游调用：non_negative_finite。
    @staticmethod
    def decimal_mb_to_megabits(data_mb: Real) -> float:
        return (
            non_negative_finite("data_size_mb", data_mb)
            * MEGABITS_PER_DECIMAL_MEGABYTE
        )


class TariffConverter:
    """Electricity tariff conversions; negative external prices remain valid."""

    # 作用：把元每千瓦时换算为元每兆瓦时。
    # 上游（直接调用）：shared.pricing_manager.PricingManager.get_external_tariff_yuan_per_mwh。
    # 下游调用：finite_number。
    @staticmethod
    def yuan_per_kwh_to_yuan_per_mwh(value: Real) -> float:
        return finite_number("yuan_per_kwh", value) * YUAN_PER_MWH_PER_YUAN_PER_KWH


# 作用：计算 CPU 需求与仿真执行时长的乘积。
# 上游（直接调用）：tests.v1.test_domain_foundations.DomainFoundationTest.test_unit_005_cpu_work_units。
# 下游调用：positive_finite。
def cpu_work_sim_units(cpu_demand: Real, execution_duration_sim: Real) -> float:
    cpu = positive_finite("cpu_demand", cpu_demand)
    duration = positive_finite("execution_duration_sim", execution_duration_sim)
    return cpu * duration


# 作用：把 CPU 工作量换算为 CPU 小时。
# 上游（直接调用）：tests.v1.test_domain_foundations.DomainFoundationTest.test_unit_005_cpu_work_units。
# 下游调用：positive_finite、converter.sim_to_hours。
def cpu_work_cpu_hours(
    cpu_demand: Real,
    execution_duration_sim: Real,
    converter: TimeConverter,
) -> float:
    return positive_finite("cpu_demand", cpu_demand) * converter.sim_to_hours(
        positive_finite("execution_duration_sim", execution_duration_sim)
    )
