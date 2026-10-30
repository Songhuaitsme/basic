# 文件功能：计算配对 t 统计、Bootstrap 区间、相对变化及负载摘要。
# 主要内容：PairedStatus、PairedSummary、BootstrapSummary、UtilizationInterval、LoadMetrics、paired_t_summary、_invalid_pair、paired_bootstrap 等。
# 模块上游（导入关系）：v1.evaluation_v1、v1.evaluation_v1.diagnostics、tests.v1.test_evaluation_v1、tests.v1.test_v1_acceptance。
# 模块下游（项目依赖）：v1.domain.models、v1.domain.units。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Seed-paired inference and event-duration-weighted load metrics."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Mapping, Tuple

import numpy as np
from scipy import stats

from v1.domain.models import MetricStatus, MetricValue
from v1.domain.units import finite_number, non_negative_finite, positive_finite


class PairedStatus(str, Enum):
    VALID = "VALID"
    INVALID_PAIR = "INVALID_PAIR"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class PairedSummary:
    status: PairedStatus
    seeds: Tuple[int, ...]
    differences: Tuple[float, ...]
    mean_difference: MetricValue
    sample_standard_deviation: MetricValue
    ci_low: MetricValue
    ci_high: MetricValue
    reason: str = ""


# 作用：根据配对样本计算差值均值及 t 统计。
# 上游（直接调用）：tests.v1.test_evaluation_v1.EvaluationV1Test.test_pair_invalidity_is_symmetric_and_bootstrap_is_reproducible、tests.v1.test_evaluation_v1.EvaluationV1Test.test_seed_pairing_uses_seed_ids_and_sample_ddof_one。
# 下游调用：finite_number、_invalid_pair、MetricValue.not_applicable、MetricValue.valid、PairedSummary 等。
def paired_t_summary(
    baseline_by_seed: Mapping[int, MetricValue],
    treatment_by_seed: Mapping[int, MetricValue],
    confidence_level: float = 0.95,
) -> PairedSummary:
    confidence = finite_number("confidence_level", confidence_level)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be in (0,1)")
    baseline_seeds = set(baseline_by_seed)
    treatment_seeds = set(treatment_by_seed)
    if baseline_seeds != treatment_seeds:
        missing = sorted(baseline_seeds ^ treatment_seeds)
        return _invalid_pair(tuple(sorted(baseline_seeds | treatment_seeds)), f"unpaired seeds: {missing}")
    seeds = tuple(sorted(baseline_seeds))
    differences = []
    for seed in seeds:
        baseline = baseline_by_seed[seed]
        treatment = treatment_by_seed[seed]
        if baseline.status is not MetricStatus.VALID or treatment.status is not MetricStatus.VALID:
            return _invalid_pair(seeds, f"seed {seed} contains non-VALID metric")
        differences.append(treatment.value - baseline.value)
    if not differences:
        na = MetricValue.not_applicable("zero paired seeds")
        return PairedSummary(PairedStatus.NOT_APPLICABLE, (), (), na, na, na, na, "zero paired seeds")
    mean = float(np.mean(differences))
    mean_metric = MetricValue.valid(mean)
    if len(differences) < 2:
        na = MetricValue.not_applicable("paired t interval requires at least two seeds")
        return PairedSummary(
            PairedStatus.NOT_APPLICABLE,
            seeds,
            tuple(differences),
            mean_metric,
            na,
            na,
            na,
            "paired t interval requires at least two seeds",
        )
    sample_sd = float(np.std(differences, ddof=1))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(differences) - 1))
    half_width = critical * sample_sd / math.sqrt(len(differences))
    return PairedSummary(
        PairedStatus.VALID,
        seeds,
        tuple(differences),
        mean_metric,
        MetricValue.valid(sample_sd),
        MetricValue.valid(mean - half_width),
        MetricValue.valid(mean + half_width),
    )


# 作用：生成样本不足或配对无效的统计结果。
# 上游（直接调用）：v1.evaluation_v1.statistics.paired_t_summary。
# 下游调用：MetricValue.invalid、PairedSummary。
def _invalid_pair(seeds, reason):
    invalid = MetricValue.invalid(reason)
    return PairedSummary(
        PairedStatus.INVALID_PAIR,
        tuple(seeds),
        (),
        invalid,
        invalid,
        invalid,
        invalid,
        reason,
    )


@dataclass(frozen=True)
class BootstrapSummary:
    mean_difference: float
    ci_low: float
    ci_high: float
    resample_count: int
    random_seed: int


# 作用：对配对差值重采样并估计置信区间。
# 上游（直接调用）：tests.v1.test_evaluation_v1.EvaluationV1Test.test_pair_invalidity_is_symmetric_and_bootstrap_is_reproducible。
# 下游调用：finite_number、rng.integers、values[indices].mean、BootstrapSummary、values.mean。
def paired_bootstrap(
    differences: Iterable[float],
    *,
    resample_count: int,
    random_seed: int,
    confidence_level: float = 0.95,
) -> BootstrapSummary:
    values = np.asarray([finite_number("paired_difference", value) for value in differences])
    if values.size == 0:
        raise ValueError("paired bootstrap requires at least one seed difference")
    if isinstance(resample_count, bool) or not isinstance(resample_count, int) or resample_count <= 0:
        raise ValueError("resample_count must be a positive integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
    confidence = finite_number("confidence_level", confidence_level)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be in (0,1)")
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, values.size, size=(resample_count, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, (alpha, 1.0 - alpha), method="linear")
    return BootstrapSummary(
        float(values.mean()),
        float(low),
        float(high),
        resample_count,
        random_seed,
    )


# 作用：计算相对基线变化并处理不可计算情况。
# 上游（直接调用）：tests.v1.test_evaluation_v1.EvaluationV1Test.test_pair_invalidity_is_symmetric_and_bootstrap_is_reproducible、tests.v1.test_v1_acceptance.V1AcceptanceContractTest.test_not_applicable_wait_relative_change_and_finite_json。
# 下游调用：finite_number、MetricValue.not_applicable、MetricValue.valid。
def relative_change(treatment: float, baseline: float) -> MetricValue:
    treatment_value = finite_number("treatment", treatment)
    baseline_value = finite_number("baseline", baseline)
    if baseline_value <= 0.0:
        return MetricValue.not_applicable("non-positive baseline denominator")
    return MetricValue.valid(
        (treatment_value - baseline_value) / baseline_value,
        numerator=treatment_value - baseline_value,
        denominator=baseline_value,
    )


# 作用：统计可配对有效样本数。
# 上游（直接调用）：tests.v1.test_v1_acceptance.V1AcceptanceContractTest.test_formal_quality_gate_and_power_rules。
# 下游调用：non_negative_finite、positive_finite、finite_number、stats.norm.ppf。
def paired_sample_size(
    pilot_difference_sd: float,
    minimum_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    sd = non_negative_finite("pilot_difference_sd", pilot_difference_sd)
    effect = positive_finite("minimum_effect", abs(minimum_effect))
    alpha_value = finite_number("alpha", alpha)
    power_value = finite_number("power", power)
    if not 0.0 < alpha_value < 1.0 or not 0.0 < power_value < 1.0:
        raise ValueError("alpha and power must be in (0,1)")
    if sd == 0.0:
        return 2
    estimate = (
        (stats.norm.ppf(1.0 - alpha_value / 2.0) + stats.norm.ppf(power_value))
        * sd
        / effect
    ) ** 2
    return max(2, int(math.ceil(estimate)))


@dataclass(frozen=True)
class UtilizationInterval:
    duration_seconds: float
    node_utilizations: Tuple[float, ...]

    # 作用：在数据类构造后校验字段或建立派生数据。
    # 调用约定：由 dataclass 生成的构造流程调用。
    # 下游调用：positive_finite、non_negative_finite。
    def __post_init__(self):
        object.__setattr__(
            self,
            "duration_seconds",
            positive_finite("duration_seconds", self.duration_seconds),
        )
        values = tuple(
            non_negative_finite("node_utilization", value)
            for value in self.node_utilizations
        )
        if not values:
            raise ValueError("node_utilizations cannot be empty")
        object.__setattr__(self, "node_utilizations", values)


@dataclass(frozen=True)
class LoadMetrics:
    time_node_mean_utilization: float
    weighted_p95_utilization: float
    maximum_utilization: float
    time_weighted_node_cv: float
    hotspot_time_ratio: float
    physical_overcapacity_time_ratio: float


# 作用：汇总任务负载及容量统计。
# 上游（直接调用）：v1.evaluation_v1.diagnostics._resource_records。
# 下游调用：values.mean、values.std、values.max、weighted_values.sort、LoadMetrics。
def summarize_load(intervals: Iterable[UtilizationInterval]) -> LoadMetrics:
    items = tuple(intervals)
    if not items:
        raise ValueError("load summary requires event intervals")
    total_duration = sum(item.duration_seconds for item in items)
    weighted_values = []
    mean_accumulator = 0.0
    cv_accumulator = 0.0
    hotspot_duration = 0.0
    over_duration = 0.0
    maximum = 0.0
    for item in items:
        values = np.asarray(item.node_utilizations, dtype=float)
        interval_mean = float(values.mean())
        interval_cv = 0.0 if interval_mean == 0.0 else float(values.std(ddof=0) / interval_mean)
        mean_accumulator += interval_mean * item.duration_seconds
        cv_accumulator += interval_cv * item.duration_seconds
        maximum = max(maximum, float(values.max()))
        for value in values:
            weighted_values.append((float(value), item.duration_seconds / len(values)))
        if float(values.max()) > 0.85:
            hotspot_duration += item.duration_seconds
        if float(values.max()) > 1.0:
            over_duration += item.duration_seconds
    weighted_values.sort(key=lambda pair: pair[0])
    threshold = 0.95 * sum(weight for _, weight in weighted_values)
    cumulative = 0.0
    p95 = weighted_values[-1][0]
    for value, weight in weighted_values:
        cumulative += weight
        if cumulative >= threshold:
            p95 = value
            break
    return LoadMetrics(
        mean_accumulator / total_duration,
        p95,
        maximum,
        cv_accumulator / total_duration,
        hotspot_duration / total_duration,
        over_duration / total_duration,
    )
