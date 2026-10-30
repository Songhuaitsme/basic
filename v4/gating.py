# 文件功能：按目标收益减等待惩罚判断是否接受主动等待，未通过时回退最早候选。
# 主要内容：WaitGateConfig、WaitGateDiagnostic、NetBenefitWaitGate、default_objective_config。
# 模块上游（导入关系）：v4、v4._entrypoint、v4.v4_runtime、v4.tests.test_wait_gate、v5.tests.test_candidate_generator 等。
# 模块下游（项目依赖）：shared.config、v1.domain.models、v1.scheduler.objectives。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V4 post-selection net-benefit gate for active waiting."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Mapping

from shared import config
from v1.domain.models import SlaType
from v1.scheduler.objectives import ObjectiveConfig, ObjectiveScorer


@dataclass(frozen=True)
class WaitGateConfig:
    """Frozen V4 gate parameters in normalized objective units."""

    time_scale_sim: float = config.TRAFFIC_DAY_DURATION_IN_SIM
    penalty_weight_by_sla: Mapping[SlaType, float] = field(
        default_factory=lambda: {
            SlaType.HARD: 1.0,
            SlaType.SOFT: config.V1_SOFT_TARDINESS_WEIGHT,
            SlaType.FLEXIBLE: config.V1_FLEXIBLE_TARDINESS_WEIGHT,
        }
    )
    min_gain_by_sla: Mapping[SlaType, float] = field(
        default_factory=lambda: {
            SlaType.HARD: 0.0,
            SlaType.SOFT: 0.0,
            SlaType.FLEXIBLE: 0.0,
        }
    )
    tolerance: float = 1e-12

    # 作用：在数据类构造后校验字段或建立派生数据。
    # 调用约定：由 dataclass 生成的构造流程调用。
    def __post_init__(self):
        if not math.isfinite(self.time_scale_sim) or self.time_scale_sim <= 0.0:
            raise ValueError("time_scale_sim must be positive and finite")
        if not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        for name, values in (
            ("penalty_weight_by_sla", self.penalty_weight_by_sla),
            ("min_gain_by_sla", self.min_gain_by_sla),
        ):
            missing = set(SlaType) - set(values)
            if missing:
                raise ValueError(f"{name} is missing SLA values: {sorted(item.value for item in missing)}")
            for sla_type, value in values.items():
                numeric = float(value)
                if not math.isfinite(numeric) or numeric < 0.0:
                    raise ValueError(f"{name}[{sla_type.value}] must be finite and non-negative")


@dataclass(frozen=True)
class WaitGateDiagnostic:
    task_id: str
    proposed_candidate_id: str
    final_candidate_id: str
    proposed_active_wait_sim: float
    objective_gain_before_wait_penalty: float
    wait_penalty: float
    estimated_net_wait_gain: float
    minimum_required_gain: float
    applied: bool
    passed: bool
    reason: str


class NetBenefitWaitGate:
    """Accept delayed selections only when normalized net gain clears a floor."""

    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v4.v4_runtime.create_v4_runtime。
    # 调用约定：构造实例时由 Python 调用。
    # 下游调用：ObjectiveScorer、WaitGateConfig。
    def __init__(self, objective_config: ObjectiveConfig, gate_config=None):
        self.scorer = ObjectiveScorer(objective_config)
        self.config = gate_config or WaitGateConfig()

    # 作用：比较选中与最早候选的目标收益，扣除 SLA 加权等待惩罚，返回门控诊断。
    # 上游（直接调用）：v4.gating.NetBenefitWaitGate.apply。
    # 下游调用：WaitGateDiagnostic、self.scorer.score。
    def evaluate(self, selected, earliest, task=None) -> WaitGateDiagnostic:
        sla_type = SlaType.HARD if task is None else task.sla_type
        task_id = "" if task is None else task.task_id
        wait = float(selected.compute_start_sim - earliest.compute_start_sim)
        if wait <= self.config.tolerance:
            return WaitGateDiagnostic(
                task_id,
                selected.candidate_id,
                selected.candidate_id,
                max(0.0, wait),
                0.0,
                0.0,
                0.0,
                float(self.config.min_gain_by_sla[sla_type]),
                False,
                True,
                "NO_ACTIVE_WAIT",
            )

        selected_score = self.scorer.score(selected, sla_type).total_score
        earliest_score = self.scorer.score(earliest, sla_type).total_score
        objective_gain = selected_score - earliest_score
        wait_penalty = (
            float(self.config.penalty_weight_by_sla[sla_type])
            * wait
            / self.config.time_scale_sim
        )
        net_gain = objective_gain - wait_penalty
        minimum = float(self.config.min_gain_by_sla[sla_type])
        passed = net_gain > minimum + self.config.tolerance
        return WaitGateDiagnostic(
            task_id,
            selected.candidate_id,
            selected.candidate_id if passed else earliest.candidate_id,
            wait,
            objective_gain,
            wait_penalty,
            net_gain,
            minimum,
            True,
            passed,
            "NET_GAIN_ACCEPTED" if passed else "NET_GAIN_BELOW_THRESHOLD",
        )

    # 作用：应用净收益判定，未通过则把选中候选替换为最早可行候选。
    # 调用上下文：由策略包装器调用；输出最终选择与诊断，诊断由 V4Scheduler 写入决策记录。
    # 下游调用：self.evaluate。
    def apply(self, selection, task=None):
        diagnostic = self.evaluate(
            selection.selected_candidate,
            selection.earliest_candidate,
            task,
        )
        if diagnostic.passed:
            return selection, diagnostic
        return replace(
            selection,
            selected_candidate=selection.earliest_candidate,
        ), diagnostic


# 作用：从共享配置构造成本、绿电和 SLA 目标参数。
# 上游（直接调用）：v4.v4_runtime.create_v4_runtime。
# 下游调用：ObjectiveConfig。
def default_objective_config() -> ObjectiveConfig:
    return ObjectiveConfig(
        config.V1_COST_REFERENCE_YUAN,
        config.V1_COST_SCALE_YUAN,
        config.V1_GREEN_ABSORPTION_DELTA_SCALE,
        config.V1_OBJECTIVE_COST_WEIGHT,
        config.V1_OBJECTIVE_GREEN_WEIGHT,
        config.V1_OBJECTIVE_BALANCE_WEIGHT,
        config.V1_SOFT_TARDINESS_WEIGHT,
        config.V1_FLEXIBLE_TARDINESS_WEIGHT,
    )
