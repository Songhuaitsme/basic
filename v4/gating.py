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

    def __init__(self, objective_config: ObjectiveConfig, gate_config=None):
        self.scorer = ObjectiveScorer(objective_config)
        self.config = gate_config or WaitGateConfig()

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
