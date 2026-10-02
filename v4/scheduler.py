"""V4 scheduler diagnostics for the post-selection wait gate."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

from v1.scheduler.v1_scheduler import SchedulingDecision, V1Scheduler


@dataclass(frozen=True)
class V4SchedulingDecision(SchedulingDecision):
    proposed_candidate_id: Optional[str] = None
    proposed_active_wait_sim: Optional[float] = None
    wait_gate_applied: Optional[bool] = None
    wait_gate_passed: Optional[bool] = None
    wait_gate_reason: str = ""
    objective_gain_before_wait_penalty: Optional[float] = None
    wait_penalty: Optional[float] = None
    estimated_net_wait_gain: Optional[float] = None
    minimum_required_wait_gain: Optional[float] = None


class V4Scheduler(V1Scheduler):
    """V1-compatible scheduler that exposes V4 gate decisions in audit rows."""

    def _schedule_one(self, task, *args, **kwargs):
        clear = getattr(self.policy, "clear_gate_diagnostic", None)
        if clear is not None:
            clear(task.task_id)
        return super()._schedule_one(task, *args, **kwargs)

    def _decision_record(self, task_id, *args, **kwargs):
        base = V1Scheduler._decision_record(task_id, *args, **kwargs)
        pop = getattr(self.policy, "pop_gate_diagnostic", None)
        diagnostic = None if pop is None else pop(task_id)
        base_values = {
            item.name: getattr(base, item.name)
            for item in fields(SchedulingDecision)
        }
        if diagnostic is None:
            return V4SchedulingDecision(**base_values)
        return V4SchedulingDecision(
            **base_values,
            proposed_candidate_id=diagnostic.proposed_candidate_id,
            proposed_active_wait_sim=diagnostic.proposed_active_wait_sim,
            wait_gate_applied=diagnostic.applied,
            wait_gate_passed=diagnostic.passed,
            wait_gate_reason=diagnostic.reason,
            objective_gain_before_wait_penalty=(
                diagnostic.objective_gain_before_wait_penalty
            ),
            wait_penalty=diagnostic.wait_penalty,
            estimated_net_wait_gain=diagnostic.estimated_net_wait_gain,
            minimum_required_wait_gain=diagnostic.minimum_required_gain,
        )
