import unittest
from types import SimpleNamespace

from v1.domain.models import SlaType
from v1.learning.candidate_dqn import CandidateSelectionTrace
from v1.scheduler.policies import CandidateStreamSelection
from v4.gating import NetBenefitWaitGate, WaitGateConfig, default_objective_config
from v4.policy import NetBenefitGatePolicy
from v4.scheduler import V4Scheduler
from v4.v4_runtime import create_v4_runtime


def candidate(candidate_id, start, cost, *, green=0.5, absorption=0.0, margin=0.5):
    return SimpleNamespace(
        candidate_id=candidate_id,
        compute_start_sim=float(start),
        estimated_candidate_marginal_system_cost_yuan=float(cost),
        estimated_green_coverage=float(green),
        estimated_green_absorption_delta=float(absorption),
        estimated_green_opportunity=True,
        capacity_margin=float(margin),
        preferred_start_tardiness_ratio=0.0,
        preferred_start_tardiness_applicable=True,
    )


def task(sla_type=SlaType.SOFT):
    return SimpleNamespace(task_id="task-1", sla_type=sla_type)


def selection(selected, earliest):
    return CandidateStreamSelection(selected, earliest, 2, "hash", None)


class FixedPolicy:
    name = "fixed"

    def __init__(self, result):
        self.result = result

    def select_stream(self, candidates, task=None, context=None):
        tuple(candidates)
        return self.result


class FixedEncoder:
    def encode(self, candidate, earliest_compute_start_sim):
        return (candidate.compute_start_sim, earliest_compute_start_sim)


class WaitGateTest(unittest.TestCase):
    def test_gate_rejects_wait_when_only_wait_penalty_changes(self):
        earliest = candidate("early", 10, 100.0)
        delayed = candidate("late", 20, 100.0)
        gate = NetBenefitWaitGate(default_objective_config(), WaitGateConfig())

        gated, diagnostic = gate.apply(selection(delayed, earliest), task())

        self.assertIs(gated.selected_candidate, earliest)
        self.assertTrue(diagnostic.applied)
        self.assertFalse(diagnostic.passed)
        self.assertGreater(diagnostic.wait_penalty, 0.0)
        self.assertLess(diagnostic.estimated_net_wait_gain, 0.0)
        self.assertEqual(diagnostic.reason, "NET_GAIN_BELOW_THRESHOLD")

    def test_gate_accepts_wait_when_cost_gain_clears_penalty(self):
        objective = default_objective_config()
        scale = objective.cost_scale_yuan
        reference = objective.reference_marginal_cost_yuan
        earliest = candidate("early", 10, reference + scale)
        delayed = candidate("late", 20, reference - scale)
        gate = NetBenefitWaitGate(objective, WaitGateConfig())

        gated, diagnostic = gate.apply(selection(delayed, earliest), task())

        self.assertIs(gated.selected_candidate, delayed)
        self.assertTrue(diagnostic.passed)
        self.assertGreater(diagnostic.estimated_net_wait_gain, 0.0)
        self.assertEqual(diagnostic.reason, "NET_GAIN_ACCEPTED")

    def test_gate_does_not_override_same_start_candidate(self):
        earliest = candidate("early", 10, 100.0)
        selected = candidate("same-time", 10, 90.0)
        gate = NetBenefitWaitGate(default_objective_config(), WaitGateConfig())

        gated, diagnostic = gate.apply(selection(selected, earliest), task())

        self.assertIs(gated.selected_candidate, selected)
        self.assertFalse(diagnostic.applied)
        self.assertTrue(diagnostic.passed)
        self.assertEqual(diagnostic.reason, "NO_ACTIVE_WAIT")

    def test_policy_wrapper_exposes_gate_diagnostic(self):
        earliest = candidate("early", 10, 100.0)
        delayed = candidate("late", 20, 100.0)
        gate = NetBenefitWaitGate(default_objective_config(), WaitGateConfig())
        policy = NetBenefitGatePolicy(
            FixedPolicy(selection(delayed, earliest)),
            gate,
        )

        result = policy.select_stream((earliest, delayed), task=task())
        diagnostic = policy.pop_gate_diagnostic("task-1")

        self.assertIs(result.selected_candidate, earliest)
        self.assertEqual(diagnostic.proposed_candidate_id, "late")
        self.assertEqual(diagnostic.final_candidate_id, "early")

    def test_policy_wrapper_rewrites_training_trace_after_fallback(self):
        earliest = candidate("early", 10, 100.0)
        delayed = candidate("late", 20, 100.0)
        base = FixedPolicy(selection(delayed, earliest))
        base.feature_encoder = FixedEncoder()
        base._selection_traces = [CandidateSelectionTrace(
            task_id="task-1",
            decision_time_sim=0.0,
            global_state=(0.0,),
            candidate_count=2,
            selected_candidate_id="late",
            selected_candidate_features=(20.0, 10.0),
            selected_candidate=delayed,
        )]
        policy = NetBenefitGatePolicy(
            base,
            NetBenefitWaitGate(default_objective_config(), WaitGateConfig()),
        )

        policy.select_stream((earliest, delayed), task=task())
        trace = base._selection_traces[-1]

        self.assertIs(trace.selected_candidate, earliest)
        self.assertEqual(trace.selected_candidate_id, "early")
        self.assertEqual(trace.selected_candidate_features, (10.0, 10.0))

    def test_v4_runtime_keeps_v3_engine_and_installs_v4_scheduler(self):
        runtime = create_v4_runtime(policy_name="earliest_feasible")

        self.assertIsInstance(runtime.scheduler, V4Scheduler)
        self.assertIsInstance(runtime.scheduler.policy, NetBenefitGatePolicy)
        self.assertTrue(runtime.config_view_overrides["V4_WAIT_GATE_ENABLED"])


if __name__ == "__main__":
    unittest.main()
