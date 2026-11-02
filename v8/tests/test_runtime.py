"""V8 runtime integration and process-local entrypoint isolation."""

import unittest

import v1.evaluate_v1 as evaluation
import v1.train_v1 as training

from v7.scheduler import V7CandidateGenerator
from v7.v7_runtime import create_v7_runtime
from v8._entrypoint import use_v8_runtime
from v8.scheduler import V8CandidateGenerator
from v8.v8_runtime import create_v8_runtime


class V8RuntimeTest(unittest.TestCase):
    def test_runtime_installs_v8_without_changing_v7(self):
        kwargs = dict(policy_name="earliest_feasible", forecast_end_sim=30)
        baseline = create_v7_runtime(**kwargs)
        optimized = create_v8_runtime(**kwargs)
        self.assertIs(type(baseline.scheduler.candidate_generator), V7CandidateGenerator)
        self.assertIs(type(optimized.scheduler.candidate_generator), V8CandidateGenerator)
        self.assertTrue(optimized.config_view_overrides["V8_STATIC_HARD_PRUNING"])
        self.assertEqual(
            optimized.config_view_overrides["V8_THEORETICAL_SLOT_COUNT_SCOPE"],
            "post_static_hard_pruning_search_slots",
        )
        with self.assertRaisesRegex(ValueError, "path cache"):
            create_v8_runtime(path_search_cache=False, **kwargs)

    def test_entrypoint_restores_factories_and_config(self):
        original = (
            training.create_v1_runtime, evaluation.create_v1_runtime,
            training._training_config_view,
        )
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with use_v8_runtime():
                self.assertIs(training.create_v1_runtime, create_v8_runtime)
                self.assertIs(evaluation.create_v1_runtime, create_v8_runtime)
                self.assertTrue(training._training_config_view()["V8_STATIC_HARD_PRUNING"])
                raise RuntimeError("expected")
        self.assertEqual(original, (
            training.create_v1_runtime, evaluation.create_v1_runtime,
            training._training_config_view,
        ))


if __name__ == "__main__":
    unittest.main()
