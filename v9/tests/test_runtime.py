import unittest
import pickle

import v1.evaluate_v1 as evaluation
import v1.train_v1 as training

from v8.scheduler import V8CandidateGenerator
from v9._entrypoint import use_v9_runtime
from v9.scheduler import V9CandidateGenerator
from v9.v9_runtime import create_v9_runtime


class V9RuntimeTest(unittest.TestCase):
    def test_stage_switch_and_complete_reference(self):
        kwargs = dict(policy_name="earliest_feasible", forecast_end_sim=30)
        baseline = create_v9_runtime(search_stage=0, **kwargs)
        optimized = create_v9_runtime(search_stage=3, **kwargs)
        self.assertIs(type(baseline.scheduler.candidate_generator), V8CandidateGenerator)
        self.assertIs(type(optimized.scheduler.candidate_generator), V9CandidateGenerator)
        self.assertIs(optimized.scheduler.candidate_generator.energy_accounting,
                      optimized.accounting)
        self.assertEqual(optimized.config_view_overrides["V9_SEARCH_STAGE"], "E3")
        restored = pickle.loads(pickle.dumps(optimized.scheduler.candidate_generator))
        self.assertIs(type(restored), V9CandidateGenerator)

    def test_entrypoint_restores_factories(self):
        original = (training.create_v1_runtime, evaluation.create_v1_runtime,
                    training._training_config_view)
        with use_v9_runtime(2):
            self.assertIs(type(training.create_v1_runtime(
                policy_name="earliest_feasible", forecast_end_sim=30
            ).scheduler.candidate_generator), V9CandidateGenerator)
        self.assertEqual(original, (training.create_v1_runtime,
                                    evaluation.create_v1_runtime,
                                    training._training_config_view))


if __name__ == "__main__":
    unittest.main()
