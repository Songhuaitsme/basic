import unittest

from v2.compare_five_policies import _best_policy, mean_ci


class CompareFivePoliciesTest(unittest.TestCase):
    def test_mean_ci_marks_single_seed_as_not_estimable(self):
        mean, low, high = mean_ci([3.0])
        self.assertEqual(mean, 3.0)
        self.assertIsNone(low)
        self.assertIsNone(high)

    def test_mean_ci_contains_constant_multi_seed_mean(self):
        mean, low, high = mean_ci([2.0, 2.0, 2.0])
        self.assertEqual((mean, low, high), (2.0, 2.0, 2.0))

    def test_best_policy_respects_direction_and_ties(self):
        values = {"earliest_feasible": 2.0, "candidate_dqn": 1.0}
        self.assertEqual(_best_policy(values, "lower"), ("Candidate DQN", 1.0))
        tied, value = _best_policy({"equal_weight": 1.0, "candidate_dqn": 1.0}, "higher")
        self.assertIn("并列", tied)
        self.assertEqual(value, 1.0)
        self.assertEqual(_best_policy(values, "neutral"), ("不判优", None))


if __name__ == "__main__":
    unittest.main()
