import unittest

from v2.compare_pricing_models import _comparison_rows


class PricingModelComparisonTests(unittest.TestCase):
    def test_comparison_rows_use_regional_minus_uniform(self):
        uniform = {
            "cost_yuan_per_completed_cpu_hour": 10.0,
            "completed_task_green_coverage": 0.6,
            "system_green_absorption_rate": 0.1,
            "completion_rate": 0.9,
            "reservation_reliability": 1.0,
            "soft_preferred_on_time_rate": 0.8,
            "flexible_preferred_on_time_rate": 0.7,
            "hard_start_delay_p95_hours": 1.0,
            "mean_completion_delay_hours": 2.0,
            "allocated_cpu_hours_cv": 0.5,
            "active_wait_count": 10.0,
            "mean_active_wait_hours": 3.0,
        }
        regional = dict(uniform)
        regional["cost_yuan_per_completed_cpu_hour"] = 8.0
        rows = _comparison_rows({
            "tou_uniform": uniform,
            "tou_region": regional,
        })
        cost = next(
            row
            for row in rows
            if row["metric"] == "cost_yuan_per_completed_cpu_hour"
        )
        self.assertEqual(cost["absolute_delta"], -2.0)
        self.assertAlmostEqual(cost["relative_delta"], -0.2)


if __name__ == "__main__":
    unittest.main()
