import unittest

from v2.compare_v1_v2_models import extract_quality_metrics


class V1V2ComparisonTests(unittest.TestCase):
    def test_extracts_required_quality_metrics(self):
        report = {
            "status": "VALID",
            "metrics": {
                "cost_yuan_per_completed_cpu_hour": {
                    "value": 12.5,
                    "status": "VALID",
                },
                "completed_task_green_coverage": {
                    "value": 0.75,
                    "status": "VALID",
                },
            },
            "task_records": [
                {
                    "final_state": "Completed",
                    "target_node": "a",
                    "cpu_work_cpu_hours": 10.0,
                    "completion_delay_sim": 12.0,
                },
                {
                    "final_state": "Completed",
                    "target_node": "b",
                    "cpu_work_cpu_hours": 30.0,
                    "completion_delay_sim": 24.0,
                },
            ],
        }
        metrics = extract_quality_metrics(report)
        self.assertEqual(metrics["cost_yuan_per_completed_cpu_hour"], 12.5)
        self.assertEqual(metrics["completed_task_green_coverage"], 0.75)
        self.assertAlmostEqual(metrics["allocated_cpu_hours_cv"], 0.5)
        self.assertAlmostEqual(metrics["mean_completion_delay_hours"], 1.5)


if __name__ == "__main__":
    unittest.main()
