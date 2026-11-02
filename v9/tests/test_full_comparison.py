"""Report aggregation checks without starting a simulation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from v9.evaluate_full_comparison import _paths, summarize


def report(seed, completed, trace_hash="trace"):
    valid = lambda value: {"status": "VALID", "value": value}
    sla = {name: {"expired_rate": valid(0), "start_delay_p95_sim": valid(2)}
           for name in ("Hard", "Soft", "Flexible")}
    return {
        "status": "VALID",
        "metadata": {"seed": seed, "model_hash": "model",
                     "task_trace_hash": trace_hash, "exogenous_trace_hash": "energy",
                     "topology_hash": "topology", "dependency_lock_hash": "dependencies",
                     "candidate_mode": "layered_pool", "warmup_days": 2,
                     "arrival_cutoff_sim": 864, "tariff_mode": "tou_region",
                     "code_hash": "code", "config_hash": "config"},
        "metrics": {"arrival_count": 10, "completed_count": completed,
                    "expired_count": 0, "failed_count": 0,
                    "total_economic_cost_yuan": 100,
                    "completed_cpu_hours": 2,
                    "completion_rate": valid(completed / 10),
                    "completed_task_green_coverage": valid(0.5),
                    "system_green_absorption_rate": valid(0.4),
                    "cost_yuan_per_completed_cpu_hour": valid(50),
                    "sla_metrics": sla,
                    "active_wait_metrics": {"mean_active_wait_sim": valid(1)}},
        "diagnostics": {"runtime_summary": {
            "sections_seconds": {"candidate_slot_generation": 3},
            "decision_wall_seconds": {"mean": 0.1, "p95": 0.2},
            "total_wall_seconds": 10,
        }},
    }


class FullComparisonTest(unittest.TestCase):
    def test_overall_summary_uses_paired_seeds(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (42, 43):
                paths = _paths(root, seed)
                paths["trace"].parent.mkdir(parents=True)
                paths["trace"].write_text(json.dumps({"task_trace_hash": "trace"}))
                paths["v7"].write_text(json.dumps(report(seed, 7)))
                paths["e3"].write_text(json.dumps(report(seed, 8)))
            result = summarize(root, (42, 43), "overall")
            self.assertEqual(result["median"]["e3"]["completion_rate"], 0.8)
            self.assertEqual(result["paired_delta_from_reference"]["e3"]["completed_count"], [1, 1])
            self.assertTrue((root / "overall_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
