"""Benchmark receipts must reject unequal models, decisions, and bad timers."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from v6.profile_candidates import PAIR_FIELDS, summarize


class ProfileComparisonTest(unittest.TestCase):
    def _reports(self, folder):
        report = {
            "status": "VALID", "metadata": {key: "same" for key in PAIR_FIELDS},
            "metrics": {"completed_count": 1},
            "decision_records": [{"task_id": "t", "selected_candidate_id": "a", "candidate_set_hash": "hash", "decision_wall_seconds": 1}],
            "diagnostics": {"runtime_summary": {
                "total_wall_seconds": 10, "sections_seconds": {"candidate_slot_generation": 9},
                "decision_wall_seconds": {"mean": 1, "p95": 2},
            }},
        }
        baseline = folder / "trial_01_off.json"
        baseline.write_text(json.dumps(report), encoding="utf-8")
        detailed = deepcopy(report)
        detailed["decision_records"][0]["decision_wall_seconds"] = 2
        detailed["diagnostics"]["runtime_summary"].update({
            "total_wall_seconds": 10.2,
            "candidate_detail": {"status": "VALID", "pipeline_seconds": 9, "unattributed_percent": 2, "rows": []},
        })
        target = folder / "trial_01_on.json"
        target.write_text(json.dumps(detailed), encoding="utf-8")
        return target, detailed

    def test_timing_only_changes_pass_and_exports_are_written(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._reports(folder)
            result = summarize(folder, 1)
            self.assertAlmostEqual(result["instrumentation_overhead_ratio"], 0.02)
            self.assertTrue(result["overhead_within_5_percent"])
            self.assertFalse(result["cprofile_recommended"])
            for name in ("comparison.json", "comparison.csv", "candidate_detail.csv"):
                self.assertTrue((folder / name).exists())

    def test_changed_model_or_decision_and_invalid_timers_fail(self):
        for kind in ("model", "decision", "timers"):
            with self.subTest(kind=kind), TemporaryDirectory() as tmp:
                folder = Path(tmp)
                path, detailed = self._reports(folder)
                if kind == "model":
                    detailed["metadata"]["model_hash"] = "different"
                elif kind == "decision":
                    detailed["decision_records"][0]["selected_candidate_id"] = "different"
                else:
                    detailed["diagnostics"]["runtime_summary"]["candidate_detail"]["status"] = "INVALID"
                path.write_text(json.dumps(detailed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    summarize(folder, 1)


if __name__ == "__main__":
    unittest.main()
