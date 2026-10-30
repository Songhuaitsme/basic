"""V6/V7 comparison allows declared version changes but rejects behavior drift."""

from copy import deepcopy
import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from v6.profile_candidates import PAIR_FIELDS
from v7.compare_v6_v7 import prepare_resume, summarize


class ComparisonTest(unittest.TestCase):
    def reports(self, folder):
        baseline = {
            "status": "VALID", "metadata": {k: "same" for k in PAIR_FIELDS},
            "metrics": {"completed_count": 1},
            "decision_records": [{"selected_candidate_id": "a", "candidate_set_hash": "x", "decision_wall_seconds": 1}],
            "diagnostics": {"runtime_summary": {
                "total_wall_seconds": 10, "sections_seconds": {"candidate_slot_generation": 9},
                "decision_wall_seconds": {"mean": 1, "p95": 2}, "path_search_cache": {"enabled": False},
            }},
        }
        baseline["metadata"]["system_version"] = "6.0"
        optimized = deepcopy(baseline)
        optimized["metadata"].update(system_version="7.0", config_hash="v7-config", code_hash="v7-code")
        optimized["diagnostics"]["runtime_summary"].update({
            "total_wall_seconds": 7, "sections_seconds": {"candidate_slot_generation": 6},
            "path_search_cache": {"enabled": True, "provider": {"path_cache_hits": 5}},
        })
        optimized["decision_records"][0]["decision_wall_seconds"] = 0.5
        for version, report in (("v6", baseline), ("v7", optimized)):
            (folder / f"trial_01_{version}.json").write_text(json.dumps(report), encoding="utf-8")
        return optimized

    def test_versions_can_differ_but_model_decisions_must_match(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            optimized = self.reports(folder)
            result = summarize(folder, 1)
            self.assertAlmostEqual(result["total_time_reduction_ratio"], 0.3)
            self.assertTrue((folder / "comparison.csv").exists())
            optimized["decision_records"][0]["candidate_set_hash"] = "changed"
            (folder / "trial_01_v7.json").write_text(json.dumps(optimized), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "decisions"):
                summarize(folder, 1)

    def test_detailed_profiling_and_wrong_cache_state_are_rejected(self):
        for kind in ("detail", "cache"):
            with self.subTest(kind=kind), TemporaryDirectory() as tmp:
                folder = Path(tmp)
                optimized = self.reports(folder)
                runtime = optimized["diagnostics"]["runtime_summary"]
                if kind == "detail":
                    runtime["candidate_detail"] = {}
                else:
                    runtime["path_search_cache"]["enabled"] = False
                (folder / "trial_01_v7.json").write_text(json.dumps(optimized), encoding="utf-8")
                with self.assertRaises(ValueError):
                    summarize(folder, 1)

    def _resume_args(self, folder):
        return SimpleNamespace(output_dir=folder, repeats=3)

    @staticmethod
    def _receipt(version, args, model_hash):
        report = json.loads((args.output_dir / f"trial_01_{version}.json").read_text(encoding="utf-8"))
        return dict(report["metadata"])

    def test_resume_skips_valid_reports_and_preserves_incomplete_json(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.reports(folder)
            completed = prepare_resume(self._resume_args(folder), "same", self._receipt)
            self.assertEqual(completed, {(1, "v6"), (1, "v7")})
            incomplete = folder / "trial_02_v7.json"
            incomplete.write_text('{"status":', encoding="utf-8")
            with patch("builtins.print"):
                completed = prepare_resume(self._resume_args(folder), "same", self._receipt)
            self.assertEqual(completed, {(1, "v6"), (1, "v7")})
            self.assertFalse(incomplete.exists())
            backups = list(folder.glob("trial_02_v7.json.interrupted-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), '{"status":')

    def test_resume_rejects_changed_code_model_or_requested_settings(self):
        for changed in ("code_hash", "model_hash", "seed", "config_hash"):
            with self.subTest(changed=changed), TemporaryDirectory() as tmp:
                folder = Path(tmp)
                self.reports(folder)
                incomplete = folder / "trial_02_v6.json"
                incomplete.write_text("{", encoding="utf-8")

                def different(version, args, model_hash):
                    receipt = self._receipt(version, args, model_hash)
                    receipt[changed] = "different"
                    return receipt

                with self.assertRaisesRegex(ValueError, changed):
                    prepare_resume(self._resume_args(folder), "same", different)
                # Validation fails before any files are moved or overwritten.
                self.assertTrue(incomplete.exists())

    def test_cli_resume_launches_only_missing_trials(self):
        import v7.compare_v6_v7 as comparison

        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            optimized = self.reports(folder)
            (folder / "trial_01_v7.json").unlink()
            model = folder / "model.pt"
            model.write_bytes(b"fixture")
            baseline = json.loads((folder / "trial_01_v6.json").read_text(encoding="utf-8"))
            baseline["metadata"]["model_hash"] = hashlib.sha256(model.read_bytes()).hexdigest()
            optimized["metadata"]["model_hash"] = baseline["metadata"]["model_hash"]
            (folder / "trial_01_v6.json").write_text(json.dumps(baseline), encoding="utf-8")

            def fake_run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                self.assertEqual(output.name, "trial_01_v7.json")
                output.write_text(json.dumps(optimized), encoding="utf-8")

            with patch("sys.argv", ["compare", "--resume", "--repeats", "1", "--model-path", str(model), "--output-dir", str(folder)]), \
                    patch.object(comparison, "_current_receipt", side_effect=self._receipt), \
                    patch.object(comparison.subprocess, "run", side_effect=fake_run) as run, patch("builtins.print"):
                comparison.main()
            self.assertEqual(run.call_count, 1)
            self.assertTrue((folder / "comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
