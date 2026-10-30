"""Candidate profiling accounting, lifecycle, and frozen-decision equivalence."""

import csv
from dataclasses import dataclass, replace
from enum import Enum
import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from v1.candidate_profiling import (
    CandidateDetailProfiler, candidate_stage, write_candidate_detail_csv,
)
from v1.learning.candidate_dqn import CandidateDQNPolicy, SharedCandidateQNetwork
from v1.learning.features import CandidateFeatureConfig, CandidateFeatureEncoder
from v1.profiling import TrainingPerformanceProfiler
from v1.scheduler.path_provider import StaticPathProvider
from v1.scheduler.transmission import TransmissionModel
from v4.gating import NetBenefitWaitGate, WaitGateConfig, default_objective_config
from v4.policy import NetBenefitGatePolicy
from v5.scheduler import V5CandidateGenerator
from v6.accounting import V6EnergyAccounting
from v6.tests import test_energy as energy_fixtures


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class CandidateProfilingTest(unittest.TestCase):
    def test_nested_inclusive_exclusive_and_legacy_leaves(self):
        clock = Clock()
        profiler = TrainingPerformanceProfiler()
        profiler.enable_candidate_detail()
        detail = profiler.candidate_detail
        detail.clock = clock
        with detail.scope("candidate_prepare"):
            clock.value = 2
            with detail.scope("feasibility", phase="anchor"):
                profiler.add("candidate_cpu_feasibility_seconds", 1.0)
                clock.value = 5
            clock.value = 10
        report = detail.summary(11.0)
        own = {row["stage"]: row["self_seconds"] for row in report["rows"]}
        self.assertEqual(own, {"candidate_prepare": 7.0, "feasibility": 2.0, "cpu_check": 1.0})
        self.assertEqual(report["scoped_seconds"], 10.0)
        self.assertEqual(report["exclusive_seconds"], 10.0)
        self.assertEqual(report["outside_scopes_seconds"], 1.0)
        self.assertEqual(report["status"], "VALID")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "detail.csv"
            write_candidate_detail_csv(report, path)
            with path.open(encoding="utf-8-sig") as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(sum(float(row["self_seconds"]) for row in exported), 11.0)

    def test_exception_and_early_return_close_scopes(self):
        clock = Clock()
        detail = CandidateDetailProfiler(clock)
        with self.assertRaisesRegex(ValueError, "expected"):
            with detail.scope("candidate_prepare"):
                clock.value = 4
                raise ValueError("expected")
        self.assertEqual(detail.stack, [])
        self.assertEqual(detail.summary(4)["status"], "VALID")

        class Operation:
            profiler = SimpleNamespace(candidate_detail=detail)

            @candidate_stage("result_assembly")
            def empty(self):
                clock.value = 6
                return ()

        self.assertEqual(Operation().empty(), ())
        self.assertFalse(detail.stack)
        self.assertEqual(detail.summary(6)["exclusive_seconds"], 6)

    def test_overlapping_timers_are_reported_not_hidden(self):
        clock = Clock()
        detail = CandidateDetailProfiler(clock)
        with detail.scope("candidate_prepare"):
            detail.record_leaf("candidate_metric_evaluation_seconds", 3)
            clock.value = 2
        report = detail.summary(2)
        self.assertEqual(report["status"], "INVALID")
        self.assertEqual(report["negative_self_count"], 1)
        self.assertTrue(any(row["self_seconds"] < 0 for row in report["rows"]))

    def test_disabled_and_old_pickle_and_empty_csv(self):
        profiler = TrainingPerformanceProfiler()
        self.assertNotIn("candidate_detail", profiler.summary(0))
        old = dict(profiler.__dict__)
        old.pop("candidate_detail")
        restored = TrainingPerformanceProfiler.__new__(TrainingPerformanceProfiler)
        restored.__setstate__(old)
        self.assertIsNone(restored.candidate_detail)
        self.assertEqual(pickle.loads(pickle.dumps(restored)).summary(0), restored.summary(0))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "detail.csv"
            write_candidate_detail_csv(CandidateDetailProfiler().summary(0), path)
            with path.open(encoding="utf-8-sig") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])

    def test_v6_pool_metrics_decisions_and_stream_boundaries_match(self):
        fixture = energy_fixtures.EnergyTest()
        fixture.setUp()
        encoder = CandidateFeatureEncoder(
            {"N": 0}, CandidateFeatureConfig(30.0, 1000.0, 1.0, 10.0, 100.0),
        )
        network = SharedCandidateQNetwork(2, encoder.feature_dim, hidden_dim=8)
        for sla in ("Hard", "Soft", "Flexible"):
            task = fixture.task(sla)
            outputs = []
            for enabled in (False, True):
                profiler = TrainingPerformanceProfiler()
                if enabled:
                    profiler.enable_candidate_detail()
                accounting = fixture.accounting(V6EnergyAccounting)
                generator = V5CandidateGenerator(
                    ("N",), 1.0, StaticPathProvider(fixture.graph),
                    TransmissionModel(fixture.converter, 200000.0), fixture.calendar,
                    candidate_mode="layered_pool", pool_max_by_sla={s: 8 for s in ("Hard", "Soft", "Flexible")},
                )
                generator.profiler = profiler
                stream = generator.prepare_stream(
                    task, 0.0, reservation_snapshot=fixture.snapshot,
                    forecast_covered_until_sim=40.0,
                    metric_evaluator=accounting.candidate_metric_evaluator(fixture.snapshot),
                )
                base = CandidateDQNPolicy(network, encoder, lambda _: (0.0, 0.0), candidate_chunk_size=2)
                base.profiler = profiler
                policy = NetBenefitGatePolicy(base, NetBenefitWaitGate(default_objective_config(), WaitGateConfig()))
                selection = policy.select_complete_stream(stream, task)
                outputs.append((stream.context.selected_records, stream.context.selected_metrics, selection, policy.pop_gate_diagnostic(task.task_id)))
                self.assertGreater(profiler.timings["candidate_feasibility_prepare_seconds"], 0)
                self.assertGreater(profiler.counters["prepare_feasibility_check_count"], 0)
                if enabled:
                    # Direct generator tests have no scheduler prepare timer;
                    # use the measured root plus stream outer interval here.
                    roots = profiler.candidate_detail.root_seconds
                    detail = profiler.candidate_detail.summary(roots)
                    self.assertEqual(detail["status"], "VALID")
                    self.assertAlmostEqual(detail["exclusive_seconds"], roots)
                    phases = {row["phase"] for row in detail["rows"]}
                    self.assertTrue({"anchor", "second", "final_stream"}.issubset(phases))
                    stages = {row["stage"] for row in detail["rows"]}
                    self.assertIn("resource_occupancy", stages)
                    self.assertIn("pareto", stages)
                    self.assertNotIn("selection_inference_seconds", stages)
                    self.assertFalse(profiler.candidate_detail.stack)
            self.assertEqual(outputs[0], outputs[1])

    def test_empty_pool_keeps_counts_and_closes_root(self):
        fixture = energy_fixtures.EnergyTest()
        fixture.setUp()
        task = replace(fixture.task(), cpu_demand=100.0)
        generator = V5CandidateGenerator(
            ("N",), 1.0, StaticPathProvider(fixture.graph),
            TransmissionModel(fixture.converter, 200000.0), fixture.calendar,
            candidate_mode="layered_pool",
        )
        profiler = TrainingPerformanceProfiler()
        profiler.enable_candidate_detail()
        generator.profiler = profiler
        stream = generator.prepare_stream(
            task, 0.0, reservation_snapshot=fixture.snapshot,
            forecast_covered_until_sim=40.0,
            metric_evaluator=fixture.accounting(V6EnergyAccounting).candidate_metric_evaluator(fixture.snapshot),
        )
        self.assertEqual(tuple(stream.iter_candidates()), ())
        self.assertFalse(profiler.candidate_detail.stack)
        detail = profiler.candidate_detail.summary(profiler.candidate_detail.root_seconds)
        self.assertEqual(detail["status"], "VALID")
        self.assertGreater(detail["counters"]["anchor/static_cpu_rejections"], 0)

    def test_cli_flag_is_forwarded_and_json_csv_are_exported(self):
        import v1.evaluate_v1 as evaluation

        class Status(str, Enum):
            VALID = "VALID"

        @dataclass
        class Report:
            status: Status
            diagnostics: dict
            cycle_results: tuple = ()

        detail = CandidateDetailProfiler().summary(0)
        report = Report(Status.VALID, {"runtime_summary": {"candidate_detail": detail}})
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "detail.json"
            with patch("sys.argv", [
                "evaluate_v6", "--policy", "candidate_dqn", "--candidate-profile-detail",
                "--output", str(output),
            ]), patch.object(evaluation, "run_evaluation", return_value=report) as run, patch("builtins.print"):
                evaluation.main(system_version="6.0", default_report_mode="compact")
            self.assertTrue(run.call_args.kwargs["candidate_profile_detail"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["diagnostics"]["runtime_summary"]["candidate_detail"]["status"], "VALID")
            self.assertTrue(output.with_suffix(".candidate_detail.csv").exists())


if __name__ == "__main__":
    unittest.main()
