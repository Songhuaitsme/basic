from contextlib import contextmanager
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import torch

from v1.domain.models import MetricValue
from v2.evaluate_coordination import (
    TASK_FIELDS,
    build_system_metrics,
    build_task_rows,
    discover_v2_policy_model,
    resolve_model_path,
)
from v2.coordination_metrics import (
    COUNTERFACTUAL_LABEL,
    build_green_coverage_gain_distribution,
    build_mutually_exclusive_coordination_groups,
)


class CoordinationEvaluationTest(unittest.TestCase):
    @contextmanager
    def _temporary_directory(self):
        path = Path(__file__).resolve().parent / f"_tmp_{uuid.uuid4().hex}"
        path.mkdir()
        try:
            yield str(path)
        finally:
            shutil.rmtree(path)

    def test_discovery_selects_highest_step_formal_v2_policy(self):
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            formal = root / "artifacts" / "v2" / "formal"
            formal.mkdir(parents=True)
            for name, steps in (
                ("candidate_12000_tou_region.pt", 12_000),
                ("candidate_600000_tou_region.pt", 600_000),
            ):
                torch.save({
                    "model_state_dict": {"weight": torch.tensor([1.0])},
                    "metadata": {"model_schema_version": "1.0"},
                    "training_steps": steps,
                    "run_config": {"V1_TARIFF_MODE": "tou_region"},
                }, formal / name)
            torch.save(
                {"checkpoint_type": "v1_training_resume"},
                formal / "candidate_700000_tou_region.last.pt",
            )

            selected = discover_v2_policy_model(root)

            self.assertEqual(selected.name, "candidate_600000_tou_region.pt")

    def test_explicit_uniform_model_is_rejected(self):
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            path = root / "candidate_uniform.pt"
            torch.save({
                "model_state_dict": {"weight": torch.tensor([1.0])},
                "metadata": {"model_schema_version": "1.0"},
                "run_config": {"V1_TARIFF_MODE": "tou_uniform"},
            }, path)

            with self.assertRaisesRegex(ValueError, "tou_region"):
                resolve_model_path(root, path)

    def _report(self):
        record = SimpleNamespace(
            task_id="task-1",
            source_node="A0",
            target_node="B1",
            arrival_time_sim=2.0,
            earliest_compute_start_sim=3.0,
            compute_start_sim=5.0,
            compute_end_sim=7.0,
            cpu_demand=10.0,
            execution_duration_sim=2.0,
            task_energy_mwh=4.0,
            earliest_candidate_marginal_system_cost_yuan=12.0,
            candidate_marginal_system_cost_yuan=9.0,
            earliest_green_coverage=0.25,
            selected_green_coverage=0.75,
            earliest_candidate_marginal_green_energy_mwh=1.0,
            candidate_marginal_green_energy_mwh=3.0,
            earliest_green_absorption_delta=0.1,
            selected_green_absorption_delta=0.4,
            sla_type="Flexible",
            final_state="Completed",
            start_delay_sim=3.0,
            preferred_start_limit_sim=2.0,
            latest_start_limit_sim=4.0,
            terminal_reason=None,
            task_attributed_cost_yuan=8.5,
            task_attributed_green_energy_mwh=3.0,
            earliest_target_node="A1",
        )
        metrics = SimpleNamespace(
            total_economic_cost_yuan=8.5,
            completed_task_green_coverage=MetricValue.valid(0.75),
            system_green_absorption_rate=MetricValue.valid(0.5),
        )
        accounting = SimpleNamespace(
            total_task_attributed_green_energy_mwh=3.0,
        )
        return SimpleNamespace(
            task_records=(record,),
            metrics=metrics,
            accounting_report=accounting,
        )

    def test_task_rows_preserve_same_decision_counterfactual(self):
        row = build_task_rows(self._report())[0]

        self.assertTrue(set(TASK_FIELDS).issubset(row))
        self.assertEqual(row["active_wait"], 2.0)
        self.assertEqual(row["cost_saving"], 3.0)
        self.assertEqual(row["green_coverage_gain"], 0.5)
        self.assertEqual(row["green_energy_gain"], 2.0)
        self.assertAlmostEqual(row["green_absorption_gain"], 0.3)
        self.assertTrue(row["node_adjustment"])
        self.assertTrue(row["region_adjustment"])
        self.assertTrue(row["temporal_adjustment"])
        self.assertTrue(row["node_temporal_adjustment"])
        self.assertTrue(row["region_temporal_adjustment"])
        self.assertEqual(row["selected_region"], "B")
        self.assertEqual(row["earliest_region"], "A")
        self.assertEqual(row["comparison_basis"], COUNTERFACTUAL_LABEL)
        self.assertNotIn("spatial_migration", row)
        self.assertTrue(row["sla_satisfied"])

    def test_system_metrics_use_committed_tasks_as_adjustment_denominator(self):
        report = self._report()
        rows = build_task_rows(report)

        metrics = build_system_metrics(report, rows)

        self.assertEqual(metrics["total_tasks"], 1)
        self.assertEqual(metrics["reservation_success_rate"], 1.0)
        self.assertEqual(metrics["node_adjustment_ratio"], 1.0)
        self.assertEqual(metrics["region_adjustment_ratio"], 1.0)
        self.assertEqual(metrics["temporal_adjustment_ratio"], 1.0)
        self.assertEqual(metrics["region_temporal_adjustment_ratio"], 1.0)
        self.assertNotIn("spatial_migration_ratio", metrics)
        self.assertEqual(metrics["total_cost_saving"], 3.0)
        self.assertEqual(metrics["total_green_energy_gain"], 2.0)
        self.assertEqual(metrics["average_additional_wait"], 2.0)
        self.assertEqual(metrics["sla_change"], 0.0)

    def test_mutually_exclusive_groups_cover_every_scheduled_task_once(self):
        rows = build_task_rows(self._report())

        groups = build_mutually_exclusive_coordination_groups(rows)

        self.assertEqual(sum(row["task_count"] for row in groups), 1)
        joint = next(
            row for row in groups
            if row["coordination_type"] == "region_temporal_adjustment"
        )
        self.assertEqual(joint["task_count"], 1)
        self.assertEqual(joint["total_green_energy_gain"], 2.0)

    def test_green_gain_distribution_uses_zero_tolerance(self):
        rows = build_task_rows(self._report())
        almost_zero = dict(rows[0])
        almost_zero["task_id"] = "task-2"
        almost_zero["green_coverage_gain"] = 1e-15

        distribution = build_green_coverage_gain_distribution(
            [rows[0], almost_zero]
        )

        self.assertEqual(distribution["gain_positive_tasks"], 1)
        self.assertEqual(distribution["gain_zero_tasks"], 1)
        self.assertEqual(distribution["gain_negative_tasks"], 0)


if __name__ == "__main__":
    unittest.main()
