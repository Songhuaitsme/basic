"""Small losslessness checks for the V3 sweep-line accounting path."""

import unittest

import networkx as nx
import numpy as np

from v1.accounting import (
    ExogenousEnergyAccounting,
    ForecastSegment,
    LinearPowerModel,
    PiecewiseConstantForecast,
)
from v1.domain.models import TaskSpec
from v1.domain.reservations import ReservationRequest, TimeInterval
from v1.domain.units import TimeConverter
from v1.scheduler.resource_calendar import ReservationCalendar
from v1.scheduler.transmission import build_path_spec
from v3.accounting import SweepLineEnergyAccounting


class SweepLineEnergyAccountingTest(unittest.TestCase):
    def test_batch_metrics_match_v1_with_overlapping_allocations(self):
        tariff = PiecewiseConstantForecast.tariff_yuan_per_mwh((
            ForecastSegment(TimeInterval(0.0, 2.0), 100.0),
            ForecastSegment(TimeInterval(2.0, 8.0), 250.0),
            ForecastSegment(TimeInterval(8.0, 20.0), 80.0),
        ))
        green = PiecewiseConstantForecast.green_power_mw((
            ForecastSegment(TimeInterval(0.0, 4.0), 0.5),
            ForecastSegment(TimeInterval(4.0, 10.0), 3.0),
            ForecastSegment(TimeInterval(10.0, 20.0), 1.0),
        ))
        converter = TimeConverter(3600.0)
        power_model = LinearPowerModel(1.0)
        baseline = ExogenousEnergyAccounting(
            converter, power_model, {"N": tariff}, {"N": green}
        )
        optimized = SweepLineEnergyAccounting(
            converter, power_model, {"N": tariff}, {"N": green}
        )

        graph = nx.Graph()
        graph.add_node("N")
        path = build_path_spec(graph, ["N"])
        calendar = ReservationCalendar({"N": 10.0}, {})
        for task_id, cpu, start, end in (
            ("existing-a", 1.0, 1.0, 6.0),
            ("existing-b", 2.0, 3.0, 9.0),
            ("existing-c", 0.5, 7.0, 12.0),
        ):
            request = ReservationRequest(
                task_id=task_id,
                committed_candidate_id=f"candidate-{task_id}",
                committed_at_sim=0.0,
                reservation_snapshot_version=calendar.version,
                target_node="N",
                path=path,
                transmission_interval_sim=None,
                compute_interval_sim=TimeInterval(start, end),
                bandwidth_amount_mbps=1.0,
                cpu_amount=cpu,
            )
            calendar.try_commit(request, calendar.version)
        snapshot = calendar.snapshot()
        task = TaskSpec.create(
            task_id="candidate",
            arrival_time_sim=0.0,
            source_node="N",
            cpu_demand=1.5,
            execution_duration_sim=2.5,
            data_size_mb=0.0,
            bandwidth_demand_mbps=1.0,
            sla_type="Hard",
            latest_start_limit_sim=15.0,
        )
        starts = np.asarray((0.0, 1.0, 2.5, 4.0, 6.5, 9.0, 12.0))
        ends = starts + task.execution_duration_sim
        v3_evaluator = optimized.candidate_metric_evaluator(snapshot)
        actual = v3_evaluator.evaluate_batch(
            task=task,
            path=path,
            target_node="N",
            compute_start_sim=starts,
            compute_end_sim=ends,
            reservation_snapshot=snapshot,
        )
        expected = [
            baseline.evaluate_candidate(
                task=task,
                target_node="N",
                compute_start_sim=float(start),
                compute_end_sim=float(end),
                reservation_snapshot=snapshot,
            ).as_candidate_metrics()
            for start, end in zip(starts, ends)
        ]

        for key in (
            "system_cost_yuan",
            "green_coverage",
            "marginal_green_energy_mwh",
            "green_absorption_delta",
        ):
            np.testing.assert_allclose(
                actual[key],
                [row[key] for row in expected],
                rtol=0.0,
                atol=1e-12,
            )
        np.testing.assert_array_equal(
            actual["green_opportunity"],
            [row["green_opportunity"] for row in expected],
        )

        first_index = optimized._candidate_integral_index(
            task, "N", snapshot, 1.0, 3.5
        )
        second_index = optimized._candidate_integral_index(
            task, "N", snapshot, 9.0, 11.5
        )
        self.assertIs(first_index, second_index)


if __name__ == "__main__":
    unittest.main()
