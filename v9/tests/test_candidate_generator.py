"""Small deterministic regressions; no simulation or benchmark runs."""

import unittest
from dataclasses import replace

import networkx as nx

from v1.accounting import ForecastSegment, LinearPowerModel, PiecewiseConstantForecast
from v1.domain.models import TaskSpec
from v1.domain.reservations import ReservationRequest, TimeInterval
from v1.domain.candidates import CandidateGenerationStatus
from v1.domain.units import TimeConverter
from v1.scheduler.resource_calendar import ReservationCalendar
from v1.scheduler.transmission import TransmissionModel, build_path_spec
from v7.scheduler import CachedStaticPathProvider
from v6.accounting import V6EnergyAccounting
from v8.scheduler import V8CandidateGenerator
from v9.scheduler import V9CandidateGenerator


class V9SearchTest(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edge("S", "N", capacity=100, cost=1, distance_km=1)
        self.calendar = ReservationCalendar({"S": 10, "N": 10}, {("S", "N"): 100})

    def generator(self, stage=1, mode="layered_pool", samples=8):
        return V9CandidateGenerator(
            ("N",), 1, CachedStaticPathProvider(self.graph),
            TransmissionModel(TimeConverter(3600), 200000), self.calendar,
            candidate_mode=mode, search_stage=stage,
            pool_time_samples_by_sla={s: samples for s in ("Hard", "Soft", "Flexible")},
        )

    def accounting(self, green_start=6):
        tariff = PiecewiseConstantForecast.tariff_yuan_per_mwh((
            ForecastSegment(TimeInterval(0, green_start), 100),
            ForecastSegment(TimeInterval(green_start, 30), 10),
        ))
        green = PiecewiseConstantForecast.green_power_mw((
            ForecastSegment(TimeInterval(0, green_start), 0),
            ForecastSegment(TimeInterval(green_start, 30), 20),
        ))
        return V6EnergyAccounting(TimeConverter(3600), LinearPowerModel(1),
                                  {"N": tariff}, {"N": green})

    @staticmethod
    def task(latest=10, sla="Hard"):
        return TaskSpec.create(
            task_id=f"task-{latest}-{sla}", arrival_time_sim=0, source_node="S",
            cpu_demand=10, execution_duration_sim=1, data_size_mb=1,
            bandwidth_demand_mbps=10, sla_type=sla,
            latest_start_limit_sim=latest if sla == "Hard" else None,
            preferred_start_limit_sim=None if sla == "Hard" else latest / 2,
        )

    def reserve_cpu(self, end):
        self.calendar.try_commit(ReservationRequest(
            task_id="busy", committed_candidate_id="busy", committed_at_sim=0,
            reservation_snapshot_version=self.calendar.version, target_node="N",
            path=build_path_spec(self.graph, ["N"]), transmission_interval_sim=None,
            compute_interval_sim=TimeInterval(0, end), bandwidth_amount_mbps=1,
            cpu_amount=10,
        ), self.calendar.version)

    def test_future_release_and_wait_earliest_are_feasible(self):
        self.reserve_cpu(6)
        for stage in (1, 2, 3):
            with self.subTest(stage=stage):
                generator = self.generator(stage)
                stream = generator.prepare_stream(self.task(), 0,
                    reservation_snapshot=self.calendar.snapshot())
                records = tuple(stream.iter_candidate_records())
                self.assertTrue(records)
                self.assertEqual(stream.earliest_compute_start_sim,
                                 min(r.compute_start_sim for r in records))
                self.assertGreaterEqual(records[0].compute_start_sim, 6)
                self.assertLessEqual(generator.last_search_stats["time_probes"],
                                     generator.last_search_stats["time_probe_budget"])

    def test_short_window_and_complete_mode(self):
        for mode in ("layered_pool", "complete"):
            generator = self.generator(mode=mode)
            stream = generator.prepare_stream(self.task(latest=1), 0,
                reservation_snapshot=self.calendar.snapshot())
            self.assertTrue(tuple(stream.iter_candidate_records()))

    def test_complete_matches_v8_reference(self):
        task = self.task(latest=4)
        snapshot = self.calendar.snapshot()
        baseline = V8CandidateGenerator(
            ("N",), 1, CachedStaticPathProvider(self.graph),
            TransmissionModel(TimeConverter(3600), 200000), self.calendar,
            candidate_mode="complete",
        )
        optimized = self.generator(stage=3, mode="complete")
        old = baseline.prepare_stream(task, 0, reservation_snapshot=snapshot)
        new = optimized.prepare_stream(task, 0, reservation_snapshot=snapshot)
        self.assertEqual(tuple(old.iter_candidate_records()),
                         tuple(new.iter_candidate_records()))

    def test_retry_uses_new_snapshot(self):
        generator = self.generator()
        task = self.task()
        before = generator.prepare_stream(task, 0,
            reservation_snapshot=self.calendar.snapshot())
        self.reserve_cpu(6)
        after = generator.prepare_stream(task, 0,
            reservation_snapshot=self.calendar.snapshot())
        self.assertLess(before.earliest_compute_start_sim,
                        after.earliest_compute_start_sim)

    def test_energy_preview_uses_full_interval_and_existing_load(self):
        generator = self.generator(stage=2)
        generator.energy_accounting = self.accounting()
        task = self.task()
        snapshot = self.calendar.snapshot()
        context = generator.prepare_stream(task, 0, reservation_snapshot=snapshot,
            metric_evaluator=generator.energy_accounting.candidate_metric_evaluator(snapshot))
        self.assertTrue(tuple(context.iter_candidate_records()))
        self.assertGreater(generator.last_search_stats["energy_preview_count"], 0)
        self.assertGreater(generator.last_search_stats["exact_energy_count"], 0)
        # The interval [5, 6) has no green; [6, 7) does.
        before = generator.energy_accounting.evaluate_candidate(
            task=task, target_node="N", compute_start_sim=5,
            compute_end_sim=6, reservation_snapshot=snapshot)
        after = generator.energy_accounting.evaluate_candidate(
            task=task, target_node="N", compute_start_sim=6,
            compute_end_sim=7, reservation_snapshot=snapshot)
        self.assertGreater(after.candidate_marginal_green_energy_mwh,
                           before.candidate_marginal_green_energy_mwh)
        self.reserve_cpu(8)
        busy_snapshot = self.calendar.snapshot()
        blocked = generator.prepare_stream(task, 0, reservation_snapshot=busy_snapshot,
            metric_evaluator=generator.energy_accounting.candidate_metric_evaluator(busy_snapshot))
        self.assertTrue(all(r.compute_start_sim >= 8 for r in blocked.iter_candidate_records()))

    def test_green_supply_already_consumed(self):
        accounting = self.accounting(green_start=2)
        task = self.task()
        # Keep CPU capacity available while the existing task consumes all
        # green power supplied over [2, 4).
        accounting.green_by_node["N"] = PiecewiseConstantForecast.green_power_mw((
            ForecastSegment(TimeInterval(0, 2), 0),
            ForecastSegment(TimeInterval(2, 30), 5),
        ))
        self.calendar.try_commit(ReservationRequest(
            task_id="half", committed_candidate_id="half", committed_at_sim=0,
            reservation_snapshot_version=self.calendar.version, target_node="N",
            path=build_path_spec(self.graph, ["N"]), transmission_interval_sim=None,
            compute_interval_sim=TimeInterval(2, 4), bandwidth_amount_mbps=1,
            cpu_amount=5,
        ), self.calendar.version)
        snapshot = self.calendar.snapshot()
        occupied = accounting.evaluate_candidate(task=task, target_node="N",
            compute_start_sim=2, compute_end_sim=3, reservation_snapshot=snapshot)
        free = accounting.evaluate_candidate(task=task, target_node="N",
            compute_start_sim=5, compute_end_sim=6, reservation_snapshot=snapshot)
        self.assertEqual(occupied.candidate_marginal_green_energy_mwh, 0)
        self.assertGreater(free.candidate_marginal_green_energy_mwh, 0)

    def test_sla_grid_budget_and_no_duplicate_candidates(self):
        for sla in ("Hard", "Soft", "Flexible"):
            for stage in (1, 2, 3):
                with self.subTest(sla=sla, stage=stage):
                    generator = self.generator(stage=stage, samples=4)
                    stream = generator.prepare_stream(self.task(latest=8, sla=sla), 0,
                        reservation_snapshot=self.calendar.snapshot())
                    records = tuple(stream.iter_candidate_records())
                    self.assertEqual(len(records), len({r.candidate_id for r in records}))
                    self.assertLessEqual(len(records),
                                         generator.last_search_stats["candidate_pool_capacity"])
                    self.assertTrue(all(r.compute_start_sim <= 8 for r in records))
                    self.assertLessEqual(generator.last_search_stats["time_probes"],
                                         generator.last_search_stats["time_probe_budget"])

    def test_budget_exhaustion_is_not_proof_of_infeasibility(self):
        self.reserve_cpu(100)
        generator = self.generator(stage=3, samples=4)
        stream = generator.prepare_stream(self.task(latest=30), 0,
            reservation_snapshot=self.calendar.snapshot())
        self.assertEqual(stream.status, CandidateGenerationStatus.EMPTY_PHYSICAL)
        self.assertEqual(generator.last_search_stats["feasible_nodes"], 0)
        self.assertGreater(generator.last_search_stats["unresolved_nodes"], 0)
        self.assertLessEqual(generator.last_search_stats["time_probes"],
                             generator.last_search_stats["time_probe_budget"])

    def test_bandwidth_release_and_check_counters(self):
        path = build_path_spec(self.graph, ["S", "N"])
        self.calendar.try_commit(ReservationRequest(
            task_id="link", committed_candidate_id="link", committed_at_sim=0,
            reservation_snapshot_version=self.calendar.version, target_node="N",
            path=path, transmission_interval_sim=TimeInterval(0, 6),
            compute_interval_sim=TimeInterval(6, 7), bandwidth_amount_mbps=100,
            cpu_amount=1,
        ), self.calendar.version)
        generator = self.generator(stage=1)
        stream = generator.prepare_stream(self.task(), 0,
            reservation_snapshot=self.calendar.snapshot())
        self.assertTrue(tuple(stream.iter_candidate_records()))
        self.assertGreater(generator.last_search_stats["cpu_checks"], 0)
        self.assertGreater(generator.last_search_stats["bandwidth_checks"], 0)

    def test_forecast_version_and_duration_boundary_are_not_cached(self):
        generator = self.generator(stage=2)
        generator.energy_accounting = self.accounting(green_start=6)
        task = replace(self.task(), execution_duration_sim=2)
        snapshot = self.calendar.snapshot()
        first = generator.prepare_stream(task, 0, reservation_snapshot=snapshot,
            forecast_version="forecast-a", forecast_covered_until_sim=12,
            metric_evaluator=generator.energy_accounting.candidate_metric_evaluator(snapshot))
        first_ids = {r.candidate_id for r in first.iter_candidate_records()}
        second = generator.prepare_stream(task, 0, reservation_snapshot=snapshot,
            forecast_version="forecast-b", forecast_covered_until_sim=8,
            metric_evaluator=generator.energy_accounting.candidate_metric_evaluator(snapshot))
        second_records = tuple(second.iter_candidate_records())
        self.assertTrue(first_ids)
        self.assertTrue(second_records)
        self.assertTrue(first_ids.isdisjoint(r.candidate_id for r in second_records))
        self.assertTrue(all(r.compute_end_sim <= 8 for r in second_records))
        self.assertTrue(all(r.compute_start_sim == int(r.compute_start_sim)
                            for r in second_records))


if __name__ == "__main__":
    unittest.main()
