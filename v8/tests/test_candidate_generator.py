"""Experimental static admission must preserve V7 candidates and decisions."""

from dataclasses import replace
import pickle
import unittest

import networkx as nx

from v1.domain.models import TaskSpec
from v1.domain.reservations import ReservationRequest, TimeInterval
from v1.domain.units import TimeConverter
from v1.scheduler.resource_calendar import ReservationCalendar
from v1.scheduler.transmission import TransmissionModel, build_path_spec
from v7.scheduler import CachedStaticPathProvider, V7CandidateGenerator
from v8.scheduler import V8CandidateGenerator


class StaticHardPruningTest(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        for node in ("S", "N", "small", "blocked"):
            self.graph.add_node(node)
        for node, bandwidth in (("N", 100), ("small", 100), ("blocked", 1)):
            self.graph.add_edge("S", node, capacity=bandwidth, cost=1, distance_km=1)
        self.calendar = ReservationCalendar(
            {"S": 10, "N": 10, "small": 1, "blocked": 10},
            {tuple(edge): self.graph.edges[edge]["capacity"] for edge in self.graph.edges},
        )

    def generator(self, kind, mode="layered_pool"):
        return kind(
            ("small", "N", "blocked", "S"), 1,
            CachedStaticPathProvider(self.graph),
            TransmissionModel(TimeConverter(3600), 200000), self.calendar,
            candidate_mode=mode,
            pool_max_by_sla={sla: 8 for sla in ("Hard", "Soft", "Flexible")},
        )

    @staticmethod
    def task(sla="Hard", cpu=2, latest=8, source="S"):
        return TaskSpec.create(
            task_id=f"{sla}-{cpu}-{latest}-{source}", arrival_time_sim=0,
            source_node=source, cpu_demand=cpu, execution_duration_sim=1,
            data_size_mb=1, bandwidth_demand_mbps=10, sla_type=sla,
            latest_start_limit_sim=latest if sla == "Hard" else None,
            preferred_start_limit_sim=None if sla == "Hard" else latest,
        )

    def assert_equal_streams(self, task, now, mode="layered_pool"):
        old = self.generator(V7CandidateGenerator, mode)
        new = self.generator(V8CandidateGenerator, mode)
        snapshot = self.calendar.snapshot()
        a = old.prepare_stream(task, now, reservation_snapshot=snapshot)
        b = new.prepare_stream(task, now, reservation_snapshot=snapshot)
        self.assertEqual(a.status, b.status)
        self.assertEqual(tuple(a.iter_candidates()), tuple(b.iter_candidates()))
        self.assertEqual(tuple(a.iter_candidate_records()), tuple(b.iter_candidate_records()))
        self.assertLessEqual(b.theoretical_slot_count, a.theoretical_slot_count)
        return old, new, a, b

    def test_cpu_capacity_boundary_and_path_queries(self):
        for cpu in (1, 1 + 5e-10, 2, 10, 10 + 5e-10, 11):
            with self.subTest(cpu=cpu):
                old, new, a, b = self.assert_equal_streams(self.task(cpu=cpu), 0)
                stats = new.path_search_statistics()["static_hard_pruning"]
                self.assertEqual(stats["path_queries_avoided"], stats["static_cpu_nodes_rejected"])
                self.assertEqual(
                    old.path_provider.stats["path_cache_misses"] - new.path_provider.stats["path_cache_misses"],
                    stats["path_queries_avoided"],
                )
                self.assertEqual(stats["theoretical_slot_count_scope"], "post_static_hard_pruning_search_slots")
                if cpu == 2:
                    self.assertGreater(a.theoretical_slot_count, b.theoretical_slot_count)

    def test_sla_boundary_no_path_local_path_all_slas_and_complete(self):
        for mode in ("layered_pool", "complete"):
            for sla in ("Hard", "Soft", "Flexible"):
                for latest, now, source in (
                    (0, 0, "S"), (1, 0, "S"), (8, 7, "S"),
                    (8, 0, "blocked"), (8, 0, "isolated"),
                ):
                    with self.subTest(mode=mode, sla=sla, latest=latest, now=now, source=source):
                        # Non-Hard preferred limit must be positive.
                        if sla != "Hard" and latest == 0:
                            continue
                        self.assert_equal_streams(self.task(sla, latest=latest, source=source), now, mode)

    def test_retry_topology_and_tolerance_invalidation(self):
        old = self.generator(V7CandidateGenerator)
        new = self.generator(V8CandidateGenerator)
        task = self.task()
        for now in (0, 3, 8, 10):
            self.assertEqual(
                [g for g in old._declared_path_grids(task, now)
                 if g[0] != "small" and g[3] <= g[4]],
                list(new._declared_path_grids(task, now)),
            )
        self.assertEqual(new._grid_stats["task_grid_hits"], 3)
        changed = self.graph.copy()
        changed["S"]["N"]["capacity"] = 5
        old.path_provider.replace_topology(changed)
        new.path_provider.replace_topology(changed)
        self.assertEqual(
            [g for g in old._declared_path_grids(task, 0) if g[0] != "small" and g[3] <= g[4]],
            list(new._declared_path_grids(task, 0)),
        )
        self.assertEqual(new._grid_stats["task_grid_misses"], 1)
        self.assertEqual(list(new._declared_path_grids(replace(task, data_size_mb=2), 0)),
                         list(new._declared_path_grids(replace(task, data_size_mb=2), 0)))
        old.active_wait_enabled = new.active_wait_enabled = False
        self.assertEqual(list(new._declared_path_grids(task, 10)), [])
        boundary = self.task(cpu=10 + 5e-10)
        list(new._declared_path_grids(boundary, 0))
        old.time_tolerance = new.time_tolerance = 0
        self.assertEqual(list(new._declared_path_grids(boundary, 0)), [])
        self.assertEqual(new._grid_stats["task_grid_misses"], 1)
        restored = pickle.loads(pickle.dumps(new))
        self.assertEqual(list(new._declared_path_grids(task, 0)), list(restored._declared_path_grids(task, 0)))

    def test_future_cpu_release_is_not_pruned(self):
        self.calendar.try_commit(ReservationRequest(
            task_id="busy", committed_candidate_id="busy", committed_at_sim=0,
            reservation_snapshot_version=self.calendar.version, target_node="N",
            path=build_path_spec(self.graph, ["N"]), transmission_interval_sim=None,
            compute_interval_sim=TimeInterval(0, 2), bandwidth_amount_mbps=1, cpu_amount=10,
        ), self.calendar.version)
        _, _, baseline, optimized = self.assert_equal_streams(self.task(latest=8), 0, "complete")
        self.assertTrue(any(c.target_node == "N" and c.compute_start_sim >= 2 for c in optimized.iter_candidates()))
        self.assertEqual(tuple(baseline.iter_candidates()), tuple(optimized.iter_candidates()))


if __name__ == "__main__":
    unittest.main()
