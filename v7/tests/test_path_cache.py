"""Path identity/order, capacity thresholds, invalidation, bounds and reuse."""

from dataclasses import replace
import pickle
import random
import unittest
from unittest.mock import patch

import networkx as nx

from v1.accounting import ForecastSegment, LinearPowerModel, PiecewiseConstantForecast
from v1.domain.models import TaskSpec
from v1.domain.reservations import ReservationRequest, TimeInterval
from v1.domain.units import TimeConverter
from v1.learning.candidate_dqn import CandidateDQNPolicy, SharedCandidateQNetwork
from v1.learning.features import CandidateFeatureConfig, CandidateFeatureEncoder
from v1.scheduler.path_provider import StaticPathProvider
from v1.scheduler.resource_calendar import ReservationCalendar
from v1.scheduler.transmission import TransmissionModel, build_path_spec
from v4.gating import NetBenefitWaitGate, WaitGateConfig, default_objective_config
from v4.policy import NetBenefitGatePolicy
from v5.scheduler import V5CandidateGenerator
from v6.accounting import V6EnergyAccounting
from v6.v6_runtime import create_v6_runtime
from v7._entrypoint import use_v7_runtime
from v7.scheduler import CachedStaticPathProvider, V7CandidateGenerator
from v7.v7_runtime import create_v7_runtime


def edge(graph, u, v, capacity=100.0, cost=1.0):
    graph.add_edge(u, v, capacity=capacity, cost=cost, distance_km=1.0)


class PathCacheTest(unittest.TestCase):
    def test_cache_reuses_pathspecs_and_capacity_equivalence_classes(self):
        graph = nx.Graph()
        edge(graph, "S", "T")
        edge(graph, "S", "U")
        provider = CachedStaticPathProvider(graph)
        first = provider.candidate_paths("S", "T", 10)
        again = provider.candidate_paths("S", "T", 80)
        self.assertIs(first[0], again[0])
        self.assertEqual(provider.stats["path_cache_hits"], 1)
        again.clear()
        self.assertEqual(len(provider.candidate_paths("S", "T", 80)), 1)
        provider.candidate_paths("S", "U", 10)
        self.assertEqual(provider.stats["filtered_graph_builds"], 1)
        self.assertEqual(provider.stats["single_source_runs"], 1)
        self.assertEqual(provider.candidate_paths("S", "T", 100.00001), [])
        self.assertEqual(provider.candidate_paths("S", "T", 101), [])
        self.assertEqual(provider.stats["negative_cache_hits"], 1)
        self.assertEqual(provider.candidate_paths("missing", "missing", 10), [])
        self.assertEqual(provider.candidate_paths("S", "S", 10000)[0].ordered_nodes, ("S",))

    def test_equal_shortest_paths_fall_back_to_original_tie_order(self):
        graph = nx.Graph()
        for u, v in [("S", "A"), ("S", "B"), ("B", "T"), ("A", "T"), ("A", "B")]:
            edge(graph, u, v)
        baseline = StaticPathProvider(graph)
        expected = baseline.candidate_paths("S", "T", 10)
        provider = CachedStaticPathProvider(graph)
        self.assertEqual(provider.candidate_paths("S", "T", 10), expected)
        self.assertGreater(provider.stats["tie_or_unsafe_fallbacks"], 0)
        self.assertEqual(provider.stats["path_searches"], 1)

    def test_float_and_zero_weights_and_k_paths_keep_original_algorithm(self):
        for cost in (0.0, 0.1, 1.0):
            for k in (1, 2, 3):
                graph = nx.Graph()
                edge(graph, "S", "A", cost=cost)
                edge(graph, "A", "T", cost=1)
                edge(graph, "S", "T", cost=4)
                old = StaticPathProvider(graph, k)
                new = CachedStaticPathProvider(graph, k)
                self.assertEqual(new.candidate_paths("S", "T", 10), old.candidate_paths("S", "T", 10))
                if cost != 1 or k > 1:
                    self.assertEqual(new.stats["single_source_runs"], 0)

    def test_random_graphs_match_baseline_including_bandwidth_boundaries(self):
        rng = random.Random(42)
        for directed in (False, True):
            for case in range(12):
                graph = nx.DiGraph() if directed else nx.Graph()
                nodes = list("ABCDE")
                rng.shuffle(nodes)
                graph.add_nodes_from(nodes)
                pairs = [(u, v) for u in nodes for v in nodes if u != v and (directed or u < v)]
                rng.shuffle(pairs)
                for u, v in pairs:
                    if rng.random() < 0.65:
                        edge(graph, u, v, rng.choice((20, 50, 100)), rng.choice((1, 2, 3)) * (0.1 if case % 2 else 1))
                for k in (1, 2):
                    old, new = StaticPathProvider(graph, k), CachedStaticPathProvider(graph, k)
                    for source in nodes:
                        for target in nodes:
                            for bandwidth in (10, 20, 20.00001, 50, 100, 101):
                                self.assertEqual(
                                    new.candidate_paths(source, target, bandwidth),
                                    old.candidate_paths(source, target, bandwidth),
                                    (directed, case, k, source, target, bandwidth),
                                )

    def test_topology_update_lru_and_pickle_clear_transient_state(self):
        graph = nx.Graph()
        edge(graph, "S", "T")
        edge(graph, "S", "U")
        provider = CachedStaticPathProvider(graph, path_capacity=1, graph_capacity=1, source_capacity=1)
        provider.candidate_paths("S", "T", 10)
        provider.candidate_paths("S", "U", 200)
        self.assertLessEqual(len(provider._paths), 1)
        self.assertLessEqual(len(provider._graphs), 1)
        # External mutation does not change the private snapshot; explicit
        # replacement atomically switches all cached topology-derived values.
        graph["S"]["T"]["capacity"] = 5
        graph["S"]["T"]["distance_km"] = 9
        provider.replace_topology(graph)
        self.assertEqual(provider.topology_revision, 1)
        self.assertEqual(provider.candidate_paths("S", "T", 10), [])
        self.assertEqual(provider.candidate_paths("S", "T", 1)[0].total_distance_km, 9)
        restored = pickle.loads(pickle.dumps(provider))
        self.assertEqual(len(restored._paths), 0)
        self.assertEqual(len(restored._graphs), 0)
        self.assertEqual(restored.candidate_paths("S", "T", 1), provider.candidate_paths("S", "T", 1))


class GridAndDecisionTest(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        for u, v in [("S", "A"), ("S", "B"), ("B", "N"), ("A", "N"), ("S", "small")]:
            edge(self.graph, u, v)
        self.nodes = ("N", "small")
        self.calendar = ReservationCalendar({"N": 10, "small": 0.5}, {tuple(e): 100 for e in self.graph.edges})
        self.calendar.try_commit(ReservationRequest(
            task_id="existing", committed_candidate_id="existing", committed_at_sim=0,
            reservation_snapshot_version=self.calendar.version, target_node="N",
            path=build_path_spec(self.graph, ["N"]), transmission_interval_sim=None,
            compute_interval_sim=TimeInterval(2, 5), bandwidth_amount_mbps=1, cpu_amount=9,
        ), self.calendar.version)
        self.converter = TimeConverter(3600)

    def task(self, sla="Hard"):
        return TaskSpec.create(
            task_id="task-" + sla, arrival_time_sim=0, source_node="S", cpu_demand=2,
            execution_duration_sim=2, data_size_mb=1, bandwidth_demand_mbps=10,
            sla_type=sla, latest_start_limit_sim=20 if sla == "Hard" else None,
            preferred_start_limit_sim=None if sla == "Hard" else 20,
        )

    def generators(self):
        common = dict(candidate_mode="layered_pool", pool_max_by_sla={s: 8 for s in ("Hard", "Soft", "Flexible")})
        return [kind(
            self.nodes, 1, provider(self.graph), TransmissionModel(self.converter, 200000),
            self.calendar, **common,
        ) for kind, provider in ((V5CandidateGenerator, StaticPathProvider), (V7CandidateGenerator, CachedStaticPathProvider))]

    def test_retry_reuses_static_work_and_recomputes_time_bounds(self):
        old, new = self.generators()
        task = self.task()
        for now in (0, 3, 7):
            expected = list(old._declared_path_grids(task, now))
            with patch.object(new.transmission_model, "duration", wraps=new.transmission_model.duration) as duration:
                actual = list(new._declared_path_grids(task, now))
            self.assertEqual(actual, expected)
            self.assertEqual(duration.call_count, 0 if now else 2)
        self.assertEqual(new._grid_stats["task_grid_hits"], 2)
        # Waiting switch still changes the grid on a cache hit.
        old.active_wait_enabled = new.active_wait_enabled = False
        self.assertEqual(list(old._declared_path_grids(task, 8)), list(new._declared_path_grids(task, 8)))
        self.assertTrue(all(first == last for *_, first, last in new._declared_path_grids(task, 8)))
        restored = pickle.loads(pickle.dumps(new))
        self.assertFalse(hasattr(restored, "_static_grid_cache"))
        self.assertEqual(list(old._declared_path_grids(task, 8)), list(restored._declared_path_grids(task, 8)))

    def test_topology_replacement_and_changed_task_invalidate_retry_cache(self):
        _, new = self.generators()
        task = self.task()
        initial = list(new._declared_path_grids(task, 0))
        changed = self.graph.copy()
        changed["S"]["A"]["cost"] = 100
        new.path_provider.replace_topology(changed)
        old = StaticPathProvider(changed)
        actual = list(new._declared_path_grids(task, 0))
        self.assertEqual(actual[0][1], old.candidate_paths("S", "N", 10)[0])
        self.assertEqual(new._grid_stats["task_grid_misses"], 1)
        list(new._declared_path_grids(replace(task, data_size_mb=2), 0))
        self.assertEqual(new._grid_stats["task_grid_misses"], 2)
        self.assertTrue(initial)

    def test_pools_features_gated_decisions_and_hashes_match_all_slas(self):
        tariff = PiecewiseConstantForecast.tariff_yuan_per_mwh((
            ForecastSegment(TimeInterval(0, 8), 250), ForecastSegment(TimeInterval(8, 80), 80),
        ))
        green = PiecewiseConstantForecast.green_power_mw((
            ForecastSegment(TimeInterval(0, 10), 0.5), ForecastSegment(TimeInterval(10, 80), 3),
        ))
        accounting = V6EnergyAccounting(self.converter, LinearPowerModel(1),
                                      {n: tariff for n in self.nodes}, {n: green for n in self.nodes})
        snapshot = self.calendar.snapshot()
        encoder = CandidateFeatureEncoder({n: i for i, n in enumerate(self.nodes)}, CandidateFeatureConfig(60, 1000, 1, 10, 100))
        network = SharedCandidateQNetwork(2, encoder.feature_dim, hidden_dim=8)
        old, new = self.generators()
        for sla in ("Hard", "Soft", "Flexible"):
            task = self.task(sla)
            for now in (0, 3):
                streams = [g.prepare_stream(task, now, reservation_snapshot=snapshot,
                                           forecast_covered_until_sim=70,
                                           metric_evaluator=accounting.candidate_metric_evaluator(snapshot)) for g in (old, new)]
                self.assertEqual(tuple(streams[0].iter_candidates()), tuple(streams[1].iter_candidates()))
                for epsilon in (0, 1):
                    results = []
                    for stream in streams:
                        policy = NetBenefitGatePolicy(
                            CandidateDQNPolicy(network, encoder, lambda _: (0, 0), epsilon=epsilon, random_seed=7),
                            NetBenefitWaitGate(default_objective_config(), WaitGateConfig()),
                        )
                        results.append((policy.select_complete_stream(stream, task), policy.pop_gate_diagnostic(task.task_id)))
                    self.assertEqual(results[0], results[1])

    def test_runtime_isolation_and_entrypoint_restoration(self):
        import v1.evaluate_v1 as evaluation
        import v1.train_v1 as training
        v6 = create_v6_runtime(policy_name="earliest_feasible", forecast_end_sim=30)
        self.assertNotIsInstance(v6.scheduler.candidate_generator, V7CandidateGenerator)
        v7 = create_v7_runtime(policy_name="earliest_feasible", forecast_end_sim=30)
        self.assertIsInstance(v7.scheduler.candidate_generator, V7CandidateGenerator)
        self.assertIsInstance(v7.accounting, V6EnergyAccounting)
        baseline = create_v7_runtime(policy_name="earliest_feasible", forecast_end_sim=30, path_search_cache=False)
        self.assertNotIsInstance(baseline.scheduler.candidate_generator, V7CandidateGenerator)
        original = (evaluation.create_v1_runtime, training.create_v1_runtime, training._training_config_view)
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with use_v7_runtime():
                self.assertIs(evaluation.create_v1_runtime, create_v7_runtime)
                raise RuntimeError("expected")
        self.assertEqual(original, (evaluation.create_v1_runtime, training.create_v1_runtime, training._training_config_view))


if __name__ == "__main__":
    unittest.main()
