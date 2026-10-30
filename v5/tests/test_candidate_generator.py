# 文件功能：验证 候选生成器 的行为、边界条件和回归约束。
# 主要内容：CandidateGeneratorTest。
# 使用入口：由 unittest 发现并执行，构造依赖后验证断言。
# 模块下游（项目依赖）：v1.accounting.forecast、v1.accounting.energy、v1.domain.models、v1.domain.reservations、v1.domain.units、v1.learning.candidate_dqn 等。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Focused pool/decision equivalence checks, without long simulation runs."""

import random
import unittest
from types import SimpleNamespace

import networkx as nx
from v1.accounting import ForecastSegment, LinearPowerModel, PiecewiseConstantForecast
from v1.domain.models import TaskSpec
from v1.domain.reservations import ReservationRequest, TimeInterval
from v1.domain.units import TimeConverter
from v1.learning.candidate_dqn import CandidateDQNPolicy, SharedCandidateQNetwork
from v1.learning.features import CandidateFeatureConfig, CandidateFeatureEncoder
from v1.profiling import TrainingPerformanceProfiler
from v1.scheduler.path_provider import StaticPathProvider
from v1.scheduler.resource_calendar import ReservationCalendar
from v1.scheduler.transmission import TransmissionModel, build_path_spec
from v3.accounting import SweepLineEnergyAccounting
from v3.scheduler import V3CandidateGenerator
from v4.gating import NetBenefitWaitGate, WaitGateConfig, default_objective_config
from v4.policy import NetBenefitGatePolicy
from v5.scheduler import V5CandidateGenerator
from v5.v5_runtime import create_v5_runtime


class CandidateGeneratorTest(unittest.TestCase):
    # 作用：验证 pool_metrics_and_gated_dqn_decisions_match_with_less_work 场景，覆盖 候选生成器 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：ReservationCalendar、build_path_spec、PiecewiseConstantForecast.tariff_yuan_per_mwh、PiecewiseConstantForecast.green_power_mw、SweepLineEnergyAccounting 等。
    def test_pool_metrics_and_gated_dqn_decisions_match_with_less_work(self):
        graph = nx.Graph()
        graph.add_edge("S", "N", capacity=100.0, distance_km=0.0, cost=1.0)
        graph.add_edge("S", "small", capacity=100.0, distance_km=0.0, cost=1.0)
        nodes = ("S", "N", "small")
        calendar = ReservationCalendar(
            {"S": 10.0, "N": 10.0, "small": 0.5},
            {("S", "N"): 100.0, ("S", "small"): 100.0},
        )
        calendar.try_commit(ReservationRequest(
            task_id="existing", committed_candidate_id="existing-candidate",
            committed_at_sim=0.0, reservation_snapshot_version=calendar.version,
            target_node="N", path=build_path_spec(graph, ["N"]),
            transmission_interval_sim=None, compute_interval_sim=TimeInterval(2.0, 5.0),
            bandwidth_amount_mbps=1.0, cpu_amount=9.0,
        ), calendar.version)
        snapshot = calendar.snapshot()
        converter = TimeConverter(3600.0)
        tariff = PiecewiseConstantForecast.tariff_yuan_per_mwh((
            ForecastSegment(TimeInterval(0.0, 8.0), 250.0),
            ForecastSegment(TimeInterval(8.0, 40.0), 80.0),
        ))
        green = PiecewiseConstantForecast.green_power_mw((
            ForecastSegment(TimeInterval(0.0, 10.0), 0.5),
            ForecastSegment(TimeInterval(10.0, 40.0), 3.0),
        ))
        accounting = SweepLineEnergyAccounting(
            converter, LinearPowerModel(1.0),
            {node: tariff for node in nodes}, {node: green for node in nodes},
        )
        generators = [kind(
            nodes, 1.0, StaticPathProvider(graph), TransmissionModel(converter, 200000.0),
            calendar, candidate_mode="layered_pool",
            pool_max_by_sla={sla: 8 for sla in ("Hard", "Soft", "Flexible")},
            pool_node_limit_by_sla={sla: 3 for sla in ("Hard", "Soft", "Flexible")},
        ) for kind in (V3CandidateGenerator, V5CandidateGenerator)]
        encoder = CandidateFeatureEncoder(
            {node: index for index, node in enumerate(nodes)},
            CandidateFeatureConfig(30.0, 1000.0, 1.0, 10.0, 100.0),
        )
        network = SharedCandidateQNetwork(2, encoder.feature_dim, hidden_dim=8)
        for sla in ("Hard", "Soft", "Flexible"):
            with self.subTest(sla=sla):
                task = TaskSpec.create(
                    task_id="task-" + sla, arrival_time_sim=0.0, source_node="S",
                    cpu_demand=2.0, execution_duration_sim=2.0, data_size_mb=100.0,
                    bandwidth_demand_mbps=10.0, sla_type=sla,
                    latest_start_limit_sim=20.0 if sla == "Hard" else None,
                    preferred_start_limit_sim=None if sla == "Hard" else 20.0,
                )
                streams, counts = [], []
                for generator in generators:
                    generator.profiler = TrainingPerformanceProfiler()
                    streams.append(generator.prepare_stream(
                        task, 0.0, reservation_snapshot=snapshot,
                        forecast_covered_until_sim=25.0,
                        metric_evaluator=accounting.candidate_metric_evaluator(snapshot),
                    ))
                    counts.append(generator.profiler.counters["candidate_metric_evaluation_count"])
                expected, actual = streams
                self.assertEqual(expected.status, actual.status)
                self.assertEqual(expected.theoretical_slot_count, actual.theoretical_slot_count)
                self.assertEqual(tuple(expected.iter_candidates()), tuple(actual.iter_candidates()))
                self.assertLess(counts[1], counts[0])
                self.assertGreater(generators[1].last_generation_stats["record_cache_hits"], 0)
                self.assertGreater(generators[1].last_generation_stats["static_cpu_rejections"], 0)
                self.assertIsNone(generators[1]._generation_cache)
                # Check both greedy inference and seeded exploration through
                # the unchanged V4 gate, including its earliest fallback.
                for epsilon in (0.0, 1.0):
                    selections, diagnostics = [], []
                    for stream in streams:
                        policy = NetBenefitGatePolicy(
                            CandidateDQNPolicy(network, encoder, lambda _: (0.0, 0.0),
                                               epsilon=epsilon, random_seed=7),
                            NetBenefitWaitGate(default_objective_config(), WaitGateConfig()),
                        )
                        selections.append(policy.select_complete_stream(stream, task))
                        diagnostics.append(policy.pop_gate_diagnostic(task.task_id))
                    self.assertEqual(selections[0].selected_candidate, selections[1].selected_candidate)
                    self.assertEqual(selections[0].candidate_set_hash, selections[1].candidate_set_hash)
                    self.assertEqual(diagnostics[0], diagnostics[1])

    # 作用：验证 layer_quotas_and_pareto_ties_match_original 场景，覆盖 候选生成器 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：rng.randrange、V3CandidateGenerator.__new__、V5CandidateGenerator.__new__、baseline._select_layered_pool、optimized._select_layered_pool 等。
    def test_layer_quotas_and_pareto_ties_match_original(self):
        rng = random.Random(42)
        items = tuple(SimpleNamespace(
            candidate_id=f"candidate-{index:03}", target_node="N",
            path=SimpleNamespace(path_id="path"),
            compute_start_sim=float(index % 12), compute_end_sim=float(index % 12 + 1),
            estimated_candidate_marginal_system_cost_yuan=float(rng.randrange(5)),
            estimated_green_coverage=rng.randrange(4) / 4.0,
            estimated_green_absorption_delta=0.0,
            preferred_start_tardiness_ratio=rng.randrange(3) / 3.0,
            projected_node_utilization=rng.randrange(3) / 3.0,
            projected_path_peak_utilization=0.0,
        ) for index in range(150))
        baseline = V3CandidateGenerator.__new__(V3CandidateGenerator)
        optimized = V5CandidateGenerator.__new__(V5CandidateGenerator)
        for limit in (8, 32, 200):
            self.assertEqual(
                baseline._select_layered_pool(items, limit),
                optimized._select_layered_pool(items, limit),
            )
        self.assertEqual(optimized._pareto_candidates((items[0], items[0])), (items[0], items[0]))

    # 作用：验证 runtime_installs_v5_generator_and_preserves_v4_gate 场景，覆盖 候选生成器 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：create_v5_runtime。
    def test_runtime_installs_v5_generator_and_preserves_v4_gate(self):
        runtime = create_v5_runtime(policy_name="earliest_feasible", forecast_end_sim=30.0)
        self.assertIsInstance(runtime.scheduler.candidate_generator, V5CandidateGenerator)
        self.assertIsInstance(runtime.scheduler.policy, NetBenefitGatePolicy)
        self.assertTrue(runtime.config_view_overrides["V4_WAIT_GATE_ENABLED"])
        self.assertTrue(runtime.config_view_overrides["V5_DECISION_LOCAL_CANDIDATE_CACHE"])


if __name__ == "__main__":
    unittest.main()
