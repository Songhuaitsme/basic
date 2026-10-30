# 文件功能：验证 能源核算 的行为、边界条件和回归约束。
# 主要内容：EnergyTest。
# 使用入口：由 unittest 发现并执行，构造依赖后验证断言。
# 模块下游（项目依赖）：v1.accounting.forecast、v1.accounting.energy、v1.domain.models、v1.domain.reservations、v1.domain.units、v1.learning.candidate_dqn 等。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Small exactness, cache-lifecycle, and runtime integration checks."""

import pickle
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from v1.accounting import ForecastSegment, LinearPowerModel, PiecewiseConstantForecast
from v1.domain.models import TaskSpec
from v1.domain.reservations import ReservationRequest, TimeInterval
from v1.domain.units import TimeConverter
from v1.learning.candidate_dqn import CandidateDQNPolicy, SharedCandidateQNetwork
from v1.learning.features import CandidateFeatureConfig, CandidateFeatureEncoder
from v1.scheduler.path_provider import StaticPathProvider
from v1.scheduler.resource_calendar import ReservationCalendar, ReservationSnapshot
from v1.scheduler.transmission import TransmissionModel, build_path_spec
from v3.accounting import SweepLineEnergyAccounting
from v4.gating import NetBenefitWaitGate, WaitGateConfig, default_objective_config
from v4.policy import NetBenefitGatePolicy
from v5.scheduler import V5CandidateGenerator
from v6.accounting import V6EnergyAccounting
from v6.accounting.energy import V6CandidateIntegralIndex
from v6.v6_runtime import create_v6_runtime


class EnergyTest(unittest.TestCase):
    # 作用：为每个测试建立独立的依赖和初始数据。
    # 调用约定：由 unittest 在每个测试前调用。
    # 下游调用：build_path_spec、ReservationCalendar、PiecewiseConstantForecast.tariff_yuan_per_mwh、PiecewiseConstantForecast.green_power_mw、self.graph.add_node 等。
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_node("N")
        self.path = build_path_spec(self.graph, ["N"])
        self.calendar = ReservationCalendar({"N": 10.0}, {})
        for task_id, cpu, start, end in (
            ("a", 3.0, 1.0, 6.0), ("b", 2.0, 4.0, 9.0),
        ):
            self.calendar.try_commit(ReservationRequest(
                task_id=task_id, committed_candidate_id=task_id,
                committed_at_sim=0.0, reservation_snapshot_version=self.calendar.version,
                target_node="N", path=self.path, transmission_interval_sim=None,
                compute_interval_sim=TimeInterval(start, end),
                bandwidth_amount_mbps=1.0, cpu_amount=cpu,
            ), self.calendar.version)
        self.snapshot = self.calendar.snapshot()
        self.converter = TimeConverter(3600.0)
        self.tariff = PiecewiseConstantForecast.tariff_yuan_per_mwh((
            ForecastSegment(TimeInterval(0.0, 8.0), 100.0),
            ForecastSegment(TimeInterval(8.0, 40.0), 250.0),
        ))
        self.green = PiecewiseConstantForecast.green_power_mw((
            ForecastSegment(TimeInterval(0.0, 4.0), 0.0),
            ForecastSegment(TimeInterval(4.0, 10.0), 0.5),
            ForecastSegment(TimeInterval(10.0, 40.0), 5.0),
        ))

    # 作用：构造或处理 accounting 测试辅助数据，供本文件用例复用。
    # 上游（直接调用）：v6.tests.test_energy.EnergyTest.test_identity_cache_avoids_hash_and_refreshes_forecasts_snapshots、v6.tests.test_energy.EnergyTest.test_runtime_and_nonlinear_billing_keep_existing_semantics、v6.tests.test_energy.EnergyTest.test_scalar_batch_pool_and_gated_dqn_match_v5_exactly。
    # 下游调用：LinearPowerModel。
    def accounting(self, kind, **kwargs):
        return kind(self.converter, LinearPowerModel(1.0),
                    {"N": self.tariff}, {"N": self.green}, **kwargs)

    # 作用：构造或处理 task 测试辅助数据，供本文件用例复用。
    # 上游（直接调用）：v6.tests.test_energy.EnergyTest.test_identity_cache_avoids_hash_and_refreshes_forecasts_snapshots、v6.tests.test_energy.EnergyTest.test_runtime_and_nonlinear_billing_keep_existing_semantics、v6.tests.test_energy.EnergyTest.test_scalar_batch_pool_and_gated_dqn_match_v5_exactly。
    # 下游调用：TaskSpec.create。
    def task(self, sla="Hard"):
        return TaskSpec.create(
            task_id="task-" + sla, arrival_time_sim=0.0, source_node="N",
            cpu_demand=1.5, execution_duration_sim=1.5, data_size_mb=0.0,
            bandwidth_demand_mbps=1.0, sla_type=sla,
            latest_start_limit_sim=20.0 if sla == "Hard" else None,
            preferred_start_limit_sim=None if sla == "Hard" else 20.0,
        )

    # 作用：验证 scalar_batch_pool_and_gated_dqn_match_v5_exactly 场景，覆盖 能源核算 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：self.accounting、CandidateFeatureEncoder、SharedCandidateQNetwork、self.task、StaticPathProvider 等。
    def test_scalar_batch_pool_and_gated_dqn_match_v5_exactly(self):
        baseline = self.accounting(SweepLineEnergyAccounting)
        optimized = self.accounting(V6EnergyAccounting)
        encoder = CandidateFeatureEncoder(
            {"N": 0}, CandidateFeatureConfig(30.0, 1000.0, 1.0, 10.0, 100.0)
        )
        network = SharedCandidateQNetwork(2, encoder.feature_dim, hidden_dim=8)
        for sla in ("Hard", "Soft", "Flexible"):
            with self.subTest(sla=sla):
                task = self.task(sla)
                starts = np.asarray([0.0, 2.0, 4.0, 6.0, 10.0, task.absolute_latest_start_sim])
                kwargs = dict(task=task, target_node="N", path=self.path,
                              reservation_snapshot=self.snapshot)
                evaluators = [item.candidate_metric_evaluator(self.snapshot)
                              for item in (baseline, optimized)]
                expected, actual = [e.evaluate_batch(
                    **kwargs, compute_start_sim=starts,
                    compute_end_sim=starts + task.execution_duration_sim,
                ) for e in evaluators]
                for key in expected:
                    np.testing.assert_array_equal(expected[key], actual[key])
                for start in starts:
                    args = dict(kwargs, compute_start_sim=float(start),
                                compute_end_sim=float(start + task.execution_duration_sim))
                    self.assertEqual(evaluators[0](**args), evaluators[1](**args))
                streams = []
                for evaluator in evaluators:
                    generator = V5CandidateGenerator(
                        ("N",), 1.0, StaticPathProvider(self.graph),
                        TransmissionModel(self.converter, 200000.0), self.calendar,
                        candidate_mode="layered_pool",
                        pool_max_by_sla={key: 8 for key in ("Hard", "Soft", "Flexible")},
                    )
                    streams.append(generator.prepare_stream(
                        task, 0.0, reservation_snapshot=self.snapshot,
                        forecast_covered_until_sim=40.0, metric_evaluator=evaluator,
                    ))
                self.assertEqual(tuple(streams[0].iter_candidates()), tuple(streams[1].iter_candidates()))
                selections = []
                for stream in streams:
                    policy = NetBenefitGatePolicy(
                        CandidateDQNPolicy(network, encoder, lambda _: (0.0, 0.0)),
                        NetBenefitWaitGate(default_objective_config(), WaitGateConfig()),
                    )
                    selections.append(policy.select_complete_stream(stream, task))
                self.assertEqual(selections[0].selected_candidate, selections[1].selected_candidate)
                self.assertEqual(selections[0].candidate_set_hash, selections[1].candidate_set_hash)

    # 作用：验证 identity_cache_avoids_hash_and_refreshes_forecasts_snapshots 场景，覆盖 能源核算 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：self.accounting、self.task、PiecewiseConstantForecast.tariff_yuan_per_mwh、optimized.candidate_metric_evaluator、self.accounting(SweepLineEnergyAccounting).candidate_metric_evaluator(self.snapshot) 等。
    def test_identity_cache_avoids_hash_and_refreshes_forecasts_snapshots(self):
        optimized = self.accounting(V6EnergyAccounting)
        task = self.task()
        evaluator = optimized.candidate_metric_evaluator(self.snapshot)
        kwargs = dict(task=task, path=self.path, target_node="N",
                      compute_start_sim=4.0, compute_end_sim=5.5)
        expected = self.accounting(SweepLineEnergyAccounting).candidate_metric_evaluator(
            self.snapshot
        )(**kwargs)
        with patch.object(ReservationSnapshot, "__hash__", side_effect=AssertionError("snapshot hashed")):
            self.assertEqual(evaluator(**kwargs), expected)
            evaluator.evaluate_batch(**dict(kwargs, compute_start_sim=[4.0], compute_end_sim=[5.5]))
        index = optimized._candidate_integral_index(task, "N", self.snapshot, 4.0, 5.5)
        self.assertIs(index, optimized._candidate_integral_index(task, "N", self.snapshot, 10.0, 11.5))
        self.assertIsInstance(index, V6CandidateIntegralIndex)
        for array in (index._boundaries_array, index._rates_array, index._prefixes_array):
            self.assertFalse(array.flags.writeable)
        # Matching numeric versions do not mean matching physical allocations.
        empty = ReservationSnapshot(self.snapshot.reservation_version, (), ())
        changed = evaluator(**dict(kwargs, reservation_snapshot=empty))
        fresh = self.accounting(SweepLineEnergyAccounting).candidate_metric_evaluator(empty)(**kwargs)
        self.assertEqual(changed, fresh)
        self.assertNotEqual(changed["green_coverage"], expected["green_coverage"])
        replacement = PiecewiseConstantForecast.tariff_yuan_per_mwh((
            ForecastSegment(TimeInterval(0.0, 40.0), 500.0),
        ), version=self.tariff.version)
        optimized.tariff_by_node["N"] = replacement
        self.assertNotEqual(evaluator(**kwargs)["system_cost_yuan"], expected["system_cost_yuan"])
        restored = pickle.loads(pickle.dumps(optimized))
        self.assertEqual(len(restored._candidate_index_cache), 0)
        self.assertEqual(len(restored._snapshot_allocations_by_node_cache), 0)
        self.assertEqual(restored.candidate_metric_evaluator(self.snapshot)(**kwargs), evaluator(**kwargs))

    # 作用：验证 runtime_and_nonlinear_billing_keep_existing_semantics 场景，覆盖 能源核算 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：create_v6_runtime、self.accounting、self.task、self.accounting(kind, node_bill_rate_model=bill).candidate_metric_evaluator。
    def test_runtime_and_nonlinear_billing_keep_existing_semantics(self):
        runtime = create_v6_runtime(policy_name="earliest_feasible", forecast_end_sim=30.0)
        self.assertIsInstance(runtime.accounting, V6EnergyAccounting)
        self.assertIs(runtime.scheduler.metrics_ledger.accounting, runtime.accounting)
        self.assertIsInstance(runtime.scheduler.candidate_generator, V5CandidateGenerator)
        self.assertIsInstance(runtime.scheduler.policy, NetBenefitGatePolicy)
        self.assertTrue(runtime.config_view_overrides["V6_READONLY_INTEGRAL_ARRAYS"])

        # 作用：构造或处理 bill 测试辅助数据，供本文件用例复用。
        # 调用约定：由外层 v6.tests.test_energy.EnergyTest.test_runtime_and_nonlinear_billing_keep_existing_semantics 创建，作为局部辅助或回调使用。
        def bill(**kwargs):
            return kwargs["tariff_yuan_per_mwh"] * kwargs["total_task_power_mw"] ** 2

        expected, actual = [self.accounting(kind, node_bill_rate_model=bill).candidate_metric_evaluator(
            self.snapshot
        ).evaluate_batch(task=self.task(), path=self.path, target_node="N",
                         compute_start_sim=[2.0, 8.0], compute_end_sim=[3.5, 9.5])
                           for kind in (SweepLineEnergyAccounting, V6EnergyAccounting)]
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])


if __name__ == "__main__":
    unittest.main()
