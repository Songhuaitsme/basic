# 文件功能：验证 等待门控 的行为、边界条件和回归约束。
# 主要内容：FixedPolicy、FixedEncoder、WaitGateTest、candidate、task、selection。
# 使用入口：由 unittest 发现并执行，构造依赖后验证断言。
# 模块下游（项目依赖）：v1.domain.models、v1.learning.candidate_dqn、v1.scheduler.policies、v4.gating、v4.policy、v4.scheduler 等。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

import unittest
from types import SimpleNamespace

from v1.domain.models import SlaType
from v1.learning.candidate_dqn import CandidateSelectionTrace
from v1.scheduler.policies import CandidateStreamSelection
from v4.gating import NetBenefitWaitGate, WaitGateConfig, default_objective_config
from v4.policy import NetBenefitGatePolicy
from v4.scheduler import V4Scheduler
from v4.v4_runtime import create_v4_runtime


# 作用：构造或处理 candidate 测试辅助数据，供本文件用例复用。
# 上游（直接调用）：v4.tests.test_wait_gate.WaitGateTest.test_gate_accepts_wait_when_cost_gain_clears_penalty、v4.tests.test_wait_gate.WaitGateTest.test_gate_does_not_override_same_start_candidate、v4.tests.test_wait_gate.WaitGateTest.test_gate_rejects_wait_when_only_wait_penalty_changes 等。
def candidate(candidate_id, start, cost, *, green=0.5, absorption=0.0, margin=0.5):
    return SimpleNamespace(
        candidate_id=candidate_id,
        compute_start_sim=float(start),
        estimated_candidate_marginal_system_cost_yuan=float(cost),
        estimated_green_coverage=float(green),
        estimated_green_absorption_delta=float(absorption),
        estimated_green_opportunity=True,
        capacity_margin=float(margin),
        preferred_start_tardiness_ratio=0.0,
        preferred_start_tardiness_applicable=True,
    )


# 作用：构造或处理 task 测试辅助数据，供本文件用例复用。
# 上游（直接调用）：v4.tests.test_wait_gate.WaitGateTest.test_gate_accepts_wait_when_cost_gain_clears_penalty、v4.tests.test_wait_gate.WaitGateTest.test_gate_does_not_override_same_start_candidate、v4.tests.test_wait_gate.WaitGateTest.test_gate_rejects_wait_when_only_wait_penalty_changes 等。
def task(sla_type=SlaType.SOFT):
    return SimpleNamespace(task_id="task-1", sla_type=sla_type)


# 作用：构造或处理 selection 测试辅助数据，供本文件用例复用。
# 上游（直接调用）：v4.tests.test_wait_gate.WaitGateTest.test_gate_accepts_wait_when_cost_gain_clears_penalty、v4.tests.test_wait_gate.WaitGateTest.test_gate_does_not_override_same_start_candidate、v4.tests.test_wait_gate.WaitGateTest.test_gate_rejects_wait_when_only_wait_penalty_changes 等。
# 下游调用：CandidateStreamSelection。
def selection(selected, earliest):
    return CandidateStreamSelection(selected, earliest, 2, "hash", None)


class FixedPolicy:
    name = "fixed"

    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v4.tests.test_wait_gate.WaitGateTest.test_policy_wrapper_exposes_gate_diagnostic、v4.tests.test_wait_gate.WaitGateTest.test_policy_wrapper_rewrites_training_trace_after_fallback。
    # 调用约定：构造实例时由 Python 调用。
    def __init__(self, result):
        self.result = result

    # 作用：遍历候选流，汇总选中与最早候选及决策信息。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    def select_stream(self, candidates, task=None, context=None):
        tuple(candidates)
        return self.result


class FixedEncoder:
    # 作用：把单候选转换为模型特征。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def encode(self, candidate, earliest_compute_start_sim):
        return (candidate.compute_start_sim, earliest_compute_start_sim)


class WaitGateTest(unittest.TestCase):
    # 作用：验证 gate_rejects_wait_when_only_wait_penalty_changes 场景，覆盖 等待门控 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：candidate、NetBenefitWaitGate、default_objective_config、selection、task 等。
    def test_gate_rejects_wait_when_only_wait_penalty_changes(self):
        earliest = candidate("early", 10, 100.0)
        delayed = candidate("late", 20, 100.0)
        gate = NetBenefitWaitGate(default_objective_config(), WaitGateConfig())

        gated, diagnostic = gate.apply(selection(delayed, earliest), task())

        self.assertIs(gated.selected_candidate, earliest)
        self.assertTrue(diagnostic.applied)
        self.assertFalse(diagnostic.passed)
        self.assertGreater(diagnostic.wait_penalty, 0.0)
        self.assertLess(diagnostic.estimated_net_wait_gain, 0.0)
        self.assertEqual(diagnostic.reason, "NET_GAIN_BELOW_THRESHOLD")

    # 作用：验证 gate_accepts_wait_when_cost_gain_clears_penalty 场景，覆盖 等待门控 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：default_objective_config、candidate、NetBenefitWaitGate、selection、task 等。
    def test_gate_accepts_wait_when_cost_gain_clears_penalty(self):
        objective = default_objective_config()
        scale = objective.cost_scale_yuan
        reference = objective.reference_marginal_cost_yuan
        earliest = candidate("early", 10, reference + scale)
        delayed = candidate("late", 20, reference - scale)
        gate = NetBenefitWaitGate(objective, WaitGateConfig())

        gated, diagnostic = gate.apply(selection(delayed, earliest), task())

        self.assertIs(gated.selected_candidate, delayed)
        self.assertTrue(diagnostic.passed)
        self.assertGreater(diagnostic.estimated_net_wait_gain, 0.0)
        self.assertEqual(diagnostic.reason, "NET_GAIN_ACCEPTED")

    # 作用：验证 gate_does_not_override_same_start_candidate 场景，覆盖 等待门控 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：candidate、NetBenefitWaitGate、default_objective_config、selection、task 等。
    def test_gate_does_not_override_same_start_candidate(self):
        earliest = candidate("early", 10, 100.0)
        selected = candidate("same-time", 10, 90.0)
        gate = NetBenefitWaitGate(default_objective_config(), WaitGateConfig())

        gated, diagnostic = gate.apply(selection(selected, earliest), task())

        self.assertIs(gated.selected_candidate, selected)
        self.assertFalse(diagnostic.applied)
        self.assertTrue(diagnostic.passed)
        self.assertEqual(diagnostic.reason, "NO_ACTIVE_WAIT")

    # 作用：验证 policy_wrapper_exposes_gate_diagnostic 场景，覆盖 等待门控 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：candidate、NetBenefitWaitGate、default_objective_config、NetBenefitGatePolicy、FixedPolicy 等。
    def test_policy_wrapper_exposes_gate_diagnostic(self):
        earliest = candidate("early", 10, 100.0)
        delayed = candidate("late", 20, 100.0)
        gate = NetBenefitWaitGate(default_objective_config(), WaitGateConfig())
        policy = NetBenefitGatePolicy(
            FixedPolicy(selection(delayed, earliest)),
            gate,
        )

        result = policy.select_stream((earliest, delayed), task=task())
        diagnostic = policy.pop_gate_diagnostic("task-1")

        self.assertIs(result.selected_candidate, earliest)
        self.assertEqual(diagnostic.proposed_candidate_id, "late")
        self.assertEqual(diagnostic.final_candidate_id, "early")

    # 作用：验证 policy_wrapper_rewrites_training_trace_after_fallback 场景，覆盖 等待门控 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：candidate、FixedPolicy、selection、NetBenefitGatePolicy、NetBenefitWaitGate 等。
    def test_policy_wrapper_rewrites_training_trace_after_fallback(self):
        earliest = candidate("early", 10, 100.0)
        delayed = candidate("late", 20, 100.0)
        base = FixedPolicy(selection(delayed, earliest))
        base.feature_encoder = FixedEncoder()
        base._selection_traces = [CandidateSelectionTrace(
            task_id="task-1",
            decision_time_sim=0.0,
            global_state=(0.0,),
            candidate_count=2,
            selected_candidate_id="late",
            selected_candidate_features=(20.0, 10.0),
            selected_candidate=delayed,
        )]
        policy = NetBenefitGatePolicy(
            base,
            NetBenefitWaitGate(default_objective_config(), WaitGateConfig()),
        )

        policy.select_stream((earliest, delayed), task=task())
        trace = base._selection_traces[-1]

        self.assertIs(trace.selected_candidate, earliest)
        self.assertEqual(trace.selected_candidate_id, "early")
        self.assertEqual(trace.selected_candidate_features, (10.0, 10.0))

    # 作用：验证 v4_runtime_keeps_v3_engine_and_installs_v4_scheduler 场景，覆盖 等待门控 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：create_v4_runtime。
    def test_v4_runtime_keeps_v3_engine_and_installs_v4_scheduler(self):
        runtime = create_v4_runtime(policy_name="earliest_feasible")

        self.assertIsInstance(runtime.scheduler, V4Scheduler)
        self.assertIsInstance(runtime.scheduler.policy, NetBenefitGatePolicy)
        self.assertTrue(runtime.config_view_overrides["V4_WAIT_GATE_ENABLED"])


if __name__ == "__main__":
    unittest.main()
