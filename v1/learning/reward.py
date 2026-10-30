# 文件功能：按物理时间折扣组织提交奖励、完成修正及终止惩罚，形成经验转移。
# 主要内容：DecisionRecord、GammaClock、EventRewardBuffer、RewardAssembler。
# 模块上游（导入关系）：v1.learning、v1.train_v1、tests.v1.test_ablation_settings_v1、tests.v1.test_learning_v1、tests.v1.test_training_monitoring_v1 等。
# 模块下游（项目依赖）：v1.domain.units、v1.learning.replay。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Physical-time event reward assembly without retroactive replay mutation."""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from v1.domain.units import TimeConverter, finite_number

from .replay import ReplayTransition, TimestampedReward


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    task_id: str
    candidate_id: str
    decision_time_sim: float
    estimated_local_utility: float


class GammaClock:
    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.train_v1.V1TrainingLoop.__init__。
    # 调用约定：构造实例时由 Python 调用。
    # 下游调用：finite_number。
    def __init__(
        self,
        gamma_per_second: float,
        time_converter: TimeConverter,
        mode: str = "physical_time",
        decision_gamma: float = 0.95,
    ):
        gamma = finite_number("gamma_per_second", gamma_per_second)
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma_per_second must be in (0,1]")
        self.gamma_per_second = gamma
        self.time_converter = time_converter
        if mode not in {"physical_time", "decision_step", "none"}:
            raise ValueError(
                "discount mode must be physical_time, decision_step, or none"
            )
        decision = finite_number("decision_gamma", decision_gamma)
        if not 0.0 < decision <= 1.0:
            raise ValueError("decision_gamma must be in (0,1]")
        self.mode = mode
        self.decision_gamma = decision

    # 作用：把决策间隔换算为物理秒。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：finite_number、self.time_converter.sim_to_seconds。
    def elapsed_seconds(self, start_sim: float, end_sim: float) -> float:
        start = finite_number("start_sim", start_sim)
        end = finite_number("end_sim", end_sim)
        if end < start:
            raise ValueError("time cannot move backwards")
        return self.time_converter.sim_to_seconds(end - start)

    # 作用：按间隔及折扣模式计算折扣因子。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：finite_number。
    def discount(self, elapsed_seconds: float) -> float:
        seconds = finite_number("elapsed_seconds", elapsed_seconds)
        if seconds < 0.0:
            raise ValueError("elapsed_seconds cannot be negative")
        if self.mode == "none":
            return 1.0
        if self.mode == "decision_step":
            return self.decision_gamma
        return self.gamma_per_second ** seconds


class EventRewardBuffer:
    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.learning.reward.RewardAssembler.__init__。
    # 调用约定：构造实例时由 Python 调用。
    def __init__(self):
        self._events = []

    # 作用：向带时间戳的奖励缓冲追加事件。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self._events.sort。
    def append(self, event: TimestampedReward):
        self._events.append(event)
        self._events.sort(key=lambda item: (item.event_time_sim, item.decision_id, item.event_type))

    # 作用：取出指定时间区间内的奖励。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def pop_interval(self, start_sim: float, end_sim: float) -> Tuple[TimestampedReward, ...]:
        selected = tuple(
            event for event in self._events
            if start_sim < event.event_time_sim <= end_sim
        )
        selected_set = set(selected)
        self._events = [event for event in self._events if event not in selected_set]
        return selected


class RewardAssembler:
    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.train_v1.V1TrainingLoop.__init__。
    # 调用约定：构造实例时由 Python 调用。
    # 下游调用：EventRewardBuffer。
    def __init__(self, clock: GammaClock, event_buffer: Optional[EventRewardBuffer] = None):
        self.clock = clock
        self.event_buffer = event_buffer or EventRewardBuffer()

    # 作用：计算资源提交时的估计奖励。
    # 上游（直接调用）：tests.v1.test_learning_v1.CandidateDqnV1Test.test_estimate_plus_realization_correction_recovers_realized_utility、tests.v1.test_v1_acceptance.V1AcceptanceContractTest.test_reward_assembly_is_deterministic_and_has_no_hidden_duplicate。
    # 下游调用：finite_number。
    @staticmethod
    def commit_reward(decision_record: DecisionRecord) -> float:
        return finite_number("estimated_local_utility", decision_record.estimated_local_utility)

    # 作用：按实际结算与提交估计的差异计算奖励修正。
    # 上游（直接调用）：tests.v1.test_learning_v1.CandidateDqnV1Test.test_estimate_plus_realization_correction_recovers_realized_utility、tests.v1.test_v1_acceptance.V1AcceptanceContractTest.test_reward_assembly_is_deterministic_and_has_no_hidden_duplicate。
    # 下游调用：finite_number。
    @staticmethod
    def realization_correction(
        decision_record: DecisionRecord,
        realized_local_utility: float,
        completion_outcome_reward: float,
    ) -> float:
        return (
            finite_number("realized_local_utility", realized_local_utility)
            - finite_number("estimated_local_utility", decision_record.estimated_local_utility)
            + finite_number("completion_outcome_reward", completion_outcome_reward)
        )

    # 作用：将奖励事件加入时间归因缓冲。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def buffer_event(self, event: TimestampedReward):
        self.event_buffer.append(event)

    # 作用：聚合区间奖励和折扣，形成回放转移。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：finite_number、self.clock.elapsed_seconds、self.event_buffer.pop_interval、self.clock.discount、ReplayTransition。
    def build_transition(
        self,
        *,
        global_state_before: Iterable[float],
        selected_candidate_id: Optional[str],
        selected_candidate_features: Optional[Iterable[float]],
        immediate_reward: float,
        global_state_after: Iterable[float],
        next_candidate_features: Iterable[Iterable[float]],
        decision_time_sim: float,
        next_transition_time_sim: float,
        terminal: bool,
        next_candidate_context=None,
    ) -> ReplayTransition:
        elapsed = self.clock.elapsed_seconds(decision_time_sim, next_transition_time_sim)
        events = self.event_buffer.pop_interval(decision_time_sim, next_transition_time_sim)
        reward = finite_number("immediate_reward", immediate_reward)
        for event in events:
            event_elapsed = self.clock.elapsed_seconds(decision_time_sim, event.event_time_sim)
            reward += self.clock.discount(event_elapsed) * event.reward
        return ReplayTransition(
            tuple(float(value) for value in global_state_before),
            selected_candidate_id,
            None if selected_candidate_features is None else tuple(float(value) for value in selected_candidate_features),
            reward,
            tuple(float(value) for value in global_state_after),
            tuple(tuple(float(value) for value in item) for item in next_candidate_features),
            decision_time_sim,
            next_transition_time_sim,
            elapsed,
            self.clock.discount(elapsed),
            bool(terminal),
            events,
            next_candidate_context,
        )
