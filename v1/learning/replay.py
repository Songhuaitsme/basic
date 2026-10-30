# 文件功能：定义决策和经验转移记录，管理经验采样、序列化及恢复。
# 主要内容：TimestampedReward、ReplayTransition、CandidateReplayBuffer。
# 模块上游（导入关系）：v1.learning、v1.learning.candidate_dqn、v1.learning.reward、v1.train_v1、tests.v1.test_learning_v1 等。
# 模块下游（项目依赖）：v1.domain.units。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Immutable variable-candidate replay contracts and JSON round trips."""

from dataclasses import asdict, dataclass
import json
import random
from typing import Optional, Tuple

from v1.domain.units import finite_number, non_negative_finite


@dataclass(frozen=True)
class TimestampedReward:
    event_time_sim: float
    reward: float
    decision_id: str
    event_type: str

    # 作用：在数据类构造后校验字段或建立派生数据。
    # 调用约定：由 dataclass 生成的构造流程调用。
    # 下游调用：finite_number。
    def __post_init__(self):
        object.__setattr__(self, "event_time_sim", finite_number("event_time_sim", self.event_time_sim))
        object.__setattr__(self, "reward", finite_number("event_reward", self.reward))
        if not self.decision_id or not self.event_type:
            raise ValueError("decision_id and event_type must be non-empty")


@dataclass(frozen=True)
class ReplayTransition:
    global_state_before: Tuple[float, ...]
    selected_candidate_id: Optional[str]
    selected_candidate_features: Optional[Tuple[float, ...]]
    reward: float
    global_state_after: Tuple[float, ...]
    next_candidate_features: Tuple[Tuple[float, ...], ...]
    decision_time_sim: float
    next_transition_time_sim: float
    elapsed_seconds: float
    gamma_elapsed: float
    terminal: bool
    timestamped_event_rewards: Tuple[TimestampedReward, ...] = ()
    next_candidate_context: object = None

    # 作用：在数据类构造后校验字段或建立派生数据。
    # 调用约定：由 dataclass 生成的构造流程调用。
    # 下游调用：finite_number。
    def __post_init__(self):
        for name in ("reward", "decision_time_sim", "next_transition_time_sim", "elapsed_seconds", "gamma_elapsed"):
            object.__setattr__(self, name, finite_number(name, getattr(self, name)))
        if self.next_transition_time_sim < self.decision_time_sim:
            raise ValueError("transition time cannot move backwards")
        # A very long physical-time interval can legitimately underflow
        # gamma**elapsed to exactly zero in IEEE-754 arithmetic.  Zero means
        # that the bootstrap contribution is numerically negligible.
        if self.elapsed_seconds < 0.0 or not 0.0 <= self.gamma_elapsed <= 1.0:
            raise ValueError("invalid elapsed time or discount")
        if (self.selected_candidate_id is None) != (self.selected_candidate_features is None):
            raise ValueError("candidate id and selected features must be both present or absent")

    # 作用：将经验记录序列化为 JSON。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def to_json(self) -> str:
        if self.next_candidate_context is not None:
            raise ValueError(
                "context-backed replay transitions use checkpoint serialization, not JSON"
            )
        values = asdict(self)
        values["next_candidate_features"] = [
            [float(value) for value in row]
            for row in self.next_candidate_features
        ]
        return json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)

    # 作用：从 JSON 恢复经验记录。
    # 上游（直接调用）：tests.v1.test_learning_v1.CandidateDqnV1Test.test_numpy_replay_features_serialize、tests.v1.test_learning_v1.CandidateDqnV1Test.test_replay_round_trip_and_append_only_history。
    # 下游调用：TimestampedReward。
    @classmethod
    def from_json(cls, payload: str) -> "ReplayTransition":
        values = json.loads(payload)
        events = tuple(TimestampedReward(**event) for event in values.pop("timestamped_event_rewards", ()))
        return cls(
            global_state_before=tuple(values["global_state_before"]),
            selected_candidate_id=values["selected_candidate_id"],
            selected_candidate_features=(
                None if values["selected_candidate_features"] is None
                else tuple(values["selected_candidate_features"])
            ),
            reward=values["reward"],
            global_state_after=tuple(values["global_state_after"]),
            next_candidate_features=tuple(tuple(item) for item in values["next_candidate_features"]),
            decision_time_sim=values["decision_time_sim"],
            next_transition_time_sim=values["next_transition_time_sim"],
            elapsed_seconds=values["elapsed_seconds"],
            gamma_elapsed=values["gamma_elapsed"],
            terminal=values["terminal"],
            timestamped_event_rewards=events,
        )


class CandidateReplayBuffer:
    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.train_v1.V1TrainingLoop.__init__。
    # 调用约定：构造实例时由 Python 调用。
    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._items = []
        self._position = 0

    # 作用：向缓冲区或统计容器追加记录。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def add(self, transition: ReplayTransition):
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    # 作用：返回当前保存的记录数量。
    # 调用约定：由 len() 或容器协议调用。
    def __len__(self):
        return len(self._items)

    # 作用：返回缓冲区保存的经验转移。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def items(self):
        return tuple(self._items)

    # 作用：从经验缓冲区随机采样训练批次。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：generator.sample。
    def sample(self, batch_size: int, rng: Optional[random.Random] = None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > len(self._items):
            raise ValueError("batch_size exceeds replay population")
        generator = rng or random
        return tuple(generator.sample(self._items, batch_size))

    # 作用：导出可恢复的经验缓冲区状态。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def state_dict(self):
        return {
            "capacity": self.capacity,
            "items": tuple(self._items),
            "position": self._position,
        }

    # 作用：由保存状态恢复经验缓冲区。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def load_state_dict(self, state):
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("replay capacity mismatch")
        items = list(state.get("items", ()))
        if len(items) > self.capacity or any(
            not isinstance(item, ReplayTransition) for item in items
        ):
            raise ValueError("invalid replay checkpoint items")
        position = int(state.get("position", 0))
        if not 0 <= position < self.capacity:
            raise ValueError("invalid replay checkpoint position")
        self._items = items
        self._position = position
