# 文件功能：维护任务生命周期、合法状态转移、状态索引及重试计数。
# 主要内容：StateTransitionError、StateTransition、TaskStateMachine。
# 模块上游（导入关系）：v1.scheduler.queue_manager、v1.scheduler.reservation_manager、v1.scheduler.v1_scheduler、v1.simulation、v1.simulation.event_engine 等。
# 模块下游（项目依赖）：v1.domain.models、v1.domain.units。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Deterministic v1.0 task lifecycle transitions."""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Set

from v1.domain.models import TaskRuntime, TaskSpec, TaskState
from v1.domain.units import finite_number


TERMINAL_STATES = {
    TaskState.COMPLETED,
    TaskState.REJECTED,
    TaskState.EXPIRED,
    TaskState.FAILED,
}

ALLOWED_TRANSITIONS = {
    TaskState.ARRIVED: {TaskState.QUEUED, TaskState.REJECTED},
    TaskState.QUEUED: {
        TaskState.RESERVED,
        TaskState.PENDING_UNCOMMITTED,
        TaskState.EXPIRED,
    },
    TaskState.PENDING_UNCOMMITTED: {TaskState.QUEUED, TaskState.EXPIRED},
    TaskState.RESERVED: {
        TaskState.TRANSMITTING,
        TaskState.RUNNING,
        TaskState.FAILED,
    },
    TaskState.TRANSMITTING: {TaskState.RUNNING, TaskState.FAILED},
    TaskState.RUNNING: {TaskState.COMPLETED, TaskState.FAILED},
    TaskState.COMPLETED: set(),
    TaskState.REJECTED: set(),
    TaskState.EXPIRED: set(),
    TaskState.FAILED: set(),
}

TERMINAL_REASON_BY_STATE = {
    TaskState.COMPLETED: {"COMPLETED"},
    TaskState.REJECTED: {
        "INVALID_TASK",
        "STATICALLY_UNSERVICEABLE",
        "SCHEDULER_QUEUE_CAPACITY",
    },
    TaskState.EXPIRED: {"ABSOLUTE_START_DEADLINE"},
    TaskState.FAILED: {
        "TRANSMISSION_FAILURE",
        "EXECUTION_FAILURE",
        "RESERVATION_BROKEN",
    },
}


class StateTransitionError(ValueError):
    pass


@dataclass(frozen=True)
class StateTransition:
    task_id: str
    event_time_sim: float
    previous_state: TaskState
    new_state: TaskState
    state_version: int
    terminal_reason: Optional[str] = None
    reservation_id: Optional[str] = None


class TaskStateMachine:
    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.scheduler.v1_scheduler.V1Scheduler.__init__、v1.simulation.event_engine.EventEngine.__init__。
    # 调用约定：构造实例时由 Python 调用。
    def __init__(self):
        self._specs: Dict[str, TaskSpec] = {}
        self._runtimes: Dict[str, TaskRuntime] = {}
        self._task_ids_by_state = {state: set() for state in TaskState}
        self._registration_index = {}

    # 作用：恢复序列化状态，补齐必要索引或默认字段。
    # 调用约定：由 pickle/检查点反序列化流程调用。
    # 下游调用：self._rebuild_state_index。
    def __setstate__(self, state):
        """Restore old checkpoints and rebuild disposable V2 state indices."""

        self.__dict__.update(state)
        self._rebuild_state_index()

    # 作用：由运行记录重建按状态分类的任务索引。
    # 上游（直接调用）：v1.simulation.state_machine.TaskStateMachine.__setstate__。
    def _rebuild_state_index(self) -> None:
        indexed = {state: set() for state in TaskState}
        registration_index = {}
        for position, (task_id, runtime) in enumerate(self._runtimes.items()):
            indexed[runtime.state].add(task_id)
            registration_index[task_id] = position
        self._task_ids_by_state = indexed
        self._registration_index = registration_index

    # 作用：注册任务规格并建立初始运行状态。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：StateTransitionError、TaskRuntime。
    def register(self, task_spec: TaskSpec) -> TaskRuntime:
        if task_spec.task_id in self._specs:
            raise StateTransitionError(f"duplicate task_id: {task_spec.task_id}")
        runtime = TaskRuntime(
            task_id=task_spec.task_id,
            state=TaskState.ARRIVED,
            last_state_change_sim=task_spec.arrival_time_sim,
        )
        self._specs[task_spec.task_id] = task_spec
        self._runtimes[task_spec.task_id] = runtime
        self._registration_index[task_spec.task_id] = len(
            self._registration_index
        )
        self._task_ids_by_state[TaskState.ARRIVED].add(task_spec.task_id)
        return replace(runtime)

    # 作用：查询不可变任务规格。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：StateTransitionError。
    def task_spec(self, task_id: str) -> TaskSpec:
        try:
            return self._specs[task_id]
        except KeyError:
            raise StateTransitionError(f"unknown task_id: {task_id}")

    # 作用：查询任务当前运行记录。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：StateTransitionError。
    def runtime(self, task_id: str) -> TaskRuntime:
        try:
            return replace(self._runtimes[task_id])
        except KeyError:
            raise StateTransitionError(f"unknown task_id: {task_id}")

    # 作用：校验并执行合法状态转移，同时维护索引。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：finite_number、StateTransitionError、self._task_ids_by_state[previous].remove、StateTransition。
    def transition(
        self,
        task_id: str,
        new_state: TaskState,
        event_time_sim: float,
        terminal_reason: Optional[str] = None,
        reservation_id: Optional[str] = None,
    ) -> Optional[StateTransition]:
        if not isinstance(new_state, TaskState):
            raise StateTransitionError("new_state must be a TaskState")
        try:
            runtime = self._runtimes[task_id]
        except KeyError:
            raise StateTransitionError(f"unknown task_id: {task_id}")
        event_time = finite_number("event_time_sim", event_time_sim)
        if event_time < runtime.last_state_change_sim - 1e-12:
            raise StateTransitionError("task state cannot move backwards in time")
        if runtime.state is new_state:
            return None
        if new_state not in ALLOWED_TRANSITIONS[runtime.state]:
            raise StateTransitionError(
                f"illegal transition {runtime.state.value}->{new_state.value}"
            )

        if new_state in TERMINAL_STATES:
            allowed_reasons: Set[str] = TERMINAL_REASON_BY_STATE[new_state]
            if terminal_reason not in allowed_reasons:
                raise StateTransitionError(
                    f"{new_state.value} requires one of {sorted(allowed_reasons)}"
                )
        elif terminal_reason is not None:
            raise StateTransitionError("non-terminal transition cannot set terminal_reason")
        if (
            reservation_id is not None
            and runtime.reservation_id not in (None, reservation_id)
        ):
            raise StateTransitionError("reservation_id is immutable once assigned")

        previous = runtime.state
        self._task_ids_by_state[previous].remove(task_id)
        self._task_ids_by_state[new_state].add(task_id)
        runtime.state = new_state
        runtime.state_version += 1
        runtime.last_state_change_sim = event_time
        if reservation_id is not None:
            runtime.reservation_id = reservation_id
        if terminal_reason is not None:
            runtime.terminal_reason = terminal_reason
        return StateTransition(
            task_id=task_id,
            event_time_sim=event_time,
            previous_state=previous,
            new_state=new_state,
            state_version=runtime.state_version,
            terminal_reason=runtime.terminal_reason,
            reservation_id=runtime.reservation_id,
        )

    # 作用：返回各状态的任务数量。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def count_by_state(self) -> Dict[TaskState, int]:
        return {
            state: len(self._task_ids_by_state[state])
            for state in TaskState
        }

    # 作用：汇集指定状态集合中的任务标识。
    # 上游（直接调用）：v1.simulation.state_machine.TaskStateMachine.task_ids_in_state。
    # 下游调用：StateTransitionError。
    def task_ids_in_states(self, states) -> tuple:
        """Return deterministic ids from the maintained state membership index."""

        requested = tuple(states)
        unknown = [state for state in requested if not isinstance(state, TaskState)]
        if unknown:
            raise StateTransitionError("states must contain TaskState values")
        selected = set()
        for state in requested:
            selected.update(self._task_ids_by_state[state])
        return tuple(sorted(selected, key=self._registration_index.__getitem__))

    # 作用：返回指定状态中的任务标识。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self.task_ids_in_states。
    def task_ids_in_state(self, state: TaskState) -> tuple:
        return self.task_ids_in_states((state,))

    # 作用：增加待重试次数。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：StateTransitionError。
    def increment_pending_attempts(self, task_id: str) -> int:
        try:
            runtime = self._runtimes[task_id]
        except KeyError:
            raise StateTransitionError(f"unknown task_id: {task_id}")
        runtime.pending_attempts += 1
        return runtime.pending_attempts

    # 作用：增加提交尝试次数。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：StateTransitionError。
    def increment_commit_attempts(self, task_id: str) -> int:
        try:
            runtime = self._runtimes[task_id]
        except KeyError:
            raise StateTransitionError(f"unknown task_id: {task_id}")
        runtime.commit_attempts_current_decision += 1
        return runtime.commit_attempts_current_decision

    # 作用：重置提交重试计数。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：StateTransitionError。
    def reset_commit_attempts(self, task_id: str) -> None:
        try:
            self._runtimes[task_id].commit_attempts_current_decision = 0
        except KeyError:
            raise StateTransitionError(f"unknown task_id: {task_id}")

    # 作用：返回已注册任务总数。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    @property
    def task_count(self) -> int:
        return len(self._specs)

    # 作用：返回已注册任务标识。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    @property
    def task_ids(self):
        return tuple(self._specs.keys())
