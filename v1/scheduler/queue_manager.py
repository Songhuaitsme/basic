# 文件功能：维护任务队列、SLA 排序、待重试任务和截止时间索引。
# 主要内容：TaskQueueManager、queue_order_key。
# 模块上游（导入关系）：v1.scheduler、v1.scheduler.v1_scheduler、tests.v1.test_candidate_queue。
# 模块下游（项目依赖）：v1.domain.models、v1.simulation.state_machine。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Deterministic EDF queue and event-triggered Pending management."""

from collections import Counter
import heapq
from typing import Iterable, List, Tuple

from v1.domain.models import SlaType, TaskSpec, TaskState
from v1.simulation.state_machine import StateTransition, TaskStateMachine


SLA_TIE_RANK = {
    SlaType.HARD: 0,
    SlaType.SOFT: 1,
    SlaType.FLEXIBLE: 2,
}

PHYSICAL_REACTIVATION_EVENTS = {
    "CPU_INTERVAL_ENDED",
    "BANDWIDTH_INTERVAL_ENDED",
    "RESERVATION_RELEASED",
    "TOPOLOGY_CAPACITY_CHANGED",
    "FORECAST_COVERAGE_EXTENDED",
}


# 作用：生成 SLA 优先级与时间约束组成的稳定队列键。
# 上游（直接调用）：v1.scheduler.queue_manager.TaskQueueManager._activate_queued。
def queue_order_key(task: TaskSpec) -> Tuple:
    preferred = task.absolute_preferred_start_sim
    return (
        task.absolute_latest_start_sim,
        SLA_TIE_RANK[task.sla_type],
        float("inf") if preferred is None else preferred,
        task.arrival_time_sim,
        task.task_id,
    )


class TaskQueueManager:
    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.scheduler.v1_scheduler.V1Scheduler.__init__。
    # 调用约定：构造实例时由 Python 调用。
    def __init__(self, state_machine: TaskStateMachine, max_queue_length: int):
        if (
            isinstance(max_queue_length, bool)
            or not isinstance(max_queue_length, int)
            or max_queue_length <= 0
        ):
            raise ValueError("max_queue_length must be a positive integer")
        self.state_machine = state_machine
        self.max_queue_length = max_queue_length
        self.reason_counts = Counter()
        self._edf_heap = []
        self._queue_generation = Counter()
        self._pending_task_ids = set()
        self._deadline_heap = []
        self._deadline_task_ids = set()

    # 作用：恢复序列化状态，补齐必要索引或默认字段。
    # 调用约定：由 pickle/检查点反序列化流程调用。
    # 下游调用：self._rebuild_indices。
    def __setstate__(self, state):
        """Rebuild disposable V2 queue indices for legacy resume checkpoints."""

        self.__dict__.update(state)
        self._rebuild_indices()

    # 作用：由现有任务状态重建队列和截止时间索引。
    # 上游（直接调用）：v1.scheduler.queue_manager.TaskQueueManager.__setstate__。
    # 下游调用：self._register_deadline、self._activate_queued、self.state_machine.task_ids_in_state、self.state_machine.task_ids_in_states、self.state_machine.task_spec 等。
    def _rebuild_indices(self):
        self._edf_heap = []
        self._queue_generation = Counter()
        self._pending_task_ids = set(
            self.state_machine.task_ids_in_state(
                TaskState.PENDING_UNCOMMITTED
            )
        )
        self._deadline_heap = []
        self._deadline_task_ids = set()
        for task_id in self.state_machine.task_ids_in_states(
            (TaskState.QUEUED, TaskState.PENDING_UNCOMMITTED)
        ):
            task = self.state_machine.task_spec(task_id)
            self._register_deadline(task)
            if self.state_machine.runtime(task_id).state is TaskState.QUEUED:
                self._activate_queued(task)

    # 作用：将任务加入可调度队列索引。
    # 上游（直接调用）：v1.scheduler.queue_manager.TaskQueueManager._rebuild_indices、v1.scheduler.queue_manager.TaskQueueManager.enqueue_new、v1.scheduler.queue_manager.TaskQueueManager.reactivate_pending。
    # 下游调用：queue_order_key。
    def _activate_queued(self, task: TaskSpec) -> None:
        token = self._queue_generation[task.task_id] + 1
        self._queue_generation[task.task_id] = token
        heapq.heappush(
            self._edf_heap,
            (queue_order_key(task), token, task.task_id),
        )

    # 作用：将任务从可调度索引移除。
    # 上游（直接调用）：v1.scheduler.queue_manager.TaskQueueManager.expire_due_tasks_after_boundary_opportunity、v1.scheduler.queue_manager.TaskQueueManager.mark_pending。
    def _deactivate_queued(self, task_id: str) -> None:
        # Existing heap entries become stale and are discarded lazily.
        self._queue_generation[task_id] += 1

    # 作用：把任务最晚开始期限加入截止索引。
    # 上游（直接调用）：v1.scheduler.queue_manager.TaskQueueManager._rebuild_indices、v1.scheduler.queue_manager.TaskQueueManager.enqueue_new。
    def _register_deadline(self, task: TaskSpec) -> None:
        if task.task_id in self._deadline_task_ids:
            return
        self._deadline_task_ids.add(task.task_id)
        heapq.heappush(
            self._deadline_heap,
            (task.absolute_latest_start_sim, task.task_id),
        )

    # 作用：按堆索引取出有效任务并保持规定次序。
    # 上游（直接调用）：v1.scheduler.queue_manager.TaskQueueManager.eligible_tasks、v1.scheduler.queue_manager.TaskQueueManager.ordered_queued_tasks。
    # 下游调用：self.state_machine.runtime、self.state_machine.task_spec。
    def _ordered_from_heap(self, max_tasks=None, now_sim=None):
        selected = []
        retained = []
        while self._edf_heap:
            entry = heapq.heappop(self._edf_heap)
            _, token, task_id = entry
            if token != self._queue_generation[task_id]:
                continue
            if (
                self.state_machine.runtime(task_id).state
                is not TaskState.QUEUED
            ):
                continue
            retained.append(entry)
            task = self.state_machine.task_spec(task_id)
            if (
                now_sim is None
                or task.arrival_time_sim <= now_sim + 1e-12
            ):
                selected.append(task)
                if max_tasks is not None and len(selected) >= max_tasks:
                    break
        for entry in retained:
            heapq.heappush(self._edf_heap, entry)
        return selected

    # 作用：统计尚未成功提交资源的任务。
    # 上游（直接调用）：v1.scheduler.queue_manager.TaskQueueManager.enqueue_new。
    # 下游调用：self.state_machine.count_by_state。
    def _uncommitted_count(self) -> int:
        counts = self.state_machine.count_by_state()
        return counts[TaskState.QUEUED] + counts[TaskState.PENDING_UNCOMMITTED]

    # 作用：注册新任务，依据静态可服务性和队列容量决定入队状态。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self._uncommitted_count、self._activate_queued、self._register_deadline、self.state_machine.register、self.state_machine.transition。
    def enqueue_new(
        self,
        task: TaskSpec,
        statically_serviceable: bool = True,
    ) -> StateTransition:
        self.state_machine.register(task)
        if not statically_serviceable:
            transition = self.state_machine.transition(
                task.task_id,
                TaskState.REJECTED,
                task.arrival_time_sim,
                terminal_reason="STATICALLY_UNSERVICEABLE",
            )
            self.reason_counts["STATICALLY_UNSERVICEABLE"] += 1
            return transition
        if self._uncommitted_count() >= self.max_queue_length:
            transition = self.state_machine.transition(
                task.task_id,
                TaskState.REJECTED,
                task.arrival_time_sim,
                terminal_reason="SCHEDULER_QUEUE_CAPACITY",
            )
            self.reason_counts["SCHEDULER_QUEUE_CAPACITY"] += 1
            return transition
        transition = self.state_machine.transition(
            task.task_id,
            TaskState.QUEUED,
            task.arrival_time_sim,
        )
        self._activate_queued(task)
        self._register_deadline(task)
        return transition

    # 作用：按稳定次序返回排队任务。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self._ordered_from_heap。
    def ordered_queued_tasks(self) -> List[TaskSpec]:
        return self._ordered_from_heap()

    # 作用：选出本周期符合时刻和数量限制的任务。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self._ordered_from_heap。
    def eligible_tasks(
        self,
        max_tasks: int,
        now_sim: float = None,
    ) -> List[TaskSpec]:
        if isinstance(max_tasks, bool) or not isinstance(max_tasks, int) or max_tasks < 0:
            raise ValueError("max_tasks must be a non-negative integer")
        if max_tasks == 0:
            return []
        return self._ordered_from_heap(max_tasks=max_tasks, now_sim=now_sim)

    # 作用：将未能立即提交的任务标记为待重试。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self._deactivate_queued、self.state_machine.transition、self.state_machine.increment_pending_attempts。
    def mark_pending(self, task_id: str, now_sim: float) -> StateTransition:
        transition = self.state_machine.transition(
            task_id,
            TaskState.PENDING_UNCOMMITTED,
            now_sim,
        )
        self._deactivate_queued(task_id)
        self._pending_task_ids.add(task_id)
        self.state_machine.increment_pending_attempts(task_id)
        return transition

    # 作用：根据时刻和物理事件唤醒待重试任务。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self._activate_queued、self.state_machine.task_ids_in_state、self.state_machine.transition、self._pending_task_ids.discard、self.state_machine.task_spec。
    def reactivate_pending(
        self,
        now_sim: float,
        event_types: Iterable[str],
    ) -> List[StateTransition]:
        if not (set(event_types) & PHYSICAL_REACTIVATION_EVENTS):
            return []
        transitions = []
        for task_id in self.state_machine.task_ids_in_state(
            TaskState.PENDING_UNCOMMITTED
        ):
            if task_id not in self._pending_task_ids:
                continue
            transitions.append(self.state_machine.transition(
                task_id,
                TaskState.QUEUED,
                now_sim,
            ))
            self._pending_task_ids.discard(task_id)
            self._activate_queued(self.state_machine.task_spec(task_id))
        return transitions

    # 作用：查询未提交任务中最近的最晚开始期限。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self.state_machine.runtime、self._deadline_task_ids.discard。
    def next_uncommitted_deadline_sim(self):
        """Return the next live queue deadline, pruning terminal entries."""

        while self._deadline_heap:
            deadline_sim, task_id = self._deadline_heap[0]
            runtime = self.state_machine.runtime(task_id)
            if runtime.state in {
                TaskState.QUEUED,
                TaskState.PENDING_UNCOMMITTED,
            }:
                return deadline_sim
            heapq.heappop(self._deadline_heap)
            self._deadline_task_ids.discard(task_id)
        return None

    # 作用：在截止边界调度机会结束后使超时任务过期。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self._deactivate_queued、self._deadline_task_ids.discard、self.state_machine.runtime、self._pending_task_ids.discard、self.state_machine.transition。
    def expire_due_tasks_after_boundary_opportunity(
        self,
        now_sim: float,
    ) -> List[StateTransition]:
        transitions = []
        while (
            self._deadline_heap
            and self._deadline_heap[0][0] <= now_sim + 1e-12
        ):
            _, task_id = heapq.heappop(self._deadline_heap)
            self._deadline_task_ids.discard(task_id)
            runtime = self.state_machine.runtime(task_id)
            if runtime.state not in {
                TaskState.QUEUED,
                TaskState.PENDING_UNCOMMITTED,
            }:
                continue
            if runtime.state is TaskState.QUEUED:
                self._deactivate_queued(task_id)
            self._pending_task_ids.discard(task_id)
            transitions.append(self.state_machine.transition(
                task_id,
                TaskState.EXPIRED,
                now_sim,
                terminal_reason="ABSOLUTE_START_DEADLINE",
            ))
            self.reason_counts["ABSOLUTE_START_DEADLINE"] += 1
        return transitions

    # 作用：提取内部索引中的有效任务标识。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def _task_ids(self):
        return self.state_machine.task_ids
