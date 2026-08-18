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

    def __setstate__(self, state):
        """Rebuild disposable V2 queue indices for legacy resume checkpoints."""

        self.__dict__.update(state)
        self._rebuild_indices()

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

    def _activate_queued(self, task: TaskSpec) -> None:
        token = self._queue_generation[task.task_id] + 1
        self._queue_generation[task.task_id] = token
        heapq.heappush(
            self._edf_heap,
            (queue_order_key(task), token, task.task_id),
        )

    def _deactivate_queued(self, task_id: str) -> None:
        # Existing heap entries become stale and are discarded lazily.
        self._queue_generation[task_id] += 1

    def _register_deadline(self, task: TaskSpec) -> None:
        if task.task_id in self._deadline_task_ids:
            return
        self._deadline_task_ids.add(task.task_id)
        heapq.heappush(
            self._deadline_heap,
            (task.absolute_latest_start_sim, task.task_id),
        )

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

    def _uncommitted_count(self) -> int:
        counts = self.state_machine.count_by_state()
        return counts[TaskState.QUEUED] + counts[TaskState.PENDING_UNCOMMITTED]

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

    def ordered_queued_tasks(self) -> List[TaskSpec]:
        return self._ordered_from_heap()

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

    def _task_ids(self):
        return self.state_machine.task_ids
