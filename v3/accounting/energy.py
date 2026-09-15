"""Lossless sweep-line candidate accounting for the V3 runtime.

The frozen accounting formulas remain those from :mod:`v1.accounting.energy`.
Only construction and reuse of the candidate integral index changes:

* CPU allocations are grouped by node once per immutable reservation snapshot.
* Each allocation becomes ``start -> +power`` and ``end -> -power`` events.
* Existing load is produced by a sweep line instead of repeatedly scanning all
  allocations at every tariff/green boundary.
* The index covers the task's complete SLA window so the layered-pool anchor
  pass and second sampling pass share the same prefix arrays.
"""

from collections import OrderedDict, defaultdict
from typing import Optional

from v1.accounting.energy import (
    ExogenousEnergyAccounting,
    _CandidateIntegralIndex,
)
from v1.domain.models import TaskSpec
from v1.domain.reservations import TimeInterval
from v1.domain.units import finite_number


class SweepLineEnergyAccounting(ExogenousEnergyAccounting):
    """Exact candidate accounting backed by per-node load sweep lines."""

    SNAPSHOT_ALLOCATION_CACHE_CAPACITY = 64

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._snapshot_allocations_by_node_cache = OrderedDict()

    def __getstate__(self):
        state = super().__getstate__()
        # Snapshot identity is process-local and should never be checkpointed.
        state["_snapshot_allocations_by_node_cache"] = OrderedDict()
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if "_snapshot_allocations_by_node_cache" not in self.__dict__:
            self._snapshot_allocations_by_node_cache = OrderedDict()

    def _allocations_by_node(self, snapshot):
        """Group CPU allocations once for one immutable snapshot.

        ``id(snapshot)`` avoids hashing the full allocation tuples.  The cached
        snapshot object is retained and checked by identity, preventing an
        object-id reuse from returning stale data.
        """

        key = id(snapshot)
        cached = self._snapshot_allocations_by_node_cache.get(key)
        if cached is not None and cached[0] is snapshot:
            self._snapshot_allocations_by_node_cache.move_to_end(key)
            return cached[1]

        grouped = defaultdict(list)
        for allocation in snapshot.cpu_calendar_view:
            grouped[allocation.resource_id].append(allocation)
        result = {
            node: tuple(
                sorted(
                    allocations,
                    key=lambda item: (
                        item.interval_sim.start_sim,
                        item.interval_sim.end_sim,
                    ),
                )
            )
            for node, allocations in grouped.items()
        }
        self._snapshot_allocations_by_node_cache[key] = (snapshot, result)
        self._snapshot_allocations_by_node_cache.move_to_end(key)
        while (
            len(self._snapshot_allocations_by_node_cache)
            > self.SNAPSHOT_ALLOCATION_CACHE_CAPACITY
        ):
            self._snapshot_allocations_by_node_cache.popitem(last=False)
        return result

    def _node_power_events(
        self,
        snapshot,
        target_node: str,
        domain_start_sim: float,
        domain_end_sim: float,
    ):
        """Return clamped half-open load events for one node and interval."""

        events = defaultdict(float)
        allocations = self._allocations_by_node(snapshot).get(target_node, ())
        for allocation in allocations:
            interval = allocation.interval_sim
            if (
                interval.end_sim <= domain_start_sim
                or interval.start_sim >= domain_end_sim
            ):
                continue
            start = max(domain_start_sim, interval.start_sim)
            end = min(domain_end_sim, interval.end_sim)
            if end <= start:
                continue
            power = self.power_model.task_power_mw(allocation.amount)
            events[start] += power
            events[end] -= power
        return events

    def _build_candidate_integral_index(
        self,
        task: TaskSpec,
        target_node: str,
        reservation_snapshot,
        domain_start_sim: float,
        domain_end_sim: float,
    ) -> _CandidateIntegralIndex:
        """Build the frozen six candidate integrals with a load sweep line."""

        interval = TimeInterval(domain_start_sim, domain_end_sim)
        tariff, green = self._forecasts(target_node)
        power_events = self._node_power_events(
            reservation_snapshot,
            target_node,
            domain_start_sim,
            domain_end_sim,
        )
        boundaries = (
            set(tariff.boundaries(interval))
            | set(green.boundaries(interval))
            | set(power_events)
            | {domain_start_sim, domain_end_sim}
        )
        ordered = tuple(sorted(boundaries))
        if len(ordered) < 2:
            raise ValueError("candidate integral index has no positive interval")

        task_power = self.power_model.task_power_mw(task.cpu_demand)
        rate_rows = [[] for _ in range(6)]
        existing_power = 0.0
        for left, right in zip(ordered[:-1], ordered[1:]):
            # Applying all events at ``left`` preserves the half-open calendar
            # convention: starts are included and ends are excluded.
            existing_power += power_events.get(left, 0.0)
            if right <= left:
                continue
            probe = left + (right - left) / 2.0
            tariff_value = tariff.value_at(probe)
            green_value = green.value_at(probe)
            with_task = existing_power + task_power

            rate_rows[0].append(task_power)
            rate_rows[1].append(tariff_value * task_power)
            # Prefix indices are used only by the exogenous linear-billing
            # branch, matching the V1 implementation exactly.
            rate_rows[2].append(tariff_value * task_power)
            rate_rows[3].append(
                task_power * min(1.0, green_value / with_task)
            )
            rate_rows[4].append(
                min(green_value, with_task)
                - min(green_value, existing_power)
            )
            rate_rows[5].append(green_value)

        hours_per_sim = self.time_converter.sim_to_hours(1.0)
        rates = tuple(tuple(row) for row in rate_rows)
        prefixes = []
        for row in rates:
            values = [0.0]
            for index, rate in enumerate(row):
                duration_hours = (
                    ordered[index + 1] - ordered[index]
                ) * hours_per_sim
                values.append(values[-1] + rate * duration_hours)
            prefixes.append(tuple(values))
        return _CandidateIntegralIndex(
            ordered,
            rates,
            tuple(prefixes),
            hours_per_sim,
        )

    def _candidate_integral_index(
        self,
        task: TaskSpec,
        target_node: str,
        reservation_snapshot,
        first_start_sim: float,
        required_end_sim: Optional[float] = None,
    ) -> _CandidateIntegralIndex:
        """Return one full-SLA-window index shared by both candidate passes."""

        tariff, green = self._forecasts(target_node)
        forecast_start = max(
            tariff.segments[0].interval_sim.start_sim,
            green.segments[0].interval_sim.start_sim,
        )
        forecast_end = min(
            tariff.segments[-1].interval_sim.end_sim,
            green.segments[-1].interval_sim.end_sim,
        )
        # Validate optional arguments even though the cache domain deliberately
        # does not depend on the anchor pass's local request.
        local_start = finite_number("first_start_sim", first_start_sim)
        local_end = (
            task.absolute_latest_start_sim + task.execution_duration_sim
            if required_end_sim is None
            else finite_number("required_end_sim", required_end_sim)
        )
        domain_start = max(forecast_start, task.arrival_time_sim)
        domain_end = min(
            forecast_end,
            task.absolute_latest_start_sim + task.execution_duration_sim,
        )
        if local_start < domain_start - 1e-12 or local_end > domain_end + 1e-12:
            # Preserve the parent's defensive behavior for a request outside
            # physical forecast/SLA coverage.
            domain_start = max(forecast_start, min(domain_start, local_start))
            domain_end = min(forecast_end, max(domain_end, local_end))

        task_power = self.power_model.task_power_mw(task.cpu_demand)
        key = (
            reservation_snapshot.reservation_version,
            target_node,
            task_power,
            domain_start,
            domain_end,
        )
        cached = self._candidate_index_cache.get(key)
        if cached is not None:
            self._candidate_index_cache.move_to_end(key)
            return cached

        index = self._build_candidate_integral_index(
            task,
            target_node,
            reservation_snapshot,
            domain_start,
            domain_end,
        )
        self._candidate_index_cache[key] = index
        self._candidate_index_cache.move_to_end(key)
        while len(self._candidate_index_cache) > self._candidate_index_cache_capacity:
            self._candidate_index_cache.popitem(last=False)
        return index
