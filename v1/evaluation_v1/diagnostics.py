"""Derived four-level diagnostics for frozen-policy evaluation reports.

All values in this module are reconstructed from committed reservations,
terminal task records, accounting records, and measured scheduler decisions.
The diagnostics therefore observe a policy without changing its decisions.
"""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Iterable, Mapping

from v1.domain.reservations import Reservation, TimeInterval, canonical_edge

from .metrics import linear_percentile
from .statistics import UtilizationInterval, summarize_load


DIAGNOSTIC_SCHEMA_VERSION = "1.0"

METRIC_DEFINITIONS = {
    "task.scheduler_queue_delay": "decision_time - arrival_time for reserved tasks",
    "task.start_delay": "compute_start_time - arrival_time",
    "task.completion_delay": "compute_end_time - arrival_time for completed tasks",
    "time.cpu_utilization": "reserved CPU / configured CPU capacity, time weighted",
    "time.node_load_cv": "population coefficient of variation of node CPU utilizations",
    "node.cpu_utilization": "allocated CPU-time / available CPU-time over the evaluation interval",
    "node.allocation_share": "node allocated CPU-hours / system allocated CPU-hours",
    "network.link_utilization": "reserved bandwidth / configured link capacity, time weighted",
    "network.remote_task_rate": "remote completed reservations / completed reservations",
    "runtime.decision_wall": "wall time around one scheduler decision, including candidate generation, policy selection, and commit",
    "runtime.evaluation_wall": "wall time of the three-phase frozen-policy runner; report serialization is excluded",
}


def _summary(values: Iterable[float]) -> dict:
    items = tuple(float(value) for value in values if value is not None)
    if not items:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "maximum": None}
    return {
        "count": len(items),
        "mean": statistics.fmean(items),
        "p50": linear_percentile(items, 50.0).value,
        "p95": linear_percentile(items, 95.0).value,
        "maximum": max(items),
    }


def _weighted_percentile(value_weights, percentile):
    items = sorted(
        (float(value), float(weight))
        for value, weight in value_weights
        if weight > 0.0
    )
    if not items:
        return None
    threshold = percentile / 100.0 * math.fsum(weight for _, weight in items)
    cumulative = 0.0
    for value, weight in items:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return items[-1][0]


def _add_interval_events(events, interval, resource, amount, start, end):
    left = max(start, interval.start_sim)
    right = min(end, interval.end_sim)
    if right <= left:
        return
    events[left][resource] += amount
    events[right][resource] -= amount


def _resource_records(
    reservations: tuple[Reservation, ...],
    *,
    node_capacities: Mapping[str, float],
    link_capacities: Mapping[tuple[str, str], float],
    start_sim: float,
    end_sim: float,
    time_converter,
    accounting_report,
    energy_accounting=None,
):
    cpu_events = defaultdict(lambda: defaultdict(float))
    link_events = defaultdict(lambda: defaultdict(float))
    compute_count_events = defaultdict(int)
    transmit_count_events = defaultdict(int)
    boundaries = {start_sim, end_sim}
    if energy_accounting is not None and end_sim > start_sim:
        evaluation_interval = TimeInterval(start_sim, end_sim)
        for node in node_capacities:
            boundaries.update(
                energy_accounting.tariff_by_node[node].boundaries(
                    evaluation_interval
                )
            )
            boundaries.update(
                energy_accounting.green_by_node[node].boundaries(
                    evaluation_interval
                )
            )

    for reservation in reservations:
        compute = reservation.compute_interval_sim
        left = max(start_sim, compute.start_sim)
        right = min(end_sim, compute.end_sim)
        if right > left:
            boundaries.update((left, right))
            _add_interval_events(
                cpu_events, compute, reservation.target_node,
                reservation.cpu_amount, start_sim, end_sim,
            )
            compute_count_events[left] += 1
            compute_count_events[right] -= 1
        transmission = reservation.transmission_interval_sim
        if transmission is not None:
            left = max(start_sim, transmission.start_sim)
            right = min(end_sim, transmission.end_sim)
            if right > left:
                boundaries.update((left, right))
                for edge in reservation.path.resource_edges:
                    _add_interval_events(
                        link_events, transmission, edge,
                        reservation.bandwidth_amount_mbps, start_sim, end_sim,
                    )
                transmit_count_events[left] += 1
                transmit_count_events[right] -= 1

    ordered = sorted(boundaries)
    cpu_used = {node: 0.0 for node in node_capacities}
    link_used = {edge: 0.0 for edge in link_capacities}
    node_integrals = {node: 0.0 for node in node_capacities}
    node_peaks = {node: 0.0 for node in node_capacities}
    link_integrals = {edge: 0.0 for edge in link_capacities}
    link_peaks = {edge: 0.0 for edge in link_capacities}
    time_records = []
    load_intervals = []
    link_weighted_values = []
    active_compute = 0
    active_transmit = 0

    for left, right in zip(ordered[:-1], ordered[1:]):
        for node, delta in cpu_events.get(left, {}).items():
            cpu_used[node] = cpu_used.get(node, 0.0) + delta
        for edge, delta in link_events.get(left, {}).items():
            link_used[edge] = link_used.get(edge, 0.0) + delta
        active_compute += compute_count_events.get(left, 0)
        active_transmit += transmit_count_events.get(left, 0)
        if right <= left:
            continue
        duration_sim = right - left
        duration_seconds = time_converter.sim_to_seconds(duration_sim)
        node_utilizations = {
            node: cpu_used.get(node, 0.0) / capacity
            for node, capacity in node_capacities.items()
        }
        link_utilizations = {
            f"{edge[0]}--{edge[1]}": link_used.get(edge, 0.0) / capacity
            for edge, capacity in link_capacities.items()
        }
        node_values = tuple(node_utilizations.values())
        link_values = tuple(link_utilizations.values())
        computing_load = math.fsum(cpu_used.values())
        total_energy_demand_mw = None
        renewable_generation_mw = None
        renewable_used_mw = None
        electricity_price_yuan_per_mwh = None
        green_coverage_rate = None
        if energy_accounting is not None:
            probe = left + (right - left) / 2.0
            power_by_node = {
                node: (
                    energy_accounting.power_model.task_power_mw(amount)
                    if amount > 0.0 else 0.0
                )
                for node, amount in cpu_used.items()
            }
            total_energy_demand_mw = math.fsum(power_by_node.values())
            renewable_by_node = {
                node: energy_accounting.green_by_node[node].value_at(probe)
                for node in node_capacities
            }
            tariff_by_node = {
                node: energy_accounting.tariff_by_node[node].value_at(probe)
                for node in node_capacities
            }
            renewable_generation_mw = math.fsum(
                renewable_by_node.values()
            )
            renewable_used_mw = math.fsum(
                min(power_by_node[node], renewable_by_node[node])
                for node in node_capacities
            )
            electricity_price_yuan_per_mwh = (
                math.fsum(
                    tariff_by_node[node] * power_by_node[node]
                    for node in node_capacities
                ) / total_energy_demand_mw
                if total_energy_demand_mw > 0.0
                else statistics.fmean(tariff_by_node.values())
            )
            green_coverage_rate = (
                renewable_used_mw / total_energy_demand_mw
                if total_energy_demand_mw > 0.0 else None
            )
        load_intervals.append(UtilizationInterval(duration_seconds, node_values))
        for node, utilization in node_utilizations.items():
            node_integrals[node] += utilization * duration_seconds
            node_peaks[node] = max(node_peaks[node], utilization)
        for edge, capacity in link_capacities.items():
            utilization = link_used.get(edge, 0.0) / capacity
            link_integrals[edge] += utilization * duration_seconds
            link_peaks[edge] = max(link_peaks[edge], utilization)
            link_weighted_values.append((utilization, duration_seconds))
        node_mean = statistics.fmean(node_values) if node_values else 0.0
        node_cv = (
            0.0
            if not node_values or node_mean == 0.0
            else statistics.pstdev(node_values) / node_mean
        )
        time_records.append({
            "time": left,
            "start_sim": left,
            "end_sim": right,
            "duration_sim": duration_sim,
            "duration_seconds": duration_seconds,
            "active_compute_task_count": active_compute,
            "active_transmission_task_count": active_transmit,
            "active_tasks": active_compute,
            "computing_load": computing_load,
            "total_energy_demand": total_energy_demand_mw,
            "renewable_generation": renewable_generation_mw,
            "renewable_used": renewable_used_mw,
            "electricity_price": electricity_price_yuan_per_mwh,
            "green_coverage_rate": green_coverage_rate,
            "mean_node_cpu_utilization": node_mean,
            "max_node_cpu_utilization": max(node_values, default=0.0),
            "node_load_cv": node_cv,
            "mean_link_utilization": statistics.fmean(link_values) if link_values else 0.0,
            "max_link_utilization": max(link_values, default=0.0),
            "node_utilizations": node_utilizations,
            "link_utilizations": link_utilizations,
        })

    total_seconds = time_converter.sim_to_seconds(end_sim - start_sim)
    load = summarize_load(load_intervals) if load_intervals else None
    time_summary = {
        "interval_count": len(time_records),
        "time_node_mean_cpu_utilization": (
            None if load is None else load.time_node_mean_utilization
        ),
        "weighted_p95_node_cpu_utilization": (
            None if load is None else load.weighted_p95_utilization
        ),
        "maximum_node_cpu_utilization": None if load is None else load.maximum_utilization,
        "time_weighted_node_load_cv": None if load is None else load.time_weighted_node_cv,
        "cpu_hotspot_time_ratio": None if load is None else load.hotspot_time_ratio,
        "cpu_overcapacity_time_ratio": (
            None if load is None else load.physical_overcapacity_time_ratio
        ),
        "time_link_mean_utilization": (
            math.fsum(value * weight for value, weight in link_weighted_values)
            / math.fsum(weight for _, weight in link_weighted_values)
            if link_weighted_values else None
        ),
        "weighted_p95_link_utilization": _weighted_percentile(
            link_weighted_values, 95.0
        ),
        "maximum_link_utilization": max(link_peaks.values(), default=0.0),
    }

    accounting_by_node = defaultdict(lambda: {
        "task_count": 0,
        "task_energy_mwh": 0.0,
        "task_cost_yuan": 0.0,
        "task_green_energy_mwh": 0.0,
    })
    if accounting_report is not None:
        for record in accounting_report.task_records:
            values = accounting_by_node[record.target_node]
            values["task_count"] += 1
            values["task_energy_mwh"] += record.task_energy_mwh
            values["task_cost_yuan"] += record.task_attributed_cost_yuan
            values["task_green_energy_mwh"] += (
                record.task_attributed_green_energy_mwh
            )

    cpu_hours_by_node = defaultdict(float)
    reservation_count_by_node = defaultdict(int)
    for reservation in reservations:
        overlap = max(
            0.0,
            min(end_sim, reservation.compute_interval_sim.end_sim)
            - max(start_sim, reservation.compute_interval_sim.start_sim),
        )
        if overlap > 0.0:
            cpu_hours_by_node[reservation.target_node] += (
                reservation.cpu_amount * time_converter.sim_to_hours(overlap)
            )
            reservation_count_by_node[reservation.target_node] += 1
    total_cpu_hours = math.fsum(cpu_hours_by_node.values())
    node_records = []
    accounting_interval = (
        TimeInterval(start_sim, end_sim) if end_sim > start_sim else None
    )
    for node, capacity in sorted(node_capacities.items()):
        accounting = accounting_by_node[node]
        energy = accounting["task_energy_mwh"]
        green_available = None
        if energy_accounting is not None and accounting_interval is not None:
            forecast = energy_accounting.green_by_node[node]
            boundaries = forecast.boundaries(accounting_interval)
            green_available = math.fsum(
                forecast.value_at(left + (right - left) / 2.0)
                * time_converter.sim_to_hours(right - left)
                for left, right in zip(boundaries[:-1], boundaries[1:])
                if right > left
            )
        green_used = (
            None if accounting_report is None
            else accounting_report.node_green_used_mwh.get(node, 0.0)
        )
        electricity_cost = (
            None if accounting_report is None
            else accounting_report.node_bill_yuan.get(node, 0.0)
        )
        node_records.append({
            "node_id": node,
            "node": node,
            "capacity_cpu": capacity,
            "assigned_tasks": reservation_count_by_node[node],
            "reservation_count": reservation_count_by_node[node],
            "total_cpu_hours": cpu_hours_by_node[node],
            "allocated_cpu_hours": cpu_hours_by_node[node],
            "allocation_share": (
                cpu_hours_by_node[node] / total_cpu_hours
                if total_cpu_hours > 0.0 else None
            ),
            "time_weighted_cpu_utilization": (
                node_integrals[node] / total_seconds if total_seconds > 0.0 else None
            ),
            "average_cpu_utilization": (
                node_integrals[node] / total_seconds if total_seconds > 0.0 else None
            ),
            "peak_cpu_utilization": node_peaks[node],
            "total_energy_consumption": energy,
            "task_energy_mwh": energy,
            "electricity_cost": electricity_cost,
            "task_attributed_cost_yuan": accounting["task_cost_yuan"],
            "green_energy_available": green_available,
            "green_energy_used": green_used,
            "task_attributed_green_energy_mwh": accounting["task_green_energy_mwh"],
            "green_coverage_rate": (
                accounting["task_green_energy_mwh"] / energy
                if energy > 0.0 else None
            ),
            "task_green_coverage": (
                accounting["task_green_energy_mwh"] / energy if energy > 0.0 else None
            ),
        })

    link_records = []
    for edge, capacity in sorted(link_capacities.items()):
        link_records.append({
            "edge": f"{edge[0]}--{edge[1]}",
            "capacity_mbps": capacity,
            "time_weighted_utilization": (
                link_integrals[edge] / total_seconds if total_seconds > 0.0 else None
            ),
            "peak_utilization": link_peaks[edge],
        })
    return time_records, time_summary, node_records, link_records


def build_evaluation_diagnostics(
    report,
    *,
    reservations: Iterable[Reservation],
    node_capacities: Mapping[str, float],
    link_capacities: Mapping[tuple[str, str], float],
    time_converter,
    profiler_summary: Mapping[str, object],
    energy_accounting=None,
) -> dict:
    """Build auditable task/time/node/system diagnostics for one report."""

    items = tuple(reservations)
    start_sim = report.metadata.evaluation_start_sim
    end_sim = report.metadata.final_settlement_time_sim
    if end_sim <= start_sim:
        end_sim = report.metadata.arrival_cutoff_sim
    normalized_links = {
        canonical_edge(edge): float(capacity)
        for edge, capacity in link_capacities.items()
        if capacity is not None
    }
    (
        time_records,
        time_summary,
        node_records,
        link_records,
    ) = _resource_records(
        items,
        node_capacities=node_capacities,
        link_capacities=normalized_links,
        start_sim=start_sim,
        end_sim=end_sim,
        time_converter=time_converter,
        accounting_report=report.accounting_report,
        energy_accounting=energy_accounting,
    )

    task_records = tuple(report.task_records)
    decision_records = tuple(report.decision_records)
    decision_seconds = tuple(
        decision.decision_wall_seconds
        for decision in decision_records
        if decision.decision_wall_seconds is not None
    )
    task_summary = {
        "scheduler_queue_delay_sim": _summary(
            record.scheduler_queue_delay_sim for record in task_records
        ),
        "start_delay_sim": _summary(
            record.start_delay_sim for record in task_records
        ),
        "completion_delay_sim": _summary(
            record.completion_delay_sim for record in task_records
        ),
        "reservation_lead_sim": _summary(
            record.reservation_lead_sim for record in task_records
        ),
        "decision_wall_seconds": _summary(decision_seconds),
        "decision_count_per_arrived_task": (
            len(decision_records) / len(task_records) if task_records else None
        ),
        "candidate_count_per_decision": _summary(
            decision.candidate_count for decision in decision_records
        ),
    }

    remote = tuple(
        reservation for reservation in items if not reservation.path.is_local
    )
    transmission_seconds = tuple(
        time_converter.sim_to_seconds(
            reservation.transmission_interval_sim.duration_sim
        )
        for reservation in remote
    )
    transmitted_data_mb = math.fsum(
        reservation.bandwidth_amount_mbps * seconds / 8.0
        for reservation, seconds in zip(remote, transmission_seconds)
    )
    network_summary = {
        "reservation_count": len(items),
        "remote_reservation_count": len(remote),
        "remote_task_rate": len(remote) / len(items) if items else None,
        "transmission_duration_seconds": _summary(transmission_seconds),
        "total_transmitted_data_mb": transmitted_data_mb,
        "mean_path_hops": (
            statistics.fmean(len(item.path.ordered_edges) for item in remote)
            if remote else None
        ),
        "mean_path_distance_km": (
            statistics.fmean(item.path.total_distance_km for item in remote)
            if remote else None
        ),
        "total_data_distance_mb_km": math.fsum(
            reservation.bandwidth_amount_mbps * seconds / 8.0
            * reservation.path.total_distance_km
            for reservation, seconds in zip(remote, transmission_seconds)
        ),
        "time_link_mean_utilization": time_summary[
            "time_link_mean_utilization"
        ],
        "weighted_p95_link_utilization": time_summary[
            "weighted_p95_link_utilization"
        ],
        "maximum_link_utilization": time_summary["maximum_link_utilization"],
    }

    allocation_values = [
        record["allocated_cpu_hours"] for record in node_records
    ]
    allocation_mean = (
        statistics.fmean(allocation_values) if allocation_values else 0.0
    )
    node_summary = {
        "node_count": len(node_records),
        "used_node_count": sum(
            record["reservation_count"] > 0 for record in node_records
        ),
        "allocated_cpu_hours_cv": (
            statistics.pstdev(allocation_values) / allocation_mean
            if allocation_values and allocation_mean > 0.0 else 0.0
        ),
        "time_node_mean_cpu_utilization": time_summary[
            "time_node_mean_cpu_utilization"
        ],
        "weighted_p95_node_cpu_utilization": time_summary[
            "weighted_p95_node_cpu_utilization"
        ],
        "maximum_node_cpu_utilization": time_summary[
            "maximum_node_cpu_utilization"
        ],
        "time_weighted_node_load_cv": time_summary[
            "time_weighted_node_load_cv"
        ],
        "cpu_hotspot_time_ratio": time_summary["cpu_hotspot_time_ratio"],
        "cpu_overcapacity_time_ratio": time_summary[
            "cpu_overcapacity_time_ratio"
        ],
    }

    physical_hours = time_converter.sim_to_hours(end_sim - start_sim)
    completed_count = (
        0 if report.metrics is None else report.metrics.completed_count
    )
    system_summary = {
        "evaluation_span_sim": end_sim - start_sim,
        "evaluation_span_physical_hours": physical_hours,
        "completed_tasks_per_physical_hour": (
            completed_count / physical_hours if physical_hours > 0.0 else None
        ),
        "reservation_count": len(items),
        "time_interval_count": len(time_records),
    }
    runtime_summary = dict(profiler_summary)
    runtime_summary["decision_wall_seconds"] = _summary(decision_seconds)
    runtime_summary["decision_seconds_per_arrived_task"] = (
        math.fsum(decision_seconds) / len(task_records) if task_records else None
    )

    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "system_summary": system_summary,
        "task_summary": task_summary,
        "time_summary": time_summary,
        "node_summary": node_summary,
        "network_summary": network_summary,
        "runtime_summary": runtime_summary,
        "time_records": time_records,
        "node_records": node_records,
        "network_records": link_records,
    }
