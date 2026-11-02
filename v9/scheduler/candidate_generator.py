"""Bounded, constraint-aware search over the unchanged K=1 path grids."""

import math
import time

from v1.candidate_profiling import candidate_count, candidate_scope
from v1.domain.candidates import CandidateGenerationStatus, CandidateMode
from v1.domain.models import SlaType
from v1.domain.reservations import TimeInterval
from v1.domain.units import finite_number
from v1.scheduler.candidate_generator import (
    CandidateGenerationContext, CompleteCandidateStream,
    _grid_tick_ceil, _grid_tick_floor,
)
from v8.scheduler.candidate_generator import V8CandidateGenerator


class _CountingCalendar:
    """Transparent per-search counters around the existing exact checks."""

    def __init__(self, calendar):
        self.calendar = calendar
        self.search_stats = None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "calendar"), name)

    def cpu_feasible(self, *args, **kwargs):
        if self.search_stats is not None:
            self.search_stats["cpu_checks"] += 1
        return self.calendar.cpu_feasible(*args, **kwargs)

    def path_feasible(self, *args, **kwargs):
        if self.search_stats is not None:
            self.search_stats["bandwidth_checks"] += 1
        return self.calendar.path_feasible(*args, **kwargs)


class V9CandidateGenerator(V8CandidateGenerator):
    search_version = "bounded-energy-adaptive-v1"

    def __init__(self, *args, search_stage=3, energy_accounting=None, **kwargs):
        if "calendar" in kwargs:
            kwargs["calendar"] = _CountingCalendar(kwargs["calendar"])
        else:
            args = (*args[:4], _CountingCalendar(args[4]), *args[5:])
        super().__init__(*args, **kwargs)
        if search_stage not in (1, 2, 3):
            raise ValueError("V9 generator search_stage must be 1..3")
        self.search_stage = search_stage
        self.energy_accounting = energy_accounting
        self.last_search_stats = {}
        self.search_totals = {"decisions": 0}

    @staticmethod
    def _unique(ticks, first, last):
        return tuple(dict.fromkeys(int(tick) for tick in ticks if first <= int(tick) <= last))

    def _initial_ticks(self, task, first, last):
        middle = (first + last) // 2
        ticks = [first, last, middle]
        if task.sla_type is not SlaType.HARD:
            preferred = _grid_tick_floor(task.absolute_preferred_start_sim,
                                         self.scheduling_cycle_sim, self.time_tolerance)
            ticks.extend((preferred, preferred + 1))
        return self._unique(ticks, first, last)

    def _resource_ticks(self, context, node, path, duration, first, last):
        cycle = self.scheduling_cycle_sim
        execution = context.task.execution_duration_sim
        boundaries = []
        for allocation in context.reservation_snapshot.cpu_calendar_view:
            if allocation.resource_id == node:
                boundaries.extend((allocation.interval_sim.end_sim,
                                   allocation.interval_sim.start_sim - execution))
        for allocation in context.reservation_snapshot.link_calendar_view:
            if allocation.resource_id in path.resource_edges:
                boundaries.extend((allocation.interval_sim.end_sim + duration.total_sim,
                                   allocation.interval_sim.start_sim))
        ticks = []
        for boundary in boundaries:
            tick = _grid_tick_ceil(boundary, cycle, self.time_tolerance)
            ticks.extend((tick - 1, tick, tick + 1))
        return self._unique(sorted(ticks), first, last)

    def _energy_ticks(self, context, node, first, last):
        accounting = self.energy_accounting
        if accounting is None:
            return ()
        tariff, green = accounting._forecasts(node)
        cycle = self.scheduling_cycle_sim
        duration = context.task.execution_duration_sim
        window = TimeInterval(first * cycle, last * cycle + duration)
        boundaries = set(tariff.boundaries(window)) | set(green.boundaries(window))
        ticks = []
        for boundary in sorted(boundaries):
            for start in (boundary, boundary - duration):
                tick = _grid_tick_ceil(start, cycle, self.time_tolerance)
                ticks.extend((tick - 1, tick))
        return self._unique(ticks, first, last)

    def _energy_preview(self, context, node, tick, stats, cache):
        key = (node, tick)
        if key in cache:
            return cache[key]
        accounting = self.energy_accounting
        if accounting is None or stats["energy_preview_count"] >= stats["energy_preview_budget"]:
            return None
        start = tick * self.scheduling_cycle_sim
        end = start + context.task.execution_duration_sim
        if context.forecast_covered_until_sim is not None and end > context.forecast_covered_until_sim + self.time_tolerance:
            return None
        started = time.perf_counter()
        metrics = accounting.evaluate_candidate(
            task=context.task, target_node=node, compute_start_sim=start,
            compute_end_sim=end, reservation_snapshot=context.reservation_snapshot,
        )
        stats["energy_preview_count"] += 1
        stats["exact_energy_count"] += 1
        stats["energy_preview_seconds"] += time.perf_counter() - started
        candidate_count(self.profiler, "v9_energy_preview_count")
        cache[key] = metrics
        return metrics

    @staticmethod
    def _preview_key(metrics, tick):
        if metrics is None:
            return (math.inf, 0, tick)
        return (metrics.candidate_marginal_system_cost_yuan,
                -metrics.candidate_marginal_green_energy_mwh, tick)

    def _ordered_ticks(self, context, grid, stats, preview_cache):
        node, path, duration, first, last = grid
        initial = self._initial_ticks(context.task, first, last)
        resources = self._resource_ticks(context, node, path, duration, first, last)
        uniform = self._adaptive_ticks(context.task, first, last,
                                        self.pool_time_samples_by_sla[context.task.sla_type.value])
        if self.search_stage < 2 or self.energy_accounting is None:
            return self._unique((*initial, *resources, *uniform), first, last)
        energy = self._energy_ticks(context, node, first, last)
        # Preview only a bounded set of boundary starts; no energy-only admission.
        preview = self._unique((*initial, *energy), first, last)[:12]
        ranked = []
        for tick in preview:
            metrics = self._energy_preview(context, node, tick, stats, preview_cache)
            if metrics is not None:
                ranked.append((self._preview_key(metrics, tick), tick))
        ranked.sort()
        return self._unique((*initial, *resources[:2], *(tick for _, tick in ranked),
                             *resources[2:], *uniform), first, last)

    def _probe(self, context, grid, tick, state, stats):
        node, path, duration, _, _ = grid
        key = (node, path.path_id, tick)
        if key in state["tested"]:
            return
        state["tested"].add(key)
        stats["time_probes"] += 1
        candidate_count(self.profiler, "v9_time_probes")
        # The inherited record path performs CPU, path and SLA checks.
        record = self._record_at_tick(context, node, path, duration, tick)
        if record is not None:
            state["records"][record.candidate_id] = record
            state["feasible"] += 1

    def prepare_layered_pool_stream(self, task, decision_time_sim,
                                    reservation_snapshot=None, forecast_version="perfect-v1",
                                    forecast_covered_until_sim=None, metric_evaluator=None):
        # V5's decision-local record and batch materialization caches are reused.
        previous = getattr(self, "_generation_cache", None)
        cache = {"context": None, "records": {}, "candidates": {}, "capacities": {},
                 "stats": {"record_cache_hits": 0, "candidate_cache_hits": 0,
                           "static_cpu_rejections": 0}}
        self._generation_cache = cache
        try:
            return self._prepare_v9(task, decision_time_sim, reservation_snapshot,
                                    forecast_version, forecast_covered_until_sim,
                                    metric_evaluator)
        finally:
            self.calendar.search_stats = None
            self.last_generation_stats = dict(cache["stats"])
            self._generation_cache = previous

    def _prepare_v9(self, task, decision_time_sim, reservation_snapshot,
                    forecast_version, forecast_covered_until_sim, metric_evaluator):
        started = time.perf_counter()
        decision_time = finite_number("decision_time_sim", decision_time_sim)
        snapshot = reservation_snapshot or self.calendar.snapshot()
        forecast_limit = (None if forecast_covered_until_sim is None else
                          finite_number("forecast_covered_until_sim", forecast_covered_until_sim))
        context = CandidateGenerationContext(task, decision_time, snapshot,
                                             forecast_version, forecast_limit,
                                             decision_time, CandidateMode.LAYERED_POOL)
        stats = {"stage": f"E{self.search_stage}", "time_probes": 0,
                 "cpu_checks": 0, "bandwidth_checks": 0,
                 "energy_preview_count": 0, "exact_energy_count": 0,
                 "energy_preview_seconds": 0.0, "feasible_nodes": 0,
                 "proven_infeasible_nodes": 0, "unresolved_nodes": 0,
                 "time_probe_budget": 0, "energy_preview_budget": 0,
                 "exact_energy_budget": 0,
                 "candidate_pool_capacity": self.pool_max_by_sla[task.sla_type.value]}
        self.calendar.search_stats = stats
        try:
            if decision_time > task.absolute_latest_start_sim + self.time_tolerance:
                return CompleteCandidateStream(self, context,
                    CandidateGenerationStatus.EXPIRED_BEFORE_DECISION, 0, 0,
                    "decision time is after absolute latest start", metric_evaluator)
            with candidate_scope(self.profiler, "grid_setup"):
                grids = [g for g in self._declared_path_grids(task, decision_time) if g[3] <= g[4]]
                slots = sum(g[4] - g[3] + 1 for g in grids)
            samples = self.pool_time_samples_by_sla[task.sla_type.value]
            # E1/E2 preserve a bounded per-grid search. E3 has the same total
            # probe ceiling but allocates the unused part across nodes.
            per_grid = max(4, samples)
            stats["time_probe_budget"] = sum(min(per_grid, g[4] - g[3] + 1) for g in grids)
            stats["energy_preview_budget"] = (0 if self.search_stage < 2 else 12 * len(grids))
            stats["exact_energy_budget"] = (stats["time_probe_budget"]
                                            + stats["energy_preview_budget"])
            preview_cache = {}
            states = []
            for grid in grids:
                node, path, duration, first, last = grid
                # Forecast coverage proves all starts after this tick invalid.
                if forecast_limit is not None:
                    last = min(last, _grid_tick_floor(forecast_limit - task.execution_duration_sim,
                                                      self.scheduling_cycle_sim, self.time_tolerance))
                if first > last:
                    stats["proven_infeasible_nodes"] += 1
                    continue
                grid = (node, path, duration, first, last)
                ticks = self._ordered_ticks(context, grid, stats, preview_cache)
                energy_priority = min((self._preview_key(preview_cache[(node, tick)], tick)
                    for tick in ticks if (node, tick) in preview_cache),
                    default=(math.inf, 0, math.inf))
                states.append({"grid": grid, "ticks": ticks, "tested": set(),
                               "records": {}, "feasible": 0,
                               "energy_priority": energy_priority})
            # Every static-feasible node receives coverage of both ends and middle.
            for state in states:
                grid = state["grid"]
                base = min(len(state["ticks"]), 5 if self.search_stage < 3 else 3)
                for tick in state["ticks"][:base]:
                    if stats["time_probes"] >= stats["time_probe_budget"]:
                        break
                    self._probe(context, grid, tick, state, stats)
            if self.search_stage == 3:
                # Allocate one probe at a time. A per-grid cap prevents one
                # unresolved node from consuming the entire global budget.
                while stats["time_probes"] < stats["time_probe_budget"]:
                    pending = [s for s in states if len(s["tested"]) <
                               min(len(s["ticks"]), 2 * per_grid)]
                    if not pending:
                        break
                    def priority(state):
                        records = state["records"].values()
                        best = min((r.compute_start_sim for r in records), default=math.inf)
                        rate = state["feasible"] / max(1, len(state["tested"]))
                        unexplored = (len(state["ticks"]) - len(state["tested"])) / len(state["ticks"])
                        return (-int(not state["feasible"]), -unexplored,
                                -rate, state["energy_priority"], best, state["grid"][0])
                    state = min(pending, key=priority)
                    tick = next(t for t in state["ticks"] if
                                (state["grid"][0], state["grid"][1].path_id, t) not in state["tested"])
                    self._probe(context, state["grid"], tick, state, stats)
            else:
                for state in states:
                    for tick in state["ticks"]:
                        if len(state["tested"]) >= min(per_grid, len(state["ticks"])):
                            break
                        self._probe(context, state["grid"], tick, state, stats)
            records = {r.candidate_id: r for s in states for r in s["records"].values()}
            stats["feasible_nodes"] = len({s["grid"][0] for s in states if s["feasible"]})
            stats["unresolved_nodes"] = len({s["grid"][0] for s in states if
                not s["feasible"] and len(s["tested"]) < s["grid"][4] - s["grid"][3] + 1})
            stats["proven_infeasible_nodes"] += len({s["grid"][0] for s in states if
                not s["feasible"] and len(s["tested"]) == s["grid"][4] - s["grid"][3] + 1})
            if not records:
                return CompleteCandidateStream(self, context, CandidateGenerationStatus.EMPTY_PHYSICAL,
                    slots, 0, "bounded search found no candidate; unresolved nodes may remain",
                    metric_evaluator)
            # Materialize once through the inherited vector evaluator and cache.
            with candidate_scope(self.profiler, "materialization_prepare"):
                candidates = self.materialize_context_candidates(context, tuple(records.values()), metric_evaluator)
            stats["exact_energy_count"] += len(candidates) if metric_evaluator is not None else 0
            by_node = {}
            for candidate in candidates:
                by_node.setdefault(candidate.target_node, []).append(candidate)
            nodes = tuple(by_node)
            limit = min(self.pool_node_limit_by_sla[task.sla_type.value], len(nodes))
            regions = {}
            for node in sorted(nodes, key=lambda n: min(
                    c.estimated_candidate_marginal_system_cost_yuan for c in by_node[n])):
                region = self.path_provider.graph.nodes[node].get("region", node)
                regions.setdefault(region, node)
            ranks = (
                tuple(regions.values()),
                sorted(nodes, key=lambda n: min(c.compute_end_sim for c in by_node[n])),
                sorted(nodes, key=lambda n: min(c.estimated_candidate_marginal_system_cost_yuan for c in by_node[n])),
                sorted(nodes, key=lambda n: -max(c.estimated_green_coverage + c.estimated_green_absorption_delta for c in by_node[n])),
                sorted(nodes, key=lambda n: -max(c.capacity_margin for c in by_node[n])),
            )
            chosen_nodes = set(self._round_robin_unique(ranks, limit))
            global_earliest = min(candidates, key=lambda c: (
                c.compute_start_sim, c.target_node, c.path.path_id, c.candidate_id))
            if global_earliest.target_node not in chosen_nodes:
                # Include its node in the bounded pool; remove the last-ranked one.
                chosen_nodes.remove(next(n for n in reversed(self._round_robin_unique(ranks, limit))
                                         if n in chosen_nodes))
                chosen_nodes.add(global_earliest.target_node)
            filtered = tuple(c for c in candidates if c.target_node in chosen_nodes)
            selected = list(self._select_layered_pool(filtered,
                           self.pool_max_by_sla[task.sla_type.value]))
            # Wait Gate must see a real feasible earliest explored candidate.
            earliest = min(filtered, key=lambda c: (c.compute_start_sim, c.target_node, c.path.path_id, c.candidate_id))
            if earliest.candidate_id not in {c.candidate_id for c in selected}:
                if len(selected) >= self.pool_max_by_sla[task.sla_type.value]:
                    selected.pop()
                selected.append(earliest)
            chosen = {c.candidate_id: c for c in selected}
            selected_records = tuple(sorted((records[k] for k in chosen),
                key=lambda r: (r.compute_start_sim, r.target_node, r.path.path_id, r.candidate_id)))
            selected_metrics = tuple({
                "system_cost_yuan": chosen[r.candidate_id].estimated_candidate_marginal_system_cost_yuan,
                "green_coverage": chosen[r.candidate_id].estimated_green_coverage,
                "marginal_green_energy_mwh": chosen[r.candidate_id].estimated_candidate_marginal_green_energy_mwh,
                "green_absorption_delta": chosen[r.candidate_id].estimated_green_absorption_delta,
                "green_opportunity": chosen[r.candidate_id].estimated_green_opportunity,
            } for r in selected_records)
            final_context = CandidateGenerationContext(task, decision_time, snapshot,
                forecast_version, forecast_limit, selected_records[0].compute_start_sim,
                CandidateMode.LAYERED_POOL, selected_records, selected_metrics)
            return CompleteCandidateStream(self, final_context, CandidateGenerationStatus.OK,
                                           slots, len(selected_records), "", metric_evaluator)
        finally:
            stats["generation_seconds"] = time.perf_counter() - started
            self.last_search_stats = stats
            self.search_totals["decisions"] += 1
            for key, value in stats.items():
                if key != "stage" and key != "candidate_pool_capacity" and isinstance(value, (int, float)):
                    self.search_totals[key] = self.search_totals.get(key, 0) + value

    def path_search_statistics(self):
        result = super().path_search_statistics()
        result["v9_search"] = {
            "stage": f"E{self.search_stage}",
            "totals": dict(self.search_totals),
            "last_decision": dict(self.last_search_stats),
            "wait_earliest_scope": "earliest feasible among explored slots",
        }
        return result
