"""V8 static admission before path and time-grid search."""

from v1.candidate_profiling import candidate_scope
from v1.scheduler.candidate_generator import _time_grid_tick_bounds
from v7.scheduler.candidate_generator import V7CandidateGenerator


class V8CandidateGenerator(V7CandidateGenerator):
    grid_optimization_version = "static-hard-admission-v1"

    def _grid_cache(self):
        # The cached path set now depends on CPU capacity and tolerance too.
        cache = super()._grid_cache()
        configuration = (self.calendar, self.time_tolerance)
        if getattr(self, "_static_hard_configuration", None) != configuration:
            self._static_hard_configuration = configuration
            cache.clear()
            self._grid_stats = {"task_grid_hits": 0, "task_grid_misses": 0}
        if not hasattr(self, "_static_hard_stats"):
            self._static_hard_stats = {
                "static_cpu_nodes_rejected": 0,
                "path_queries_avoided": 0,
                "grid_preparations_avoided_empty_window": 0,
            }
        return cache

    def _declared_path_grids(self, task, decision_time):
        cache = self._grid_cache()
        if task in cache:
            cache.move_to_end(task)
            static = cache[task]
            self._grid_stats["task_grid_hits"] += 1
        else:
            self._grid_stats["task_grid_misses"] += 1
            eligible = []
            for target in self.compute_nodes:
                capacity = self.calendar.node_capacity(target)
                if capacity is None or task.cpu_demand > capacity + max(self.time_tolerance, 1e-12):
                    self._static_hard_stats["static_cpu_nodes_rejected"] += 1
                    self._static_hard_stats["path_queries_avoided"] += 1
                else:
                    eligible.append(target)
            with candidate_scope(self.profiler, "path_lookup"):
                paths = self.path_provider.candidate_paths_many(
                    task.source_node, eligible, task.bandwidth_demand_mbps,
                ) if eligible else ()
            with candidate_scope(self.profiler, "transmission_duration"):
                items = []
                for target, candidates in paths:
                    for path in candidates:
                        duration = self.transmission_model.duration(task, path)
                        if duration.static_path_feasible:
                            items.append((target, path, duration))
                static = tuple(items)
            cache[task] = static
            while len(cache) > self.static_grid_cache_capacity:
                cache.popitem(last=False)
        with candidate_scope(self.profiler, "grid_boundaries"):
            grids = []
            for target, path, duration in static:
                first, last = _time_grid_tick_bounds(
                    decision_time + duration.total_sim, task.absolute_latest_start_sim,
                    self.scheduling_cycle_sim, self.time_tolerance,
                )
                if first > last:
                    self._static_hard_stats["grid_preparations_avoided_empty_window"] += 1
                    continue
                if not self.active_wait_enabled:
                    last = first
                grids.append((target, path, duration, first, last))
        yield from grids

    def path_search_statistics(self):
        result = super().path_search_statistics()
        result["static_hard_pruning"] = dict(
            self._static_hard_stats if hasattr(self, "_static_hard_stats") else {},
            theoretical_slot_count_scope="post_static_hard_pruning_search_slots",
        )
        return result

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_static_hard_configuration", None)
        state.pop("_static_hard_stats", None)
        return state
