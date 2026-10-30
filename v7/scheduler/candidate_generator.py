"""Reuse task-static paths/durations; recompute decision-time bounds on retries."""

from collections import OrderedDict

from v1.candidate_profiling import candidate_scope
from v1.scheduler.candidate_generator import _time_grid_tick_bounds
from v5.scheduler.candidate_generator import V5CandidateGenerator


class V7CandidateGenerator(V5CandidateGenerator):
    grid_optimization_version = "task-static-path-duration-lru-v1"
    static_grid_cache_capacity = 256

    def _grid_cache(self):
        configuration = (
            self.path_provider, self.path_provider.topology_revision,
            self.path_provider.max_paths_per_target, self.path_provider.single_source_enabled,
            self.compute_nodes, self.transmission_model.converter,
            self.transmission_model.fiber_speed_km_per_second,
        )
        if getattr(self, "_grid_configuration", None) != configuration:
            self._grid_configuration = configuration
            self._static_grid_cache = OrderedDict()
            self._grid_stats = {"task_grid_hits": 0, "task_grid_misses": 0}
        return self._static_grid_cache

    def _declared_path_grids(self, task, decision_time):
        cache = self._grid_cache()
        if task in cache:
            cache.move_to_end(task)
            static = cache[task]
            self._grid_stats["task_grid_hits"] += 1
        else:
            self._grid_stats["task_grid_misses"] += 1
            with candidate_scope(self.profiler, "path_lookup"):
                paths = self.path_provider.candidate_paths_many(
                    task.source_node, self.compute_nodes, task.bandwidth_demand_mbps,
                )
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
        # Do not cache first/last ticks. The task can be retried at a later
        # time and active_wait_enabled/grid tolerance may change independently.
        with candidate_scope(self.profiler, "grid_boundaries"):
            grids = []
            for target, path, duration in static:
                first, last = _time_grid_tick_bounds(
                    decision_time + duration.total_sim, task.absolute_latest_start_sim,
                    self.scheduling_cycle_sim, self.time_tolerance,
                )
                if not self.active_wait_enabled:
                    last = first
                grids.append((target, path, duration, first, last))
        # The scope is closed before yielding, so consumer work is excluded.
        yield from grids

    def path_search_statistics(self):
        return {
            "enabled": True, "provider": self.path_provider.cache_statistics(),
            "retry_grids": dict(getattr(self, "_grid_stats", {}),
                                entries=len(getattr(self, "_static_grid_cache", ()))),
        }

    def __getstate__(self):
        state = dict(self.__dict__)
        for name in ("_grid_configuration", "_static_grid_cache", "_grid_stats"):
            state.pop(name, None)
        return state
