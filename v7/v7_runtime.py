"""Install V7 static-path/grid caches while retaining V6 accounting and policy."""

from pathlib import Path

from v6.v6_runtime import create_v6_runtime
from .scheduler import CachedStaticPathProvider, V7CandidateGenerator


def path_config_view(enabled=True):
    return {
        "V7_PATH_SEARCH_CACHE": bool(enabled),
        "V7_PATH_CACHE_VERSION": CachedStaticPathProvider.optimization_version,
        "V7_STATIC_GRID_CACHE_VERSION": V7CandidateGenerator.grid_optimization_version,
        "V7_UNIQUE_SINGLE_SOURCE_GUARD": True,
    }


def create_v7_runtime(*, path_search_cache=True, **kwargs):
    runtime = create_v6_runtime(**kwargs)
    if path_search_cache:
        old = runtime.scheduler.candidate_generator
        provider = CachedStaticPathProvider(old.path_provider.graph, old.path_provider.max_paths_per_target)
        generator = V7CandidateGenerator(
            old.compute_nodes, old.scheduling_cycle_sim, provider,
            old.transmission_model, old.calendar, old.time_tolerance,
            candidate_mode=old.candidate_mode.value,
            active_wait_enabled=old.active_wait_enabled,
            pool_max_by_sla=old.pool_max_by_sla,
            pool_node_limit_by_sla=old.pool_node_limit_by_sla,
            pool_time_samples_by_sla=old.pool_time_samples_by_sla,
        )
        generator.profiler = old.profiler
        runtime.scheduler.candidate_generator = generator
    runtime.config_view_overrides.update(path_config_view(path_search_cache))
    runtime.evaluation_code_files += (
        Path(__file__), Path(__file__).parent / "scheduler" / "path_provider.py",
        Path(__file__).parent / "scheduler" / "candidate_generator.py",
    )
    return runtime
