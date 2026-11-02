"""Install V8 static admission on the unchanged V7 scheduler stack."""

from pathlib import Path

from v7.v7_runtime import create_v7_runtime

from .scheduler import V8CandidateGenerator


def static_hard_config_view():
    return {
        "V8_STATIC_HARD_PRUNING": True,
        "V8_STATIC_HARD_VERSION": V8CandidateGenerator.grid_optimization_version,
        "V8_THEORETICAL_SLOT_COUNT_SCOPE": "post_static_hard_pruning_search_slots",
    }


def create_v8_runtime(*, path_search_cache=True, **kwargs):
    if not path_search_cache:
        raise ValueError("V8 requires the V7 path cache")
    runtime = create_v7_runtime(path_search_cache=True, **kwargs)
    old = runtime.scheduler.candidate_generator
    generator = V8CandidateGenerator(
        old.compute_nodes, old.scheduling_cycle_sim, old.path_provider,
        old.transmission_model, old.calendar, old.time_tolerance,
        candidate_mode=old.candidate_mode.value,
        active_wait_enabled=old.active_wait_enabled,
        pool_max_by_sla=old.pool_max_by_sla,
        pool_node_limit_by_sla=old.pool_node_limit_by_sla,
        pool_time_samples_by_sla=old.pool_time_samples_by_sla,
    )
    generator.profiler = old.profiler
    runtime.scheduler.candidate_generator = generator
    runtime.config_view_overrides.update(static_hard_config_view())
    runtime.evaluation_code_files += (Path(__file__), Path(__file__).parent / "scheduler" / "candidate_generator.py")
    return runtime
