"""Install V9 search without changing the V8 runtime or its accounting."""

from pathlib import Path

from v8.v8_runtime import create_v8_runtime

from .scheduler import V9CandidateGenerator


def create_v9_runtime(*, search_stage=3, **kwargs):
    if search_stage not in (0, 1, 2, 3):
        raise ValueError("search_stage must be E0..E3 (0..3)")
    runtime = create_v8_runtime(**kwargs)
    runtime.config_view_overrides.update({
        "V9_SEARCH_STAGE": f"E{search_stage}",
        "V9_SEARCH_VERSION": V9CandidateGenerator.search_version,
        "V9_WAIT_EARLIEST_SCOPE": "earliest feasible in explored slots",
    })
    if search_stage == 0:
        return runtime
    old = runtime.scheduler.candidate_generator
    generator = V9CandidateGenerator(
        old.compute_nodes, old.scheduling_cycle_sim, old.path_provider,
        old.transmission_model, old.calendar, old.time_tolerance,
        candidate_mode=old.candidate_mode.value,
        active_wait_enabled=old.active_wait_enabled,
        pool_max_by_sla=old.pool_max_by_sla,
        pool_node_limit_by_sla=old.pool_node_limit_by_sla,
        pool_time_samples_by_sla=old.pool_time_samples_by_sla,
        search_stage=search_stage, energy_accounting=runtime.accounting,
    )
    generator.profiler = old.profiler
    runtime.scheduler.candidate_generator = generator
    runtime.evaluation_code_files += (
        Path(__file__), Path(__file__).parent / "scheduler" / "candidate_generator.py",
    )
    return runtime
