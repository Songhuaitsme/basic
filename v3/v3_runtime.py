"""Composition adapter for the isolated V3 candidate-generation runtime."""

from v1.accounting import MetricsLedger
from v1.v1_runtime import V1Runtime, create_v1_runtime

from .accounting import SweepLineEnergyAccounting
from .scheduler import V3CandidateGenerator


def create_v3_runtime(**kwargs) -> V1Runtime:
    """Build a V1-compatible runtime with V3 candidate internals.

    Constructing the frozen runtime first keeps every public domain, policy,
    feature, and scheduler contract unchanged.  The untouched initial
    accounting/generator objects are then replaced before any scheduling work
    can occur.
    """

    runtime = create_v1_runtime(**kwargs)
    old_accounting = runtime.accounting
    accounting = SweepLineEnergyAccounting(
        runtime.time_converter,
        old_accounting.power_model,
        old_accounting.tariff_by_node,
        old_accounting.green_by_node,
        node_bill_rate_model=old_accounting.node_bill_rate_model,
    )

    old_generator = runtime.scheduler.candidate_generator
    generator = V3CandidateGenerator(
        old_generator.compute_nodes,
        old_generator.scheduling_cycle_sim,
        old_generator.path_provider,
        old_generator.transmission_model,
        old_generator.calendar,
        old_generator.time_tolerance,
        candidate_mode=old_generator.candidate_mode.value,
        active_wait_enabled=old_generator.active_wait_enabled,
        pool_max_by_sla=old_generator.pool_max_by_sla,
        pool_node_limit_by_sla=old_generator.pool_node_limit_by_sla,
        pool_time_samples_by_sla=old_generator.pool_time_samples_by_sla,
    )

    ledger = MetricsLedger(accounting)
    runtime.accounting = accounting
    runtime.metrics_ledger = ledger
    runtime.scheduler.candidate_generator = generator
    runtime.scheduler.metrics_ledger = ledger
    return runtime
