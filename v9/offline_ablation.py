"""Reusable fixed-snapshot candidate comparison; caller supplies small cases.

This module deliberately does not create tasks or run an experiment on import.
Use the same task, snapshot, forecast version, evaluator and K=1 path provider
for E0..E3. A complete-mode generator is the reference on small windows.
"""

import time


def compare_case(task, decision_time, snapshot, evaluator, generators, *,
                 forecast_version="perfect-v1", forecast_limit=None):
    """Return per-stage recall, quality gap and actual search costs.

    ``generators`` maps E0/E1/E2/E3/complete to configured generators.
    Quality gap is the difference from the complete stream's lowest cost and
    highest marginal green energy. Lower cost gap and green gap are better.
    """
    outputs = {}
    for name, generator in generators.items():
        started = time.perf_counter()
        stream = generator.prepare_stream(
            task, decision_time, reservation_snapshot=snapshot,
            forecast_version=forecast_version,
            forecast_covered_until_sim=forecast_limit,
            metric_evaluator=evaluator,
        )
        candidates = tuple(stream.iter_candidates())
        outputs[name] = (candidates, time.perf_counter() - started,
                         dict(getattr(generator, "last_search_stats", {})))
    reference = outputs["complete"][0]
    reference_nodes = {c.target_node for c in reference}
    def physical_key(candidate):
        return (candidate.target_node, candidate.path.path_id,
                candidate.transmission_start_sim, candidate.compute_start_sim)

    reference_ids = {physical_key(c) for c in reference}
    best_cost = min((c.estimated_candidate_marginal_system_cost_yuan for c in reference),
                    default=None)
    best_green = max((c.estimated_candidate_marginal_green_energy_mwh for c in reference),
                     default=None)
    quality_count = max(1, (len(reference) + 9) // 10)
    quality = set()
    for key in (
        lambda c: c.estimated_candidate_marginal_system_cost_yuan,
        lambda c: -c.estimated_candidate_marginal_green_energy_mwh,
        lambda c: c.compute_end_sim,
    ):
        quality.update(physical_key(c) for c in sorted(reference, key=key)[:quality_count])
    result = {}
    for name, (candidates, seconds, stats) in outputs.items():
        found_nodes = {c.target_node for c in candidates}
        found_ids = {physical_key(c) for c in candidates}
        result[name] = {
            "feasible_node_recall": (len(found_nodes & reference_nodes) / len(reference_nodes)
                                     if reference_nodes else 1.0),
            "candidate_recall": (len(found_ids & reference_ids) / len(reference_ids)
                                 if reference_ids else 1.0),
            "quality_candidate_recall": (len(found_ids & quality) / len(quality)
                                         if quality else 1.0),
            "lowest_cost_gap_yuan": (
                min(c.estimated_candidate_marginal_system_cost_yuan for c in candidates)
                - best_cost if candidates and best_cost is not None else None),
            "highest_green_gap_mwh": (
                best_green - max(c.estimated_candidate_marginal_green_energy_mwh
                                 for c in candidates)
                if candidates and best_green is not None else None),
            "candidate_generation_seconds": seconds,
            "candidate_count": len(candidates),
            "search_stats": stats,
        }
    return result
