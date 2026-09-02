"""Pure metric helpers for Selected-vs-Earliest coordination analysis.

The comparison is per decision: both candidates come from the same candidate
set and reservation snapshot.  Nothing in this module runs a scheduler or
loads a policy checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
import statistics

from shared.data_loader import DataLoader


COUNTERFACTUAL_LABEL = (
    "per-decision counterfactual vs Earliest Feasible Candidate"
)
TEMPORAL_TOLERANCE = 1e-12
GAIN_ZERO_TOLERANCE = 1e-12

ADJUSTMENT_FIELDS = (
    "node_adjustment",
    "region_adjustment",
    "temporal_adjustment",
    "node_temporal_adjustment",
    "region_temporal_adjustment",
)

REMOVED_MIGRATION_FIELDS = (
    "spatial_migration",
    "temporal_migration",
    "both_migration",
)

GROUP_METRIC_FIELDS = (
    "coordination_type",
    "definition",
    "task_count",
    "task_ratio",
    "avg_green_coverage_gain",
    "total_green_energy_gain",
    "avg_cost_saving",
    "total_cost_saving",
    "avg_additional_wait",
    "sla_violation_rate",
    "comparison_basis",
)

SLA_METRIC_FIELDS = (
    "sla_type",
    "task_count",
    "node_adjustment_ratio",
    "region_adjustment_ratio",
    "temporal_adjustment_ratio",
    "region_temporal_adjustment_ratio",
    "avg_green_coverage_gain",
    "total_green_energy_gain",
    "avg_cost_saving",
    "avg_additional_wait",
    "sla_violation_rate",
    "comparison_basis",
)


def _present(value) -> bool:
    return value is not None and value != ""


def _as_float(value):
    return float(value) if _present(value) else None


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def _values(rows: Iterable[Mapping], field: str) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if _present(row.get(field))
    ]


def _mean(rows: Iterable[Mapping], field: str):
    values = _values(rows, field)
    return statistics.fmean(values) if values else None


def _sum(rows: Iterable[Mapping], field: str) -> float:
    return math.fsum(_values(rows, field))


def node_region_map() -> dict[str, str]:
    topology = DataLoader.load_network_topology()
    return {
        str(node): str(attributes["region"])
        for node, attributes in topology.nodes(data=True)
    }


def enrich_adjustments(
    rows: Sequence[Mapping],
    *,
    regions: Mapping[str, str] | None = None,
    temporal_tolerance: float = TEMPORAL_TOLERANCE,
) -> list[dict]:
    """Replace legacy migration flags with Selected-vs-Earliest adjustments."""

    region_lookup = dict(regions or node_region_map())
    enriched: list[dict] = []
    unknown_nodes: set[str] = set()
    for original in rows:
        row = dict(original)
        for field in REMOVED_MIGRATION_FIELDS:
            row.pop(field, None)

        selected_node = row.get("target_node")
        earliest_node = row.get("earliest_target_node")
        selected_region = (
            region_lookup.get(str(selected_node)) if _present(selected_node)
            else None
        )
        earliest_region = (
            region_lookup.get(str(earliest_node)) if _present(earliest_node)
            else None
        )
        if _present(selected_node) and selected_region is None:
            unknown_nodes.add(str(selected_node))
        if _present(earliest_node) and earliest_region is None:
            unknown_nodes.add(str(earliest_node))

        selected_start = _as_float(row.get("selected_start"))
        earliest_start = _as_float(row.get("earliest_start"))
        node_adjustment = (
            _present(selected_node)
            and _present(earliest_node)
            and str(selected_node) != str(earliest_node)
        )
        region_adjustment = (
            selected_region is not None
            and earliest_region is not None
            and selected_region != earliest_region
        )
        temporal_adjustment = (
            selected_start is not None
            and earliest_start is not None
            and selected_start > earliest_start + temporal_tolerance
        )
        row.update({
            "selected_region": selected_region,
            "earliest_region": earliest_region,
            "node_adjustment": node_adjustment,
            "region_adjustment": region_adjustment,
            "temporal_adjustment": temporal_adjustment,
            "node_temporal_adjustment": (
                node_adjustment and temporal_adjustment
            ),
            "region_temporal_adjustment": (
                region_adjustment and temporal_adjustment
            ),
            "comparison_basis": COUNTERFACTUAL_LABEL,
        })
        enriched.append(row)

    if unknown_nodes:
        raise ValueError(
            "nodes missing topology region metadata: "
            + ", ".join(sorted(unknown_nodes))
        )
    return enriched


def scheduled_rows(rows: Sequence[Mapping]) -> list[Mapping]:
    return [row for row in rows if _present(row.get("selected_start"))]


def build_adjustment_metrics(rows: Sequence[Mapping]) -> dict:
    scheduled = scheduled_rows(rows)
    denominator = len(scheduled)
    metrics: dict[str, object] = {
        "scheduled_task_denominator": denominator,
        "comparison_basis": COUNTERFACTUAL_LABEL,
        "temporal_tolerance": TEMPORAL_TOLERANCE,
    }
    for field in ADJUSTMENT_FIELDS:
        tasks = sum(_as_bool(row.get(field)) for row in scheduled)
        metrics[f"{field}_tasks"] = tasks
        metrics[f"{field}_ratio"] = _ratio(tasks, denominator)
    metrics["definitions"] = {
        "node_adjustment": (
            "selected_target_node != earliest_target_node"
        ),
        "region_adjustment": (
            "region(selected_target_node) != "
            "region(earliest_target_node)"
        ),
        "temporal_adjustment": (
            "selected_start > earliest_start + temporal_tolerance"
        ),
        "ratio_denominator": "tasks with a committed reservation",
        "benefit_scope": COUNTERFACTUAL_LABEL,
    }
    return metrics


def _group_metrics(
    coordination_type: str,
    definition: str,
    group: Sequence[Mapping],
    denominator: int,
) -> dict:
    violations = sum(not _as_bool(row.get("sla_satisfied")) for row in group)
    return {
        "coordination_type": coordination_type,
        "definition": definition,
        "task_count": len(group),
        "task_ratio": _ratio(len(group), denominator),
        "avg_green_coverage_gain": _mean(group, "green_coverage_gain"),
        "total_green_energy_gain": _sum(group, "green_energy_gain"),
        "avg_cost_saving": _mean(group, "cost_saving"),
        "total_cost_saving": _sum(group, "cost_saving"),
        "avg_additional_wait": _mean(group, "active_wait"),
        "sla_violation_rate": _ratio(violations, len(group)),
        "comparison_basis": COUNTERFACTUAL_LABEL,
    }


def build_requested_coordination_groups(rows: Sequence[Mapping]) -> list[dict]:
    """Return the five requested views; cross-region includes its joint subset."""

    scheduled = scheduled_rows(rows)
    predicates = (
        (
            "no_obvious_adjustment",
            "NOT node_adjustment AND NOT temporal_adjustment",
            lambda row: not _as_bool(row["node_adjustment"])
            and not _as_bool(row["temporal_adjustment"]),
        ),
        (
            "node_only_adjustment",
            "node_adjustment AND NOT region_adjustment AND NOT temporal_adjustment",
            lambda row: _as_bool(row["node_adjustment"])
            and not _as_bool(row["region_adjustment"])
            and not _as_bool(row["temporal_adjustment"]),
        ),
        (
            "cross_region_adjustment",
            "region_adjustment; includes region_temporal_adjustment",
            lambda row: _as_bool(row["region_adjustment"]),
        ),
        (
            "temporal_only_adjustment",
            "temporal_adjustment AND NOT node_adjustment",
            lambda row: _as_bool(row["temporal_adjustment"])
            and not _as_bool(row["node_adjustment"]),
        ),
        (
            "region_temporal_adjustment",
            "region_adjustment AND temporal_adjustment",
            lambda row: _as_bool(row["region_adjustment"])
            and _as_bool(row["temporal_adjustment"]),
        ),
    )
    return [
        _group_metrics(
            name,
            definition,
            [row for row in scheduled if predicate(row)],
            len(scheduled),
        )
        for name, definition, predicate in predicates
    ]


def build_mutually_exclusive_coordination_groups(
    rows: Sequence[Mapping],
) -> list[dict]:
    """Partition every scheduled task into exactly one adjustment type."""

    scheduled = scheduled_rows(rows)

    def key(row: Mapping) -> str:
        node = _as_bool(row["node_adjustment"])
        region = _as_bool(row["region_adjustment"])
        temporal = _as_bool(row["temporal_adjustment"])
        if region and temporal:
            return "region_temporal_adjustment"
        if region:
            return "cross_region_only_adjustment"
        if node and temporal:
            return "within_region_node_temporal_adjustment"
        if node:
            return "within_region_node_only_adjustment"
        if temporal:
            return "temporal_only_adjustment"
        return "no_obvious_adjustment"

    definitions = (
        (
            "no_obvious_adjustment",
            "NOT node_adjustment AND NOT temporal_adjustment",
        ),
        (
            "within_region_node_only_adjustment",
            "node_adjustment AND NOT region_adjustment AND NOT temporal_adjustment",
        ),
        (
            "cross_region_only_adjustment",
            "region_adjustment AND NOT temporal_adjustment",
        ),
        (
            "temporal_only_adjustment",
            "temporal_adjustment AND NOT node_adjustment",
        ),
        (
            "within_region_node_temporal_adjustment",
            "node_adjustment AND NOT region_adjustment AND temporal_adjustment",
        ),
        (
            "region_temporal_adjustment",
            "region_adjustment AND temporal_adjustment",
        ),
    )
    keyed = [(row, key(row)) for row in scheduled]
    groups = [
        _group_metrics(
            name,
            definition,
            [row for row, row_key in keyed if row_key == name],
            len(scheduled),
        )
        for name, definition in definitions
    ]
    if sum(int(group["task_count"]) for group in groups) != len(scheduled):
        raise AssertionError("mutually exclusive groups do not cover denominator")
    return groups


def _quantile(sorted_values: Sequence[float], probability: float):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def build_green_coverage_gain_distribution(
    rows: Sequence[Mapping],
    *,
    zero_tolerance: float = GAIN_ZERO_TOLERANCE,
) -> dict:
    scheduled = scheduled_rows(rows)
    values = sorted(_values(scheduled, "green_coverage_gain"))
    denominator = len(values)
    positive = sum(value > zero_tolerance for value in values)
    zero = sum(abs(value) <= zero_tolerance for value in values)
    negative = sum(value < -zero_tolerance for value in values)
    return {
        "task_count": denominator,
        "gain_positive_tasks": positive,
        "gain_positive_ratio": _ratio(positive, denominator),
        "gain_zero_tasks": zero,
        "gain_zero_ratio": _ratio(zero, denominator),
        "gain_negative_tasks": negative,
        "gain_negative_ratio": _ratio(negative, denominator),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p25": _quantile(values, 0.25),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "max": max(values) if values else None,
        "min": min(values) if values else None,
        "zero_tolerance": zero_tolerance,
        "comparison_basis": COUNTERFACTUAL_LABEL,
    }


def build_sla_type_metrics(rows: Sequence[Mapping]) -> list[dict]:
    scheduled = scheduled_rows(rows)
    order = ("Flexible", "Soft", "Hard")
    observed = sorted({str(row["sla_type"]) for row in scheduled})
    sla_types = list(order) + [item for item in observed if item not in order]
    result = []
    for sla_type in sla_types:
        group = [row for row in scheduled if row.get("sla_type") == sla_type]
        if not group:
            continue
        count = len(group)
        violations = sum(
            not _as_bool(row.get("sla_satisfied")) for row in group
        )
        result.append({
            "sla_type": sla_type,
            "task_count": count,
            "node_adjustment_ratio": _ratio(
                sum(_as_bool(row["node_adjustment"]) for row in group), count
            ),
            "region_adjustment_ratio": _ratio(
                sum(_as_bool(row["region_adjustment"]) for row in group), count
            ),
            "temporal_adjustment_ratio": _ratio(
                sum(_as_bool(row["temporal_adjustment"]) for row in group), count
            ),
            "region_temporal_adjustment_ratio": _ratio(
                sum(
                    _as_bool(row["region_temporal_adjustment"])
                    for row in group
                ),
                count,
            ),
            "avg_green_coverage_gain": _mean(group, "green_coverage_gain"),
            "total_green_energy_gain": _sum(group, "green_energy_gain"),
            "avg_cost_saving": _mean(group, "cost_saving"),
            "avg_additional_wait": _mean(group, "active_wait"),
            "sla_violation_rate": _ratio(violations, count),
            "comparison_basis": COUNTERFACTUAL_LABEL,
        })
    return result
