"""Recompute V2 coordination metrics from an existing task_metrics.csv."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

from v2.coordination_metrics import (
    ADJUSTMENT_FIELDS,
    COUNTERFACTUAL_LABEL,
    GAIN_ZERO_TOLERANCE,
    GROUP_METRIC_FIELDS,
    SLA_METRIC_FIELDS,
    TEMPORAL_TOLERANCE,
    build_adjustment_metrics,
    build_green_coverage_gain_distribution,
    build_mutually_exclusive_coordination_groups,
    build_requested_coordination_groups,
    build_sla_type_metrics,
    enrich_adjustments,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fields=None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute Selected-vs-Earliest coordination metrics without "
            "loading a model or running the scheduler"
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    source_csv = source_dir / "task_metrics.csv"
    if not source_csv.is_file():
        parser.error(f"missing source task metrics: {source_csv}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else source_dir / "recomputed_adjustment_metrics"
    )
    if output_dir.exists():
        raise FileExistsError(
            f"output already exists: {output_dir}; choose a new --output-dir"
        )
    output_dir.mkdir(parents=True)

    source_rows = _read_csv(source_csv)
    task_rows = enrich_adjustments(source_rows)
    adjustment_metrics = build_adjustment_metrics(task_rows)
    requested_groups = build_requested_coordination_groups(task_rows)
    exclusive_groups = build_mutually_exclusive_coordination_groups(task_rows)
    gain_distribution = build_green_coverage_gain_distribution(task_rows)
    sla_metrics = build_sla_type_metrics(task_rows)

    task_fields = [
        field
        for field in source_rows[0]
        if field not in {"spatial_migration", "temporal_migration", "both_migration"}
    ]
    for field in (
        "selected_region",
        "earliest_region",
        *ADJUSTMENT_FIELDS,
        "comparison_basis",
    ):
        if field not in task_fields:
            task_fields.append(field)

    _write_csv(output_dir / "task_adjustment_metrics.csv", task_rows, task_fields)
    _write_json(output_dir / "adjustment_metrics.json", adjustment_metrics)
    _write_csv(
        output_dir / "coordination_type_metrics.csv",
        requested_groups,
        GROUP_METRIC_FIELDS,
    )
    _write_csv(
        output_dir / "coordination_type_metrics_mutually_exclusive.csv",
        exclusive_groups,
        GROUP_METRIC_FIELDS,
    )
    _write_csv(
        output_dir / "green_coverage_gain_distribution.csv",
        [gain_distribution],
    )
    _write_csv(
        output_dir / "sla_type_metrics.csv",
        sla_metrics,
        SLA_METRIC_FIELDS,
    )
    report = {
        "adjustment_metrics": adjustment_metrics,
        "coordination_type_metrics": requested_groups,
        "coordination_type_metrics_mutually_exclusive": exclusive_groups,
        "green_coverage_gain_distribution": gain_distribution,
        "sla_type_metrics": sla_metrics,
        "definitions": {
            "comparison_basis": COUNTERFACTUAL_LABEL,
            "temporal_tolerance": TEMPORAL_TOLERANCE,
            "green_coverage_gain_zero_tolerance": GAIN_ZERO_TOLERANCE,
            "source_data_unchanged": True,
            "model_or_scheduler_executed": False,
        },
    }
    _write_json(output_dir / "coordination_adjustment_report.json", report)

    manifest = {
        "status": "VALID",
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_directory": str(source_dir),
        "source_task_metrics": str(source_csv),
        "source_task_metrics_sha256": _sha256(source_csv),
        "source_row_count": len(source_rows),
        "scheduled_task_denominator": adjustment_metrics[
            "scheduled_task_denominator"
        ],
        "comparison_basis": COUNTERFACTUAL_LABEL,
        "model_or_scheduler_executed": False,
        "source_data_unchanged": True,
        "result_files": sorted(path.name for path in output_dir.iterdir()),
    }
    _write_json(output_dir / "recalculation_manifest.json", manifest)

    print(json.dumps({
        "status": "VALID",
        "output_dir": str(output_dir),
        "scheduled_task_denominator": adjustment_metrics[
            "scheduled_task_denominator"
        ],
        "model_or_scheduler_executed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
