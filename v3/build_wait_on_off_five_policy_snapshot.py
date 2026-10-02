"""Build one reviewed table containing V3 Wait ON and Wait OFF results."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


POLICIES = (
    ("earliest_feasible", "Earliest Feasible"),
    ("lowest_cost", "Lowest Cost"),
    ("highest_green", "Highest Green"),
    ("equal_weight", "Equal Weight"),
    ("candidate_dqn", "Candidate DQN"),
)

WAIT_ON_FILES = {
    key: f"artifacts/v3/evaluation/{key}_seed42_warmup2d.json"
    for key, _ in POLICIES
}
WAIT_OFF_FILES = {
    "earliest_feasible": "artifacts/v3/evaluation/wait_off_five_policy/earliest_feasible_seed42_warmup2d_wait_off.json",
    "lowest_cost": "artifacts/v3/evaluation/wait_off_five_policy/lowest_cost_seed42_warmup2d_wait_off.json",
    "highest_green": "artifacts/v3/evaluation/wait_off_five_policy/highest_green_seed42_warmup2d_wait_off.json",
    "equal_weight": "artifacts/v3/evaluation/wait_off_five_policy/equal_weight_seed42_warmup2d_wait_off.json",
    "candidate_dqn": "artifacts/v3/evaluation/wait_ablation/formal_600k_dqn_wait_off.json",
}

COMMON_FIELDS = (
    "seed",
    "task_trace_hash",
    "exogenous_trace_hash",
    "topology_hash",
    "code_hash",
    "warmup_days",
    "evaluation_start_sim",
    "arrival_cutoff_sim",
)


def value(metric):
    return metric.get("value") if isinstance(metric, dict) else metric


def pct(metric):
    number = value(metric)
    return None if number is None else number * 100.0


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def extract_row(key, label, wait_state, relative, report):
    metrics = report["metrics"]
    wait = metrics["active_wait_metrics"]
    hard = metrics["sla_metrics"]["Hard"]
    soft = metrics["sla_metrics"]["Soft"]
    flexible = metrics["sla_metrics"]["Flexible"]
    reserved = metrics["reserved_ever_count"]
    return {
        "wait_state": wait_state,
        "policy_key": key,
        "policy": label,
        "status": report["status"],
        "config_hash": report["metadata"].get("config_hash"),
        "arrival_count": metrics["arrival_count"],
        "reserved_count": reserved,
        "completed_count": metrics["completed_count"],
        "expired_count": metrics["expired_count"],
        "failed_count": metrics["failed_count"],
        "acceptance_pct": pct(metrics["acceptance_rate"]),
        "completion_pct": pct(metrics["completion_rate"]),
        "reservation_reliability_pct": pct(metrics["reservation_reliability"]),
        "total_cost_yuan": metrics["total_economic_cost_yuan"],
        "completed_cpu_hours": metrics["completed_cpu_hours"],
        "cost_per_completed_cpu_hour": value(metrics["cost_yuan_per_completed_cpu_hour"]),
        "cost_per_arrived_cpu_hour": value(metrics["cost_yuan_per_arrived_cpu_hour"]),
        "task_green_pct": pct(metrics["completed_task_green_coverage"]),
        "system_green_absorption_pct": pct(metrics["system_green_absorption_rate"]),
        "active_wait_count": wait["count"],
        "active_wait_pct": wait["count"] / reserved * 100.0 if reserved else None,
        "mean_active_wait_sim": value(wait["mean_active_wait_sim"]),
        "p95_active_wait_sim": value(wait["p95_active_wait_sim"]),
        "positive_benefit_pct": pct(wait["positive_benefit_rate"]),
        "hard_count": hard["count"],
        "hard_expired_pct": pct(hard["expired_rate"]),
        "hard_start_delay_p50_sim": value(hard["start_delay_p50_sim"]),
        "hard_start_delay_p95_sim": value(hard["start_delay_p95_sim"]),
        "soft_count": soft["count"],
        "soft_on_time_pct": pct(soft["preferred_on_time_rate"]),
        "soft_tardy_pct": pct(soft["acceptable_tardy_rate"]),
        "soft_expired_pct": pct(soft["expired_rate"]),
        "soft_start_delay_p50_sim": value(soft["start_delay_p50_sim"]),
        "soft_start_delay_p95_sim": value(soft["start_delay_p95_sim"]),
        "soft_tardiness_p50_sim": value(soft["preferred_start_tardiness_p50"]),
        "soft_tardiness_p95_sim": value(soft["preferred_start_tardiness_p95"]),
        "flexible_count": flexible["count"],
        "flexible_on_time_pct": pct(flexible["preferred_on_time_rate"]),
        "flexible_tardy_pct": pct(flexible["acceptable_tardy_rate"]),
        "flexible_expired_pct": pct(flexible["expired_rate"]),
        "flexible_start_delay_p50_sim": value(flexible["start_delay_p50_sim"]),
        "flexible_start_delay_p95_sim": value(flexible["start_delay_p95_sim"]),
        "flexible_tardiness_p50_sim": value(flexible["preferred_start_tardiness_p50"]),
        "flexible_tardiness_p95_sim": value(flexible["preferred_start_tardiness_p95"]),
        "source_file": relative,
    }


def build_snapshot(root: Path) -> dict:
    rows = []
    manifest = []
    reports = []
    source_files = []
    for wait_state, file_map in (("ON", WAIT_ON_FILES), ("OFF", WAIT_OFF_FILES)):
        for key, label in POLICIES:
            relative = file_map[key]
            path = root / relative
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("status") != "VALID":
                raise ValueError(f"report is not VALID: {path}")
            rows.append(extract_row(key, label, wait_state, relative, report))
            reports.append((wait_state, key, report))
            source_files.append(relative)
            manifest.append({
                "wait_state": wait_state,
                "policy_key": key,
                "policy": label,
                "source_file": relative,
                "sha256": digest(path),
                "status": report["status"],
                "model_hash": report["metadata"].get("model_hash"),
                "config_hash": report["metadata"].get("config_hash"),
            })

    reference = reports[0][2]["metadata"]
    for field in COMMON_FIELDS:
        observed = {report["metadata"].get(field) for _, _, report in reports}
        if observed != {reference.get(field)}:
            raise ValueError(f"common field mismatch: {field}: {observed}")

    config_hashes = {}
    for wait_state in ("ON", "OFF"):
        observed = {
            report["metadata"].get("config_hash")
            for state, _, report in reports if state == wait_state
        }
        if len(observed) != 1:
            raise ValueError(f"config hash mismatch within Wait {wait_state}: {observed}")
        config_hashes[wait_state] = next(iter(observed))
    if config_hashes["ON"] == config_hashes["OFF"]:
        raise ValueError("Wait ON and Wait OFF unexpectedly share one config hash")

    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    evidence = [{
        "title": "Formal V3 Wait ON/OFF evaluation JSON",
        "detail": (
            "Direct extraction from ten VALID reports. Seed, task trace, exogenous "
            "trace, topology, code, warm-up, and measurement window match exactly."
        ),
    }]
    source = {
        "label": "V3 Wait ON and Wait OFF five-policy evaluation reports",
        "files": source_files,
        "executedAt": generated_at,
        "evidenceFlow": evidence,
        "metricDefinitions": [],
    }
    return {
        "id": "v3-wait-on-off-five-policy-source-table",
        "title": "V3 Wait ON / OFF 五策略源数据对比",
        "surface": "dashboard",
        "status": "reviewed",
        "buildStatus": "creating",
        "generatedAt": generated_at,
        "filters": [],
        "queries": {
            "wait_on_off_policy_metrics": {
                "label": "Wait ON / OFF 五策略源数据",
                "reportingField": "policy",
                "rows": rows,
                "methods": [{
                    "language": "python",
                    "code": (
                        "Read metrics directly from each report; convert MetricValue "
                        "fractions to percent; compute active_wait_pct as "
                        "active_wait_count / reserved_count * 100."
                    ),
                }],
                "source": source,
            },
            "experiment_identity": {
                "label": "评估基础配置",
                "rows": [{
                    **{field: reference.get(field) for field in COMMON_FIELDS},
                    "wait_on_config_hash": config_hashes["ON"],
                    "wait_off_config_hash": config_hashes["OFF"],
                    "report_count": 10,
                    "all_reports_valid": True,
                    "paired_metadata_match": True,
                }],
                "source": source,
            },
            "source_manifest": {
                "label": "原始报告清单",
                "reportingField": "policy",
                "rows": manifest,
                "source": source,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_snapshot(root), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
