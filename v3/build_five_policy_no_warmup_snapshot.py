"""Build reviewed report data from five V3 no-warmup 600k evaluations."""

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
PAIRED_FIELDS = (
    "system_version", "seed", "task_trace_hash", "exogenous_trace_hash",
    "topology_hash", "code_hash", "config_hash", "warmup_days",
    "evaluation_start_sim", "arrival_cutoff_sim", "tariff_mode",
    "gamma_per_second", "percentile_method",
)


def value(metric):
    return metric.get("value") if isinstance(metric, dict) else metric


def ratio_fields(metric, prefix):
    if not isinstance(metric, dict):
        return {prefix: metric, f"{prefix}_numerator": None, f"{prefix}_denominator": None}
    return {
        prefix: metric.get("value"),
        f"{prefix}_numerator": metric.get("numerator"),
        f"{prefix}_denominator": metric.get("denominator"),
        f"{prefix}_status": metric.get("status"),
    }


def build_snapshot(root: Path) -> dict:
    base = root / "artifacts" / "v3" / "evaluation" / "no_warmup_600k"
    reports = {}
    manifest = []
    files = []
    for key, label in POLICIES:
        path = base / f"{key}_seed42.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "VALID":
            raise ValueError(f"report is not VALID: {path}")
        reports[key] = payload
        relative = path.relative_to(root).as_posix()
        files.append(relative)
        manifest.append({
            "policy_key": key,
            "policy": label,
            "source_file": relative,
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "status": payload["status"],
            "model_hash": payload["metadata"].get("model_hash"),
        })

    reference = reports[POLICIES[0][0]]["metadata"]
    identity = {field: reference.get(field) for field in PAIRED_FIELDS}
    for field in PAIRED_FIELDS:
        observed = {reports[key]["metadata"].get(field) for key, _ in POLICIES}
        if observed != {reference.get(field)}:
            raise ValueError(f"paired field mismatch: {field}: {observed}")
    identity.update({"report_count": 5, "all_reports_valid": True, "paired_metadata_match": True})

    core_rows, sla_rows, wait_rows, phase_rows = [], [], [], []
    for key, label in POLICIES:
        report = reports[key]
        metrics = report["metrics"]
        row = {
            "policy_key": key, "policy": label,
            "arrival_count": metrics["arrival_count"],
            "reserved_ever_count": metrics["reserved_ever_count"],
            "rejected_count": metrics["rejected_count"],
            "expired_count": metrics["expired_count"],
            "completed_count": metrics["completed_count"],
            "failed_count": metrics["failed_count"],
            "total_economic_cost_yuan": metrics["total_economic_cost_yuan"],
            "completed_cpu_hours": metrics["completed_cpu_hours"],
            "arrived_requested_cpu_hours": metrics["arrived_requested_cpu_hours"],
            "final_state_counts": metrics["final_state_counts"],
        }
        for field in (
            "acceptance_rate", "completion_rate", "reservation_reliability",
            "cost_yuan_per_completed_cpu_hour", "cost_yuan_per_arrived_cpu_hour",
            "completed_task_green_coverage", "system_green_absorption_rate",
        ):
            row.update(ratio_fields(metrics[field], field))
        core_rows.append(row)

        wait = metrics["active_wait_metrics"]
        wait_row = {"policy_key": key, "policy": label, "count": wait["count"]}
        for field in ("mean_active_wait_sim", "p95_active_wait_sim", "positive_benefit_rate"):
            wait_row.update(ratio_fields(wait[field], field))
        wait_rows.append(wait_row)

        for sla_name in ("Hard", "Soft", "Flexible"):
            sla = metrics["sla_metrics"][sla_name]
            sla_row = {"policy_key": key, "policy": label, "sla_type": sla_name, "count": sla["count"]}
            for field in (
                "preferred_on_time_rate", "acceptable_tardy_rate", "expired_rate",
                "start_delay_p50_sim", "start_delay_p95_sim",
                "preferred_start_tardiness_p50", "preferred_start_tardiness_p95",
            ):
                sla_row.update(ratio_fields(sla[field], field))
            sla_rows.append(sla_row)

        phases = report["phase_batch_counts"]
        phase_rows.append({"policy_key": key, "policy": label, **phases,
                           "cycle_result_count": report.get("cycle_result_count")})

    source = {
        "label": "V3 600k no-warmup five-policy evaluation JSON",
        "files": files,
        "executedAt": "2026-09-19",
        "evidenceFlow": [{
            "title": "Formal V3 evaluation output",
            "detail": "Direct extraction from five VALID JSON reports; no task-level records were aggregated into the comparison tables.",
        }],
        "metricDefinitions": [
            {"field": "completion_rate", "label": "Completion rate", "definition": "completed_count / arrival_count; source numerator and denominator retained."},
            {"field": "cost_yuan_per_completed_cpu_hour", "label": "Cost per completed CPU-hour", "definition": "total economic cost divided by completed CPU-hours; yuan/CPU-hour."},
            {"field": "completed_task_green_coverage", "label": "Completed-task green coverage", "definition": "Attributed green energy divided by total energy for completed measurement-cohort tasks."},
            {"field": "system_green_absorption_rate", "label": "System green absorption", "definition": "Green energy used by all running load divided by available green supply in the measurement window."},
        ],
    }

    def query(label, rows, method):
        return {"label": label, "reportingField": "policy", "rows": rows,
                "methods": [{"language": "python", "code": method}], "source": source}

    return {
        "title": "V3 600k 无 Warm-up 五策略评测：完整源指标对比",
        "surface": "report", "status": "reviewed", "buildStatus": "creating",
        "generatedAt": datetime.now().astimezone().isoformat(), "filters": [],
        "report": {"asOf": "2026-09-19", "originalQuestion": "根据五种策略评测结果生成 HTML，重点展示源数据。"},
        "queries": {
            "policy_core": query("策略级完整核心指标", core_rows, "Direct extraction of metrics; MetricValue values, status, numerator, and denominator are retained."),
            "sla_detail": query("策略与 SLA 级完整指标", sla_rows, "One row per policy and SLA type, directly extracted from metrics.sla_metrics."),
            "wait_detail": query("主动等待完整指标", wait_rows, "Direct extraction from metrics.active_wait_metrics."),
            "phase_counts": query("仿真阶段批次数", phase_rows, "Direct extraction from phase_batch_counts and cycle_result_count."),
            "experiment_identity": {"label": "配对实验口径", "rows": [identity], "source": source},
            "source_manifest": {"label": "原始文件与校验值", "reportingField": "policy", "rows": manifest, "source": source},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_snapshot(root), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
