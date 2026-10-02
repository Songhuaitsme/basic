"""Build a cohort-aligned comparison of V3 warm-up and no-warm-up reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

POLICIES = (("earliest_feasible", "Earliest Feasible"), ("lowest_cost", "Lowest Cost"),
            ("highest_green", "Highest Green"), ("equal_weight", "Equal Weight"),
            ("candidate_dqn", "Candidate DQN"))
METRICS = (
    ("arrival_count", "Arrival count", "count", "neutral"),
    ("completed_count", "Completed count", "count", "higher"),
    ("completion_rate", "Completion rate", "%", "higher"),
    ("reservation_reliability", "Reservation reliability", "%", "higher"),
    ("total_economic_cost_yuan", "Total economic cost", "yuan", "lower"),
    ("completed_cpu_hours", "Completed CPU-hours", "CPU-h", "higher"),
    ("cost_yuan_per_completed_cpu_hour", "Cost / completed CPU-hour", "yuan/CPU-h", "lower"),
    ("cost_yuan_per_arrived_cpu_hour", "Cost / arrived CPU-hour", "yuan/CPU-h", "lower"),
    ("completed_task_green_coverage", "Completed-task green coverage", "%", "higher"),
    ("system_green_absorption_rate", "System green absorption", "%", "higher"),
    ("active_wait_count", "Active-wait count", "count", "context"),
    ("active_wait_rate", "Active-wait / reserved", "%", "context"),
    ("positive_benefit_rate", "Positive wait benefit", "%", "higher"),
)
START_SIM, END_SIM, SECONDS_PER_SIM = 576.0, 864.0, 300.0


def recompute_window(report: dict, start_sim: float = START_SIM,
                     end_sim: float = END_SIM) -> tuple[dict, dict]:
    tasks = [r for r in report["task_records"] if start_sim <= r["arrival_time_sim"] < end_sim]
    task_ids = {r["task_id"] for r in tasks}
    decisions = {r["task_id"]: r for r in report.get("decision_records", ()) if r["task_id"] in task_ids}
    completed = [r for r in tasks if r["final_state"] == "Completed"]
    reserved = [r for r in tasks if r.get("target_node") is not None]
    total_cost = sum(float(r.get("task_attributed_cost_yuan") or 0) for r in completed)
    completed_cpu = sum(float(r.get("cpu_work_cpu_hours") or 0) for r in completed)
    arrived_cpu = sum(float(r.get("cpu_work_cpu_hours") or 0) for r in tasks)
    completed_energy = sum(float(r.get("task_energy_mwh") or 0) for r in completed)
    completed_green = sum(float(r.get("task_attributed_green_energy_mwh") or 0) for r in completed)
    waits = [r for r in tasks if float(r.get("active_wait_sim") or 0) > 0]
    positives = sum(decisions.get(r["task_id"], {}).get("benefit_positive") is True for r in waits)

    green_supply = green_used = 0.0
    interval_count = 0
    for interval in report["diagnostics"]["time_records"]:
        left = max(start_sim, float(interval["start_sim"]))
        right = min(end_sim, float(interval["end_sim"]))
        if right <= left:
            continue
        hours = (right - left) * SECONDS_PER_SIM / 3600.0
        green_supply += float(interval["renewable_generation"]) * hours
        green_used += float(interval["renewable_used"]) * hours
        interval_count += 1

    n, accepted = len(tasks), len(reserved)
    values = {
        "arrival_count": n, "completed_count": len(completed),
        "completion_rate": len(completed) / n * 100 if n else None,
        "reservation_reliability": len(completed) / accepted * 100 if accepted else None,
        "total_economic_cost_yuan": total_cost, "completed_cpu_hours": completed_cpu,
        "cost_yuan_per_completed_cpu_hour": total_cost / completed_cpu if completed_cpu else None,
        "cost_yuan_per_arrived_cpu_hour": total_cost / arrived_cpu if arrived_cpu else None,
        "completed_task_green_coverage": completed_green / completed_energy * 100 if completed_energy else None,
        "system_green_absorption_rate": green_used / green_supply * 100 if green_supply else None,
        "active_wait_count": len(waits),
        "active_wait_rate": len(waits) / accepted * 100 if accepted else None,
        "positive_benefit_rate": positives / len(waits) * 100 if waits else None,
    }
    audit = {"cohort_start_sim": start_sim, "cohort_end_sim": end_sim,
             "task_id_count": n, "task_id_unique_count": len(task_ids),
             "reserved_count": accepted, "time_interval_count": interval_count,
             "system_green_supply_mwh": green_supply, "system_green_used_mwh": green_used}
    return values, audit


def fmt(value, unit):
    if value is None:
        return "—"
    if unit == "count":
        return f"{value:,.0f}"
    if unit == "%":
        return f"{value:.2f}%"
    return f"{value:,.4f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_warmup_recalculation(report: dict, values: dict, policy_key: str) -> None:
    """The aligned window equals the original warm-up measurement window."""
    percent_fields = {"completion_rate", "reservation_reliability",
                      "completed_task_green_coverage", "system_green_absorption_rate"}
    fields = ("arrival_count", "completed_count", "completion_rate", "reservation_reliability",
              "total_economic_cost_yuan", "completed_cpu_hours",
              "cost_yuan_per_completed_cpu_hour", "cost_yuan_per_arrived_cpu_hour",
              "completed_task_green_coverage", "system_green_absorption_rate")
    for field in fields:
        published = report["metrics"][field]
        if isinstance(published, dict):
            published = published.get("value")
        if field in percent_fields and published is not None:
            published *= 100.0
        if published is None or abs(values[field] - published) > 1e-6 * max(1.0, abs(published)):
            raise ValueError(f"Warm-up recalculation mismatch: {policy_key}/{field}: {values[field]} vs {published}")


def build(root: Path, output: Path) -> None:
    conditions = (
        ("no_warmup", "No warm-up (last day)", root / "artifacts/v3/evaluation/no_warmup_600k", "{key}_seed42.json"),
        ("warmup_2d", "Warm-up 2d (measurement day)", root / "artifacts/v3/evaluation", "{key}_seed42_warmup2d.json"),
    )
    reports, wide_rows, audit_rows = {}, [], []
    for condition, label, folder, pattern in conditions:
        for policy_key, policy in POLICIES:
            path = folder / pattern.format(key=policy_key)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("status") != "VALID":
                raise ValueError(f"Expected VALID report: {path}")
            values, audit = recompute_window(report)
            if condition == "warmup_2d":
                validate_warmup_recalculation(report, values, policy_key)
            reports[(condition, policy_key)] = (report, values)
            wide_rows.append({"condition": condition, "condition_label": label, "policy_key": policy_key,
                              "policy": policy, "cohort_start_sim": START_SIM, "cohort_end_sim": END_SIM, **values})
            audit_rows.append({"condition": condition, "condition_label": label, "policy_key": policy_key,
                               "policy": policy, "model_hash": report["metadata"].get("model_hash"),
                               "source_file": path.relative_to(root).as_posix(), **audit})

    expected_ids = None
    for condition, _, _, _ in conditions:
        for policy_key, _ in POLICIES:
            report = reports[(condition, policy_key)][0]
            ids = {r["task_id"] for r in report["task_records"] if START_SIM <= r["arrival_time_sim"] < END_SIM}
            expected_ids = ids if expected_ids is None else expected_ids
            if ids != expected_ids:
                raise ValueError(f"Cohort task IDs do not match: {condition}/{policy_key}")

    paired_rows = []
    for policy_key, policy in POLICIES:
        no_values, warm_values = reports[("no_warmup", policy_key)][1], reports[("warmup_2d", policy_key)][1]
        for key, label, unit, direction in METRICS:
            a, b = no_values[key], warm_values[key]
            delta = None if a is None or b is None else b - a
            paired_rows.append({"policy_key": policy_key, "policy": policy, "metric": key,
                                "metric_label": label, "unit": unit, "preferred_direction": direction,
                                "no_warmup_last_day": a, "warmup_2d_measurement_day": b,
                                "delta_warmup_minus_no_warmup": delta,
                                "relative_delta_pct": None if delta is None or a == 0 else delta / abs(a) * 100})

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "ten_strategy_metrics.csv", wide_rows)
    write_csv(output / "paired_metric_deltas.csv", paired_rows)
    write_csv(output / "cohort_audit.csv", audit_rows)

    table_rows = []
    for row in paired_rows:
        delta_class = "pos" if (row["delta_warmup_minus_no_warmup"] or 0) > 0 else "neg"
        table_rows.append("<tr>" + "".join((
            f"<td>{html.escape(row['policy'])}</td>", f"<td>{html.escape(row['metric_label'])}</td>",
            f"<td>{fmt(row['no_warmup_last_day'], row['unit'])}</td>",
            f"<td>{fmt(row['warmup_2d_measurement_day'], row['unit'])}</td>",
            f"<td class='{delta_class}'>{fmt(row['delta_warmup_minus_no_warmup'], row['unit'])}</td>",
            f"<td>{fmt(row['relative_delta_pct'], '%')}</td>", f"<td>{row['preferred_direction']}</td>")) + "</tr>")
    model_rows = "".join(
        f"<tr><td>{html.escape(policy)}</td><td><code>{reports[('no_warmup', key)][0]['metadata'].get('model_hash')}</code></td>"
        f"<td><code>{reports[('warmup_2d', key)][0]['metadata'].get('model_hash')}</code></td>"
        f"<td>{'same' if reports[('no_warmup', key)][0]['metadata'].get('model_hash') == reports[('warmup_2d', key)][0]['metadata'].get('model_hash') else 'different'}</td></tr>"
        for key, policy in POLICIES)
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Cohort-aligned warm-up comparison</title>
<style>body{{font:14px system-ui;margin:32px;color:#17202a}}h1{{margin-bottom:6px}}.note{{background:#fff4ce;border-left:5px solid #d39e00;padding:12px;max-width:1100px}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #d9dee5;padding:7px 9px;text-align:right}}th{{background:#eef2f6;position:sticky;top:0}}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){{text-align:left}}.pos{{color:#087f23}}.neg{{color:#b42318}}code{{font-size:11px}}</style></head><body>
<h1>V3 600k：统一最后一天 cohort 的五策略对比</h1>
<p>两组均按绝对时间窗口 [576, 864) 重算，均为同一批 {len(expected_ids):,} 个到达任务；Δ = Warm-up 2d − No warm-up。</p>
<div class='note'><b>解释边界：</b>四种启发式策略在统一 cohort 后逐任务结果一致。Candidate DQN 使用不同 model_hash，因此差值表示两个训练模型在同一 cohort 上的差异，不能解释为单纯的评估 warm-up 效应。</div>
<table><thead><tr><th>策略</th><th>指标</th><th>No warm-up 最后一天</th><th>Warm-up 2d 测量日</th><th>Δ</th><th>相对变化</th><th>偏好方向</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>模型身份核对</h2><table><thead><tr><th>策略</th><th>No warm-up model hash</th><th>Warm-up model hash</th><th>检查</th></tr></thead><tbody>{model_rows}</tbody></table>
</body></html>"""
    (output / "warmup_vs_no_warmup_report.html").write_text(page, encoding="utf-8")
    (output / "README.md").write_text(
        "# Cohort-aligned warm-up comparison\n\nBoth conditions are recomputed on `[576, 864)` and the same 2,501 task IDs.\n\n"
        "- `warmup_vs_no_warmup_report.html`: paired comparison\n- `ten_strategy_metrics.csv`: 10 condition-policy rows\n"
        "- `paired_metric_deltas.csv`: metric-level deltas\n- `cohort_audit.csv`: cohort, ledger-window, source, and model audit\n\n"
        "Candidate DQN uses different model hashes, so its delta is a same-cohort model comparison, not a pure evaluation warm-up effect.\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v3/evaluation/warmup_comparison"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build(root, root / args.output if not args.output.is_absolute() else args.output)


if __name__ == "__main__":
    main()
