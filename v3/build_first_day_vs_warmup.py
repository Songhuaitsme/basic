"""Compare no-warm-up day 1 with the post-warm-up measurement day."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from build_warmup_comparison import METRICS, POLICIES, fmt, recompute_window


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(root: Path, output: Path) -> None:
    specs = (
        ("no_warmup_day1", "No warm-up day 1", 0.0, 288.0,
         root / "artifacts/v3/evaluation/no_warmup_600k", "{key}_seed42.json"),
        ("warmup_2d_day3", "Warm-up 2d measurement day", 576.0, 864.0,
         root / "artifacts/v3/evaluation", "{key}_seed42_warmup2d.json"),
    )
    results, wide, audit = {}, [], []
    for condition, label, start, end, folder, pattern in specs:
        for policy_key, policy in POLICIES:
            path = folder / pattern.format(key=policy_key)
            report = json.loads(path.read_text(encoding="utf-8"))
            values, checks = recompute_window(report, start, end)
            results[(condition, policy_key)] = values
            wide.append({"condition": condition, "condition_label": label,
                         "policy_key": policy_key, "policy": policy,
                         "window_start_sim": start, "window_end_sim": end, **values})
            audit.append({"condition": condition, "condition_label": label,
                          "policy_key": policy_key, "policy": policy,
                          "model_hash": report["metadata"].get("model_hash"),
                          "task_trace_hash": report["metadata"].get("task_trace_hash"),
                          "exogenous_trace_hash": report["metadata"].get("exogenous_trace_hash"),
                          "source_file": path.relative_to(root).as_posix(), **checks})

    paired = []
    for policy_key, policy in POLICIES:
        first = results[("no_warmup_day1", policy_key)]
        warm = results[("warmup_2d_day3", policy_key)]
        for key, label, unit, direction in METRICS:
            a, b = first[key], warm[key]
            delta = None if a is None or b is None else b - a
            paired.append({"policy_key": policy_key, "policy": policy, "metric": key,
                           "metric_label": label, "unit": unit, "preferred_direction": direction,
                           "no_warmup_day1": a, "warmup_2d_measurement_day": b,
                           "delta_warmup_minus_day1": delta,
                           "relative_delta_pct": None if delta is None or a == 0 else delta / abs(a) * 100})

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "ten_strategy_metrics.csv", wide)
    write_csv(output / "cross_window_deltas.csv", paired)
    write_csv(output / "window_audit.csv", audit)

    rows = []
    for row in paired:
        delta_class = "pos" if (row["delta_warmup_minus_day1"] or 0) > 0 else "neg"
        rows.append("<tr>" + "".join((
            f"<td>{html.escape(row['policy'])}</td>", f"<td>{html.escape(row['metric_label'])}</td>",
            f"<td>{fmt(row['no_warmup_day1'], row['unit'])}</td>",
            f"<td>{fmt(row['warmup_2d_measurement_day'], row['unit'])}</td>",
            f"<td class='{delta_class}'>{fmt(row['delta_warmup_minus_day1'], row['unit'])}</td>",
            f"<td>{fmt(row['relative_delta_pct'], '%')}</td>")) + "</tr>")
    first_counts = sorted({int(r["arrival_count"]) for r in wide if r["condition"] == "no_warmup_day1"})
    warm_counts = sorted({int(r["arrival_count"]) for r in wide if r["condition"] == "warmup_2d_day3"})
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Day 1 vs warm-up day</title>
<style>body{{font:14px system-ui;margin:32px;color:#17202a}}.note{{background:#fff4ce;border-left:5px solid #d39e00;padding:12px;max-width:1100px}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #d9dee5;padding:7px 9px;text-align:right}}th{{background:#eef2f6;position:sticky;top:0}}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){{text-align:left}}.pos{{color:#087f23}}.neg{{color:#b42318}}</style></head><body>
<h1>无 Warm-up 第一天 vs Warm-up 2d 测量日</h1>
<p>No warm-up 使用 [0,288)，Warm-up 使用 [576,864)。任务数分别为 {first_counts} 与 {warm_counts}。</p>
<div class='note'><b>解释限制：</b>这是跨日窗口对比，不是同任务 cohort。负载、任务组成、价格和绿电条件可能随天变化；差值只能描述两个窗口的观测差异，不能单独归因为 warm-up。</div>
<table><thead><tr><th>策略</th><th>指标</th><th>No warm-up 第一天</th><th>Warm-up 测量日</th><th>Δ</th><th>相对变化</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    (output / "first_day_vs_warmup_report.html").write_text(page, encoding="utf-8")
    (output / "README.md").write_text(
        "# No-warm-up day 1 vs warm-up measurement day\n\n"
        "This is a cross-window comparison: `[0,288)` versus `[576,864)`. It is not a paired task-cohort comparison, so deltas are descriptive rather than causal.\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/v3/evaluation/first_day_vs_warmup"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build(root, root / args.output if not args.output.is_absolute() else args.output)


if __name__ == "__main__":
    main()
