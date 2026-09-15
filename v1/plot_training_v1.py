"""Render CandidateDQN training and fixed-validation curves from CSV logs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#2F6B9A"
GOLD = "#C6922C"
INK = "#263238"
GRID = "#D9DEE3"


def _read_rows(path):
    target = Path(path)
    if not target.is_file():
        return ()
    with target.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = tuple(csv.DictReader(handle))
    # A resumed run can write the same boundary again.  The last observation
    # is the one represented by the latest checkpoint.
    by_cycle = {}
    for row in rows:
        try:
            cycle = int(float(row.get("cycle", "")))
        except (TypeError, ValueError):
            continue
        by_cycle[cycle] = row
    return tuple(by_cycle[key] for key in sorted(by_cycle))


def _number(row, field):
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _plot_curve(path, title, ylabel, series, *, unit_note, ratio=False):
    fig, axis = plt.subplots(figsize=(9.6, 5.4))
    plotted = 0
    for label, rows, field, color, linestyle, marker in series:
        points = [
            (int(float(row["cycle"])), _number(row, field))
            for row in rows
            if row.get("cycle") not in (None, "")
        ]
        points = [(x, y) for x, y in points if y is not None]
        if not points:
            continue
        x_values, y_values = zip(*points)
        axis.plot(
            x_values,
            y_values,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker if len(points) < 20 else None,
            linewidth=2.0,
            markersize=4.5,
        )
        plotted += 1
    axis.set_title(title, loc="left", color=INK, fontsize=14, pad=14)
    axis.set_xlabel("Training Step (scheduler cycle)", color=INK)
    axis.set_ylabel(ylabel, color=INK)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(colors=INK)
    if ratio:
        axis.set_ylim(-0.02, 1.02)
    if plotted > 1:
        axis.legend(frameon=False, loc="best")
    if plotted == 0:
        axis.text(
            0.5,
            0.5,
            "No observations yet",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#6B747C",
        )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    fig.text(0.01, 0.015, unit_note, color="#6B747C", fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def generate_training_curves(training_csv, validation_csv, output_dir):
    """Create the eight required monitoring figures and return their paths."""

    training_rows = _read_rows(training_csv)
    validation_rows = _read_rows(validation_csv)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    definitions = (
        (
            "loss_vs_training_step.png",
            "Loss vs Training Step",
            "Mean Smooth L1 loss",
            (("Mean loss", training_rows, "mean_loss", BLUE, "-", "o"),),
            "Each point summarizes updates since the previous training log.",
            False,
        ),
        (
            "reward_vs_training_step.png",
            "Reward vs Training Step",
            "Reward per transition",
            (
                ("Stage mean", training_rows, "mean_reward", BLUE, "-", "o"),
                ("Rolling mean", training_rows, "rolling_reward", GOLD, "--", "s"),
            ),
            "Stage mean uses the log interval; rolling mean uses the configured transition window.",
            False,
        ),
        (
            "q_mean_range_vs_training_step.png",
            "Q Mean / Q Range vs Training Step",
            "Candidate Q value",
            (
                ("Q mean", training_rows, "q_mean", BLUE, "-", "o"),
                ("Q range", training_rows, "q_range", GOLD, "--", "s"),
            ),
            "Q summaries reuse policy selection traces; Q range is interval max minus interval min.",
            False,
        ),
        (
            "completion_rate_vs_training_step.png",
            "Completion Rate vs Training Step",
            "Completion rate",
            (("Completion rate", validation_rows, "completion_rate", BLUE, "-", "o"),),
            "Fixed Task Trace, model.eval(), epsilon=0; formal v1 metric definition.",
            True,
        ),
        (
            "sla_violation_vs_training_step.png",
            "SLA Violation vs Training Step",
            "Expired rate",
            (("Expired rate", validation_rows, "expired_rate", GOLD, "-", "o"),),
            "SLA violation monitoring is the arrived-task expired rate.",
            True,
        ),
        (
            "cost_per_completed_cpu_hour_vs_training_step.png",
            "Cost per Completed CPU Hour vs Training Step",
            "Yuan per completed CPU hour",
            ((
                "Cost / completed CPU hour",
                validation_rows,
                "cost_yuan_per_completed_cpu_hour",
                BLUE,
                "-",
                "o",
            ),),
            "Fixed-validation value from the formal v1 evaluation metric implementation.",
            False,
        ),
        (
            "task_green_coverage_vs_training_step.png",
            "Task Green Coverage vs Training Step",
            "Completed-task green coverage",
            (("Task green coverage", validation_rows, "task_green_coverage", BLUE, "-", "o"),),
            "Completed-task green energy divided by completed-task energy.",
            True,
        ),
        (
            "system_green_absorption_vs_training_step.png",
            "System Green Absorption vs Training Step",
            "System green absorption",
            ((
                "System green absorption",
                validation_rows,
                "system_green_absorption",
                GOLD,
                "-",
                "o",
            ),),
            "System green use divided by green supply over the settled validation interval.",
            True,
        ),
    )
    paths = []
    for filename, title, ylabel, series, note, ratio in definitions:
        paths.append(
            _plot_curve(
                output / filename,
                title,
                ylabel,
                series,
                unit_note=note,
                ratio=ratio,
            )
        )
    return tuple(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Render v1 CandidateDQN training and validation curves"
    )
    parser.add_argument("--training-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = generate_training_curves(
        args.training_csv, args.validation_csv, args.output_dir
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
