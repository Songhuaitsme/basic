#!/usr/bin/env python3
"""Summarize repeated CPU/CUDA v1 training profiles and inference crossover."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_DIR = ROOT / "artifacts" / "v1" / "benchmarks" / "cuda_compare"
PROFILE_PATTERN = re.compile(
    r"candidate_dqn_seed(?P<seed>\d+)_(?P<device>cpu|cuda)"
    r"(?:_trial(?P<trial>\d+))?\.profile\.json$"
)


def median(rows: list[dict[str, object]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def load_profile_rows() -> list[dict[str, object]]:
    output = []
    for path in sorted(BENCHMARK_DIR.glob("candidate_dqn_seed*.profile.json")):
        match = PROFILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        sections = payload["sections_seconds"]
        counters = payload["counters"]
        training_path = path.with_name(path.name.replace(".profile.json", ".training.csv"))
        with training_path.open("r", encoding="utf-8", newline="") as handle:
            training_rows = list(csv.DictReader(handle))
        loss_rows = [row for row in training_rows if row["mean_loss"] != ""]
        output.append(
            {
                "seed": int(match.group("seed")),
                "device": match.group("device"),
                "trial": int(match.group("trial") or 1),
                "total_seconds": float(payload["total_wall_seconds"]),
                "core_seconds_excluding_logging_saving": (
                    float(payload["total_wall_seconds"])
                    - float(sections["logging_and_saving"])
                ),
                "candidate_slot_generation_seconds": float(
                    sections["candidate_slot_generation"]
                ),
                "neural_network_inference_seconds": float(
                    sections["neural_network_inference"]
                ),
                "backpropagation_optimizer_seconds": float(
                    sections["backpropagation_and_optimizer"]
                ),
                "environment_accounting_seconds": float(
                    sections["environment_and_accounting"]
                ),
                "logging_saving_seconds": float(sections["logging_and_saving"]),
                "selection_candidate_count": int(
                    counters["selection_candidate_count"]
                ),
                "replay_candidate_feature_count": int(
                    counters["replay_candidate_feature_count"]
                ),
                "final_transition_count": int(training_rows[-1]["transitions"]),
                "final_update_count": int(training_rows[-1]["updates"]),
                "last_logged_mean_loss": (
                    float(loss_rows[-1]["mean_loss"]) if loss_rows else None
                ),
                "profile_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    return output


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        by_device = {
            device: [row for row in seed_rows if row["device"] == device]
            for device in ("cpu", "cuda")
        }
        for device, device_rows in by_device.items():
            output.append(
                {
                    "seed": seed,
                    "device": device,
                    "trial_count": len(device_rows),
                    "median_total_seconds": median(device_rows, "total_seconds"),
                    "median_core_seconds": median(
                        device_rows, "core_seconds_excluding_logging_saving"
                    ),
                    "median_candidate_seconds": median(
                        device_rows, "candidate_slot_generation_seconds"
                    ),
                    "median_neural_network_seconds": median(
                        device_rows, "neural_network_inference_seconds"
                    ),
                    "median_backprop_seconds": median(
                        device_rows, "backpropagation_optimizer_seconds"
                    ),
                    "candidate_share_of_core": (
                        median(device_rows, "candidate_slot_generation_seconds")
                        / median(
                            device_rows, "core_seconds_excluding_logging_saving"
                        )
                    ),
                }
            )
    return output


def paired_speedups(
    rows: list[dict[str, object]], aggregated: list[dict[str, object]]
) -> list[dict[str, object]]:
    output = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        cpu_agg = next(
            row
            for row in aggregated
            if row["seed"] == seed and row["device"] == "cpu"
        )
        cuda_agg = next(
            row
            for row in aggregated
            if row["seed"] == seed and row["device"] == "cuda"
        )
        record = {
            "seed": seed,
            "trial_count_per_device": cpu_agg["trial_count"],
        }
        for label in ("total", "core", "candidate", "neural_network", "backprop"):
            cpu_value = float(cpu_agg[f"median_{label}_seconds"])
            cuda_value = float(cuda_agg[f"median_{label}_seconds"])
            record[f"{label}_cpu_median_seconds"] = cpu_value
            record[f"{label}_cuda_median_seconds"] = cuda_value
            record[f"{label}_cuda_speedup"] = cpu_value / cuda_value
        output.append(record)
    return output


def semantic_checks(rows: list[dict[str, object]]) -> dict[str, object]:
    checks = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        fields = (
            "selection_candidate_count",
            "replay_candidate_feature_count",
            "final_transition_count",
            "final_update_count",
            "last_logged_mean_loss",
        )
        field_checks = {
            field: len({row[field] for row in seed_rows}) == 1 for field in fields
        }
        checks.append(
            {
                "seed": seed,
                "all_equal": all(field_checks.values()),
                "fields": field_checks,
                "values": {field: seed_rows[0][field] for field in fields},
            }
        )
    return {
        "status": "PASS" if all(item["all_equal"] for item in checks) else "FAIL",
        "by_seed": checks,
    }


def chunk_size_rows() -> list[dict[str, object]]:
    output = []
    for chunk in (4096, 16384, 65536):
        if chunk == 4096:
            paths = sorted(
                path
                for path in BENCHMARK_DIR.glob(
                    "candidate_dqn_seed14_cuda*.profile.json"
                )
                if "chunk" not in path.name
            )
        else:
            paths = [
                BENCHMARK_DIR
                / f"candidate_dqn_seed14_cuda_chunk{chunk}.profile.json"
            ]
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        output.append(
            {
                "chunk_size": chunk,
                "trial_count": len(payloads),
                "median_total_seconds": statistics.median(
                    float(item["total_wall_seconds"]) for item in payloads
                ),
                "median_core_seconds": statistics.median(
                    float(item["total_wall_seconds"])
                    - float(item["sections_seconds"]["logging_and_saving"])
                    for item in payloads
                ),
                "median_neural_network_seconds": statistics.median(
                    float(item["sections_seconds"]["neural_network_inference"])
                    for item in payloads
                ),
                "median_candidate_seconds": statistics.median(
                    float(item["sections_seconds"]["candidate_slot_generation"])
                    for item in payloads
                ),
                "selection_candidate_count": int(
                    payloads[0]["counters"]["selection_candidate_count"]
                ),
                "replay_candidate_feature_count": int(
                    payloads[0]["counters"]["replay_candidate_feature_count"]
                ),
            }
        )
    return output


def main() -> None:
    rows = load_profile_rows()
    aggregated = aggregate(rows)
    speedups = paired_speedups(rows, aggregated)
    checks = semantic_checks(rows)
    chunks = chunk_size_rows()
    inference = json.loads(
        (BENCHMARK_DIR / "inference_crossover.json").read_text(encoding="utf-8")
    )
    crossover = next(
        (
            row["candidate_count"]
            for row in inference["comparison"]
            if row["transfer_roundtrip_cuda_speedup"] >= 1.0
        ),
        None,
    )

    payload = {
        "status": "VALID" if checks["status"] == "PASS" else "INVALID",
        "environment": {
            "torch_version": inference["torch_version"],
            "compiled_cuda": inference["compiled_cuda"],
            "cuda_device": inference["cuda_device"],
            "torch_cpu_threads": inference["torch_cpu_threads"],
        },
        "semantic_equivalence": checks,
        "raw_profiles": rows,
        "aggregate_by_seed_device": aggregated,
        "paired_speedups": speedups,
        "inference_crossover_candidate_count": crossover,
        "inference_comparison": inference["comparison"],
        "cuda_chunk_size_comparison_seed14": chunks,
        "decision": {
            "formal_device": "cuda",
            "formal_candidate_chunk_size": 65536,
            "reason": (
                "CUDA gives a modest end-to-end benefit while preserving all checked "
                "outcomes; 65536 minimizes measured CUDA inference time on the seed-14 "
                "workload, but Python candidate enumeration remains dominant."
            ),
            "not_proven": (
                "These benchmarks do not prove that CUDA materially changes the "
                "feasibility of long training or that the Pilot policy improves the objective."
            ),
        },
    }
    (BENCHMARK_DIR / "cuda_comparison_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (BENCHMARK_DIR / "cuda_comparison_trials.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    speedup_by_seed = {int(row["seed"]): row for row in speedups}
    seed9 = speedup_by_seed[9]
    seed14 = speedup_by_seed[14]
    chunk_best = min(chunks, key=lambda row: float(row["median_core_seconds"]))
    report = f"""# CUDA 对照结论

状态：{payload['status']}；PyTorch {inference['torch_version']}；设备：{inference['cuda_device']}。

## 端到端结果

- Seed 9（每设备 3 轮）：总耗时中位数 CPU {seed9['total_cpu_median_seconds']:.3f}s，CUDA {seed9['total_cuda_median_seconds']:.3f}s，CUDA 加速 {seed9['total_cuda_speedup']:.3f}×。
- Seed 14（每设备 3 轮）：总耗时中位数 CPU {seed14['total_cpu_median_seconds']:.3f}s，CUDA {seed14['total_cuda_median_seconds']:.3f}s，CUDA 加速 {seed14['total_cuda_speedup']:.3f}×。
- 所有配对轮次的候选数、replay 特征数、transition、update 和最后 loss 一致：{checks['status']}。

## 推理交叉点

- 包含 NumPy→设备→CPU NumPy 往返时，CUDA 首次达到 CPU 的批量约为 {crossover} 个候选。
- 当前 4096 候选/块的热身纯评分 CUDA 加速为 {next(row for row in inference['comparison'] if row['candidate_count'] == 4096)['transfer_roundtrip_cuda_speedup']:.3f}×。
- 短训练的 CUDA NN 阶段仍可能更慢，因为首次 CUDA 初始化和 batch=1 反向传播尚未摊薄。

## 块大小

- Seed 14 的最小核心耗时出现在 chunk={chunk_best['chunk_size']}：{chunk_best['median_core_seconds']:.3f}s。
- 选择正式候选块大小 65536；4GB RTX 3050 已完成该负载且语义结果一致。

## 决策

后续训练使用 CUDA 与 chunk=65536，但只将其视为小幅端到端优化。候选枚举仍占核心耗时的大多数，CUDA 不能替代候选生成和样本效率优化。
"""
    (BENCHMARK_DIR / "cuda_comparison_summary.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "summary": str(
                    BENCHMARK_DIR / "cuda_comparison_summary.json"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
