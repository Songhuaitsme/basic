#!/usr/bin/env python3
"""CPU/CUDA inference crossover benchmark for the frozen v1 Q-network."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v1.learning.candidate_dqn import SharedCandidateQNetwork


SIZES_AND_REPEATS = (
    (1, 400),
    (32, 300),
    (256, 160),
    (1024, 80),
    (4096, 40),
    (16384, 16),
    (65536, 6),
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_samples(operation, device: torch.device, repeats: int) -> list[float]:
    for _ in range(5):
        operation()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        synchronize(device)
        samples.append(time.perf_counter() - started)
    return samples


def summarize(samples: list[float], candidates: int) -> dict[str, float]:
    median = statistics.median(samples)
    ordered = sorted(samples)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    return {
        "median_seconds": median,
        "p90_seconds": p90,
        "median_candidates_per_second": candidates / median,
    }


def benchmark_device(
    *,
    device_name: str,
    metadata: dict[str, object],
    state_dict: dict[str, torch.Tensor],
    seed: int,
) -> list[dict[str, object]]:
    device = torch.device(device_name)
    network = SharedCandidateQNetwork(
        int(metadata["global_state_dim"]),
        int(metadata["candidate_feature_dim"]),
    )
    network.load_state_dict(state_dict)
    network.eval().to(device)

    rng = np.random.default_rng(seed)
    global_state_np = rng.normal(
        size=int(metadata["global_state_dim"])
    ).astype(np.float32)
    output = []

    for candidates, repeats in SIZES_AND_REPEATS:
        features_np = rng.normal(
            size=(candidates, int(metadata["candidate_feature_dim"]))
        ).astype(np.float32)

        def transfer_roundtrip() -> None:
            state = torch.as_tensor(global_state_np, device=device)
            features = torch.as_tensor(features_np, device=device)
            with torch.no_grad():
                network(state, features).cpu().numpy()

        transfer_samples = timed_samples(
            transfer_roundtrip, device=device, repeats=repeats
        )

        state_resident = torch.as_tensor(global_state_np, device=device)
        features_resident = torch.as_tensor(features_np, device=device)

        def resident_forward() -> None:
            with torch.no_grad():
                network(state_resident, features_resident)

        resident_samples = timed_samples(
            resident_forward, device=device, repeats=repeats
        )

        output.append(
            {
                "device": device_name,
                "candidate_count": candidates,
                "repeats": repeats,
                "transfer_roundtrip": summarize(transfer_samples, candidates),
                "resident_forward": summarize(resident_samples, candidates),
            }
        )
        del features_resident
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="artifacts/v1/pilot/candidate_dqn_pilot_seed7_b1.pt",
    )
    parser.add_argument(
        "--output",
        default="artifacts/v1/benchmarks/cuda_compare/inference_crossover.json",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    results = []
    for device in devices:
        results.extend(
            benchmark_device(
                device_name=device,
                metadata=checkpoint["metadata"],
                state_dict=checkpoint["model_state_dict"],
                seed=args.seed,
            )
        )

    by_size: dict[int, dict[str, dict[str, object]]] = {}
    for row in results:
        by_size.setdefault(int(row["candidate_count"]), {})[str(row["device"])] = row
    comparison = []
    for size in sorted(by_size):
        cpu = by_size[size]["cpu"]
        cuda = by_size[size].get("cuda")
        record: dict[str, object] = {"candidate_count": size}
        if cuda is not None:
            for mode in ("transfer_roundtrip", "resident_forward"):
                cpu_seconds = float(cpu[mode]["median_seconds"])  # type: ignore[index]
                cuda_seconds = float(cuda[mode]["median_seconds"])  # type: ignore[index]
                record[f"{mode}_cpu_seconds"] = cpu_seconds
                record[f"{mode}_cuda_seconds"] = cuda_seconds
                record[f"{mode}_cuda_speedup"] = cpu_seconds / cuda_seconds
        comparison.append(record)

    payload = {
        "status": "VALID",
        "torch_version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "torch_cpu_threads": torch.get_num_threads(),
        "model": args.model,
        "metadata": checkpoint["metadata"],
        "measurement": {
            "transfer_roundtrip": "numpy input -> device -> Q-network -> CPU numpy output",
            "resident_forward": "input tensors and output remain on device",
            "statistic": "median of synchronized per-call wall time",
        },
        "results": results,
        "comparison": comparison,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "VALID", "output": str(output)}))


if __name__ == "__main__":
    main()
