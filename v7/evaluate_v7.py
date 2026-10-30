"""Evaluate V7 with existing V3/V4/V5/V6 model checkpoints."""

from v1.evaluate_v1 import main as evaluation_main
from ._entrypoint import use_v7_runtime


def main():
    with use_v7_runtime():
        evaluation_main(
            system_version="7.0", default_output="artifacts/v7/evaluation/report.json",
            default_audit_mode="periodic", default_report_mode="compact",
            default_candidate_chunk_size=65536,
        )


if __name__ == "__main__":
    main()
