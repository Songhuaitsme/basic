"""Evaluate V8 with existing V7/V6 Candidate DQN checkpoints."""

from v1.evaluate_v1 import main as evaluation_main

from ._entrypoint import use_v8_runtime


def main():
    with use_v8_runtime():
        evaluation_main(
            system_version="8.0", default_output="artifacts/v8/evaluation/report.json",
            default_audit_mode="periodic", default_report_mode="compact",
            default_candidate_chunk_size=65536,
        )


if __name__ == "__main__":
    main()
