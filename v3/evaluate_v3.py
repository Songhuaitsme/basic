"""V3 evaluation entry point."""

from v1.evaluate_v1 import main as _evaluation_main

from ._entrypoint import use_v3_runtime


def main():
    with use_v3_runtime():
        _evaluation_main(
            system_version="3.0",
            default_output="artifacts/v3/evaluation/report.json",
            default_audit_mode="periodic",
            default_report_mode="compact",
            default_candidate_chunk_size=65536,
        )


if __name__ == "__main__":
    main()
