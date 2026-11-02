"""Evaluate V9 stages with a frozen V7/V8 compatible checkpoint."""

import argparse
import sys

from v1.evaluate_v1 import main as evaluation_main

from ._entrypoint import use_v9_runtime


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--search-stage", type=int, choices=range(4), default=3)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    with use_v9_runtime(known.search_stage):
        evaluation_main(
            system_version="9.0", default_output=f"artifacts/v9/e{known.search_stage}/report.json",
            default_audit_mode="periodic", default_report_mode="compact",
            default_candidate_chunk_size=65536,
        )


if __name__ == "__main__":
    main()
