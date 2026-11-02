"""Optional V9 retraining entrypoint; keep frozen-model comparisons separate."""

import argparse
import sys

from v1.train_v1 import main as training_main

from ._entrypoint import use_v9_runtime


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--search-stage", type=int, choices=range(4), default=3)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if not any(arg in {"--skip-idle-cycles", "--no-skip-idle-cycles"}
               for arg in sys.argv[1:]):
        sys.argv.append("--skip-idle-cycles")
    with use_v9_runtime(known.search_stage):
        training_main(
            system_version="9.0",
            default_output=f"artifacts/v9/e{known.search_stage}/candidate_dqn.pt",
            default_candidate_chunk_size=65536,
            default_checkpoint_every=50000,
        )


if __name__ == "__main__":
    main()
