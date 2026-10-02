"""V4 training entry point."""

import sys

from v1.train_v1 import main as _training_main

from ._entrypoint import use_v4_runtime


def main():
    has_idle_choice = any(
        arg in {"--skip-idle-cycles", "--no-skip-idle-cycles"}
        for arg in sys.argv[1:]
    )
    if not has_idle_choice:
        sys.argv.append("--skip-idle-cycles")
    with use_v4_runtime():
        _training_main(
            system_version="4.0",
            default_output="artifacts/v4/logs/candidate_dqn.pt",
            default_candidate_chunk_size=65536,
            default_checkpoint_every=50000,
        )


if __name__ == "__main__":
    main()
