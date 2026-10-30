"""V7 training; inference can reuse V6 checkpoints, exact resume uses V7."""

import sys
from v1.train_v1 import main as training_main
from ._entrypoint import use_v7_runtime


def main():
    if not any(arg in {"--skip-idle-cycles", "--no-skip-idle-cycles"} for arg in sys.argv[1:]):
        sys.argv.append("--skip-idle-cycles")
    with use_v7_runtime():
        training_main(
            system_version="7.0", default_output="artifacts/v7/logs/candidate_dqn.pt",
            default_candidate_chunk_size=65536, default_checkpoint_every=50000,
        )


if __name__ == "__main__":
    main()
