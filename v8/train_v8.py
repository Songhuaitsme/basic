"""Train V8 while keeping V7 model architecture and checkpoint format."""

import sys

from v1.train_v1 import main as training_main

from ._entrypoint import use_v8_runtime


def main():
    if not any(arg in {"--skip-idle-cycles", "--no-skip-idle-cycles"} for arg in sys.argv[1:]):
        sys.argv.append("--skip-idle-cycles")
    with use_v8_runtime():
        training_main(
            system_version="8.0", default_output="artifacts/v8/logs/candidate_dqn.pt",
            default_candidate_chunk_size=65536, default_checkpoint_every=50000,
        )


if __name__ == "__main__":
    main()
