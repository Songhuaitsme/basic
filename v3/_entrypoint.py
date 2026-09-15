"""Process-local adapters for reusing frozen V1 CLI orchestration in V3."""

from contextlib import contextmanager

import v1.evaluate_v1 as evaluation
import v1.train_v1 as training

from .v3_runtime import create_v3_runtime


@contextmanager
def use_v3_runtime():
    """Temporarily inject the V3 factory without modifying V1/V2 source."""

    old_training_factory = training.create_v1_runtime
    old_evaluation_factory = evaluation.create_v1_runtime
    training.create_v1_runtime = create_v3_runtime
    evaluation.create_v1_runtime = create_v3_runtime
    try:
        yield
    finally:
        training.create_v1_runtime = old_training_factory
        evaluation.create_v1_runtime = old_evaluation_factory
