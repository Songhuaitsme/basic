"""Process-local V8 adapters with V7/V6 restoration."""

from contextlib import contextmanager

import v1.evaluate_v1 as evaluation
import v1.train_v1 as training

from v7._entrypoint import use_v7_runtime

from .v8_runtime import create_v8_runtime, static_hard_config_view


@contextmanager
def use_v8_runtime():
    with use_v7_runtime():
        old_config = training._training_config_view
        old_training_factory = training.create_v1_runtime
        old_evaluation_factory = evaluation.create_v1_runtime

        def config_view(**overrides):
            values = old_config(**overrides)
            values.update(static_hard_config_view())
            return values

        training.create_v1_runtime = create_v8_runtime
        evaluation.create_v1_runtime = create_v8_runtime
        training._training_config_view = config_view
        try:
            yield
        finally:
            training.create_v1_runtime = old_training_factory
            evaluation.create_v1_runtime = old_evaluation_factory
            training._training_config_view = old_config
