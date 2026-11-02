"""Process-local V9 adapter, preserving V8 entrypoint restoration."""

from contextlib import contextmanager

import v1.evaluate_v1 as evaluation
import v1.train_v1 as training

from v8._entrypoint import use_v8_runtime

from .v9_runtime import create_v9_runtime


@contextmanager
def use_v9_runtime(search_stage=3):
    with use_v8_runtime():
        old_config = training._training_config_view
        old_training_factory = training.create_v1_runtime
        old_evaluation_factory = evaluation.create_v1_runtime

        def factory(**kwargs):
            return create_v9_runtime(search_stage=search_stage, **kwargs)

        def config_view(**overrides):
            values = old_config(**overrides)
            values.update(V9_SEARCH_STAGE=f"E{search_stage}")
            return values

        training.create_v1_runtime = factory
        evaluation.create_v1_runtime = factory
        training._training_config_view = config_view
        try:
            yield
        finally:
            training.create_v1_runtime = old_training_factory
            evaluation.create_v1_runtime = old_evaluation_factory
            training._training_config_view = old_config
