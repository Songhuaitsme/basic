"""Process-local adapters for V4 CLI entry points."""

from contextlib import contextmanager

import v1.evaluate_v1 as evaluation
import v1.train_v1 as training

from .v4_runtime import create_v4_runtime


def _gate_config_view():
    from .gating import WaitGateConfig

    gate = WaitGateConfig()
    return {
        "V4_WAIT_GATE_ENABLED": True,
        "V4_WAIT_GATE_TIME_SCALE_SIM": gate.time_scale_sim,
        "V4_WAIT_GATE_PENALTY_WEIGHT_BY_SLA": {
            key.value: float(value)
            for key, value in gate.penalty_weight_by_sla.items()
        },
        "V4_WAIT_GATE_MIN_GAIN_BY_SLA": {
            key.value: float(value)
            for key, value in gate.min_gain_by_sla.items()
        },
        "V4_WAIT_GATE_TOLERANCE": gate.tolerance,
    }


@contextmanager
def use_v4_runtime():
    old_training_factory = training.create_v1_runtime
    old_evaluation_factory = evaluation.create_v1_runtime
    old_training_config_view = training._training_config_view

    def v4_training_config_view(**overrides):
        values = old_training_config_view(**overrides)
        values.update(_gate_config_view())
        return values

    training.create_v1_runtime = create_v4_runtime
    evaluation.create_v1_runtime = create_v4_runtime
    training._training_config_view = v4_training_config_view
    try:
        yield
    finally:
        training.create_v1_runtime = old_training_factory
        evaluation.create_v1_runtime = old_evaluation_factory
        training._training_config_view = old_training_config_view
