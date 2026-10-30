# 文件功能：通过上下文临时把 V1 训练与评估切换到 V4 工厂及配置，退出时沿嵌套上下文恢复入口。
# 主要内容：_gate_config_view、use_v4_runtime。
# 模块上游（导入关系）：v4.evaluate_v4、v4.train_v4、v5._entrypoint。
# 模块下游（项目依赖）：v1.evaluate_v1、v1.train_v1、v4.v4_runtime、v4.gating。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Process-local adapters for V4 CLI entry points."""

from contextlib import contextmanager

import v1.evaluate_v1 as evaluation
import v1.train_v1 as training

from .v4_runtime import create_v4_runtime


# 作用：提取等待门控配置供元数据与恢复兼容性检查使用。
# 上游（直接调用）：v4._entrypoint.use_v4_runtime.v4_training_config_view。
# 下游调用：WaitGateConfig。
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


# 作用：临时注入 V4 工厂与门控配置视图，退出时恢复。
# 上游（直接调用）：v4.evaluate_v4.main、v4.train_v4.main、v5._entrypoint.use_v5_runtime。
@contextmanager
def use_v4_runtime():
    old_training_factory = training.create_v1_runtime
    old_evaluation_factory = evaluation.create_v1_runtime
    old_training_config_view = training._training_config_view

    # 作用：在原训练配置中补充 V4 门控参数。
    # 调用约定：由外层 v4._entrypoint.use_v4_runtime 创建，作为局部辅助或回调使用。
    # 下游调用：_gate_config_view。
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
