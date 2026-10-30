# 文件功能：通过上下文临时把 V1 训练与评估切换到 V5 工厂及配置，退出时沿嵌套上下文恢复入口。
# 主要内容：use_v5_runtime。
# 模块上游（导入关系）：v5.evaluate_v5、v5.train_v5、v6._entrypoint、v5.tests.test_profiling。
# 模块下游（项目依赖）：v1.evaluate_v1、v1.train_v1、v4._entrypoint、v5.profiling、v5.v5_runtime。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Process-local adapters for V5 training and evaluation."""

from contextlib import contextmanager

import v1.evaluate_v1 as evaluation
import v1.train_v1 as training
from v4._entrypoint import use_v4_runtime

from .profiling import V5TrainingPerformanceProfiler
from .v5_runtime import candidate_config_view, create_v5_runtime


# 作用：嵌套 V4 适配，注入 V5 工厂、配置和性能统计器。
# 上游（直接调用）：v5.evaluate_v5.main、v5.train_v5.main、v6._entrypoint.use_v6_runtime。
# 下游调用：use_v4_runtime。
@contextmanager
def use_v5_runtime():
    # Reuse V4's gate metadata and restoration of all patched entry points.
    with use_v4_runtime():
        old_config_view = training._training_config_view
        old_training_profiler = training.TrainingPerformanceProfiler
        old_evaluation_profiler = evaluation.TrainingPerformanceProfiler

        # 作用：在上层训练配置中补充 V5 候选优化参数。
        # 调用约定：由外层 v5._entrypoint.use_v5_runtime 创建，作为局部辅助或回调使用。
        # 下游调用：candidate_config_view。
        def v5_training_config_view(**overrides):
            values = old_config_view(**overrides)
            values.update(candidate_config_view())
            return values

        training.create_v1_runtime = create_v5_runtime
        evaluation.create_v1_runtime = create_v5_runtime
        training._training_config_view = v5_training_config_view
        training.TrainingPerformanceProfiler = V5TrainingPerformanceProfiler
        evaluation.TrainingPerformanceProfiler = V5TrainingPerformanceProfiler
        try:
            yield
        finally:
            training.TrainingPerformanceProfiler = old_training_profiler
            evaluation.TrainingPerformanceProfiler = old_evaluation_profiler
