# 文件功能：通过上下文临时把 V1 训练与评估切换到 V6 工厂及配置，退出时沿嵌套上下文恢复入口。
# 主要内容：use_v6_runtime。
# 模块上游（导入关系）：v6.evaluate_v6、v6.train_v6。
# 模块下游（项目依赖）：v1.evaluate_v1、v1.train_v1、v5._entrypoint、v6.v6_runtime。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Process-local V6 adapters preserving V5 configuration and profiler fixes."""

from contextlib import contextmanager

import v1.evaluate_v1 as evaluation
import v1.train_v1 as training
from v5._entrypoint import use_v5_runtime

from .v6_runtime import accounting_config_view, create_v6_runtime


# 作用：嵌套 V5 适配，注入 V6 工厂和能源优化配置。
# 上游（直接调用）：v6.evaluate_v6.main、v6.train_v6.main。
# 下游调用：use_v5_runtime。
@contextmanager
def use_v6_runtime():
    with use_v5_runtime():
        old_config_view = training._training_config_view

        # 作用：在上层训练配置中补充 V6 能源优化参数。
        # 调用约定：由外层 v6._entrypoint.use_v6_runtime 创建，作为局部辅助或回调使用。
        # 下游调用：accounting_config_view。
        def v6_training_config_view(**overrides):
            values = old_config_view(**overrides)
            values.update(accounting_config_view())
            return values

        training.create_v1_runtime = create_v6_runtime
        evaluation.create_v1_runtime = create_v6_runtime
        training._training_config_view = v6_training_config_view
        yield
