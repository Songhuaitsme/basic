# 文件功能：通过上下文临时把 V1 训练与评估切换到 V3 工厂及配置，退出时沿嵌套上下文恢复入口。
# 主要内容：use_v3_runtime。
# 模块上游（导入关系）：v3.evaluate_v3、v3.train_v3。
# 模块下游（项目依赖）：v1.evaluate_v1、v1.train_v1、v3.v3_runtime。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Process-local adapters for reusing frozen V1 CLI orchestration in V3."""

from contextlib import contextmanager

import v1.evaluate_v1 as evaluation
import v1.train_v1 as training

from .v3_runtime import create_v3_runtime


# 作用：临时把训练与评估工厂切换为 V3，退出时恢复。
# 上游（直接调用）：v3.evaluate_v3.main、v3.train_v3.main。
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
