# 文件功能：声明 v5 包，汇集对外导出的类型与组件。
# 主要内容：导出或引用 create_v5_runtime。
# 使用入口：包导入或外部调用方使用。
# 模块下游（项目依赖）：v5.v5_runtime。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V5 lossless heuristic candidate-pool acceleration."""

from .v5_runtime import create_v5_runtime

SYSTEM_VERSION = "5.0"

__all__ = ["SYSTEM_VERSION", "create_v5_runtime"]
