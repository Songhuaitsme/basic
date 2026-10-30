# 文件功能：声明 v3 包，汇集对外导出的类型与组件。
# 主要内容：导出或引用 create_v3_runtime。
# 使用入口：包导入或外部调用方使用。
# 模块下游（项目依赖）：v3.v3_runtime。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V3 candidate-generation performance runtime.

V3 reuses the frozen V1 domain, scheduler, feature, and model contracts.  Its
changes are isolated to this package and are injected only by the V3 entry
points.
"""

SYSTEM_VERSION = "3.0"

from .v3_runtime import create_v3_runtime

__all__ = ["SYSTEM_VERSION", "create_v3_runtime"]
