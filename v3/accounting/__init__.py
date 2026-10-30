# 文件功能：声明 v3.accounting 包，汇集对外导出的类型与组件。
# 主要内容：导出或引用 SweepLineEnergyAccounting。
# 使用入口：包导入或外部调用方使用。
# 模块下游（项目依赖）：v3.accounting.energy。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V3 accounting implementations."""

from .energy import SweepLineEnergyAccounting

__all__ = ["SweepLineEnergyAccounting"]
