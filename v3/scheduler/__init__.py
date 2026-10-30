# 文件功能：声明 v3.scheduler 包，汇集对外导出的类型与组件。
# 主要内容：导出或引用 V3CandidateGenerator。
# 使用入口：包导入或外部调用方使用。
# 模块下游（项目依赖）：v3.scheduler.candidate_generator。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V3 scheduler components."""

from .candidate_generator import V3CandidateGenerator

__all__ = ["V3CandidateGenerator"]
