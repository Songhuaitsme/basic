# 文件功能：声明 v1.simulation 包，汇集对外导出的类型与组件。
# 主要内容：导出或引用 ALLOWED_TRANSITIONS、TERMINAL_STATES、StateTransition、StateTransitionError、TaskStateMachine、DomainEvent、EventEngine。
# 使用入口：包导入或外部调用方使用。
# 模块下游（项目依赖）：v1.simulation.state_machine、v1.simulation.event_engine。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""v1.0 discrete-event simulation primitives."""

from .state_machine import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    StateTransition,
    StateTransitionError,
    TaskStateMachine,
)
from .event_engine import DomainEvent, EventEngine


__all__ = [
    "ALLOWED_TRANSITIONS",
    "DomainEvent",
    "EventEngine",
    "TERMINAL_STATES",
    "StateTransition",
    "StateTransitionError",
    "TaskStateMachine",
]
