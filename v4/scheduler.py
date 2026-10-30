# 文件功能：扩展 V1 调度决策记录，把等待门控诊断写入 V4 决策。
# 主要内容：V4SchedulingDecision、V4Scheduler。
# 模块上游（导入关系）：v4.v4_runtime、v4.tests.test_wait_gate。
# 模块下游（项目依赖）：v1.scheduler.v1_scheduler。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V4 scheduler diagnostics for the post-selection wait gate."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

from v1.scheduler.v1_scheduler import SchedulingDecision, V1Scheduler


@dataclass(frozen=True)
class V4SchedulingDecision(SchedulingDecision):
    proposed_candidate_id: Optional[str] = None
    proposed_active_wait_sim: Optional[float] = None
    wait_gate_applied: Optional[bool] = None
    wait_gate_passed: Optional[bool] = None
    wait_gate_reason: str = ""
    objective_gain_before_wait_penalty: Optional[float] = None
    wait_penalty: Optional[float] = None
    estimated_net_wait_gain: Optional[float] = None
    minimum_required_wait_gain: Optional[float] = None


class V4Scheduler(V1Scheduler):
    """V1-compatible scheduler that exposes V4 gate decisions in audit rows."""

    # 作用：清除任务旧门控诊断，再委托 V1 完成单任务调度。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：super()._schedule_one。
    def _schedule_one(self, task, *args, **kwargs):
        clear = getattr(self.policy, "clear_gate_diagnostic", None)
        if clear is not None:
            clear(task.task_id)
        return super()._schedule_one(task, *args, **kwargs)

    # 作用：在 V1 记录上附加门控收益、惩罚、阈值及原因。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：V1Scheduler._decision_record、V4SchedulingDecision。
    def _decision_record(self, task_id, *args, **kwargs):
        base = V1Scheduler._decision_record(task_id, *args, **kwargs)
        pop = getattr(self.policy, "pop_gate_diagnostic", None)
        diagnostic = None if pop is None else pop(task_id)
        base_values = {
            item.name: getattr(base, item.name)
            for item in fields(SchedulingDecision)
        }
        if diagnostic is None:
            return V4SchedulingDecision(**base_values)
        return V4SchedulingDecision(
            **base_values,
            proposed_candidate_id=diagnostic.proposed_candidate_id,
            proposed_active_wait_sim=diagnostic.proposed_active_wait_sim,
            wait_gate_applied=diagnostic.applied,
            wait_gate_passed=diagnostic.passed,
            wait_gate_reason=diagnostic.reason,
            objective_gain_before_wait_penalty=(
                diagnostic.objective_gain_before_wait_penalty
            ),
            wait_penalty=diagnostic.wait_penalty,
            estimated_net_wait_gain=diagnostic.estimated_net_wait_gain,
            minimum_required_wait_gain=diagnostic.minimum_required_gain,
        )
