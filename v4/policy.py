# 文件功能：包装原策略选择接口，应用等待门控并修正选择轨迹及诊断记录。
# 主要内容：NetBenefitGatePolicy。
# 模块上游（导入关系）：v4.v4_runtime、v4.tests.test_wait_gate、v5.tests.test_candidate_generator、v6.tests.test_energy。
# 模块下游（项目依赖）：v1.scheduler.policies。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Transparent policy wrapper applying the V4 wait gate."""

from __future__ import annotations

from dataclasses import replace

from v1.scheduler.policies import CandidateStreamSelection


class NetBenefitGatePolicy:
    """Preserve the wrapped policy API while gating delayed selections."""

    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v4.v4_runtime.create_v4_runtime。
    # 调用约定：构造实例时由 Python 调用。
    def __init__(self, base_policy, gate):
        object.__setattr__(self, "base_policy", base_policy)
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "_gate_diagnostics", {})

    # 作用：返回被包装策略名称。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    @property
    def name(self):
        return self.base_policy.name

    # 作用：将未直接定义的属性读取转发给原策略。
    # 调用约定：由 Python 属性读取协议调用。
    def __getattr__(self, name):
        return getattr(self.base_policy, name)

    # 作用：控制属性写入，维护不可变性或包装器转发约定。
    # 调用约定：由 Python 属性赋值协议调用。
    def __setattr__(self, name, value):
        if name in {"base_policy", "gate", "_gate_diagnostics"}:
            object.__setattr__(self, name, value)
        elif hasattr(self.base_policy, name):
            setattr(self.base_policy, name, value)
        else:
            object.__setattr__(self, name, value)

    # 作用：清除任务上次选择的门控诊断。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def clear_gate_diagnostic(self, task_id):
        self._gate_diagnostics.pop(task_id, None)

    # 作用：取出并移除任务的门控诊断。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def pop_gate_diagnostic(self, task_id):
        return self._gate_diagnostics.pop(task_id, None)

    # 作用：门控回退后同步修正最近的 DQN 选择轨迹及特征。
    # 上游（直接调用）：v4.policy.NetBenefitGatePolicy._apply_gate。
    # 下游调用：encoder.encode。
    def _rewrite_latest_trace(self, task, selection, original_selected):
        traces = getattr(self.base_policy, "_selection_traces", None)
        if not traces or selection.selected_candidate is original_selected:
            return
        trace = traces[-1]
        task_id = "" if task is None else task.task_id
        if trace.task_id != task_id or trace.selected_candidate is not original_selected:
            return
        encoder = getattr(self.base_policy, "feature_encoder", None)
        if encoder is None:
            return
        candidate = selection.selected_candidate
        features = encoder.encode(
            candidate,
            selection.earliest_candidate.compute_start_sim,
        )
        traces[-1] = replace(
            trace,
            selected_candidate_id=candidate.candidate_id,
            selected_candidate_features=tuple(features),
            selected_candidate=candidate,
        )

    # 作用：应用门控，保存诊断并修正训练轨迹。
    # 上游（直接调用）：v4.policy.NetBenefitGatePolicy.select、v4.policy.NetBenefitGatePolicy.select_complete_stream、v4.policy.NetBenefitGatePolicy.select_stream。
    # 调用上下文：原策略选出候选后调用；门控回退时改写最新选择轨迹，保证经验对应实际候选。
    # 下游调用：self._rewrite_latest_trace、self.gate.apply。
    def _apply_gate(self, selection, task):
        original_selected = selection.selected_candidate
        gated, diagnostic = self.gate.apply(selection, task)
        if task is not None:
            self._gate_diagnostics[task.task_id] = diagnostic
        self._rewrite_latest_trace(task, gated, original_selected)
        return gated

    # 作用：遍历候选流，汇总选中与最早候选及决策信息。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    # 下游调用：self._apply_gate、self.base_policy.select_stream。
    def select_stream(self, candidates, task=None, context=None):
        selection = self.base_policy.select_stream(
            candidates,
            task=task,
            context=context,
        )
        return self._apply_gate(selection, task)

    # 作用：选择完整候选流，保留最早反事实及决策信息。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    # 下游调用：self._apply_gate、self.base_policy.select_complete_stream、self.base_policy.select_stream、stream.iter_candidates。
    def select_complete_stream(self, stream, task=None):
        if hasattr(self.base_policy, "select_complete_stream"):
            selection = self.base_policy.select_complete_stream(
                stream,
                task=task,
            )
        else:
            selection = self.base_policy.select_stream(
                stream.iter_candidates(),
                task=task,
                context=stream.context,
            )
        return self._apply_gate(selection, task)

    # 作用：按当前策略从候选集合返回最终候选。
    # 调用约定：由调度器或包装器按所选策略动态调用。
    # 下游调用：self._apply_gate、self.base_policy.select、CandidateStreamSelection。
    def select(self, candidates, task=None):
        items = tuple(candidates)
        if not items:
            raise ValueError("candidate set cannot be empty")
        selected = self.base_policy.select(items, task=task)
        earliest = min(
            items,
            key=lambda item: (
                item.compute_start_sim,
                item.target_node,
                item.path.path_id,
                item.candidate_id,
            ),
        )
        selection = CandidateStreamSelection(
            selected,
            earliest,
            len(items),
            "",
            None,
        )
        return self._apply_gate(selection, task).selected_candidate
