"""Transparent policy wrapper applying the V4 wait gate."""

from __future__ import annotations

from dataclasses import replace

from v1.scheduler.policies import CandidateStreamSelection


class NetBenefitGatePolicy:
    """Preserve the wrapped policy API while gating delayed selections."""

    def __init__(self, base_policy, gate):
        object.__setattr__(self, "base_policy", base_policy)
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "_gate_diagnostics", {})

    @property
    def name(self):
        return self.base_policy.name

    def __getattr__(self, name):
        return getattr(self.base_policy, name)

    def __setattr__(self, name, value):
        if name in {"base_policy", "gate", "_gate_diagnostics"}:
            object.__setattr__(self, name, value)
        elif hasattr(self.base_policy, name):
            setattr(self.base_policy, name, value)
        else:
            object.__setattr__(self, name, value)

    def clear_gate_diagnostic(self, task_id):
        self._gate_diagnostics.pop(task_id, None)

    def pop_gate_diagnostic(self, task_id):
        return self._gate_diagnostics.pop(task_id, None)

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

    def _apply_gate(self, selection, task):
        original_selected = selection.selected_candidate
        gated, diagnostic = self.gate.apply(selection, task)
        if task is not None:
            self._gate_diagnostics[task.task_id] = diagnostic
        self._rewrite_latest_trace(task, gated, original_selected)
        return gated

    def select_stream(self, candidates, task=None, context=None):
        selection = self.base_policy.select_stream(
            candidates,
            task=task,
            context=context,
        )
        return self._apply_gate(selection, task)

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
