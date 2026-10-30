"""Optional, synchronous candidate-stage wall times with exclusive accounting.

Existing leaf timers feed this collector rather than being timed a second time.
Scopes must finish before a generator yields control to its consumer.
"""

from contextlib import nullcontext
from functools import wraps
import time


_NO_SCOPE = nullcontext()


LEAF_STAGES = {
    "candidate_metric_evaluation_seconds": "physical_metrics",
    "candidate_id_hash_seconds": "candidate_id",
    "candidate_object_construction_seconds": "candidate_object",
    "candidate_transmission_schedule_seconds": "transmission",
    "candidate_cpu_feasibility_seconds": "cpu_check",
    "candidate_path_feasibility_seconds": "path_check",
    "candidate_sla_and_item_seconds": "sla_and_item",
}

STAGE_LABELS = {
    "candidate_prepare": "候选准备外层及缓存生命周期",
    "grid_setup": "节点路径网格准备",
    "path_lookup": "静态路径查询与缓存",
    "transmission_duration": "任务静态传输时长准备",
    "grid_boundaries": "当前决策的时间网格边界",
    "time_sampling": "采样时间点生成",
    "anchor_pass": "Anchor循环与调度",
    "anchor_node_ranking": "Anchor节点排名与选择",
    "second_pass": "第二轮循环与记录去重",
    "record_cache_and_assembly": "记录缓存、容量判断与轻量记录组装",
    "feasibility": "可行性检查其他部分",
    "resource_occupancy": "资源占用情况查询",
    "transmission": "传输时间安排",
    "cpu_check": "CPU区间检查",
    "path_check": "路径区间检查",
    "sla_and_item": "SLA检查与信息组装",
    "materialization_prepare": "候选缓存过滤、分组与物化准备",
    "physical_metrics": "精确物理指标评估",
    "layered_selection": "归一化、分层排序与候选去重",
    "pareto": "Pareto筛选",
    "result_assembly": "最终记录、指标与上下文整理",
    "stream_consume": "候选流批次消费其他开销",
    "candidate_id": "候选ID生成",
    "candidate_object": "候选对象构造",
}


class CandidateDetailProfiler:
    def __init__(self, clock=None):
        self.clock = clock or time.perf_counter
        self.stack = []
        self.rows = {}
        self.counters = {}
        self.root_seconds = 0.0
        self.negative_self_count = 0

    def scope(self, stage, phase=None):
        return _StageScope(self, stage, phase)

    def _row(self, stage, phase, path):
        key = (phase, path)
        row = self.rows.get(key)
        if row is None:
            row = self.rows[key] = {
                "phase": phase, "stage": stage,
                "label": STAGE_LABELS.get(stage, stage), "path": path,
                "calls": 0, "inclusive_seconds": 0.0, "self_seconds": 0.0,
            }
        return row

    def record_leaf(self, legacy_key, seconds):
        stage = LEAF_STAGES.get(legacy_key)
        if stage is None or not self.stack:
            return
        parent = self.stack[-1]
        row = self._row(stage, parent.phase, parent.path + "/" + stage)
        row["calls"] += 1
        row["inclusive_seconds"] += seconds
        row["self_seconds"] += seconds
        parent.children += seconds

    def count(self, name, amount=1):
        phase = self.stack[-1].phase if self.stack else "outside"
        key = phase + "/" + name
        self.counters[key] = self.counters.get(key, 0) + amount

    def summary(self, pipeline_seconds):
        rows = []
        for source in self.rows.values():
            row = dict(source)
            row["mean_self_ms"] = row["self_seconds"] * 1000 / row["calls"]
            row["percent_of_pipeline"] = (
                row["self_seconds"] / pipeline_seconds * 100 if pipeline_seconds else 0.0
            )
            rows.append(row)
        rows.sort(key=lambda row: row["self_seconds"], reverse=True)
        exclusive = sum(row["self_seconds"] for row in rows)
        outside = pipeline_seconds - self.root_seconds
        # These container self-times still need explanation if unusually large.
        containers = {"candidate_prepare", "anchor_pass", "second_pass", "stream_consume"}
        unattributed = max(0.0, outside) + sum(
            max(0.0, row["self_seconds"]) for row in rows if row["stage"] in containers
        )
        tolerance = max(1e-6, pipeline_seconds * 1e-8)
        return {
            "schema_version": "1.0", "enabled": True,
            "pipeline_seconds": pipeline_seconds, "scoped_seconds": self.root_seconds,
            "exclusive_seconds": exclusive, "outside_scopes_seconds": outside,
            "reconciliation_error_seconds": self.root_seconds - exclusive,
            "unattributed_seconds": unattributed,
            "unattributed_percent": unattributed / pipeline_seconds * 100 if pipeline_seconds else 0.0,
            "negative_self_count": self.negative_self_count,
            "status": "VALID" if (
                not self.stack and not self.negative_self_count
                and outside >= -tolerance and abs(self.root_seconds - exclusive) <= tolerance
            ) else "INVALID",
            "rows": rows, "counters": dict(self.counters),
            "measurement_note": (
                "Sum self_seconds only; inclusive times contain children. "
                "calls counts timed regions, not candidate count. "
                "Whole evaluation including warmup and settlement; instrumentation overhead is included."
            ),
        }


class _StageScope:
    __slots__ = ("owner", "stage", "phase", "path", "row", "started", "children")

    def __init__(self, owner, stage, phase):
        self.owner = owner
        self.stage = stage
        parent = owner.stack[-1] if owner.stack else None
        self.phase = phase or (parent.phase if parent else "prepare")
        self.path = (parent.path + "/" if parent else "") + stage
        self.row = owner._row(stage, self.phase, self.path)
        self.children = 0.0

    def __enter__(self):
        self.owner.stack.append(self)
        self.started = self.owner.clock()
        return self

    def __exit__(self, *exc):
        elapsed = self.owner.clock() - self.started
        owner = self.owner
        owner.stack.pop()
        own = elapsed - self.children
        if own < -max(1e-6, elapsed * 1e-8):
            owner.negative_self_count += 1
        self.row["calls"] += 1
        self.row["inclusive_seconds"] += elapsed
        self.row["self_seconds"] += own
        if owner.stack:
            owner.stack[-1].children += elapsed
        else:
            owner.root_seconds += elapsed
        return False


def candidate_scope(profiler, stage, phase=None):
    detail = getattr(profiler, "candidate_detail", None)
    return detail.scope(stage, phase) if detail is not None else _NO_SCOPE


def candidate_count(profiler, name, amount=1):
    detail = getattr(profiler, "candidate_detail", None)
    if detail is not None:
        detail.count(name, amount)


def candidate_stage(stage, *, phase=None, feasibility=False):
    """Instrument synchronous methods; never apply to generator functions."""
    def decorate(method):
        @wraps(method)
        def measured(self, *args, **kwargs):
            profiler = getattr(self, "profiler", None)
            detail = getattr(profiler, "candidate_detail", None)
            if detail is None and not feasibility:
                return method(self, *args, **kwargs)
            # Only layered_pool lacked this aggregate timer; complete mode
            # already times feasibility elsewhere, so must not be double counted.
            context = args[0] if args else kwargs.get("context")
            aggregate = feasibility and profiler is not None and (
                getattr(getattr(context, "candidate_mode", None), "value", None) == "layered_pool"
            )
            if detail is None and not aggregate:
                return method(self, *args, **kwargs)
            started = time.perf_counter() if aggregate else None
            try:
                with candidate_scope(profiler, stage, phase):
                    result = method(self, *args, **kwargs)
                    if feasibility and detail is not None:
                        detail.count("feasibility_passed" if result[0] is not None else "feasibility_rejected")
                    return result
            finally:
                if aggregate:
                    profiler.add("candidate_feasibility_prepare_seconds", time.perf_counter() - started)
                    profiler.increment("prepare_feasibility_check_count")
        return measured
    return decorate


def write_candidate_detail_csv(detail, path):
    """UTF-8 BOM permits direct opening in Excel; rows are exclusive-time sorted."""
    import csv

    columns = (
        "phase", "stage", "label", "path", "calls", "inclusive_seconds",
        "self_seconds", "mean_self_ms", "percent_of_pipeline",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        rows = list(detail["rows"])
        outside = detail.get("outside_scopes_seconds", 0.0)
        if outside:
            pipeline = detail["pipeline_seconds"]
            rows.append({
                "phase": "outside", "stage": "outside_scopes",
                "label": "未覆盖的外层计时与统计开销", "path": "outside_scopes",
                "calls": 0, "inclusive_seconds": outside, "self_seconds": outside,
                "mean_self_ms": None,
                "percent_of_pipeline": outside / pipeline * 100 if pipeline else 0.0,
            })
        rows.sort(key=lambda row: row["self_seconds"], reverse=True)
        writer.writerows(rows)
