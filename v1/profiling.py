# 文件功能：累计训练阶段耗时与计数，提供可序列化的性能摘要。
# 主要内容：TrainingPerformanceProfiler。
# 模块上游（导入关系）：v1.evaluate_v1、v1.train_v1、v5.profiling、tests.v1.test_training_profile_v1、v5.tests.test_candidate_generator。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Stable, pickle-safe wall-clock profiling data for v1 training."""

import math

from v1.candidate_profiling import CandidateDetailProfiler


PROFILE_DETAIL_KEYS = (
    "scheduler_decision_seconds",
    "policy_selection_seconds",
    "reservation_commit_seconds",
    "candidate_prepare_seconds",
    "candidate_stream_seconds",
    "replay_candidate_generation_seconds",
    "transition_feature_cache_seconds",
    "candidate_feature_encoding_seconds",
    "selection_inference_seconds",
    "bootstrap_inference_seconds",
    "training_forward_seconds",
    "backpropagation_seconds",
    "environment_update_seconds",
    "logging_seconds",
    "checkpoint_seconds",
    "artifact_save_seconds",
    "candidate_feasibility_prepare_seconds",
    "candidate_feasibility_stream_seconds",
    "candidate_metric_evaluation_seconds",
    "candidate_id_hash_seconds",
    "candidate_object_construction_seconds",
    "candidate_transmission_schedule_seconds",
    "candidate_cpu_feasibility_seconds",
    "candidate_path_feasibility_seconds",
    "candidate_sla_and_item_seconds",
)

PROFILE_NESTED_STAGE_KEYS = (
    "candidate_prepare_seconds",
    "candidate_stream_seconds",
    "replay_candidate_generation_seconds",
    "transition_feature_cache_seconds",
    "candidate_feature_encoding_seconds",
    "selection_inference_seconds",
    "bootstrap_inference_seconds",
    "training_forward_seconds",
    "backpropagation_seconds",
)


class TrainingPerformanceProfiler:
    """Accumulates exclusive wall-clock stages for a profiling training run."""

    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.evaluate_v1.run_evaluation、v1.train_v1.run_training、v5.profiling.V5TrainingPerformanceProfiler.__init__。
    # 调用约定：构造实例时由 Python 调用。
    def __init__(self):
        self.candidate_detail = None
        self.timings = {key: 0.0 for key in PROFILE_DETAIL_KEYS}
        self.counters = {
            "scheduler_decision_count": 0,
            "policy_selection_count": 0,
            "reservation_commit_attempt_count": 0,
            "prepare_theoretical_slot_count": 0,
            "selection_candidate_count": 0,
            "replay_candidate_feature_count": 0,
            "scheduler_cycle_count": 0,
            "virtual_cycle_count": 0,
            "idle_cycle_skip_count": 0,
            "training_process_cycle_count": 0,
            "prepare_feasibility_check_count": 0,
            "stream_feasibility_check_count": 0,
            "candidate_metric_evaluation_count": 0,
            "candidate_object_count": 0,
        }

    # 作用：恢复序列化状态，补齐必要索引或默认字段。
    # 上游（直接调用）：v5.profiling.V5TrainingPerformanceProfiler.__setstate__。
    # 调用约定：由 pickle/检查点反序列化流程调用。
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.setdefault("candidate_detail", None)
        for key in PROFILE_DETAIL_KEYS:
            self.timings.setdefault(key, 0.0)
        self.counters.setdefault("virtual_cycle_count", 0)
        self.counters.setdefault("idle_cycle_skip_count", 0)
        self.counters.setdefault("scheduler_decision_count", 0)
        self.counters.setdefault("policy_selection_count", 0)
        self.counters.setdefault("reservation_commit_attempt_count", 0)

    # 作用：向缓冲区或统计容器追加记录。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def add(self, key, seconds):
        if key not in self.timings:
            raise KeyError(f"unknown profiling timing: {key}")
        value = float(seconds)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError("profiling seconds must be finite and non-negative")
        self.timings[key] += value
        if self.candidate_detail is not None:
            self.candidate_detail.record_leaf(key, value)

    # 作用：增加指定性能计数器。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def increment(self, key, amount=1):
        if key not in self.counters:
            raise KeyError(f"unknown profiling counter: {key}")
        value = int(amount)
        if value < 0:
            raise ValueError("profiling counter increment cannot be negative")
        self.counters[key] += value
        if self.candidate_detail is not None:
            self.candidate_detail.count(key, value)

    def enable_candidate_detail(self):
        self.candidate_detail = CandidateDetailProfiler()

    # 作用：返回可嵌套阶段的累计耗时。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def nested_stage_seconds(self):
        return sum(self.timings[key] for key in PROFILE_NESTED_STAGE_KEYS)

    # 作用：生成当前统计对象的汇总。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def summary(self, total_wall_seconds):
        total = max(0.0, float(total_wall_seconds))
        sections = {
            "candidate_slot_generation": (
                self.timings["candidate_prepare_seconds"]
                + self.timings["candidate_stream_seconds"]
                + self.timings["replay_candidate_generation_seconds"]
                + self.timings["transition_feature_cache_seconds"]
            ),
            "candidate_feature_encoding": self.timings[
                "candidate_feature_encoding_seconds"
            ],
            "neural_network_inference": (
                self.timings["selection_inference_seconds"]
                + self.timings["bootstrap_inference_seconds"]
                + self.timings["training_forward_seconds"]
            ),
            "backpropagation_and_optimizer": self.timings[
                "backpropagation_seconds"
            ],
            "environment_and_accounting": self.timings[
                "environment_update_seconds"
            ],
            "logging_and_saving": (
                self.timings["logging_seconds"]
                + self.timings["checkpoint_seconds"]
                + self.timings["artifact_save_seconds"]
            ),
        }
        accounted = sum(sections.values())
        sections["other_or_profiler_overhead"] = max(0.0, total - accounted)
        percentages = {
            key: (100.0 * seconds / total if total > 0.0 else 0.0)
            for key, seconds in sections.items()
        }
        selected = self.counters["selection_candidate_count"]
        selection_scan_seconds = (
            self.timings["candidate_prepare_seconds"]
            + self.timings["candidate_stream_seconds"]
            + self.timings["candidate_feature_encoding_seconds"]
            + self.timings["selection_inference_seconds"]
        )
        candidate_internal = {
            "prepare_feasibility": self.timings[
                "candidate_feasibility_prepare_seconds"
            ],
            "stream_and_replay_feasibility": self.timings[
                "candidate_feasibility_stream_seconds"
            ],
            "physical_metric_evaluation": self.timings[
                "candidate_metric_evaluation_seconds"
            ],
            "candidate_id_hash": self.timings["candidate_id_hash_seconds"],
            "candidate_object_construction": self.timings[
                "candidate_object_construction_seconds"
            ],
        }
        candidate_internal["candidate_pipeline_other"] = max(
            0.0,
            sections["candidate_slot_generation"]
            - sum(candidate_internal.values()),
        )
        feasibility_internal = {
            "transmission_schedule": self.timings[
                "candidate_transmission_schedule_seconds"
            ],
            "cpu_interval_feasibility": self.timings[
                "candidate_cpu_feasibility_seconds"
            ],
            "path_interval_feasibility": self.timings[
                "candidate_path_feasibility_seconds"
            ],
            "sla_and_item_assembly": self.timings[
                "candidate_sla_and_item_seconds"
            ],
        }
        feasibility_total = (
            self.timings["candidate_feasibility_prepare_seconds"]
            + self.timings["candidate_feasibility_stream_seconds"]
        )
        feasibility_internal["feasibility_other"] = max(
            0.0, feasibility_total - sum(feasibility_internal.values())
        )
        result = {
            "total_wall_seconds": total,
            "sections_seconds": sections,
            "sections_percent": percentages,
            "detail_seconds": dict(self.timings),
            "candidate_internal_seconds": candidate_internal,
            "candidate_internal_percent_of_candidate_pipeline": {
                key: (
                    100.0 * seconds / sections["candidate_slot_generation"]
                    if sections["candidate_slot_generation"] > 0.0 else 0.0
                )
                for key, seconds in candidate_internal.items()
            },
            "candidate_feasibility_internal_seconds": feasibility_internal,
            "candidate_feasibility_internal_percent": {
                key: (
                    100.0 * seconds / feasibility_total
                    if feasibility_total > 0.0 else 0.0
                )
                for key, seconds in feasibility_internal.items()
            },
            "counters": dict(self.counters),
            "decision_stage_seconds": {
                "scheduler_decision_total": self.timings[
                    "scheduler_decision_seconds"
                ],
                "policy_selection_total": self.timings[
                    "policy_selection_seconds"
                ],
                "reservation_commit_total": self.timings[
                    "reservation_commit_seconds"
                ],
            },
            "selection_candidates_per_second": (
                selected / selection_scan_seconds
                if selection_scan_seconds > 0.0 else None
            ),
            "measurement_note": (
                "CUDA is synchronized at timed neural-network boundaries when "
                "profiling is enabled; percentages are exclusive wall-clock stages."
            ),
        }
        if self.candidate_detail is not None:
            result["candidate_detail"] = self.candidate_detail.summary(
                sections["candidate_slot_generation"]
            )
        return result
