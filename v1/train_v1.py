"""Production-oriented v1.0 candidate-DQN training entry point."""

import argparse
from collections import deque
from contextlib import contextmanager
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from shared import config
from v1.ablation_settings import apply_ablation_variant, variant_names
from v1.audit_v1 import scan_scheduler_invariants
from v1.domain.models import SlaType, TaskSpec, TaskState
from v1.learning import (
    CandidateDQNTrainer,
    CandidateDqnMetadata,
    CandidateReplayBuffer,
    DecisionRecord,
    GammaClock,
    RewardAssembler,
    TimestampedReward,
    validate_checkpoint_metadata,
)
from v1.profiling import TrainingPerformanceProfiler
from v1.scheduler import ObjectiveConfig, ObjectiveScorer
from v1.v1_runtime import (
    create_v1_runtime,
    ensure_v1_runtime_forecasts_for_tasks,
    extend_v1_runtime_forecasts,
    v1_runtime_forecast_end,
)


DEFAULT_REWARD_ROLLING_WINDOW = 1000
DEFAULT_VALIDATION_CUTOFF_SIM = 12.0
DEFAULT_VALIDATION_SAFETY_CAP = 1000000
REWARD_COMPONENT_FIELDS = (
    "immediate_reward",
    "realized_reward",
    "failure_penalty",
    "expiration_penalty",
    "completion_reward",
    "realization_correction",
)


def _empty_reward_components():
    return {field: 0.0 for field in REWARD_COMPONENT_FIELDS}


@contextmanager
def _preserve_rng_state():
    """Keep validation and trace generation invisible to training RNG streams."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_positive_int(name, value):
    if value is None:
        return None
    return _positive_int(name, value)


def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _metadata(runtime):
    architecture = "shared_candidate_q_v1"
    if not config.V1_DQN_USE_GLOBAL_STATE or not config.V1_DQN_DOUBLE_DQN:
        architecture += (
            f":global={int(config.V1_DQN_USE_GLOBAL_STATE)}"
            f":double={int(config.V1_DQN_DOUBLE_DQN)}"
        )
    return CandidateDqnMetadata(
        "1.0",
        "1.0",
        runtime.candidate_feature_encoder.feature_schema_hash,
        runtime.candidate_q_network.global_state_dim,
        runtime.candidate_q_network.candidate_feature_dim,
        config.V1_GAMMA_PER_SECOND,
        architecture=architecture,
    )


def _training_config_view(**overrides):
    names = (
        "SYSTEM_VERSION", "REQUIREMENTS_VERSION", "ALGORITHM_VERSION", "CANDIDATE_MODE",
        "V1_CANDIDATE_MODE", "SCHEDULING_CYCLE", "V1_CANDIDATE_PATH_K",
        "V1_CANDIDATE_POOL_MAX_BY_SLA",
        "V1_CANDIDATE_POOL_NODE_LIMIT_BY_SLA",
        "V1_CANDIDATE_POOL_TIME_SAMPLES_BY_SLA", "V1_GAMMA_PER_SECOND",
        "V1_CANDIDATE_DQN_HIDDEN_DIM", "LEARNING_RATE", "MEMORY_CAPACITY",
        "EPSILON_START", "EPSILON_MIN", "EPSILON_DECAY",
        "V1_TARGET_UPDATE_INTERVAL", "V1_COST_REFERENCE_YUAN",
        "V1_COST_SCALE_YUAN", "V1_GREEN_ABSORPTION_DELTA_SCALE",
        "V1_OBJECTIVE_COST_WEIGHT", "V1_OBJECTIVE_GREEN_WEIGHT",
        "V1_OBJECTIVE_BALANCE_WEIGHT", "V1_SOFT_TARDINESS_WEIGHT",
        "V1_FLEXIBLE_TARDINESS_WEIGHT",
        "V1_ABLATION_VARIANT", "V1_ACTIVE_WAIT_ENABLED",
        "V1_DISCOUNT_MODE", "V1_DECISION_GAMMA",
        "V1_REWARD_ESTIMATE_ENABLED",
        "V1_REWARD_REALIZATION_CORRECTION_ENABLED",
        "V1_REWARD_TERMINAL_PENALTIES_ENABLED",
        "V1_DISABLED_CANDIDATE_FEATURE_GROUPS",
        "V1_DQN_USE_GLOBAL_STATE", "V1_DQN_DOUBLE_DQN",
        "V1_TARIFF_MODE",
    )
    values = {name: getattr(config, name) for name in names}
    values.update(overrides)
    return values


def _canonical_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resume_config_compatible(saved, requested):
    if not isinstance(saved, dict) or not isinstance(requested, dict):
        return False
    legacy_defaults = {
        "V1_ABLATION_VARIANT": "reference",
        "V1_ACTIVE_WAIT_ENABLED": True,
        "V1_DISCOUNT_MODE": "physical_time",
        "V1_DECISION_GAMMA": 0.95,
        "V1_REWARD_ESTIMATE_ENABLED": True,
        "V1_REWARD_REALIZATION_CORRECTION_ENABLED": True,
        "V1_REWARD_TERMINAL_PENALTIES_ENABLED": True,
        "V1_DISABLED_CANDIDATE_FEATURE_GROUPS": (),
        "V1_DQN_USE_GLOBAL_STATE": True,
        "V1_DQN_DOUBLE_DQN": True,
        "V1_TARIFF_MODE": "tou_uniform",
    }
    if any(
        key not in saved and key in requested and requested.get(key) != value
        for key, value in legacy_defaults.items()
    ):
        return False
    performance_only = {
        "candidate_chunk_size",
        "invariant_check_every",
        "skip_idle_cycles",
        "SYSTEM_VERSION",
    }
    # Older checkpoints do not contain semantic keys added in later versions.
    # Validate every setting they did record without rejecting absent metadata.
    semantic_keys = set(saved) | {"bootstrap_candidate_limit"}
    saved_semantics = {
        key: saved.get(key)
        for key in semantic_keys
        if key not in performance_only
    }
    requested_semantics = {
        key: requested.get(key)
        for key in semantic_keys
        if key not in performance_only
    }
    return _canonical_hash(saved_semantics) == _canonical_hash(
        requested_semantics
    )


def _run_profiled_operation(profiler, counter_key, operation):
    if profiler is None:
        return operation()
    nested_before = profiler.nested_stage_seconds()
    started = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - started
    nested_delta = profiler.nested_stage_seconds() - nested_before
    profiler.add("environment_update_seconds", max(0.0, elapsed - nested_delta))
    profiler.increment(counter_key)
    return result


def _assert_scheduler_invariants(scheduler, cycle_result=None):
    violations = scan_scheduler_invariants(scheduler, cycle_result)
    if violations:
        detail = "; ".join(
            f"{item.invariant_id}:{item.detail}" for item in violations
        )
        raise RuntimeError(f"v1.0 invariant gate failed: {detail}")


def _should_run_invariant_check(
    completed_cycle,
    *,
    invariant_check_every,
    checkpoint_every,
    final_cycle,
):
    return (
        completed_cycle % invariant_check_every == 0
        or completed_cycle % checkpoint_every == 0
        or completed_cycle == final_cycle
    )


def _write_profile_outputs(profiler, total_wall_seconds, json_path):
    summary = profiler.summary(total_wall_seconds)
    target = Path(json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    csv_path = target.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("section", "seconds", "percent")
        )
        writer.writeheader()
        for section, seconds in summary["sections_seconds"].items():
            writer.writerow({
                "section": section,
                "seconds": seconds,
                "percent": summary["sections_percent"][section],
            })
    return target, csv_path, summary


class V1TrainingLoop:
    def __init__(
        self,
        runtime,
        *,
        device="cpu",
        candidate_chunk_size=None,
        batch_size=None,
        min_replay_size=None,
        updates_per_transition=None,
        bootstrap_candidate_limit=None,
        random_seed=0,
        profiler=None,
    ):
        self.runtime = runtime
        self.policy = runtime.scheduler.policy
        # Formal evaluation retains the exact complete-set digest. Training
        # only consumes the selected candidate, so hashing every candidate ID
        # is pure audit overhead here.
        self.policy.audit_candidate_set_hash = False
        if runtime.candidate_q_network is None:
            raise ValueError("v1 training requires candidate_dqn policy")
        self.device = _resolve_device(device)
        self.candidate_chunk_size = _positive_int(
            "candidate_chunk_size",
            config.V1_CANDIDATE_CHUNK_SIZE
            if candidate_chunk_size is None else candidate_chunk_size,
        )
        self.batch_size = _positive_int(
            "batch_size", config.BATCH_SIZE if batch_size is None else batch_size
        )
        self.min_replay_size = _positive_int(
            "min_replay_size",
            config.V1_REPLAY_MIN_SIZE
            if min_replay_size is None else min_replay_size,
        )
        self.min_replay_size = max(self.batch_size, self.min_replay_size)
        self.updates_per_transition = _positive_int(
            "updates_per_transition",
            config.V1_TRAIN_UPDATES_PER_TRANSITION
            if updates_per_transition is None else updates_per_transition,
        )
        self.bootstrap_candidate_limit = _optional_positive_int(
            "bootstrap_candidate_limit", bootstrap_candidate_limit
        )
        self.replay_random = random.Random(random_seed + 7919)
        self.target = type(runtime.candidate_q_network)(
            runtime.candidate_q_network.global_state_dim,
            runtime.candidate_q_network.candidate_feature_dim,
            runtime.candidate_q_network.hidden_dim,
        )
        self.trainer = CandidateDQNTrainer(
            runtime.candidate_q_network,
            self.target,
            config.LEARNING_RATE,
            device=self.device,
            candidate_chunk_size=self.candidate_chunk_size,
            bootstrap_candidate_limit=self.bootstrap_candidate_limit,
            next_candidate_provider=self._next_candidate_feature_chunks,
            double_dqn=config.V1_DQN_DOUBLE_DQN,
        )
        self.trainer.update_target()
        self.replay = CandidateReplayBuffer(config.MEMORY_CAPACITY)
        self.reward_assembler = RewardAssembler(
            GammaClock(
                config.V1_GAMMA_PER_SECOND,
                runtime.time_converter,
                mode=config.V1_DISCOUNT_MODE,
                decision_gamma=config.V1_DECISION_GAMMA,
            )
        )
        self.objective = ObjectiveScorer(ObjectiveConfig(
            config.V1_COST_REFERENCE_YUAN,
            config.V1_COST_SCALE_YUAN,
            config.V1_GREEN_ABSORPTION_DELTA_SCALE,
            config.V1_OBJECTIVE_COST_WEIGHT,
            config.V1_OBJECTIVE_GREEN_WEIGHT,
            config.V1_OBJECTIVE_BALANCE_WEIGHT,
            config.V1_SOFT_TARDINESS_WEIGHT,
            config.V1_FLEXIBLE_TARDINESS_WEIGHT,
        ))
        self.pending = None
        self.decision_by_task = {}
        self.transition_count = 0
        self.update_count = 0
        self.losses = []
        self.q_summaries = []
        self.reward_rolling_window = DEFAULT_REWARD_ROLLING_WINDOW
        self.rolling_rewards = deque(maxlen=self.reward_rolling_window)
        self.rewards_since_log = []
        self.reward_components_since_log = _empty_reward_components()
        self.td_errors_since_log = []
        self.completion_reward_details = {}
        self.candidate_count = 0
        self.profiler = profiler
        self.runtime.scheduler.profiler = profiler
        self.runtime.scheduler.candidate_generator.profiler = profiler
        self.policy.profiler = profiler
        self.trainer.profiler = profiler

    def ensure_monitoring_state(self, reward_rolling_window=None):
        """Add monitoring state when resuming checkpoints made by older code."""

        window = (
            getattr(self, "reward_rolling_window", DEFAULT_REWARD_ROLLING_WINDOW)
            if reward_rolling_window is None
            else _positive_int("reward_rolling_window", reward_rolling_window)
        )
        old_rolling = tuple(getattr(self, "rolling_rewards", ()))
        self.reward_rolling_window = window
        self.rolling_rewards = deque(old_rolling[-window:], maxlen=window)
        if not hasattr(self, "rewards_since_log"):
            self.rewards_since_log = []
        if not hasattr(self, "reward_components_since_log"):
            self.reward_components_since_log = _empty_reward_components()
        else:
            for field in REWARD_COMPONENT_FIELDS:
                self.reward_components_since_log.setdefault(field, 0.0)
        if not hasattr(self, "td_errors_since_log"):
            self.td_errors_since_log = []
        if not hasattr(self, "completion_reward_details"):
            self.completion_reward_details = {}
        if not hasattr(self.trainer, "last_td_errors"):
            self.trainer.last_td_errors = ()

    def consume_monitoring_interval(self):
        rewards = tuple(self.rewards_since_log)
        q_summaries = tuple(self.q_summaries)
        td_errors = (
            np.concatenate(self.td_errors_since_log)
            if self.td_errors_since_log
            else np.asarray((), dtype=np.float32)
        )
        components = dict(self.reward_components_since_log)
        self.rewards_since_log.clear()
        self.q_summaries.clear()
        self.td_errors_since_log.clear()
        self.reward_components_since_log = _empty_reward_components()
        return {
            "rewards": rewards,
            "rolling_rewards": tuple(self.rolling_rewards),
            "reward_components": components,
            "q_summaries": q_summaries,
            "td_errors": td_errors,
        }

    def _next_candidate_feature_chunks(self, context):
        evaluator = self.runtime.accounting.candidate_metric_evaluator(
            context.reservation_snapshot
        )
        chunk_size = self.candidate_chunk_size
        if self.bootstrap_candidate_limit is not None:
            chunk_size = min(chunk_size, self.bootstrap_candidate_limit)
        return self.runtime.scheduler.candidate_generator.feature_chunks_from_context(
            context,
            self.runtime.candidate_feature_encoder,
            metric_evaluator=evaluator,
            chunk_size=chunk_size,
        )

    def to_device(self, device):
        self.device = _resolve_device(device)
        self.trainer.device = torch.device(self.device)
        self.trainer.online.to(self.trainer.device)
        self.trainer.target.to(self.trainer.device)
        self.policy.device = torch.device(self.device)
        self.policy.network.to(self.policy.device)
        for state in self.trainer.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.trainer.device)

    def set_candidate_chunk_size(self, candidate_chunk_size):
        chunk_size = _positive_int(
            "candidate_chunk_size", candidate_chunk_size
        )
        self.candidate_chunk_size = chunk_size
        self.policy.candidate_chunk_size = chunk_size
        self.trainer.candidate_chunk_size = chunk_size

    def set_bootstrap_candidate_limit(self, bootstrap_candidate_limit):
        limit = _optional_positive_int(
            "bootstrap_candidate_limit", bootstrap_candidate_limit
        )
        self.bootstrap_candidate_limit = limit
        self.trainer.bootstrap_candidate_limit = limit
        self.trainer.clear_next_feature_cache()

    def process_cycle(self, result, *, check_invariants=True):
        if check_invariants:
            _assert_scheduler_invariants(self.runtime.scheduler, result)
        for event in result.domain_events:
            if event.event_type == "TASK_COMPLETED":
                self._buffer_completion(event.task_id, event.event_time_sim)
            elif (
                event.event_type == "TASK_FAILED"
                and config.V1_REWARD_TERMINAL_PENALTIES_ENABLED
                and event.task_id in self.decision_by_task
            ):
                self.reward_assembler.buffer_event(TimestampedReward(
                    event.event_time_sim,
                    config.V1_FAILURE_PENALTY,
                    self._credit_decision_id(event.task_id),
                    event.event_type,
                ))
        for transition in result.state_transitions:
            if (
                transition.new_state is TaskState.EXPIRED
                and config.V1_REWARD_TERMINAL_PENALTIES_ENABLED
                and transition.task_id in self.decision_by_task
            ):
                self.reward_assembler.buffer_event(TimestampedReward(
                    transition.event_time_sim,
                    config.V1_EXPIRATION_PENALTY,
                    self._credit_decision_id(transition.task_id),
                    "TASK_EXPIRED",
                ))

        for trace in self.policy.pop_selection_traces():
            if self.pending is not None:
                self._close_pending(
                    next_state=trace.global_state,
                    next_context=trace.candidate_context,
                    next_time=trace.decision_time_sim,
                    terminal=False,
                )
            candidate = trace.selected_candidate
            task = self.runtime.scheduler.state_machine.task_spec(trace.task_id)
            estimated_utility = self.objective.score(
                candidate, task.sla_type
            ).total_score
            record = DecisionRecord(
                "decision-" + candidate.candidate_id,
                task.task_id,
                candidate.candidate_id,
                trace.decision_time_sim,
                estimated_utility,
            )
            self.decision_by_task[task.task_id] = (record, candidate)
            self.pending = {
                "record": record,
                "state": trace.global_state,
                "candidate_id": candidate.candidate_id,
                "features": trace.selected_candidate_features,
                "immediate_reward": (
                    estimated_utility
                    if config.V1_REWARD_ESTIMATE_ENABLED else 0.0
                ),
            }
            self.candidate_count += trace.candidate_count
            if trace.q_min is not None:
                self.q_summaries.append(
                    (trace.q_min, trace.q_max, trace.q_mean)
                )

    def _credit_decision_id(self, task_id):
        linked = self.decision_by_task.get(task_id)
        if linked is not None:
            return linked[0].decision_id
        if self.pending is not None:
            return self.pending["record"].decision_id
        return "system-event"

    def _buffer_completion(self, task_id, event_time_sim):
        linked = self.decision_by_task.get(task_id)
        if linked is None:
            return
        record, candidate = linked
        # Only reservations that overlap this task on the same node can affect
        # its physical counterfactual. This avoids repeatedly realizing all
        # historical reservations as the run grows.
        completed_reservation = self.runtime.calendar.get_reservation(
            self.runtime.scheduler.state_machine.runtime(
                task_id
            ).reservation_id
        )
        relevant = self.runtime.calendar.reservations_overlapping_node(
            candidate.target_node,
            completed_reservation.compute_interval_sim,
        )
        report = self.runtime.accounting.realize(relevant)
        realized = next(item for item in report.task_records if item.task_id == task_id)
        task = self.runtime.scheduler.state_machine.task_spec(task_id)
        realized_candidate = replace(
            candidate,
            estimated_candidate_marginal_system_cost_yuan=(
                realized.task_attributed_cost_yuan
            ),
            estimated_green_coverage=realized.green_coverage,
        )
        realized_utility = self.objective.score(
            realized_candidate, task.sla_type
        ).total_score
        if config.V1_REWARD_REALIZATION_CORRECTION_ENABLED:
            realization_correction = (
                realized_utility
                - (
                    record.estimated_local_utility
                    if config.V1_REWARD_ESTIMATE_ENABLED else 0.0
                )
            )
            correction = (
                realization_correction
                + config.V1_COMPLETION_OUTCOME_REWARD
            )
        else:
            realization_correction = 0.0
            correction = config.V1_COMPLETION_OUTCOME_REWARD
        self.completion_reward_details[record.decision_id] = {
            # Realized utility is diagnostic context.  The additive reward
            # path is completion_reward + realization_correction.
            "realized_reward": realized_utility,
            "completion_reward": config.V1_COMPLETION_OUTCOME_REWARD,
            "realization_correction": realization_correction,
        }
        self.reward_assembler.buffer_event(TimestampedReward(
            event_time_sim, correction, record.decision_id, "TASK_COMPLETED"
        ))

    def _record_transition_reward(self, data, transition):
        components = _empty_reward_components()
        components["immediate_reward"] = float(data["immediate_reward"])
        for event in transition.timestamped_event_rewards:
            elapsed = self.reward_assembler.clock.elapsed_seconds(
                data["record"].decision_time_sim,
                event.event_time_sim,
            )
            discount = self.reward_assembler.clock.discount(elapsed)
            if event.event_type == "TASK_FAILED":
                components["failure_penalty"] += discount * event.reward
            elif event.event_type == "TASK_EXPIRED":
                components["expiration_penalty"] += discount * event.reward
            elif event.event_type == "TASK_COMPLETED":
                detail = self.completion_reward_details.pop(
                    event.decision_id,
                    {
                        "realized_reward": event.reward,
                        "completion_reward": 0.0,
                        "realization_correction": event.reward,
                    },
                )
                components["realized_reward"] += float(
                    detail["realized_reward"]
                )
                for field in (
                    "completion_reward",
                    "realization_correction",
                ):
                    components[field] += discount * float(detail[field])
        self.rewards_since_log.append(float(transition.reward))
        self.rolling_rewards.append(float(transition.reward))
        for field, value in components.items():
            self.reward_components_since_log[field] += value

    def _close_pending(self, *, next_state, next_context, next_time, terminal):
        data = self.pending
        cache_started = time.perf_counter()
        if terminal:
            cached_features = np.empty(
                (0, self.runtime.candidate_feature_encoder.feature_dim),
                dtype=np.float32,
            )
        else:
            chunks = tuple(
                np.asarray(chunk, dtype=np.float32)
                for chunk in self._next_candidate_feature_chunks(next_context)
                if len(chunk)
            )
            cached_features = (
                np.concatenate(chunks, axis=0)
                if chunks
                else np.empty(
                    (0, self.runtime.candidate_feature_encoder.feature_dim),
                    dtype=np.float32,
                )
            )
        if self.profiler is not None:
            self.profiler.add(
                "transition_feature_cache_seconds",
                time.perf_counter() - cache_started,
            )
        transition = self.reward_assembler.build_transition(
            global_state_before=data["state"],
            selected_candidate_id=data["candidate_id"],
            selected_candidate_features=data["features"],
            immediate_reward=data["immediate_reward"],
            global_state_after=next_state,
            next_candidate_features=(),
            next_candidate_context=None,
            decision_time_sim=data["record"].decision_time_sim,
            next_transition_time_sim=next_time,
            terminal=terminal,
        )
        transition = replace(
            transition,
            next_candidate_features=cached_features,
        )
        self._record_transition_reward(data, transition)
        self.replay.add(transition)
        self.transition_count += 1
        if len(self.replay) >= self.min_replay_size:
            for _ in range(self.updates_per_transition):
                batch = self.replay.sample(self.batch_size, self.replay_random)
                loss = self.trainer.train_batch(batch)
                self.losses.append(loss)
                self.td_errors_since_log.append(
                    self.trainer.last_td_errors
                )
                self.update_count += 1
                self.policy.epsilon = max(
                    config.EPSILON_MIN,
                    self.policy.epsilon * config.EPSILON_DECAY,
                )
                if (
                    self.update_count
                    % max(1, config.V1_TARGET_UPDATE_INTERVAL)
                    == 0
                ):
                    self.trainer.update_target()
        self.pending = None

    def finalize(self, final_time):
        if self.pending is not None:
            self._close_pending(
                next_state=self.runtime.global_state(None),
                next_context=None,
                next_time=final_time,
                terminal=True,
            )


def _settle(runtime, loop, current_time, safety_cap=1000000, profiler=None):
    iterations = 0
    unresolved = {
        TaskState.QUEUED,
        TaskState.PENDING_UNCOMMITTED,
        TaskState.RESERVED,
        TaskState.TRANSMITTING,
        TaskState.RUNNING,
    }
    unsettled_task_ids = runtime.scheduler.state_machine.task_ids_in_states(
        unresolved
    )
    unsettled_tasks = (
        runtime.scheduler.state_machine.task_spec(task_id)
        for task_id in unsettled_task_ids
    )
    forecast_end = ensure_v1_runtime_forecasts_for_tasks(runtime, unsettled_tasks)
    while runtime.scheduler.state_machine.task_ids_in_states(unresolved):
        iterations += 1
        if iterations > safety_cap:
            raise RuntimeError("v1 training settlement safety cap exceeded")
        next_deadline = (
            runtime.scheduler.queue_manager.next_uncommitted_deadline_sim()
        )
        next_event = runtime.scheduler.event_engine.next_event_time_sim
        choices = []
        if (
            next_deadline is not None
            and next_deadline >= current_time - 1e-12
        ):
            choices.append(next_deadline)
        if next_event is not None and next_event >= current_time - 1e-12:
            choices.append(next_event)
        if not choices:
            raise RuntimeError("unsettled training state has no future event/deadline")
        current_time = min(choices)
        result = _run_profiled_operation(
            profiler,
            "scheduler_cycle_count",
            lambda: runtime.scheduler.run_cycle(
                current_time,
                forecast_covered_until_sim=forecast_end,
            ),
        )
        _run_profiled_operation(
            profiler,
            "training_process_cycle_count",
            lambda: loop.process_cycle(result, check_invariants=False),
        )
    return current_time


def _generate_arrivals(runtime, current, cycle, total_capacity):
    lam, _ = runtime.task_manager.get_dynamic_task_rate(current)
    return runtime.task_manager.generate_task_specs(
        np.random.poisson(lam),
        current,
        cycle,
        cpu_budget=(
            total_capacity
            * config.SCHEDULING_CYCLE
            * config.TASK_PEAK_LOAD_MULTIPLIER
        ),
    )


def _run_training_warmup(
    runtime,
    loop,
    *,
    warmup_days,
    total_capacity,
    invariant_check_every,
    profiler=None,
):
    """Advance the live runtime to T0 without creating learning experience."""

    days = float(warmup_days)
    if not np.isfinite(days) or days < 0.0:
        raise ValueError("warmup_days must be finite and non-negative")
    warmup_end = days * config.TRAFFIC_DAY_DURATION_IN_SIM
    if warmup_end <= 0.0:
        return 0.0, 0

    initial_replay_size = len(loop.replay)
    initial_transition_count = loop.transition_count
    initial_update_count = loop.update_count
    initial_candidate_count = loop.candidate_count
    record_selection_traces = loop.policy.record_selection_traces
    loop.policy.record_selection_traces = False
    current = 0.0
    warmup_cycle = 0
    forecast_end = v1_runtime_forecast_end(runtime)
    try:
        while current < warmup_end - 1e-12:
            current = min(
                warmup_end,
                current + config.SCHEDULING_CYCLE,
            )
            arrivals = _generate_arrivals(
                runtime,
                current,
                -(warmup_cycle + 1),
                total_capacity,
            )
            if arrivals:
                forecast_end = ensure_v1_runtime_forecasts_for_tasks(
                    runtime, arrivals
                )
            result = _run_profiled_operation(
                profiler,
                "warmup_scheduler_cycle_count",
                lambda: runtime.scheduler.run_cycle(
                    current,
                    arrivals=arrivals,
                    forecast_covered_until_sim=forecast_end,
                ),
            )
            warmup_cycle += 1
            if warmup_cycle % invariant_check_every == 0:
                _assert_scheduler_invariants(runtime.scheduler, result)
    finally:
        loop.policy.pop_selection_traces()
        loop.policy.record_selection_traces = record_selection_traces

    _assert_scheduler_invariants(runtime.scheduler)
    if (
        len(loop.replay) != initial_replay_size
        or loop.transition_count != initial_transition_count
        or loop.update_count != initial_update_count
        or loop.candidate_count != initial_candidate_count
        or loop.pending is not None
    ):
        raise RuntimeError("training warm-up mutated learning state")
    return current, warmup_cycle


def _estimate_candidate_work(
    task_count,
    total_slots,
    max_slots,
    *,
    batch_size,
    min_replay_size,
    updates_per_transition,
    bootstrap_candidate_limit=None,
):
    batch_size = _positive_int("batch_size", batch_size)
    min_replay_size = max(
        batch_size,
        _positive_int("min_replay_size", min_replay_size),
    )
    updates_per_transition = _positive_int(
        "updates_per_transition", updates_per_transition
    )
    transition_count = int(task_count)
    update_count = (
        max(0, transition_count - min_replay_size + 1)
        * updates_per_transition
    )
    replay_samples = update_count * batch_size
    selection_visits = 2 * int(total_slots)
    if transition_count:
        if bootstrap_candidate_limit is None:
            # Each replay sample regenerates one complete next-candidate set.
            # The exact sampled contexts are random, so the trace mean is the
            # honest pre-run expectation and max_slots gives a conservative
            # upper bound.
            bootstrap_expected = (
                replay_samples * int(total_slots) + transition_count - 1
            ) // transition_count
            bootstrap_upper = replay_samples * int(max_slots)
        else:
            limit = _positive_int(
                "bootstrap_candidate_limit", bootstrap_candidate_limit
            )
            mean_slots = (int(total_slots) + transition_count - 1) // transition_count
            bootstrap_expected = replay_samples * min(limit, mean_slots)
            bootstrap_upper = replay_samples * min(limit, int(max_slots))
    else:
        bootstrap_expected = 0
        bootstrap_upper = 0
    return {
        "estimated_transition_count": transition_count,
        "estimated_update_count": update_count,
        "estimated_replay_context_samples": replay_samples,
        "selection_candidate_passes": 2,
        "estimated_selection_candidate_visits": selection_visits,
        "estimated_bootstrap_candidate_visits": bootstrap_expected,
        "estimated_bootstrap_candidate_visits_upper_bound": bootstrap_upper,
        "estimated_total_candidate_visits": (
            selection_visits + bootstrap_expected
        ),
        "bootstrap_candidate_limit": bootstrap_candidate_limit,
    }


def preflight_training(
    steps,
    seed,
    *,
    batch_size=None,
    min_replay_size=None,
    updates_per_transition=None,
    bootstrap_candidate_limit=None,
):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    short_horizon = max(config.V1_FORECAST_STEP_SIM, 2.0 * config.V1_FORECAST_STEP_SIM)
    runtime = create_v1_runtime(forecast_end_sim=short_horizon, random_seed=seed)
    total_capacity = sum(
        runtime.calendar.node_capacity(node)
        for node in runtime.infrastructure.compute_nodes
    )
    current = 0.0
    task_count = 0
    total_slots = 0
    max_slots = 0
    declared_slots = 0
    sla_counts = {item.value: 0 for item in SlaType}
    for cycle in range(steps):
        current += config.SCHEDULING_CYCLE
        arrivals = _generate_arrivals(runtime, current, cycle, total_capacity)
        for task in arrivals:
            theoretical = runtime.scheduler.candidate_generator.theoretical_slot_count(
                task, current
            )
            declared_slots += theoretical
            slots = (
                min(
                    theoretical,
                    config.V1_CANDIDATE_POOL_MAX_BY_SLA[task.sla_type.value],
                )
                if config.V1_CANDIDATE_MODE == "layered_pool"
                else theoretical
            )
            task_count += 1
            total_slots += slots
            max_slots = max(max_slots, slots)
            sla_counts[task.sla_type.value] += 1
    batch_size = (
        config.BATCH_SIZE if batch_size is None else batch_size
    )
    min_replay_size = (
        config.V1_REPLAY_MIN_SIZE
        if min_replay_size is None else min_replay_size
    )
    updates_per_transition = (
        config.V1_TRAIN_UPDATES_PER_TRANSITION
        if updates_per_transition is None else updates_per_transition
    )
    bootstrap_candidate_limit = _optional_positive_int(
        "bootstrap_candidate_limit", bootstrap_candidate_limit
    )
    work = _estimate_candidate_work(
        task_count,
        total_slots,
        max_slots,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        updates_per_transition=updates_per_transition,
        bootstrap_candidate_limit=bootstrap_candidate_limit,
    )
    return {
        "steps": steps,
        "seed": seed,
        "task_count": task_count,
        "sla_counts": sla_counts,
        "theoretical_candidate_slots": total_slots,
        "declared_complete_candidate_slots": declared_slots,
        "max_candidate_slots_per_task": max_slots,
        "training_parameters": {
            "batch_size": batch_size,
            "min_replay_size": max(batch_size, min_replay_size),
            "updates_per_transition": updates_per_transition,
            "bootstrap_candidate_limit": bootstrap_candidate_limit,
        },
        **work,
        "objective_calibration_ready": not (
            config.V1_COST_REFERENCE_YUAN == 0.0
            and config.V1_COST_SCALE_YUAN == 1.0
            and config.V1_SOFT_TARDINESS_WEIGHT == 0.0
            and config.V1_FLEXIBLE_TARDINESS_WEIGHT == 0.0
        ),
        "objective_parameters": {
            "cost_reference_yuan": config.V1_COST_REFERENCE_YUAN,
            "cost_scale_yuan": config.V1_COST_SCALE_YUAN,
            "green_absorption_delta_scale": config.V1_GREEN_ABSORPTION_DELTA_SCALE,
            "soft_tardiness_weight": config.V1_SOFT_TARDINESS_WEIGHT,
            "flexible_tardiness_weight": config.V1_FLEXIBLE_TARDINESS_WEIGHT,
        },
    }


def _training_monitoring_values(interval):
    rewards = interval["rewards"]
    rolling = interval["rolling_rewards"]
    q_summaries = interval["q_summaries"]
    td_errors = interval["td_errors"]
    values = {
        "mean_reward": float(np.mean(rewards)) if rewards else "",
        "rolling_reward": float(np.mean(rolling)) if rolling else "",
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        **interval["reward_components"],
        "q_mean": "",
        "q_max": "",
        "q_min": "",
        "q_range": "",
        "td_error_mean": float(np.mean(td_errors)) if len(td_errors) else "",
        "td_error_p95": (
            float(np.percentile(td_errors, 95.0)) if len(td_errors) else ""
        ),
    }
    if q_summaries:
        q_mins, q_maxes, q_means = zip(*q_summaries)
        values.update({
            "q_mean": float(np.mean(q_means)),
            "q_max": float(max(q_maxes)),
            "q_min": float(min(q_mins)),
        })
        values["q_range"] = values["q_max"] - values["q_min"]
    return values


def _overall_cpu_utilization(runtime, time_sim):
    """Return current system-wide reserved CPU divided by total CPU capacity."""

    capacities = {
        node: runtime.calendar.node_capacity(node)
        for node in runtime.infrastructure.compute_nodes
    }
    total_capacity = float(sum(capacities.values()))
    if total_capacity <= 0.0:
        return 0.0
    used_cpu = sum(
        allocation.amount
        for allocation in runtime.calendar.snapshot().cpu_calendar_view
        if allocation.resource_id in capacities
        and allocation.interval_sim.contains(time_sim)
    )
    return float(used_cpu / total_capacity)


def _checkpoint_path(output):
    return output.with_name(output.stem + ".last.pt")


def _model_artifact_payload(runtime, loop, *, cycle, seed, device, run_config):
    metadata = _metadata(runtime)
    return {
        "model_state_dict": runtime.candidate_q_network.state_dict(),
        "target_state_dict": loop.target.state_dict(),
        "optimizer_state_dict": loop.trainer.optimizer.state_dict(),
        "metadata": asdict(metadata),
        "model_id": metadata.model_id,
        "training_steps": cycle,
        "transition_count": loop.transition_count,
        "update_count": loop.update_count,
        "mean_loss": float(np.mean(loop.losses)) if loop.losses else None,
        "epsilon": loop.policy.epsilon,
        "seed": seed,
        "device": device,
        "run_config": run_config,
        "config_hash": _canonical_hash(run_config),
        "training_monitoring_schema_version": "1.0",
    }


def _save_model_artifact(
    path, runtime, loop, *, cycle, seed, device, run_config
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        _model_artifact_payload(
            runtime,
            loop,
            cycle=cycle,
            seed=seed,
            device=device,
            run_config=run_config,
        ),
        temporary,
    )
    os.replace(temporary, path)
    return path


def _load_task_trace(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("validation task trace must contain a JSON task list")
    return tuple(TaskSpec.from_mapping(row) for row in rows)


def _generate_fixed_validation_trace(cutoff, seed):
    from v1.evaluate_v1 import _generate_trace

    with _preserve_rng_state():
        runtime = create_v1_runtime(
            policy_name="earliest_feasible",
            forecast_end_sim=cutoff + config.V1_MAX_FORECAST_LOOKAHEAD_SIM,
            random_seed=seed,
            device="cpu",
        )
        return _generate_trace(runtime, cutoff, seed)


def _write_validation_trace(path, trace, *, cutoff, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    task_rows = []
    for task in trace:
        row = asdict(task)
        row["sla_type"] = task.sla_type.value
        task_rows.append(row)
    payload = {
        "schema_version": "1.0",
        "validation_seed": seed,
        "arrival_cutoff_sim": cutoff,
        "tasks": task_rows,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def _metric_value(metric):
    return "" if metric is None or metric.value is None else float(metric.value)


def _validation_metrics_row(
    report,
    *,
    cycle,
    time_sim,
    checkpoint_path,
    validation_seed,
    validation_cutoff,
):
    metrics = report.metrics
    diagnostics = report.diagnostics or {}
    task_summary = diagnostics.get("task_summary", {})
    runtime_summary = diagnostics.get("runtime_summary", {})
    expired_rate = ""
    if metrics is not None and metrics.arrival_count:
        expired_rate = metrics.expired_count / metrics.arrival_count
    row = {
        "cycle": cycle,
        "time_sim": time_sim,
        "status": report.status.value,
        "checkpoint_path": str(checkpoint_path),
        "validation_seed": validation_seed,
        "arrival_cutoff_sim": validation_cutoff,
        "task_trace_hash": report.metadata.task_trace_hash,
        "completion_rate": (
            "" if metrics is None else _metric_value(metrics.completion_rate)
        ),
        # In v1, missing the latest-start SLA boundary terminates as EXPIRED.
        # Keep both names so monitoring is explicit without introducing a new
        # formal metric definition.
        "sla_violation_rate": expired_rate,
        "expired_rate": expired_rate,
        "start_delay_p95_sim": task_summary.get(
            "start_delay_sim", {}
        ).get("p95", ""),
        "cost_yuan_per_completed_cpu_hour": (
            ""
            if metrics is None
            else _metric_value(metrics.cost_yuan_per_completed_cpu_hour)
        ),
        "task_green_coverage": (
            ""
            if metrics is None
            else _metric_value(metrics.completed_task_green_coverage)
        ),
        "system_green_absorption": (
            ""
            if metrics is None
            else _metric_value(metrics.system_green_absorption_rate)
        ),
        "active_wait_positive_benefit_rate": (
            ""
            if metrics is None
            else _metric_value(
                metrics.active_wait_metrics.positive_benefit_rate
            )
        ),
        "decision_wall_p95_seconds": task_summary.get(
            "decision_wall_seconds", {}
        ).get("p95", ""),
        "evaluation_wall_seconds": runtime_summary.get(
            "total_wall_seconds", ""
        ),
    }
    return {
        key: "" if value is None else value for key, value in row.items()
    }


def _run_fixed_validation(
    *,
    snapshot_path,
    report_path,
    cycle,
    time_sim,
    trace,
    validation_seed,
    validation_cutoff,
    validation_safety_cap,
    device,
    candidate_chunk_size,
    system_version,
):
    from v1.evaluate_v1 import run_evaluation

    with _preserve_rng_state():
        report = run_evaluation(
            "candidate_dqn",
            validation_cutoff,
            validation_seed,
            validation_safety_cap,
            snapshot_path,
            device=device,
            candidate_chunk_size=candidate_chunk_size,
            system_version=system_version,
            audit_mode="final",
            task_trace=trace,
            # Training-checkpoint validation retains its existing zero-warm-up
            # fixture until the separate training warm-up runner is introduced.
            warmup_days=0.0,
        )
    row = _validation_metrics_row(
        report,
        cycle=cycle,
        time_sim=time_sim,
        checkpoint_path=snapshot_path,
        validation_seed=validation_seed,
        validation_cutoff=validation_cutoff,
    )
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "layer": "validation",
                "metric_source": "v1.evaluate_v1.run_evaluation",
                "metrics": row,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return row


def _save_checkpoint(path, runtime, loop, *, cycle, current_time, seed, run_config):
    metadata = _metadata(runtime)
    payload = {
        "checkpoint_type": "v1_training_resume",
        "checkpoint_version": 1,
        "runtime": runtime,
        "loop": loop,
        "cycle": cycle,
        "current_time_sim": current_time,
        "seed": seed,
        "run_config": run_config,
        "config_hash": _canonical_hash(run_config),
        "metadata": asdict(metadata),
        "model_id": metadata.model_id,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_checkpoint(
    path,
    *,
    device,
    seed,
    run_config,
    forecast_end_sim=None,
    profiler=None,
):
    # RNG states are CPU ByteTensors even when the resumed model will run on
    # CUDA. Load the checkpoint on CPU first, then move model and optimizer
    # tensors through the explicit, tested device migration below.
    # Historical checkpoints created with ``python -m v1.train_v1`` recorded
    # the loop class under ``__main__``. Expose the stable implementation name
    # before unpickling so V2 can resume those files as well.
    import __main__
    if not hasattr(__main__, "V1TrainingLoop"):
        __main__.V1TrainingLoop = V1TrainingLoop
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_type") != "v1_training_resume":
        raise ValueError("resume file is not a v1 training checkpoint")
    if checkpoint.get("seed") != seed:
        raise ValueError("resume seed does not match --seed")
    saved_run_config = checkpoint.get("run_config")
    if not _resume_config_compatible(saved_run_config, run_config):
        raise ValueError("resume training configuration mismatch")
    runtime = checkpoint["runtime"]
    loop = checkpoint["loop"]
    if forecast_end_sim is not None:
        extend_v1_runtime_forecasts(runtime, forecast_end_sim)
    restored_tasks = (
        runtime.scheduler.state_machine.task_spec(task_id)
        for task_id in runtime.scheduler.state_machine.task_ids
    )
    ensure_v1_runtime_forecasts_for_tasks(runtime, restored_tasks)
    generator = runtime.scheduler.candidate_generator
    if not hasattr(generator, "active_wait_enabled"):
        generator.active_wait_enabled = True
    encoder = runtime.candidate_feature_encoder
    if not hasattr(encoder, "disabled_feature_groups"):
        if run_config.get("V1_DISABLED_CANDIDATE_FEATURE_GROUPS"):
            raise ValueError(
                "legacy checkpoint cannot enable disabled candidate feature groups"
            )
        encoder.disabled_feature_groups = ()
        encoder._disabled_feature_indices = ()
    clock = loop.reward_assembler.clock
    if not hasattr(clock, "mode"):
        clock.mode = "physical_time"
        clock.decision_gamma = 0.95
    validate_checkpoint_metadata(
        checkpoint.get("metadata", {}),
        encoder.feature_schema_hash,
    )
    loop.runtime = runtime
    loop.policy = runtime.scheduler.policy
    loop.trainer.next_candidate_provider = loop._next_candidate_feature_chunks
    loop.to_device(device)
    loop.set_candidate_chunk_size(run_config["candidate_chunk_size"])
    loop.set_bootstrap_candidate_limit(run_config.get("bootstrap_candidate_limit"))
    loop.policy.audit_candidate_set_hash = False
    loop.profiler = profiler
    runtime.scheduler.profiler = profiler
    runtime.scheduler.candidate_generator.profiler = profiler
    loop.policy.profiler = profiler
    loop.trainer.profiler = profiler
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    if device == "cuda" and checkpoint.get("torch_cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state"])
    _assert_scheduler_invariants(runtime.scheduler)
    return runtime, loop, int(checkpoint["cycle"]), float(checkpoint["current_time_sim"])


def _append_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(row)
    if exists:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or ())
            if existing_fields != fieldnames:
                historical_rows = list(reader)
                fieldnames = existing_fields + [
                    field for field in fieldnames if field not in existing_fields
                ]
                with path.open(
                    "w", newline="", encoding="utf-8"
                ) as migrated:
                    writer = csv.DictWriter(
                        migrated, fieldnames=fieldnames, extrasaction="ignore"
                    )
                    writer.writeheader()
                    writer.writerows(historical_rows)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_training(
    steps,
    seed,
    output_path,
    *,
    device="auto",
    candidate_chunk_size=None,
    batch_size=None,
    min_replay_size=None,
    updates_per_transition=None,
    bootstrap_candidate_limit=None,
    checkpoint_every=None,
    log_every=None,
    invariant_check_every=None,
    reward_rolling_window=DEFAULT_REWARD_ROLLING_WINDOW,
    validation_every=None,
    validation_cutoff=DEFAULT_VALIDATION_CUTOFF_SIM,
    validation_seed=None,
    validation_trace_path=None,
    validation_safety_cap=DEFAULT_VALIDATION_SAFETY_CAP,
    enable_validation=True,
    generate_plots=True,
    resume_path=None,
    allow_large_run=False,
    allow_uncalibrated_objective=False,
    run_preflight=True,
    profile=False,
    profile_output_path=None,
    skip_idle_cycles=None,
    system_version=None,
    warmup_days=None,
):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    device = _resolve_device(device)
    system_version = str(
        config.SYSTEM_VERSION if system_version is None else system_version
    )
    if skip_idle_cycles is None:
        skip_idle_cycles = system_version.startswith("2")
    if not isinstance(skip_idle_cycles, bool):
        raise ValueError("skip_idle_cycles must be a boolean")
    warmup_days = (
        config.WARMUP_DAYS
        if warmup_days is None else float(warmup_days)
    )
    if not np.isfinite(warmup_days) or warmup_days < 0.0:
        raise ValueError("warmup_days must be finite and non-negative")
    chunk_size = _positive_int(
        "candidate_chunk_size",
        config.V1_CANDIDATE_CHUNK_SIZE
        if candidate_chunk_size is None else candidate_chunk_size,
    )
    batch_size = _positive_int(
        "batch_size", config.BATCH_SIZE if batch_size is None else batch_size
    )
    min_replay_size = _positive_int(
        "min_replay_size",
        config.V1_REPLAY_MIN_SIZE if min_replay_size is None else min_replay_size,
    )
    updates = _positive_int(
        "updates_per_transition",
        config.V1_TRAIN_UPDATES_PER_TRANSITION
        if updates_per_transition is None else updates_per_transition,
    )
    bootstrap_candidate_limit = _optional_positive_int(
        "bootstrap_candidate_limit", bootstrap_candidate_limit
    )
    checkpoint_every = _positive_int(
        "checkpoint_every",
        config.V1_CHECKPOINT_INTERVAL_CYCLES
        if checkpoint_every is None else checkpoint_every,
    )
    log_every = _positive_int(
        "log_every", config.V1_LOG_INTERVAL_CYCLES if log_every is None else log_every,
    )
    invariant_check_every = _positive_int(
        "invariant_check_every",
        config.V1_INVARIANT_CHECK_INTERVAL_CYCLES
        if invariant_check_every is None else invariant_check_every,
    )
    reward_rolling_window = _positive_int(
        "reward_rolling_window", reward_rolling_window
    )
    if not isinstance(enable_validation, bool):
        raise ValueError("enable_validation must be a boolean")
    if not isinstance(generate_plots, bool):
        raise ValueError("generate_plots must be a boolean")
    if enable_validation:
        validation_every = _positive_int(
            "validation_every",
            checkpoint_every
            if validation_every is None else validation_every,
        )
        validation_cutoff = float(validation_cutoff)
        if not np.isfinite(validation_cutoff) or validation_cutoff <= 0.0:
            raise ValueError("validation_cutoff must be positive and finite")
        validation_safety_cap = _positive_int(
            "validation_safety_cap", validation_safety_cap
        )
        validation_seed = (
            seed + 1000003 if validation_seed is None else int(validation_seed)
        )
    output = Path(output_path)
    log_path = output.with_name(output.stem + ".training.csv")
    validation_log_path = output.with_name(output.stem + ".validation.csv")
    validation_trace_output_path = output.with_name(
        output.stem + ".validation.trace.json"
    )
    validation_checkpoint_dir = output.with_name(
        output.stem + ".checkpoints"
    )
    validation_report_dir = output.with_name(output.stem + ".validation")
    curve_dir = output.with_name(output.stem + ".curves")
    profiler = TrainingPerformanceProfiler() if profile else None
    profile_path = (
        Path(profile_output_path)
        if profile_output_path is not None
        else output.with_name(output.stem + ".profile.json")
    )
    run_config = _training_config_view(
        candidate_chunk_size=chunk_size,
        batch_size=batch_size,
        min_replay_size=max(batch_size, min_replay_size),
        updates_per_transition=updates,
        bootstrap_candidate_limit=bootstrap_candidate_limit,
        invariant_check_every=invariant_check_every,
        skip_idle_cycles=skip_idle_cycles,
        warmup_days=warmup_days,
        SYSTEM_VERSION=system_version,
    )
    warmup_duration = warmup_days * config.TRAFFIC_DAY_DURATION_IN_SIM
    horizon = (
        warmup_duration
        + steps * config.SCHEDULING_CYCLE
        + config.V1_MAX_FORECAST_LOOKAHEAD_SIM
    )

    if run_preflight and resume_path is None:
        report = preflight_training(
            steps,
            seed,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            updates_per_transition=updates,
            bootstrap_candidate_limit=bootstrap_candidate_limit,
        )
        unsafe = (
            report["theoretical_candidate_slots"]
            > config.V1_PREFLIGHT_MAX_TOTAL_SLOTS
            or report["max_candidate_slots_per_task"]
            > config.V1_PREFLIGHT_MAX_SLOTS_PER_TASK
            or report["estimated_total_candidate_visits"]
            > config.V1_PREFLIGHT_MAX_ESTIMATED_CANDIDATE_VISITS
        )
        print(json.dumps({
            "preflight": report,
            "requires_large_run_override": unsafe,
            "requires_objective_override": not report["objective_calibration_ready"],
        }, ensure_ascii=False))
        if not report["objective_calibration_ready"] and not allow_uncalibrated_objective:
            raise RuntimeError(
                "objective parameters are still the documented pilot placeholders; "
                "calibrate cost/tardiness scales or rerun with "
                "--allow-uncalibrated-objective for a diagnostic smoke run"
            )
        if unsafe and not allow_large_run:
            raise RuntimeError(
                "preflight candidate volume exceeds the safety gate; inspect the "
                "report and rerun with --allow-large-run if the runtime is intentional"
            )

    if resume_path is not None:
        runtime, loop, start_cycle, current = _load_checkpoint(
            Path(resume_path),
            device=device,
            seed=seed,
            run_config=run_config,
            forecast_end_sim=horizon,
            profiler=profiler,
        )
        if start_cycle > steps:
            raise ValueError("checkpoint cycle exceeds requested total --steps")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        runtime = create_v1_runtime(
            policy_name="candidate_dqn",
            forecast_end_sim=horizon,
            random_seed=seed,
            device=device,
            candidate_chunk_size=chunk_size,
        )
        runtime.scheduler.policy.epsilon = config.EPSILON_START
        runtime.scheduler.policy.record_selection_traces = True
        loop = V1TrainingLoop(
            runtime,
            device=device,
            candidate_chunk_size=chunk_size,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            updates_per_transition=updates,
            bootstrap_candidate_limit=bootstrap_candidate_limit,
            random_seed=seed,
            profiler=profiler,
        )
        start_cycle = 0
        current = 0.0

        total_capacity = sum(
            runtime.calendar.node_capacity(node)
            for node in runtime.infrastructure.compute_nodes
        )
        current, warmup_cycle_count = _run_training_warmup(
            runtime,
            loop,
            warmup_days=warmup_days,
            total_capacity=total_capacity,
            invariant_check_every=invariant_check_every,
            profiler=profiler,
        )
        print(json.dumps({
            "training_warmup": {
                "warmup_days": warmup_days,
                "warmup_cycle_count": warmup_cycle_count,
                "measurement_start_sim": current,
                "task_count": runtime.scheduler.state_machine.task_count,
                "replay_size": len(loop.replay),
                "transition_count": loop.transition_count,
                "update_count": loop.update_count,
            }
        }, ensure_ascii=False))

    loop.ensure_monitoring_state(reward_rolling_window)
    validation_trace = None
    if enable_validation and steps > 0:
        if validation_trace_path is not None:
            validation_trace = _load_task_trace(validation_trace_path)
        elif resume_path is not None and validation_trace_output_path.is_file():
            validation_trace = _load_task_trace(validation_trace_output_path)
        else:
            validation_trace = _generate_fixed_validation_trace(
                validation_cutoff, validation_seed
            )
        if any(
            task.arrival_time_sim >= validation_cutoff
            for task in validation_trace
        ):
            raise ValueError(
                "validation task trace contains arrivals outside validation cutoff"
            )
        _write_validation_trace(
            validation_trace_output_path,
            validation_trace,
            cutoff=validation_cutoff,
            seed=validation_seed,
        )

    def run_validation_checkpoint(cycle, time_sim):
        snapshot_path = validation_checkpoint_dir / f"step_{cycle:09d}.pt"
        report_path = validation_report_dir / f"step_{cycle:09d}.json"
        _save_model_artifact(
            snapshot_path,
            runtime,
            loop,
            cycle=cycle,
            seed=seed,
            device=device,
            run_config=run_config,
        )
        validation_row = _run_fixed_validation(
            snapshot_path=snapshot_path,
            report_path=report_path,
            cycle=cycle,
            time_sim=time_sim,
            trace=validation_trace,
            validation_seed=validation_seed,
            validation_cutoff=validation_cutoff,
            validation_safety_cap=validation_safety_cap,
            device=device,
            candidate_chunk_size=chunk_size,
            system_version=system_version,
        )
        _append_log(validation_log_path, validation_row)
        print(json.dumps({"validation": validation_row}, ensure_ascii=False))
        return validation_row

    forecast_end = v1_runtime_forecast_end(runtime)

    total_capacity = sum(
        runtime.calendar.node_capacity(node)
        for node in runtime.infrastructure.compute_nodes
    )
    interval_wall_start = time.perf_counter()
    profile_wall_start = interval_wall_start
    interval_candidates = loop.candidate_count
    interval_losses = len(loop.losses)
    for cycle in range(start_cycle, steps):
        if profiler is not None:
            profiler.increment("virtual_cycle_count")
        current += config.SCHEDULING_CYCLE
        completed_cycle = cycle + 1
        arrivals = _generate_arrivals(runtime, current, cycle, total_capacity)
        if arrivals:
            forecast_end = ensure_v1_runtime_forecasts_for_tasks(runtime, arrivals)
        check_invariants = _should_run_invariant_check(
            completed_cycle,
            invariant_check_every=invariant_check_every,
            checkpoint_every=checkpoint_every,
            final_cycle=steps,
        )
        state_counts = runtime.scheduler.state_machine.count_by_state()
        next_event = runtime.scheduler.event_engine.next_event_time_sim
        next_deadline = (
            runtime.scheduler.queue_manager.next_uncommitted_deadline_sim()
        )
        should_run_scheduler = (
            not skip_idle_cycles
            or bool(arrivals)
            or state_counts[TaskState.QUEUED] > 0
            or (
                next_event is not None
                and next_event <= current + 1e-12
            )
            or (
                next_deadline is not None
                and next_deadline <= current + 1e-12
            )
        )
        if should_run_scheduler:
            result = _run_profiled_operation(
                profiler,
                "scheduler_cycle_count",
                lambda: runtime.scheduler.run_cycle(
                    current,
                    arrivals=arrivals,
                    forecast_covered_until_sim=forecast_end,
                ),
            )
            _run_profiled_operation(
                profiler,
                "training_process_cycle_count",
                lambda: loop.process_cycle(
                    result,
                    check_invariants=check_invariants,
                ),
            )
        elif check_invariants:
            # Full audits stay periodic even when the physical scheduler has
            # no work at this virtual cycle boundary.
            _assert_scheduler_invariants(runtime.scheduler)
            if profiler is not None:
                profiler.increment("idle_cycle_skip_count")
        elif profiler is not None:
            profiler.increment("idle_cycle_skip_count")
        if completed_cycle % log_every == 0 or completed_cycle == steps:
            wall = time.perf_counter() - interval_wall_start
            recent_losses = loop.losses[interval_losses:]
            monitoring = _training_monitoring_values(
                loop.consume_monitoring_interval()
            )
            row = {
                "cycle": completed_cycle,
                "time_sim": current,
                "tasks": runtime.scheduler.state_machine.task_count,
                "overall_cpu_utilization": _overall_cpu_utilization(
                    runtime, current
                ),
                "mean_loss": (
                    float(np.mean(recent_losses)) if recent_losses else ""
                ),
                **monitoring,
                "epsilon": loop.policy.epsilon,
                "replay_size": len(loop.replay),
                "transitions": loop.transition_count,
                "updates": loop.update_count,
                "candidates": loop.candidate_count,
                "candidates_since_log": loop.candidate_count - interval_candidates,
                "wall_seconds_since_log": wall,
                "device": device,
            }
            log_started = time.perf_counter()
            _append_log(log_path, row)
            progress = {"progress": row}
            if profiler is not None:
                progress["profile"] = profiler.summary(
                    time.perf_counter() - profile_wall_start
                )["sections_percent"]
            print(json.dumps(progress, ensure_ascii=False))
            if profiler is not None:
                profiler.add(
                    "logging_seconds", time.perf_counter() - log_started
                )
            interval_wall_start = time.perf_counter()
            interval_candidates = loop.candidate_count
            interval_losses = len(loop.losses)
        if completed_cycle % checkpoint_every == 0:
            checkpoint_started = time.perf_counter()
            _save_checkpoint(
                _checkpoint_path(output), runtime, loop,
                cycle=completed_cycle, current_time=current, seed=seed,
                run_config=run_config,
            )
            if profiler is not None:
                profiler.add(
                    "checkpoint_seconds",
                    time.perf_counter() - checkpoint_started,
                )
        if (
            validation_trace is not None
            and completed_cycle < steps
            and completed_cycle % validation_every == 0
        ):
            validation_started = time.perf_counter()
            run_validation_checkpoint(completed_cycle, current)
            validation_elapsed = time.perf_counter() - validation_started
            # Validation is an isolated observation layer and is intentionally
            # excluded from the training throughput interval.
            interval_wall_start += validation_elapsed

    # A resume checkpoint must represent the online state at the requested
    # cycle boundary. Settlement advances physical time to drain all accepted
    # work and is only for producing the frozen model artifact; saving after
    # settlement would make a later resume silently continue from a different
    # time line.
    _assert_scheduler_invariants(runtime.scheduler)
    checkpoint_started = time.perf_counter()
    _save_checkpoint(
        _checkpoint_path(output), runtime, loop,
        cycle=steps, current_time=current, seed=seed,
        run_config=run_config,
    )
    if profiler is not None:
        profiler.add(
            "checkpoint_seconds",
            time.perf_counter() - checkpoint_started,
        )

    current = _settle(runtime, loop, current, profiler=profiler)
    _assert_scheduler_invariants(runtime.scheduler)
    _run_profiled_operation(
        profiler,
        "training_process_cycle_count",
        lambda: loop.finalize(current),
    )
    artifact_started = time.perf_counter()
    _save_model_artifact(
        output,
        runtime,
        loop,
        cycle=steps,
        seed=seed,
        device=device,
        run_config=run_config,
    )
    if profiler is not None:
        profiler.add(
            "artifact_save_seconds", time.perf_counter() - artifact_started
        )
        total_profile_wall = time.perf_counter() - profile_wall_start
        _write_profile_outputs(profiler, total_profile_wall, profile_path)
    if validation_trace is not None:
        run_validation_checkpoint(steps, current)
    if generate_plots and steps > 0 and log_path.is_file():
        from v1.plot_training_v1 import generate_training_curves

        curve_paths = generate_training_curves(
            log_path,
            validation_log_path,
            curve_dir,
        )
        print(json.dumps(
            {"training_curves": [str(path) for path in curve_paths]},
            ensure_ascii=False,
        ))
    return output


def main(
    *,
    system_version=None,
    default_output="artifacts/v1/logs/candidate_dqn.pt",
    default_candidate_chunk_size=None,
    default_checkpoint_every=None,
):
    effective_system_version = str(
        config.SYSTEM_VERSION if system_version is None else system_version
    )
    parser = argparse.ArgumentParser(
        description=(
            f"Train candidate DQN with system V{effective_system_version}"
        )
    )
    parser.add_argument("--steps", type=int, default=config.MAX_STEPS)
    parser.add_argument(
        "--warmup-days",
        type=float,
        default=config.WARMUP_DAYS,
        help="state-only warm-up before training step 0 (simulated days)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=(
            config.V1_CANDIDATE_CHUNK_SIZE
            if default_candidate_chunk_size is None
            else default_candidate_chunk_size
        ),
    )
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--min-replay-size", type=int, default=config.V1_REPLAY_MIN_SIZE)
    parser.add_argument("--updates-per-transition", type=int, default=config.V1_TRAIN_UPDATES_PER_TRANSITION)
    parser.add_argument(
        "--bootstrap-candidate-limit",
        type=int,
        help=(
            "training-only cap on candidates scanned for each replay "
            "bootstrap target; omitted keeps exact complete bootstrap"
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=(
            config.V1_CHECKPOINT_INTERVAL_CYCLES
            if default_checkpoint_every is None
            else default_checkpoint_every
        ),
    )
    parser.add_argument("--log-every", type=int, default=config.V1_LOG_INTERVAL_CYCLES)
    parser.add_argument(
        "--reward-rolling-window",
        type=int,
        default=DEFAULT_REWARD_ROLLING_WINDOW,
        help="number of latest transitions used by rolling_reward",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        help="validation interval in cycles (default: checkpoint interval)",
    )
    parser.add_argument(
        "--validation-cutoff",
        type=float,
        default=DEFAULT_VALIDATION_CUTOFF_SIM,
        help="fixed validation trace arrival cutoff in simulation units",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        help="fixed validation seed (default: training seed + 1000003)",
    )
    parser.add_argument(
        "--validation-trace",
        type=Path,
        help="optional fixed Task Trace JSON; otherwise generated once",
    )
    parser.add_argument(
        "--validation-safety-cap",
        type=int,
        default=DEFAULT_VALIDATION_SAFETY_CAP,
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="disable the training-time validation layer",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="disable automatic training/validation curve generation",
    )
    parser.add_argument(
        "--invariant-check-every",
        type=int,
        default=config.V1_INVARIANT_CHECK_INTERVAL_CYCLES,
    )
    parser.add_argument("--allow-large-run", action="store_true")
    parser.add_argument("--allow-uncalibrated-objective", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--ablation-variant", choices=variant_names()
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="record exclusive training-stage wall-clock timings",
    )
    parser.add_argument(
        "--profile-output",
        help="profiling JSON path (default: <model>.profile.json)",
    )
    idle_group = parser.add_mutually_exclusive_group()
    idle_group.add_argument(
        "--skip-idle-cycles",
        dest="skip_idle_cycles",
        action="store_true",
        help="execute full scheduling only on arrivals or due physical events",
    )
    idle_group.add_argument(
        "--no-skip-idle-cycles",
        dest="skip_idle_cycles",
        action="store_false",
        help="execute the scheduler at every virtual cycle boundary",
    )
    parser.set_defaults(skip_idle_cycles=None)
    args = parser.parse_args()
    with apply_ablation_variant(args.ablation_variant):
        if args.preflight_only:
            print(json.dumps(preflight_training(
                args.steps,
                args.seed,
                batch_size=args.batch_size,
                min_replay_size=args.min_replay_size,
                updates_per_transition=args.updates_per_transition,
                bootstrap_candidate_limit=args.bootstrap_candidate_limit,
            ), ensure_ascii=False, indent=2))
            return
        path = run_training(
            args.steps,
            args.seed,
            args.output,
            device=args.device,
            candidate_chunk_size=args.candidate_chunk_size,
            batch_size=args.batch_size,
            min_replay_size=args.min_replay_size,
            updates_per_transition=args.updates_per_transition,
            bootstrap_candidate_limit=args.bootstrap_candidate_limit,
            checkpoint_every=args.checkpoint_every,
            log_every=args.log_every,
            invariant_check_every=args.invariant_check_every,
            reward_rolling_window=args.reward_rolling_window,
            validation_every=args.validation_every,
            validation_cutoff=args.validation_cutoff,
            validation_seed=args.validation_seed,
            validation_trace_path=args.validation_trace,
            validation_safety_cap=args.validation_safety_cap,
            enable_validation=not args.no_validation,
            generate_plots=not args.no_plots,
            resume_path=args.resume,
            allow_large_run=args.allow_large_run,
            allow_uncalibrated_objective=args.allow_uncalibrated_objective,
            run_preflight=not args.skip_preflight,
            profile=args.profile,
            profile_output_path=args.profile_output,
            skip_idle_cycles=args.skip_idle_cycles,
            system_version=effective_system_version,
            warmup_days=args.warmup_days,
        )
    completed = {
        "status": "complete",
        "model": str(path),
        "training_log": str(path.with_name(path.stem + ".training.csv")),
    }
    if not args.no_validation and args.steps > 0:
        completed.update({
            "validation_log": str(
                path.with_name(path.stem + ".validation.csv")
            ),
            "validation_trace": str(
                path.with_name(path.stem + ".validation.trace.json")
            ),
            "validation_checkpoints": str(
                path.with_name(path.stem + ".checkpoints")
            ),
        })
    if not args.no_plots and args.steps > 0:
        completed["training_curves"] = str(
            path.with_name(path.stem + ".curves")
        )
    if args.profile:
        completed["profile"] = str(
            Path(args.profile_output)
            if args.profile_output
            else path.with_name(path.stem + ".profile.json")
        )
    print(json.dumps(completed, ensure_ascii=False))


if __name__ == "__main__":
    main()
