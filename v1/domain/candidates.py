# 文件功能：定义候选、生成模式及结果，生成确定性候选标识和预留请求。
# 主要内容：CandidateMode、CandidateGenerationStatus、Candidate、CandidateSetResult、_canonical_json_string、deterministic_candidate_id、deterministic_candidate_id_v1_fields。
# 模块上游（导入关系）：v1.domain、v1.learning.candidate_dqn、v1.learning.features、v1.scheduler.approximate、v1.scheduler.candidate_generator 等。
# 模块下游（项目依赖）：v1.domain.reservations。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Immutable v1.0 node-time-path candidate schema."""

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from functools import lru_cache
from typing import Tuple

from .reservations import PathSpec, ReservationRequest, TimeInterval


CANDIDATE_SCHEMA_VERSION = "1.0"


# 作用：按规范顺序序列化候选身份。
# 上游（直接调用）：v1.domain.candidates.deterministic_candidate_id_v1_fields。
@lru_cache(maxsize=8192)
def _canonical_json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class CandidateMode(str, Enum):
    COMPLETE = "complete"
    APPROXIMATE = "approximate"
    LAYERED_POOL = "layered_pool"


class CandidateGenerationStatus(str, Enum):
    OK = "OK"
    EMPTY_PHYSICAL = "EMPTY_PHYSICAL"
    EXPIRED_BEFORE_DECISION = "EXPIRED_BEFORE_DECISION"
    INVALID_TASK = "INVALID_TASK"
    FORECAST_NOT_COVERED = "FORECAST_NOT_COVERED"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    candidate_schema_version: str
    candidate_mode: CandidateMode
    task_id: str
    decision_time_sim: float
    reservation_snapshot_version: int
    forecast_version: str
    target_node: str
    path: PathSpec
    transmission_start_sim: float
    transmission_end_sim: float
    compute_start_sim: float
    compute_end_sim: float
    cpu_demand: float
    bandwidth_demand_mbps: float
    scheduler_queue_delay_sim: float
    earliest_feasibility_lead_sim: float
    active_wait_sim: float
    reservation_lead_sim: float
    start_delay_sim: float
    preferred_start_tardiness_ratio: float
    preferred_start_tardiness_applicable: bool
    estimated_candidate_marginal_system_cost_yuan: float
    estimated_green_coverage: float
    estimated_candidate_marginal_green_energy_mwh: float
    estimated_green_absorption_delta: float
    estimated_green_opportunity: bool
    projected_node_utilization: float
    projected_path_peak_utilization: float
    capacity_margin: float

    # 作用：把已选候选转换为日历可提交的预留请求。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：TimeInterval、ReservationRequest。
    def to_reservation_request(self) -> ReservationRequest:
        tx_interval = None
        if not self.path.is_local:
            tx_interval = TimeInterval(
                self.transmission_start_sim,
                self.transmission_end_sim,
            )
        return ReservationRequest(
            task_id=self.task_id,
            committed_candidate_id=self.candidate_id,
            committed_at_sim=self.decision_time_sim,
            reservation_snapshot_version=self.reservation_snapshot_version,
            target_node=self.target_node,
            path=self.path,
            transmission_interval_sim=tx_interval,
            compute_interval_sim=TimeInterval(
                self.compute_start_sim,
                self.compute_end_sim,
            ),
            bandwidth_amount_mbps=self.bandwidth_demand_mbps,
            cpu_amount=self.cpu_demand,
        )


@dataclass(frozen=True)
class CandidateSetResult:
    status: CandidateGenerationStatus
    candidate_mode: CandidateMode
    candidates: Tuple[Candidate, ...]
    theoretical_slot_count: int
    feasible_candidate_count: int
    earliest_compute_start_sim: float
    reason: str = ""


# 作用：根据候选领域字段生成确定性标识。
# 上游（直接调用）：tests.v1.test_domain_foundations.DomainFoundationTest.test_specialized_candidate_id_matches_generic_canonical_hash。
def deterministic_candidate_id(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "candidate-" + hashlib.sha256(encoded).hexdigest()


# 作用：根据 V1 冻结身份字段生成兼容候选标识。
# 上游（直接调用）：v1.scheduler.candidate_generator.CandidateGenerator.attach_context_candidate_id、v1.scheduler.candidate_generator.CandidateGenerator.iter_context_candidate_records、v1.scheduler.candidate_generator.CandidateGenerator.sample_context_candidate_records.attach_candidate_id。
# 下游调用：_canonical_json_string。
def deterministic_candidate_id_v1_fields(
    *,
    compute_end: float,
    compute_start: float,
    forecast_version: str,
    mode: str,
    node: str,
    path: str,
    reservation_version: int,
    schema: str,
    task: str,
    transmission_end: float,
    transmission_start: float,
) -> str:
    """Fast canonical encoder for the frozen v1 candidate-id payload.

    Key order and scalar rendering exactly match ``deterministic_candidate_id``
    with ``sort_keys=True``. Candidate times are already finite validated floats.
    """

    encoded = (
        '{"compute_end":' + repr(float(compute_end))
        + ',"compute_start":' + repr(float(compute_start))
        + ',"forecast_version":' + _canonical_json_string(forecast_version)
        + ',"mode":' + _canonical_json_string(mode)
        + ',"node":' + _canonical_json_string(node)
        + ',"path":' + _canonical_json_string(path)
        + ',"reservation_version":' + str(int(reservation_version))
        + ',"schema":' + _canonical_json_string(schema)
        + ',"task":' + _canonical_json_string(task)
        + ',"transmission_end":' + repr(float(transmission_end))
        + ',"transmission_start":' + repr(float(transmission_start))
        + "}"
    ).encode("utf-8")
    return "candidate-" + hashlib.sha256(encoded).hexdigest()
