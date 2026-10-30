# 文件功能：定义时间区间、路径、资源分配和预留契约，统一链路标识及预留哈希。
# 主要内容：ReservationValidationError、TimeInterval、PathSpec、ReservationRequest、Reservation、CommitStatus、ReleaseStatus、canonical_edge 等。
# 模块上游（导入关系）：v1.accounting.energy、v1.accounting.forecast、v1.accounting.ledger、v1.domain、v1.domain.candidates 等。
# 模块下游（项目依赖）：v1.domain.units。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Immutable path and reservation contracts for the v1.0 scheduler."""

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Optional, Tuple

from .units import UnitValidationError, finite_number, non_negative_finite, positive_finite


Edge = Tuple[str, str]


# 作用：统一无向链路的端点顺序作为资源键。
# 上游（直接调用）：v1.domain.reservations.PathSpec.resource_edges、v1.evaluation_v1.diagnostics.build_evaluation_diagnostics、v1.evaluation_v1.runner.EvaluationRunner._warmup_snapshot 等。
# 下游调用：ReservationValidationError。
def canonical_edge(edge: Edge) -> Edge:
    if len(edge) != 2:
        raise ReservationValidationError("edge", "must contain two node ids")
    u, v = str(edge[0]), str(edge[1])
    if not u or not v or u == v:
        raise ReservationValidationError("edge", "must connect two distinct nodes")
    return (u, v) if u <= v else (v, u)


class ReservationValidationError(ValueError):
    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.domain.reservations.PathSpec.__post_init__、v1.domain.reservations.ReservationRequest.__post_init__、v1.domain.reservations.TimeInterval.__post_init__ 等。
    # 调用约定：构造实例时由 Python 调用。
    # 下游调用：super().__init__。
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


# 作用：解析并验证当前上下文的数值。
# 上游（直接调用）：v1.domain.reservations.PathSpec.__post_init__、v1.domain.reservations.ReservationRequest.__post_init__、v1.domain.reservations.TimeInterval.__post_init__ 等。
# 下游调用：positive_finite、non_negative_finite、finite_number、ReservationValidationError。
def _number(field: str, value, *, positive=False, non_negative=False) -> float:
    try:
        if positive:
            return positive_finite(field, value)
        if non_negative:
            return non_negative_finite(field, value)
        return finite_number(field, value)
    except UnitValidationError as exc:
        raise ReservationValidationError(exc.field, exc.message)


@dataclass(frozen=True, order=True)
class TimeInterval:
    start_sim: float
    end_sim: float

    # 作用：在数据类构造后校验字段或建立派生数据。
    # 调用约定：由 dataclass 生成的构造流程调用。
    # 下游调用：_number、ReservationValidationError。
    def __post_init__(self) -> None:
        start = _number("interval.start_sim", self.start_sim)
        end = _number("interval.end_sim", self.end_sim)
        if end <= start:
            raise ReservationValidationError(
                "interval",
                "must be a non-empty half-open interval [start, end)",
            )
        object.__setattr__(self, "start_sim", start)
        object.__setattr__(self, "end_sim", end)

    # 作用：返回区间的仿真时长。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    @property
    def duration_sim(self) -> float:
        return self.end_sim - self.start_sim

    # 作用：判断两时间区间是否重叠。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def overlaps(self, other: "TimeInterval") -> bool:
        return self.start_sim < other.end_sim and other.start_sim < self.end_sim

    # 作用：判断时刻是否落在当前时间区间。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：_number。
    def contains(self, time_sim: float) -> bool:
        time_value = _number("time_sim", time_sim)
        return self.start_sim <= time_value < self.end_sim


@dataclass(frozen=True)
class PathSpec:
    path_id: str
    source_node: str
    target_node: str
    ordered_nodes: Tuple[str, ...]
    ordered_edges: Tuple[Edge, ...]
    total_distance_km: float
    static_bottleneck_mbps: float
    route_cost: float

    # 作用：在数据类构造后校验字段或建立派生数据。
    # 调用约定：由 dataclass 生成的构造流程调用。
    # 下游调用：ReservationValidationError、_number。
    def __post_init__(self) -> None:
        if not isinstance(self.path_id, str) or not self.path_id:
            raise ReservationValidationError("path_id", "must be non-empty")
        if not isinstance(self.source_node, str) or not self.source_node:
            raise ReservationValidationError("source_node", "must be non-empty")
        if not isinstance(self.target_node, str) or not self.target_node:
            raise ReservationValidationError("target_node", "must be non-empty")
        nodes = tuple(str(node) for node in self.ordered_nodes)
        edges = tuple((str(edge[0]), str(edge[1])) for edge in self.ordered_edges)
        if not nodes or nodes[0] != self.source_node or nodes[-1] != self.target_node:
            raise ReservationValidationError(
                "ordered_nodes",
                "must start at source_node and end at target_node",
            )
        if self.is_local:
            if nodes != (self.source_node,) or edges:
                raise ReservationValidationError(
                    "ordered_edges",
                    "local paths must contain one node and no edges",
                )
        else:
            expected_edges = tuple(zip(nodes[:-1], nodes[1:]))
            if len(nodes) < 2 or edges != expected_edges:
                raise ReservationValidationError(
                    "ordered_edges",
                    "must follow each consecutive ordered node",
                )
        object.__setattr__(self, "ordered_nodes", nodes)
        object.__setattr__(self, "ordered_edges", edges)
        object.__setattr__(
            self,
            "total_distance_km",
            _number("total_distance_km", self.total_distance_km, non_negative=True),
        )
        object.__setattr__(
            self,
            "static_bottleneck_mbps",
            _number(
                "static_bottleneck_mbps",
                self.static_bottleneck_mbps,
                non_negative=True,
            ),
        )
        object.__setattr__(
            self,
            "route_cost",
            _number("route_cost", self.route_cost, non_negative=True),
        )

    # 作用：判断路径是否无需网络传输。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    @property
    def is_local(self) -> bool:
        return self.source_node == self.target_node

    # 作用：返回路径占用的规范链路资源。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：canonical_edge。
    @property
    def resource_edges(self) -> Tuple[Edge, ...]:
        return tuple(canonical_edge(edge) for edge in self.ordered_edges)


@dataclass(frozen=True)
class ReservationRequest:
    task_id: str
    committed_candidate_id: str
    committed_at_sim: float
    reservation_snapshot_version: int
    target_node: str
    path: PathSpec
    transmission_interval_sim: Optional[TimeInterval]
    compute_interval_sim: TimeInterval
    bandwidth_amount_mbps: float
    cpu_amount: float

    # 作用：在数据类构造后校验字段或建立派生数据。
    # 调用约定：由 dataclass 生成的构造流程调用。
    # 下游调用：ReservationValidationError、_number。
    def __post_init__(self) -> None:
        for field in ("task_id", "committed_candidate_id", "target_node"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ReservationValidationError(field, "must be non-empty")
        if self.target_node != self.path.target_node:
            raise ReservationValidationError(
                "target_node",
                "must equal path.target_node",
            )
        committed_at = _number("committed_at_sim", self.committed_at_sim)
        object.__setattr__(self, "committed_at_sim", committed_at)
        if (
            isinstance(self.reservation_snapshot_version, bool)
            or not isinstance(self.reservation_snapshot_version, int)
            or self.reservation_snapshot_version < 0
        ):
            raise ReservationValidationError(
                "reservation_snapshot_version",
                "must be a non-negative integer",
            )
        object.__setattr__(
            self,
            "cpu_amount",
            _number("cpu_amount", self.cpu_amount, positive=True),
        )
        object.__setattr__(
            self,
            "bandwidth_amount_mbps",
            _number(
                "bandwidth_amount_mbps",
                self.bandwidth_amount_mbps,
                positive=True,
            ),
        )
        if self.compute_interval_sim.start_sim < committed_at:
            raise ReservationValidationError(
                "compute_interval_sim",
                "cannot start before committed_at_sim",
            )
        if self.path.is_local:
            if self.transmission_interval_sim is not None:
                raise ReservationValidationError(
                    "transmission_interval_sim",
                    "must be omitted for local reservations",
                )
        else:
            if self.transmission_interval_sim is None:
                raise ReservationValidationError(
                    "transmission_interval_sim",
                    "is required for remote reservations",
                )
            if not math.isclose(
                self.transmission_interval_sim.end_sim,
                self.compute_interval_sim.start_sim,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ReservationValidationError(
                    "transmission_interval_sim",
                    "must end exactly at compute start (JIT)",
                )
            if self.transmission_interval_sim.start_sim < committed_at:
                raise ReservationValidationError(
                    "transmission_interval_sim",
                    "cannot start before committed_at_sim",
                )


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    task_id: str
    committed_candidate_id: str
    committed_at_sim: float
    committed_reservation_version: int
    target_node: str
    path: PathSpec
    transmission_interval_sim: Optional[TimeInterval]
    compute_interval_sim: TimeInterval
    bandwidth_amount_mbps: float
    cpu_amount: float
    audit_hash: str


class CommitStatus(str, Enum):
    COMMITTED = "COMMITTED"
    CONFLICT = "CONFLICT"
    CPU_INFEASIBLE = "CPU_INFEASIBLE"
    BANDWIDTH_INFEASIBLE = "BANDWIDTH_INFEASIBLE"
    CANDIDATE_STALE = "CANDIDATE_STALE"
    INTERNAL_ROLLBACK = "INTERNAL_ROLLBACK"


class ReleaseStatus(str, Enum):
    RELEASED = "RELEASED"
    ALREADY_RELEASED = "ALREADY_RELEASED"
    NOT_FOUND = "NOT_FOUND"
    TOO_EARLY = "TOO_EARLY"


# 作用：提取预留身份计算使用的规范字段。
# 上游（直接调用）：v1.domain.reservations.deterministic_reservation_hash。
def reservation_identity_payload(request: ReservationRequest, version: int) -> dict:
    tx = request.transmission_interval_sim
    return {
        "task_id": request.task_id,
        "candidate_id": request.committed_candidate_id,
        "committed_at_sim": request.committed_at_sim,
        "committed_version": version,
        "target_node": request.target_node,
        "path_id": request.path.path_id,
        "ordered_nodes": request.path.ordered_nodes,
        "tx": None if tx is None else (tx.start_sim, tx.end_sim),
        "compute": (
            request.compute_interval_sim.start_sim,
            request.compute_interval_sim.end_sim,
        ),
        "bandwidth": request.bandwidth_amount_mbps,
        "cpu": request.cpu_amount,
    }


# 作用：由规范字段计算可重复的预留哈希。
# 上游（直接调用）：v1.scheduler.resource_calendar.ReservationCalendar.try_commit。
# 下游调用：reservation_identity_payload。
def deterministic_reservation_hash(request: ReservationRequest, version: int) -> str:
    encoded = json.dumps(
        reservation_identity_payload(request, version),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
