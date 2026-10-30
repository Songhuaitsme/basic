# 文件功能：从静态拓扑提供源节点到计算节点的候选路径。
# 主要内容：StaticPathProvider。
# 模块上游（导入关系）：v1.scheduler、v1.scheduler.candidate_generator、v1.v1_runtime、tests.v1.test_ablation_settings_v1、tests.v1.test_accounting_v1 等。
# 模块下游（项目依赖）：v1.domain.reservations、v1.domain.units、v1.scheduler.transmission。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Declared deterministic candidate path set for v1.0 complete enumeration."""

from typing import List

import networkx as nx

from v1.domain.reservations import PathSpec
from v1.domain.units import positive_finite
from .transmission import build_path_spec


class StaticPathProvider:
    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.v1_runtime.create_v1_runtime。
    # 调用约定：构造实例时由 Python 调用。
    def __init__(self, graph, max_paths_per_target: int = 1):
        if (
            isinstance(max_paths_per_target, bool)
            or not isinstance(max_paths_per_target, int)
            or max_paths_per_target <= 0
        ):
            raise ValueError("max_paths_per_target must be a positive integer")
        self.graph = graph
        self.max_paths_per_target = max_paths_per_target

    # 作用：取得源节点到目标计算节点的有限条路径。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：positive_finite、build_path_spec、feasible_graph.edges、feasible_graph.remove_edges_from。
    def candidate_paths(
        self,
        source_node: str,
        target_node: str,
        bandwidth_demand_mbps: float,
    ) -> List[PathSpec]:
        bandwidth = positive_finite(
            "bandwidth_demand_mbps",
            bandwidth_demand_mbps,
        )
        if source_node == target_node:
            if source_node not in self.graph:
                return []
            return [build_path_spec(self.graph, [source_node])]

        feasible_graph = self.graph.copy()
        blocked = [
            (u, v)
            for u, v, data in feasible_graph.edges(data=True)
            if float(data.get("capacity", 0.0)) < bandwidth
        ]
        feasible_graph.remove_edges_from(blocked)
        try:
            iterator = nx.shortest_simple_paths(
                feasible_graph,
                source_node,
                target_node,
                weight="cost",
            )
            paths = []
            for ordered_nodes in iterator:
                paths.append(build_path_spec(self.graph, ordered_nodes))
                if len(paths) >= self.max_paths_per_target:
                    break
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
