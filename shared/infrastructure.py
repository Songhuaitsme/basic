# 文件功能：组装共享物理基础设施，提供拓扑、电价、计算节点和资源视图。
# 主要内容：InfrastructureContext。
# 模块上游（导入关系）：shared、v1.v1_runtime。
# 模块下游（项目依赖）：shared.config、shared.pricing_manager、shared.topology_manager。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Minimal physical infrastructure view shared with the v1 runtime.

This deliberately contains no legacy scheduling, reward, WAIT, or action logic.
"""

from dataclasses import dataclass

from . import config
from .pricing_manager import PricingManager
from .topology_manager import TopologyManager


@dataclass
class InfrastructureContext:
    topo_manager: TopologyManager
    pricing_manager: PricingManager
    all_nodes: tuple
    base_stations: tuple
    compute_nodes: tuple
    node_resources: dict

    # 作用：根据输入构造领域对象或基础设施实例。
    # 上游（直接调用）：v1.v1_runtime.create_v1_runtime。
    # 下游调用：TopologyManager、PricingManager、graph.nodes、str(node).endswith。
    @classmethod
    def create(cls) -> "InfrastructureContext":
        topology = TopologyManager()
        graph = topology.graph
        all_nodes = tuple(graph.nodes())
        base_stations = tuple(
            node for node in all_nodes if str(node).endswith("0")
        )
        compute_nodes = tuple(
            node for node in all_nodes if not str(node).endswith("0")
        )
        node_resources = {
            node: {
                "total": float(
                    graph.nodes[node].get("capacity", config.DEFAULT_NODE_CPU)
                    or config.DEFAULT_NODE_CPU
                ),
                "used": 0.0,
            }
            for node in compute_nodes
        }
        return cls(
            topology,
            PricingManager(graph),
            all_nodes,
            base_stations,
            compute_nodes,
            node_resources,
        )
