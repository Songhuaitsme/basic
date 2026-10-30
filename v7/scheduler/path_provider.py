"""V7 bounded static-path caches; preserve the baseline's equal-cost tie choice."""

from bisect import bisect_left
from collections import OrderedDict
import math

import networkx as nx

from v1.domain.units import positive_finite
from v1.scheduler.path_provider import StaticPathProvider
from v1.scheduler.transmission import build_path_spec


class CachedStaticPathProvider(StaticPathProvider):
    optimization_version = "capacity-class-lru-unique-source-v1"

    def __init__(self, graph, max_paths_per_target=1, *, path_capacity=16384,
                 graph_capacity=8, source_capacity=64, single_source_enabled=True):
        super().__init__(graph, max_paths_per_target)
        for name, capacity in (("path_capacity", path_capacity), ("graph_capacity", graph_capacity), ("source_capacity", source_capacity)):
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, capacity)
        self.single_source_enabled = bool(single_source_enabled)
        self.topology_revision = -1
        self.replace_topology(graph)

    def replace_topology(self, graph):
        """Explicit topology update; never hash or scan the graph per lookup.

        Runtime topology is static. Work on a private frozen snapshot so an
        external graph edit cannot silently poison cached PathSpecs. To adopt
        edits to capacity, cost, distance or connectivity, call this method.
        """
        self.graph = nx.freeze(graph.copy())
        self.topology_revision += 1
        self._capacity_boundaries = sorted({
            float(data.get("capacity", 0.0)) for _, _, data in self.graph.edges(data=True)
            if math.isfinite(float(data.get("capacity", 0.0)))
        })
        self.clear_cache()

    def clear_cache(self):
        self._paths = OrderedDict()
        self._graphs = OrderedDict()
        self._sources = OrderedDict()
        self.stats = {name: 0 for name in (
            "path_cache_hits", "path_cache_misses", "negative_cache_hits",
            "filtered_graph_hits", "filtered_graph_builds", "path_searches",
            "single_source_runs", "single_source_served", "tie_or_unsafe_fallbacks",
            "pathspec_builds", "bulk_queries",
        )}

    def __getstate__(self):
        state = dict(self.__dict__)
        # Transient searches/graphs must not inflate training checkpoints.
        for name in ("_paths", "_graphs", "_sources", "stats"):
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.clear_cache()

    @staticmethod
    def _remember(cache, key, value, capacity):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > capacity:
            cache.popitem(last=False)
        return value

    def _bandwidth_class(self, bandwidth):
        # Exact equivalence, not rounding: two demands in the same interval
        # remove exactly the same edges under the original capacity < demand.
        return bisect_left(self._capacity_boundaries, bandwidth)

    @staticmethod
    def _safe_single_source(graph):
        # On strictly positive integral weights, all path sums are exact and
        # the predecessor DAG can prove uniqueness. Float summation, zero-cost
        # cycles, and ties retain the original per-target algorithm.
        maximum = 0.0
        for _, _, data in graph.edges(data=True):
            raw_weight = data.get("cost", 1.0)
            if not isinstance(raw_weight, (int, float)):
                return False
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight <= 0 or not weight.is_integer():
                return False
            maximum = max(maximum, weight)
        return maximum * max(0, len(graph) - 1) <= 2 ** 52

    def _filtered_graph(self, bandwidth, band):
        if band in self._graphs:
            self._graphs.move_to_end(band)
            self.stats["filtered_graph_hits"] += 1
            return self._graphs[band]
        graph = self.graph.copy()
        graph.remove_edges_from([
            (u, v) for u, v, data in graph.edges(data=True)
            if float(data.get("capacity", 0.0)) < bandwidth
        ])
        safe = self._safe_single_source(graph)
        self.stats["filtered_graph_builds"] += 1
        return self._remember(self._graphs, band, (nx.freeze(graph), safe), self.graph_capacity)

    def _unique_source_path(self, graph, band, source, target):
        key = (band, source)
        if key in self._sources:
            self._sources.move_to_end(key)
            predecessors, distances, unique = self._sources[key]
        else:
            predecessors, distances = nx.dijkstra_predecessor_and_distance(graph, source, weight="cost")
            unique = {source: True}
            for node in sorted(distances, key=distances.__getitem__):
                if node == source:
                    continue
                parents = predecessors[node]
                unique[node] = len(parents) == 1 and unique.get(parents[0], False)
            self._remember(self._sources, key, (predecessors, distances, unique), self.source_capacity)
            self.stats["single_source_runs"] += 1
        if target not in distances:
            return []
        if not unique[target]:
            return None  # Equal-cost routes must use the old tie order.
        nodes = [target]
        while nodes[-1] != source:
            nodes.append(predecessors[nodes[-1]][0])
        nodes.reverse()
        return nodes

    def candidate_paths(self, source_node, target_node, bandwidth_demand_mbps):
        bandwidth = positive_finite("bandwidth_demand_mbps", bandwidth_demand_mbps)
        band = self._bandwidth_class(bandwidth)
        key = (source_node, target_node, band, self.max_paths_per_target, self.single_source_enabled)
        if key in self._paths:
            paths = self._paths[key]
            self._paths.move_to_end(key)
            self.stats["path_cache_hits"] += 1
            self.stats["negative_cache_hits"] += int(not paths)
            return list(paths)  # Callers cannot mutate the cached container.
        self.stats["path_cache_misses"] += 1
        paths = []
        if source_node == target_node:
            if source_node in self.graph:
                paths = [build_path_spec(self.graph, [source_node])]
        else:
            graph, safe = self._filtered_graph(bandwidth, band)
            if source_node in graph and target_node in graph:
                nodes = None
                if self.single_source_enabled and safe and self.max_paths_per_target == 1:
                    nodes = self._unique_source_path(graph, band, source_node, target_node)
                if nodes is not None:
                    if nodes:
                        paths = [build_path_spec(self.graph, nodes)]
                    self.stats["single_source_served"] += 1
                else:
                    self.stats["tie_or_unsafe_fallbacks"] += 1
                    self.stats["path_searches"] += 1
                    try:
                        for nodes in nx.shortest_simple_paths(graph, source_node, target_node, weight="cost"):
                            paths.append(build_path_spec(self.graph, nodes))
                            if len(paths) >= self.max_paths_per_target:
                                break
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        paths = []
        self.stats["pathspec_builds"] += len(paths)
        self._remember(self._paths, key, tuple(paths), self.path_capacity)
        return paths

    def candidate_paths_many(self, source_node, target_nodes, bandwidth_demand_mbps):
        self.stats["bulk_queries"] += 1
        return tuple((target, self.candidate_paths(source_node, target, bandwidth_demand_mbps)) for target in target_nodes)

    def cache_statistics(self):
        return dict(self.stats, path_entries=len(self._paths), graph_entries=len(self._graphs),
                    source_entries=len(self._sources), topology_revision=self.topology_revision)
