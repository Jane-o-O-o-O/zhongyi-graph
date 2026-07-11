from __future__ import annotations

from dataclasses import dataclass
import math

import networkx as nx

from app.models.graph import GraphEdge, GraphNode

try:
    from graspologic.partition import hierarchical_leiden as _hierarchical_leiden
except ImportError:
    _hierarchical_leiden = None


LOW_VALUE_LABELS = {"Dose", "Alias", "Source", "Category", "DistributionArea"}
DEFAULT_LEIDEN_MAX_CLUSTER_SIZE = 12
DEFAULT_LEIDEN_SEED = 0xDEADBEEF

LABEL_WEIGHTS = {
    "Formula": 1.45,
    "Prescription": 1.25,
    "Herb": 1.15,
    "Symptom": 1.15,
    "Syndrome": 1.15,
    "Treatment": 1.1,
    "Indication": 0.95,
    "Function": 0.9,
    "Meridian": 0.65,
    "Flavor": 0.6,
    "Property": 0.6,
    "Dosage": 0.18,
    "Dose": 0.12,
    "Alias": 0.25,
    "Source": 0.22,
    "TextSource": 0.22,
    "ExternalSource": 0.22,
    "DistributionArea": 0.28,
    "Category": 0.08,
    "Evidence": 0.08,
}

RELATION_WEIGHTS = {
    "COMPOSED_OF": 1.25,
    "HAS_PRESCRIPTION": 1.1,
    "TREATS": 1.0,
    "HAS_FUNCTION": 0.85,
    "MANIFESTS_AS": 1.0,
    "RECOMMENDS_TREATMENT": 1.0,
    "ENTERS_MERIDIAN": 0.45,
    "HAS_FLAVOR": 0.4,
    "HAS_PROPERTY": 0.4,
    "DISTRIBUTED_IN": 0.22,
    "HAS_ALIAS": 0.16,
    "FROM_SOURCE": 0.12,
    "HAS_DOSE": 0.08,
    "INCLUDES": 0.05,
}


@dataclass(frozen=True)
class NodeAnalytics:
    node_id: str
    degree: int
    pagerank: float
    component_id: int
    community_id: int
    community_levels: dict[str, int]
    visual_weight: float
    label_weight: float


@dataclass(frozen=True)
class GraphAnalyticsResult:
    by_node_id: dict[str, NodeAnalytics]

    def apply_to_nodes(self, nodes: list[GraphNode]) -> None:
        for node in nodes:
            analytics = self.by_node_id.get(node.id)
            if not analytics:
                continue
            node.properties.update(
                {
                    "degree": analytics.degree,
                    "pagerank": analytics.pagerank,
                    "component_id": analytics.component_id,
                    "community_id": analytics.community_id,
                    "community_levels": _community_level_tags(analytics.community_levels),
                    "visual_weight": analytics.visual_weight,
                    "label_weight": analytics.label_weight,
                }
            )


class GraphAnalyticsService:
    def analyze(self, nodes: list[GraphNode], edges: list[GraphEdge]) -> GraphAnalyticsResult:
        nodes_by_id = {node.id: node for node in nodes}
        graph = nx.Graph()
        for node in nodes:
            graph.add_node(node.id)
        for edge in edges:
            if edge.source not in nodes_by_id or edge.target not in nodes_by_id:
                continue
            graph.add_edge(
                edge.source,
                edge.target,
                weight=self.edge_weight(edge, nodes_by_id),
            )

        pagerank = _pagerank(graph)
        component_by_node = _components(graph)
        community_by_node, community_levels_by_node = _communities(graph)
        max_pagerank = max(pagerank.values(), default=0.0)

        by_node_id: dict[str, NodeAnalytics] = {}
        for node in nodes:
            degree = int(graph.degree(node.id))
            label_weight = self.label_weight(node.label)
            pagerank_value = float(pagerank.get(node.id, 0.0))
            normalized_rank = pagerank_value / max_pagerank if max_pagerank > 0 else 0.0
            visual_weight = label_weight * (math.log1p(degree) + 1.0 + normalized_rank)
            by_node_id[node.id] = NodeAnalytics(
                node_id=node.id,
                degree=degree,
                pagerank=round(pagerank_value, 10),
                component_id=component_by_node.get(node.id, 0),
                community_id=community_by_node.get(node.id, component_by_node.get(node.id, 0)),
                community_levels=community_levels_by_node.get(
                    node.id,
                    {"0": community_by_node.get(node.id, component_by_node.get(node.id, 0))},
                ),
                visual_weight=round(visual_weight, 6),
                label_weight=label_weight,
            )
        return GraphAnalyticsResult(by_node_id=by_node_id)

    @staticmethod
    def label_weight(label: str) -> float:
        return LABEL_WEIGHTS.get(label, 1.0)

    @staticmethod
    def relation_weight(relation: str) -> float:
        return RELATION_WEIGHTS.get(relation, 1.0)

    @classmethod
    def edge_weight(cls, edge: GraphEdge, nodes_by_id: dict[str, GraphNode]) -> float:
        source = nodes_by_id.get(edge.source)
        target = nodes_by_id.get(edge.target)
        source_weight = cls.label_weight(source.label) if source else 1.0
        target_weight = cls.label_weight(target.label) if target else 1.0
        semantic_weight = math.sqrt(source_weight * target_weight)
        return round(max(0.01, semantic_weight * cls.relation_weight(edge.relation)), 6)

    @classmethod
    def retrieval_edge_weight(cls, edge: GraphEdge) -> float:
        return round(max(0.01, cls.relation_weight(edge.relation)), 6)


def _pagerank(graph: nx.Graph) -> dict[str, float]:
    if not graph.nodes:
        return {}
    try:
        return nx.pagerank(graph, weight="weight", max_iter=100, tol=1.0e-06)
    except nx.NetworkXException:
        value = 1.0 / graph.number_of_nodes()
        return {str(node): value for node in graph.nodes}


def _components(graph: nx.Graph) -> dict[str, int]:
    components = [
        sorted(component)
        for component in nx.connected_components(graph)
    ]
    components.sort(key=lambda component: (-len(component), component[0] if component else ""))
    component_by_node: dict[str, int] = {}
    for component_id, component in enumerate(components):
        for node_id in component:
            component_by_node[node_id] = component_id
    return component_by_node


def _communities(graph: nx.Graph) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    community_by_node: dict[str, int] = {}
    community_levels_by_node: dict[str, dict[str, int]] = {}
    next_community_id = 0
    next_level_ids: dict[int, int] = {}
    components = [
        graph.subgraph(component).copy()
        for component in nx.connected_components(graph)
    ]
    components.sort(
        key=lambda component: (
            -component.number_of_nodes(),
            sorted(component.nodes)[0] if component.nodes else "",
        )
    )
    for component in components:
        leiden_levels = _leiden_level_communities(component)
        if leiden_levels is not None:
            communities = leiden_levels[0]
        elif component.number_of_nodes() <= 6 or component.number_of_edges() == 0:
            communities = [sorted(component.nodes)]
        else:
            try:
                raw_communities = nx.community.greedy_modularity_communities(
                    component,
                    weight="weight",
                )
                communities = [sorted(community) for community in raw_communities]
            except nx.NetworkXException:
                communities = [sorted(component.nodes)]
        communities.sort(key=lambda community: (-len(community), community[0] if community else ""))
        for community in communities:
            assigned_community_id = next_community_id
            for node_id in community:
                community_by_node[node_id] = assigned_community_id
                community_levels_by_node.setdefault(node_id, {})["0"] = assigned_community_id
            next_community_id += 1
        if leiden_levels is None:
            continue
        for level, level_communities in sorted(leiden_levels.items()):
            if level == 0:
                continue
            next_level_id = next_level_ids.get(level, 0)
            for community in sorted(
                level_communities,
                key=lambda item: (-len(item), item[0] if item else ""),
            ):
                assigned_level_id = next_level_id
                for node_id in community:
                    community_levels_by_node.setdefault(node_id, {})[str(level)] = assigned_level_id
                next_level_id += 1
            next_level_ids[level] = next_level_id
    return community_by_node, community_levels_by_node


def _community_level_tags(community_levels: dict[str, int]) -> list[str]:
    return [
        f"{level}:{community_id}"
        for level, community_id in sorted(
            community_levels.items(),
            key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
        )
    ]


def _leiden_level_communities(graph: nx.Graph) -> dict[int, list[list[str]]] | None:
    if _hierarchical_leiden is None or graph.number_of_edges() == 0:
        return None
    try:
        partitions = _hierarchical_leiden(
            graph,
            max_cluster_size=DEFAULT_LEIDEN_MAX_CLUSTER_SIZE,
            random_seed=DEFAULT_LEIDEN_SEED,
        )
    except Exception:
        return None

    clusters: dict[int, dict[int, list[str]]] = {}
    for partition in partitions:
        level = int(getattr(partition, "level", 0))
        node_id = str(getattr(partition, "node", ""))
        if node_id not in graph:
            continue
        cluster_id = int(getattr(partition, "cluster"))
        clusters.setdefault(level, {}).setdefault(cluster_id, []).append(node_id)

    graph_nodes = set(str(node_id) for node_id in graph.nodes)
    if not clusters or 0 not in clusters:
        return None
    results: dict[int, list[list[str]]] = {}
    for level, level_clusters in sorted(clusters.items()):
        covered_nodes = {node_id for nodes in level_clusters.values() for node_id in nodes}
        if covered_nodes != graph_nodes:
            continue
        results[level] = [
            sorted(nodes)
            for _, nodes in sorted(level_clusters.items())
        ]
    return results if 0 in results else None
