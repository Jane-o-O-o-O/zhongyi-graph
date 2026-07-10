from app.data.sample_graph import SAMPLE_EDGES, SAMPLE_NODES
from app.models.graph import GraphEdge, GraphNode
from app.services.graph_analytics_service import GraphAnalyticsService
from app.services.graph_community_summary_service import GraphCommunitySummaryService
from app.services.graph_entity_resolution_service import GraphEntityResolutionService


class GraphService:
    def __init__(self, nodes: list[GraphNode], edges: list[GraphEdge]):
        self.nodes = nodes
        self.edges = edges
        self.analytics = GraphAnalyticsService().analyze(nodes, edges)
        self.analytics.apply_to_nodes(self.nodes)
        self.entity_resolution = GraphEntityResolutionService().resolve(nodes, edges)
        self.entity_resolution.apply_to_nodes(self.nodes)
        self.community_summaries = GraphCommunitySummaryService().summarize(self.nodes)
        self.community_summaries.apply_to_nodes(self.nodes)

    @classmethod
    def demo(cls) -> "GraphService":
        return cls(SAMPLE_NODES, SAMPLE_EDGES)

    def matching_nodes(self, terms: list[str]) -> list[GraphNode]:
        return [node for node in self.nodes if any(term in node.name for term in terms)]

    def related_to_terms(self, terms: list[str]) -> tuple[list[GraphNode], list[GraphEdge]]:
        matched_ids = {node.id for node in self.matching_nodes(terms)}
        return self.related_to_node_ids(list(matched_ids))

    def related_to_node_ids(self, node_ids: list[str]) -> tuple[list[GraphNode], list[GraphEdge]]:
        matched_ids = set(node_ids)
        expanded_ids = set(matched_ids)
        selected_edge_ids: set[str] = set()
        changed = True
        while changed:
            changed = False
            for edge in self.edges:
                if edge.source in expanded_ids or edge.target in expanded_ids:
                    selected_edge_ids.add(edge.id)
                    before = len(expanded_ids)
                    expanded_ids.add(edge.source)
                    expanded_ids.add(edge.target)
                    changed = changed or len(expanded_ids) > before
        selected_nodes = [node for node in self.nodes if node.id in expanded_ids]
        selected_edges = [edge for edge in self.edges if edge.id in selected_edge_ids]
        return selected_nodes, selected_edges

    def overview(
        self,
        *,
        max_nodes: int = 3000,
        max_edges: int = 9000,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        selected_nodes = _balanced_overview_nodes(self.nodes, max_nodes=max_nodes)
        visible_ids = {node.id for node in selected_nodes}
        nodes_by_id = {node.id: node for node in self.nodes}
        selected_edges = [
            edge
            for edge in self.edges
            if edge.source in visible_ids and edge.target in visible_ids
        ]
        selected_edges = sorted(
            selected_edges,
            key=lambda edge: (
                -GraphAnalyticsService.edge_weight(edge, nodes_by_id),
                edge.relation,
                edge.id,
            ),
        )[:max_edges]
        return selected_nodes, selected_edges

    def neighborhood(
        self,
        node_ids: list[str],
        *,
        allowed_relations: set[str],
        terminal_labels: set[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 80,
        max_edges: int = 160,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        terminal_labels = terminal_labels or set()
        nodes_by_id = {node.id: node for node in self.nodes}
        node_labels = {node.id: node.label for node in self.nodes}
        selected_ids = set(node_ids)
        ordered_ids = [node_id for node_id in node_ids if node_id in nodes_by_id]
        frontier = set(ordered_ids)
        selected_edge_ids: set[str] = set()
        for _depth in range(max_depth):
            next_frontier: list[str] = []
            for edge in self.edges:
                if edge.relation not in allowed_relations:
                    continue
                if edge.source in frontier and edge.target not in selected_ids:
                    selected_edge_ids.add(edge.id)
                    selected_ids.add(edge.target)
                    ordered_ids.append(edge.target)
                    if node_labels.get(edge.target) not in terminal_labels:
                        next_frontier.append(edge.target)
                elif edge.target in frontier and edge.source not in selected_ids:
                    selected_edge_ids.add(edge.id)
                    selected_ids.add(edge.source)
                    ordered_ids.append(edge.source)
                    if node_labels.get(edge.source) not in terminal_labels:
                        next_frontier.append(edge.source)
                elif edge.source in selected_ids and edge.target in selected_ids:
                    selected_edge_ids.add(edge.id)
            if not next_frontier:
                break
            available = max_nodes - len(selected_ids)
            if available <= 0:
                break
            frontier = set(next_frontier[:available])

        selected_nodes = [nodes_by_id[node_id] for node_id in ordered_ids[:max_nodes]]
        visible_ids = {node.id for node in selected_nodes}
        selected_edges = [
            edge
            for edge in self.edges
            if edge.id in selected_edge_ids and edge.source in visible_ids and edge.target in visible_ids
        ][:max_edges]
        return selected_nodes, selected_edges


def _balanced_overview_nodes(nodes: list[GraphNode], *, max_nodes: int) -> list[GraphNode]:
    if max_nodes <= 0:
        return []
    groups: dict[int, list[GraphNode]] = {}
    for node in nodes:
        community_id = int(node.properties.get("community_id", 0))
        groups.setdefault(community_id, []).append(node)
    for group_nodes in groups.values():
        group_nodes.sort(key=_overview_node_sort_key)

    community_order = sorted(
        groups,
        key=lambda community_id: (
            -sum(
                float(node.properties.get("visual_weight", 0.0))
                for node in groups[community_id][: min(8, len(groups[community_id]))]
            ),
            community_id,
        ),
    )
    active_communities = community_order[:max_nodes]
    positions = {community_id: 0 for community_id in active_communities}
    selected: list[GraphNode] = []
    selected_ids: set[str] = set()
    while len(selected) < max_nodes:
        changed = False
        for community_id in active_communities:
            position = positions[community_id]
            group_nodes = groups[community_id]
            if position >= len(group_nodes):
                continue
            node = group_nodes[position]
            positions[community_id] += 1
            if node.id in selected_ids:
                continue
            selected.append(node)
            selected_ids.add(node.id)
            changed = True
            if len(selected) >= max_nodes:
                break
        if not changed:
            break
    return selected


def _overview_node_sort_key(node: GraphNode):
    return (
        -float(node.properties.get("visual_weight", 0.0)),
        -int(node.properties.get("degree", 0)),
        node.label,
        node.name,
        node.id,
    )
