from app.data.sample_graph import SAMPLE_EDGES, SAMPLE_NODES
from app.models.graph import GraphEdge, GraphNode


class GraphService:
    def __init__(self, nodes: list[GraphNode], edges: list[GraphEdge]):
        self.nodes = nodes
        self.edges = edges

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
        degree_by_id = {node.id: 0 for node in self.nodes}
        for edge in self.edges:
            if edge.source in degree_by_id:
                degree_by_id[edge.source] += 1
            if edge.target in degree_by_id:
                degree_by_id[edge.target] += 1

        selected_nodes = sorted(
            self.nodes,
            key=lambda node: (-degree_by_id.get(node.id, 0), node.label, node.name, node.id),
        )[:max_nodes]
        visible_ids = {node.id for node in selected_nodes}
        selected_edges = [
            edge
            for edge in self.edges
            if edge.source in visible_ids and edge.target in visible_ids
        ][:max_edges]
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
