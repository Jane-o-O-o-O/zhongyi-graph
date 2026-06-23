from app.data.sample_graph import SAMPLE_EDGES, SAMPLE_NODES
from app.models.graph import GraphEdge, GraphNode


class GraphService:
    def __init__(self, nodes: list[GraphNode], edges: list[GraphEdge]):
        self.nodes = nodes
        self.edges = edges

    @classmethod
    def demo(cls) -> "GraphService":
        return cls(SAMPLE_NODES, SAMPLE_EDGES)

    def related_to_terms(self, terms: list[str]) -> tuple[list[GraphNode], list[GraphEdge]]:
        matched_ids = {
            node.id for node in self.nodes if any(term in node.name for term in terms)
        }
        expanded_ids = set(matched_ids)
        selected_edges: list[GraphEdge] = []
        for edge in self.edges:
            if edge.source in expanded_ids or edge.target in expanded_ids:
                selected_edges.append(edge)
                expanded_ids.add(edge.source)
                expanded_ids.add(edge.target)
        selected_nodes = [node for node in self.nodes if node.id in expanded_ids]
        return selected_nodes, selected_edges
