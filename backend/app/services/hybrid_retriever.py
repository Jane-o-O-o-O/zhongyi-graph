from dataclasses import dataclass

from app.models.graph import GraphEdge, GraphNode
from app.services.graph_service import GraphService
from app.services.model_clients import RerankClient
from app.services.vector_service import VectorIndexService

FORMULA_QUERY_TERMS = ("汤", "方", "方剂", "组成", "主治")
HERB_QUERY_TERMS = ("药", "中药", "功效", "归经", "性味")
FORMULA_NEIGHBOR_RELATIONS = {
    "HAS_PRESCRIPTION",
    "COMPOSED_OF",
    "HAS_DOSE",
    "TREATS",
    "HAS_ALIAS",
    "FROM_SOURCE",
}
HERB_NEIGHBOR_RELATIONS = {
    "COMPOSED_OF",
    "HAS_FUNCTION",
    "TREATS",
    "HAS_ALIAS",
    "HAS_PROPERTY",
    "HAS_FLAVOR",
    "ENTERS_MERIDIAN",
    "DISTRIBUTED_IN",
    "FROM_SOURCE",
}
DEFAULT_NEIGHBOR_RELATIONS = {
    "MANIFESTS_AS",
    "RECOMMENDS_TREATMENT",
    "RECOMMENDS_FORMULA",
    "COMPOSED_OF",
    "TREATS",
    "RELATED_TO",
}
RELATED_NEIGHBOR_RELATIONS = (
    DEFAULT_NEIGHBOR_RELATIONS
    | FORMULA_NEIGHBOR_RELATIONS
    | HERB_NEIGHBOR_RELATIONS
    | {"HAS_PRESCRIPTION", "HAS_DOSE", "INCLUDES"}
)
FORMULA_TERMINAL_LABELS = {"Alias", "Dose", "Indication", "Source"}
HERB_TERMINAL_LABELS = {
    "Alias",
    "DistributionArea",
    "Flavor",
    "Function",
    "Indication",
    "Meridian",
    "Property",
    "Source",
}


@dataclass(frozen=True)
class RetrievalResult:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    evidence_ids: list[str]
    seed_node_ids: list[str]


class HybridRetriever:
    def __init__(
        self,
        graph_service: GraphService,
        vector_index: VectorIndexService,
        rerank_client: RerankClient,
    ):
        self.graph_service = graph_service
        self.vector_index = vector_index
        self.rerank_client = rerank_client

    def retrieve(self, question: str, terms: list[str], top_k: int = 8) -> RetrievalResult:
        keyword_nodes = self.graph_service.matching_nodes(terms)
        candidate_ids = {node.id for node in keyword_nodes}

        try:
            vector_hits = self.vector_index.search(question, top_k=20, content_types=["entity"])
        except Exception:
            vector_hits = []
        candidate_ids.update(
            hit.payload["node_id"] for hit in vector_hits if hit.payload.get("node_id")
        )

        candidates = [node for node in self.graph_service.nodes if node.id in candidate_ids]
        ranked_candidates = self._rerank_nodes(question, candidates)[:top_k]
        intent = _infer_intent(question)
        anchor_ids = self._select_anchor_ids(intent, ranked_candidates, terms, top_k)

        nodes, edges = self._retrieve_neighborhood(intent, anchor_ids)
        seed_ids = _ordered_visible_seed_ids(
            ranked_node_ids=[node.id for node in ranked_candidates],
            visible_nodes=nodes,
            anchor_ids=anchor_ids,
        )
        evidence_ids = _unique(
            evidence_id for edge in edges for evidence_id in edge.evidence_ids
        )
        return RetrievalResult(
            nodes=nodes,
            edges=edges,
            evidence_ids=evidence_ids,
            seed_node_ids=seed_ids,
        )

    def _retrieve_neighborhood(
        self,
        intent: str,
        anchor_ids: list[str],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        if not anchor_ids:
            return [], []
        if intent == "formula_inquiry":
            return self.graph_service.neighborhood(
                anchor_ids,
                allowed_relations=FORMULA_NEIGHBOR_RELATIONS,
                terminal_labels=FORMULA_TERMINAL_LABELS,
                max_depth=3,
                max_nodes=120,
                max_edges=240,
            )
        if intent == "herb_inquiry":
            return self.graph_service.neighborhood(
                anchor_ids,
                allowed_relations=HERB_NEIGHBOR_RELATIONS,
                terminal_labels=HERB_TERMINAL_LABELS,
                max_depth=2,
                max_nodes=120,
                max_edges=240,
            )
        return self.graph_service.neighborhood(
            anchor_ids,
            allowed_relations=RELATED_NEIGHBOR_RELATIONS,
            max_depth=3,
            max_nodes=120,
            max_edges=240,
        )

    def _select_anchor_ids(
        self,
        intent: str,
        ranked_candidates: list[GraphNode],
        terms: list[str],
        top_k: int,
    ) -> list[str]:
        if intent == "formula_inquiry":
            anchor = self._direct_or_ranked_node("Formula", ranked_candidates, terms)
            if anchor:
                return [anchor.id]
            prescription = self._direct_or_ranked_node("Prescription", ranked_candidates, terms)
            return [prescription.id] if prescription else [node.id for node in ranked_candidates[:top_k]]
        if intent == "herb_inquiry":
            anchor = self._direct_or_ranked_node("Herb", ranked_candidates, terms)
            return [anchor.id] if anchor else [node.id for node in ranked_candidates[:top_k]]

        direct_symptom = self._direct_or_ranked_node("Symptom", ranked_candidates, terms)
        if direct_symptom:
            return [direct_symptom.id]

        for label in ("Symptom", "Syndrome"):
            for node in ranked_candidates:
                if node.label == label:
                    return [node.id]

        return [ranked_candidates[0].id] if ranked_candidates else []

    def _direct_or_ranked_node(
        self,
        label: str,
        ranked_candidates: list[GraphNode],
        terms: list[str],
    ) -> GraphNode | None:
        for term in terms:
            for node in self.graph_service.nodes:
                if node.label == label and term in node.name:
                    return node

        for node in ranked_candidates:
            if node.label == label:
                return node
        return None

    def _rerank_nodes(self, question: str, nodes: list[GraphNode]) -> list[GraphNode]:
        if len(nodes) <= 1:
            return nodes
        documents = [f"{node.name} {node.label} {node.description}" for node in nodes]
        try:
            ranking = self.rerank_client.rerank(question, documents)
        except Exception:
            return nodes
        if not ranking:
            return nodes
        ranked_indices = [index for index, _score in ranking if 0 <= index < len(nodes)]
        ranked = [nodes[index] for index in ranked_indices]
        remaining = [node for index, node in enumerate(nodes) if index not in set(ranked_indices)]
        return ranked + remaining


def _infer_intent(question: str) -> str:
    if any(term in question for term in FORMULA_QUERY_TERMS):
        return "formula_inquiry"
    if any(term in question for term in HERB_QUERY_TERMS):
        return "herb_inquiry"
    return "symptom_inquiry"


def _ordered_visible_seed_ids(
    ranked_node_ids: list[str],
    visible_nodes: list[GraphNode],
    anchor_ids: list[str],
) -> list[str]:
    visible_ids = {node.id for node in visible_nodes}
    seed_ids = _unique(
        [node_id for node_id in anchor_ids + ranked_node_ids if node_id in visible_ids]
    )
    return seed_ids or [node.id for node in visible_nodes[:5]]


def _unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
