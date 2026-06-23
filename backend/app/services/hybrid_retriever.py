from dataclasses import dataclass

from app.models.graph import GraphEdge, GraphNode
from app.services.graph_service import GraphService
from app.services.model_clients import RerankClient
from app.services.vector_service import VectorIndexService

SYMPTOM_PATH_LABELS = ("Symptom", "Syndrome", "Treatment", "Formula", "Herb")
SYMPTOM_PATH_RELATIONS = (
    "MANIFESTS_AS",
    "RECOMMENDS_TREATMENT",
    "RECOMMENDS_FORMULA",
    "COMPOSED_OF",
)
FORMULA_QUERY_TERMS = ("汤", "方", "方剂", "组成", "主治")
HERB_QUERY_TERMS = ("药", "中药", "功效", "归经", "性味")


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
        keyword_nodes, _ = self.graph_service.related_to_terms(terms)
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

        nodes, edges = self.graph_service.related_to_node_ids(anchor_ids)
        if intent == "symptom_inquiry":
            nodes, edges, seed_ids = self._symptom_clinical_path(nodes, edges, anchor_ids)
        else:
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

    def _select_anchor_ids(
        self,
        intent: str,
        ranked_candidates: list[GraphNode],
        terms: list[str],
        top_k: int,
    ) -> list[str]:
        if intent == "formula_inquiry":
            anchor = self._direct_or_ranked_node("Formula", ranked_candidates, terms)
            return [anchor.id] if anchor else [node.id for node in ranked_candidates[:top_k]]
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
        direct_matches = [
            node
            for node in self.graph_service.nodes
            if node.label == label and any(term in node.name for term in terms)
        ]
        if direct_matches:
            return direct_matches[0]

        for node in ranked_candidates:
            if node.label == label:
                return node
        return None

    def _symptom_clinical_path(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        anchor_ids: list[str],
    ) -> tuple[list[GraphNode], list[GraphEdge], list[str]]:
        allowed_labels = set(SYMPTOM_PATH_LABELS)
        allowed_relations = set(SYMPTOM_PATH_RELATIONS)
        included_ids = set(anchor_ids)

        changed = True
        while changed:
            changed = False
            for edge in edges:
                if edge.relation not in allowed_relations:
                    continue
                if edge.source in included_ids and edge.target not in included_ids:
                    included_ids.add(edge.target)
                    changed = True
                if edge.target in included_ids and edge.source not in included_ids:
                    included_ids.add(edge.source)
                    changed = True

        selected_nodes = [
            node for node in nodes if node.id in included_ids and node.label in allowed_labels
        ]
        selected_ids = {node.id for node in selected_nodes}
        selected_edges = [
            edge
            for edge in edges
            if edge.relation in allowed_relations
            and edge.source in selected_ids
            and edge.target in selected_ids
        ]
        path_ids = [
            node.id
            for node in sorted(
                selected_nodes,
                key=lambda node: (SYMPTOM_PATH_LABELS.index(node.label), node.name),
            )
        ]
        return selected_nodes, selected_edges, path_ids

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
