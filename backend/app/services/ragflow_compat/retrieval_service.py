from __future__ import annotations

from collections.abc import Callable
import inspect

from app.models.query import QueryResponse
from app.services.llm import LlmClient
from app.services.ragflow_compat.evidence import assemble_evidence_cards
from app.services.ragflow_compat.fulltext import RagflowFulltextRetriever
from app.services.ragflow_compat.kg_search import RagflowKgSearch
from app.services.ragflow_compat.query import entity_keywords
from app.services.ragflow_compat.repository import RagflowRetrievalRepository


class RagflowCompatibleRetrievalService:
    def __init__(
        self,
        *,
        repository: RagflowRetrievalRepository,
        fulltext_retriever: RagflowFulltextRetriever,
        kg_search: RagflowKgSearch,
        llm_client: LlmClient,
        query_rewriter=None,
        qdrant_stats_provider: Callable[[], dict] | None = None,
    ):
        self.repository = repository
        self.fulltext_retriever = fulltext_retriever
        self.kg_search = kg_search
        self.llm_client = llm_client
        self.query_rewriter = query_rewriter
        self.qdrant_stats_provider = qdrant_stats_provider

    def answer(self, question: str, *, comm_topn: int = 1) -> QueryResponse:
        fulltext = self.fulltext_retriever.retrieve(question, top_k=8)
        fallback_answer_types = _infer_answer_types(question)
        fallback_entities = _entities_from_keywords(fulltext.keywords)
        rewrite = _query_rewrite(
            self.query_rewriter,
            question,
            type_pool=_type_pool(self.repository),
        )
        answer_type_keywords = rewrite["answer_type_keywords"] or fallback_answer_types
        entities_from_query = rewrite["entities_from_query"] or fallback_entities
        kg = self.kg_search.retrieve(
            question,
            answer_type_keywords=answer_type_keywords,
            entities_from_query=entities_from_query,
            comm_topn=comm_topn,
        )
        evidence = assemble_evidence_cards(fulltext.hits, kg.graph_edges)
        community_evidence = [
            report.summary or report.content_with_weight
            for report in kg.community_reports
        ]
        entities = _unique([entity.entity for entity in kg.entities] + entities_from_query)
        graph_paths = [
            f"{edge.source} -> {edge.display} -> {edge.target}"
            for edge in kg.graph_edges[:12]
        ]
        answer = self.llm_client.synthesize(
            question=question,
            entities=entities,
            evidence=[card.snippet for card in evidence] + community_evidence,
            graph_paths=graph_paths,
        )
        return QueryResponse(
            question=question,
            answer=answer,
            intent=_infer_intent(question),
            entities=entities,
            graph_nodes=kg.graph_nodes,
            graph_edges=kg.graph_edges,
            highlighted_path=_highlighted_path(kg.graph_edges, kg.graph_nodes),
            evidence=evidence,
            diagnostics={
                "retrieval_engine": "ragflow_compat",
                "query_rewrite_source": rewrite["source"],
                "keywords": fulltext.keywords,
                "answer_type_keywords": answer_type_keywords,
                "entities_from_query": entities_from_query,
                "chunk_hits": len(fulltext.hits),
                "kg_entities": len(kg.entities),
                "kg_relations": len(kg.relations),
                "community_reports": len(kg.community_reports),
            },
        )

    def status(self) -> dict:
        audit = self.repository.audit()
        readiness = self.repository.readiness()
        qdrant = self._qdrant_status(audit)
        if qdrant and not qdrant["point_count_matches_postgres"]:
            readiness["warnings"] = [
                *readiness.get("warnings", []),
                "qdrant_point_count_mismatch",
            ]
        return {
            "retrieval_engine": "ragflow_compat",
            "documents": audit.documents,
            "chunks": audit.chunks,
            "chunks_with_vectors": audit.chunks_with_vectors,
            "chunks_failed_vectors": audit.chunks_failed_vectors,
            "kg_entities": audit.kg_entities,
            "kg_entities_with_vectors": audit.kg_entities_with_vectors,
            "kg_entities_failed_vectors": audit.kg_entities_failed_vectors,
            "kg_relations": audit.kg_relations,
            "community_reports": audit.community_reports,
            "graph_artifacts": audit.graph_artifacts,
            "kg_relations_with_vectors": audit.kg_relations_with_vectors,
            "kg_relations_failed_vectors": audit.kg_relations_failed_vectors,
            "short_chunks": audit.short_chunks,
            "long_chunks": audit.long_chunks,
            "qdrant": qdrant,
            "readiness": readiness,
        }

    def _qdrant_status(self, audit) -> dict | None:
        if not self.qdrant_stats_provider:
            return None
        postgres_embedded_points = _postgres_embedded_points(audit)
        try:
            stats = self.qdrant_stats_provider()
        except Exception:
            return {
                "available": False,
                "postgres_embedded_points": postgres_embedded_points,
                "point_count_matches_postgres": postgres_embedded_points == 0,
            }
        points_count = int(stats.get("points_count", 0))
        return {
            **stats,
            "points_count": points_count,
            "postgres_embedded_points": postgres_embedded_points,
            "point_count_matches_postgres": points_count == postgres_embedded_points,
        }


def _infer_answer_types(question: str) -> list[str]:
    type_keywords = []
    if any(term in question for term in ["证", "证候", "辨证"]):
        type_keywords.append("Syndrome")
    if any(term in question for term in ["方", "方剂", "汤"]):
        type_keywords.append("Formula")
    if any(term in question for term in ["药", "中药", "药味"]):
        type_keywords.append("Herb")
    return type_keywords


def _entities_from_keywords(keywords: list[str]) -> list[str]:
    return entity_keywords(keywords)


def _query_rewrite(query_rewriter, question: str, *, type_pool: dict[str, list[str]]) -> dict:
    if not query_rewriter:
        return {"source": "rules", "answer_type_keywords": [], "entities_from_query": []}
    try:
        payload = _call_query_rewriter(query_rewriter, question, type_pool)
    except Exception:
        return {"source": "rules", "answer_type_keywords": [], "entities_from_query": []}
    if not isinstance(payload, dict):
        return {"source": "rules", "answer_type_keywords": [], "entities_from_query": []}
    answer_type_keywords = _string_list(payload.get("answer_type_keywords"))[:3]
    entities_from_query = _string_list(payload.get("entities_from_query"))[:5]
    if not entities_from_query:
        entities_from_query = _string_list(payload.get("entities"))[:5]
    if not answer_type_keywords and not entities_from_query:
        return {"source": "rules", "answer_type_keywords": [], "entities_from_query": []}
    return {
        "source": "llm",
        "answer_type_keywords": answer_type_keywords,
        "entities_from_query": entities_from_query,
    }


def _call_query_rewriter(query_rewriter, question: str, type_pool: dict[str, list[str]]) -> dict:
    extract_query = query_rewriter.extract_query
    signature = inspect.signature(extract_query)
    if "type_pool" in signature.parameters:
        return extract_query(question, type_pool=type_pool)
    return extract_query(question)


def _type_pool(repository: RagflowRetrievalRepository) -> dict[str, list[str]]:
    return {
        sample.entity_type: sample.sample_entities
        for sample in repository.list_type_samples()
        if sample.sample_entities
    }


def _infer_intent(question: str) -> str:
    if any(term in question for term in ["汤", "方", "方剂"]):
        return "formula_inquiry"
    if any(term in question for term in ["药", "中药", "功效", "归经"]):
        return "herb_inquiry"
    return "symptom_inquiry"


def _highlighted_path(graph_edges, graph_nodes) -> list[str]:
    path = []
    for edge in graph_edges[:6]:
        path.extend([edge.source, edge.target])
    if not path:
        path = [node.id for node in graph_nodes[:5]]
    return _unique(path)


def _unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return _unique(str(item).strip() for item in value if str(item).strip())


def _postgres_embedded_points(audit) -> int:
    return (
        audit.chunks_with_vectors
        + audit.kg_entities_with_vectors
        + audit.kg_relations_with_vectors
    )
