from __future__ import annotations

from collections.abc import Callable
import csv
import inspect
import io
import json

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

    def answer(
        self,
        question: str,
        *,
        comm_topn: int = 1,
        max_token: int = 8196,
        ent_topn: int = 6,
        rel_topn: int = 6,
        ent_sim_threshold: float = 0.0,
        rel_sim_threshold: float = 0.0,
    ) -> QueryResponse:
        fulltext = self.fulltext_retriever.retrieve(question, top_k=8)
        fallback_answer_types = _infer_answer_types(question)
        fallback_entities = _entities_from_keywords(fulltext.keywords)
        type_pool = _type_pool(self.repository)
        rewrite = _query_rewrite(
            self.query_rewriter,
            question,
            type_pool=type_pool,
        )
        answer_type_keywords = rewrite["answer_type_keywords"] or fallback_answer_types
        entities_from_query = rewrite["entities_from_query"] or fallback_entities
        kg = self.kg_search.retrieve(
            question,
            answer_type_keywords=answer_type_keywords,
            entities_from_query=entities_from_query,
            comm_topn=comm_topn,
            ent_topn=ent_topn,
            rel_topn=rel_topn,
            ent_sim_threshold=ent_sim_threshold,
            rel_sim_threshold=rel_sim_threshold,
        )
        evidence = assemble_evidence_cards(fulltext.hits, kg.graph_edges)
        kg_context = _build_kg_context(kg, max_token=max_token)
        retrieval_trace = _retrieval_trace(
            question=question,
            rewrite=rewrite,
            type_pool=type_pool,
            fallback_answer_types=fallback_answer_types,
            fallback_entities=fallback_entities,
            answer_type_keywords=answer_type_keywords,
            entities_from_query=entities_from_query,
            fulltext=fulltext,
            kg=kg,
            kg_context=kg_context,
            max_token=max_token,
            ent_topn=ent_topn,
            rel_topn=rel_topn,
            comm_topn=comm_topn,
            ent_sim_threshold=ent_sim_threshold,
            rel_sim_threshold=rel_sim_threshold,
        )
        kg_evidence = [kg_context] if kg_context else []
        entities = _unique([entity.entity for entity in kg.entities] + entities_from_query)
        graph_paths = [
            f"{edge.source} -> {edge.display} -> {edge.target}"
            for edge in kg.graph_edges[:12]
        ]
        answer = self.llm_client.synthesize(
            question=question,
            entities=entities,
            evidence=[card.snippet for card in evidence] + kg_evidence,
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
                "kg_context_docnm": "Related content in Knowledge Graph",
                "kg_context_max_token": max_token,
                "kg_context": kg_context,
                "retrieval_trace": retrieval_trace,
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


def _build_kg_context(kg, *, max_token: int = 8196) -> str:
    remaining = max(0, int(max_token))
    sections: list[str] = []
    entity_rows: list[dict[str, str]] = []
    for entity in kg.entities:
        row = {
            "Entity": entity.entity,
            "Score": f"{entity.score:.2f}",
            "Description": _description_text(entity.description),
        }
        row_tokens = _context_token_count(str(row))
        if remaining - row_tokens <= 0:
            break
        remaining -= row_tokens
        entity_rows.append(row)
    if entity_rows:
        sections.append("\n---- Entities ----\n" + _csv_rows(["Entity", "Score", "Description"], entity_rows))

    relation_rows: list[dict[str, str]] = []
    for relation in kg.relations:
        row = {
            "From Entity": relation.from_entity,
            "To Entity": relation.to_entity,
            "Score": f"{relation.score:.2f}",
            "Description": _description_text(relation.description),
        }
        row_tokens = _context_token_count(str(row))
        if remaining - row_tokens <= 0:
            break
        remaining -= row_tokens
        relation_rows.append(row)
    if relation_rows:
        sections.append(
            "\n---- Relations ----\n"
            + _csv_rows(["From Entity", "To Entity", "Score", "Description"], relation_rows)
        )

    community_texts: list[str] = []
    for index, report in enumerate(kg.community_reports, start=1):
        report_text = _community_report_text(index, report)
        report_tokens = _context_token_count(report_text)
        if remaining - report_tokens <= 0:
            break
        remaining -= report_tokens
        community_texts.append(report_text)
    if community_texts:
        sections.append("\n---- Community Report ----\n" + "\n".join(community_texts))

    return "".join(sections)


def _retrieval_trace(
    *,
    question: str,
    rewrite: dict,
    type_pool: dict[str, list[str]],
    fallback_answer_types: list[str],
    fallback_entities: list[str],
    answer_type_keywords: list[str],
    entities_from_query: list[str],
    fulltext,
    kg,
    kg_context: str,
    max_token: int,
    ent_topn: int,
    rel_topn: int,
    comm_topn: int,
    ent_sim_threshold: float,
    rel_sim_threshold: float,
) -> dict:
    return {
        "question": question,
        "rewrite": {
            "source": rewrite["source"],
            "type_pool": type_pool,
            "fallback_answer_type_keywords": fallback_answer_types,
            "fallback_entities": fallback_entities,
            "answer_type_keywords": answer_type_keywords,
            "entities_from_query": entities_from_query,
        },
        "fulltext": {
            "keywords": fulltext.keywords,
            "hits": [_trace_chunk_hit(hit) for hit in fulltext.hits],
        },
        "kg": {
            "parameters": {
                "ent_topn": ent_topn,
                "rel_topn": rel_topn,
                "comm_topn": comm_topn,
                "ent_sim_threshold": ent_sim_threshold,
                "rel_sim_threshold": rel_sim_threshold,
            },
            "entities": [_trace_entity(entity) for entity in kg.entities],
            "relations": [_trace_relation(relation) for relation in kg.relations],
            "community_reports": [
                _trace_community_report(report)
                for report in kg.community_reports
            ],
            "graph_nodes": len(kg.graph_nodes),
            "graph_edges": len(kg.graph_edges),
        },
        "context": {
            "max_token": max_token,
            "length": len(kg_context),
            "sections": _kg_context_sections(kg_context),
        },
    }


def _trace_chunk_hit(hit) -> dict:
    chunk = hit.chunk
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source_id": chunk.source_id,
        "title": chunk.title,
        "score": _round_score(hit.score),
        "keyword_score": _round_score(hit.keyword_score),
        "vector_score": _round_score(hit.vector_score),
        "rerank_score": _round_score(hit.rerank_score),
    }


def _trace_entity(entity) -> dict:
    return {
        "entity": entity.entity,
        "score": _round_score(entity.score),
        "description": _description_text(entity.description),
    }


def _trace_relation(relation) -> dict:
    return {
        "from_entity": relation.from_entity,
        "to_entity": relation.to_entity,
        "score": _round_score(relation.score),
        "description": _description_text(relation.description),
    }


def _trace_community_report(report) -> dict:
    return {
        "report_id": report.report_id,
        "title": report.title,
        "weight": _round_score(report.weight_flt),
        "entities": report.entities_kwd,
        "source_id": report.source_id,
    }


def _kg_context_sections(kg_context: str) -> list[str]:
    sections: list[str] = []
    if "---- Entities ----" in kg_context:
        sections.append("entities")
    if "---- Relations ----" in kg_context:
        sections.append("relations")
    if "---- Community Report ----" in kg_context:
        sections.append("community_reports")
    return sections


def _csv_rows(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _community_report_text(index: int, report) -> str:
    report_content = report.summary or report.content_with_weight
    evidences = report.evidences
    try:
        parsed = json.loads(report.content_with_weight)
    except (TypeError, json.JSONDecodeError):
        parsed = {}
    if isinstance(parsed, dict):
        report_content = str(parsed.get("report") or parsed.get("summary") or report_content)
        evidences = str(parsed.get("evidences") or evidences)
    return f"# {index}. {report.title}\n## Content\n{report_content}\n## Evidences\n{evidences}\n"


def _description_text(value: str) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value
    if isinstance(parsed, dict):
        return str(parsed.get("description") or value)
    return value


def _context_token_count(value: str) -> int:
    return max(1, len(value))


def _round_score(value) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


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
