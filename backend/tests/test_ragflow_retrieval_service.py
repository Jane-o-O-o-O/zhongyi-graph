from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.services.llm import LlmClient
from app.services.model_clients import EmbeddingClient, RerankClient
from app.services.ragflow_compat.doc_store import RagflowDocStore
from app.services.ragflow_compat.fulltext import RagflowFulltextRetriever
from app.services.ragflow_compat.kg_search import RagflowKgSearch
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.retrieval_service import RagflowCompatibleRetrievalService
from app.services.ragflow_compat.schemas import (
    RetrievalChunk,
    RetrievalDocument,
    RetrievalKgEntity,
    RetrievalKgRelation,
)


def test_ragflow_compatible_retrieval_service_returns_query_response_with_diagnostics():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_documents(
        [RetrievalDocument(doc_id="doc", source_id="doc", filename="doc.txt")]
    )
    repository.replace_chunks(
        [
            RetrievalChunk(
                chunk_id="chunk:doc:0001",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=1,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为心脾两虚，常用归脾汤。",
                content_with_weight="不寐 失眠 心脾两虚 归脾汤 失眠可辨为心脾两虚，常用归脾汤。",
                content_ltks="失眠 心脾两虚 归脾汤",
                token_count=18,
            )
        ]
    )
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id="entity:失眠",
                entity_name="失眠",
                entity_type="Symptom",
                source_node_id="symptom:失眠",
                content_with_weight='{"description":"失眠 Symptom"}',
                description="失眠 Symptom",
                rank_flt=2.0,
                n_hop_with_weight=[
                    {"path": ["失眠", "心脾两虚", "归脾汤"], "weights": [3, 4]}
                ],
                evidence_chunk_ids=["chunk:doc:0001"],
            ),
            RetrievalKgEntity(
                entity_id="entity:心脾两虚",
                entity_name="心脾两虚",
                entity_type="Syndrome",
                source_node_id="syndrome:心脾两虚",
                content_with_weight='{"description":"心脾两虚 Syndrome"}',
                description="心脾两虚 Syndrome",
                rank_flt=2.0,
                evidence_chunk_ids=["chunk:doc:0001"],
            ),
        ]
    )
    repository.replace_kg_relations(
        [
            RetrievalKgRelation(
                relation_id="relation:1",
                from_entity_kwd="失眠",
                to_entity_kwd="心脾两虚",
                relation_type="MANIFESTS_AS",
                display="可辨为",
                content_with_weight="失眠 可辨为 心脾两虚",
                weight_int=2,
                evidence_chunk_ids=["chunk:doc:0001"],
            )
        ]
    )
    doc_store = RagflowDocStore(repository, EmbeddingClient.demo())
    service = RagflowCompatibleRetrievalService(
        repository=repository,
        fulltext_retriever=RagflowFulltextRetriever(
            doc_store=doc_store,
            rerank_client=RerankClient.demo(),
        ),
        kg_search=RagflowKgSearch(doc_store),
        llm_client=LlmClient.demo(),
    )

    response = service.answer("睡不着从哪些证候分析？")

    assert response.intent == "symptom_inquiry"
    assert "失眠" in response.entities
    assert response.graph_edges
    assert response.evidence
    assert response.diagnostics["retrieval_engine"] == "ragflow_compat"
    assert response.diagnostics["chunk_hits"] == 1


def test_ragflow_compatible_retrieval_status_includes_readiness_report():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    doc_store = RagflowDocStore(repository, EmbeddingClient.demo())
    service = RagflowCompatibleRetrievalService(
        repository=repository,
        fulltext_retriever=RagflowFulltextRetriever(
            doc_store=doc_store,
            rerank_client=RerankClient.demo(),
        ),
        kg_search=RagflowKgSearch(doc_store),
        llm_client=LlmClient.demo(),
    )

    status = service.status()

    assert "readiness" in status
    assert status["readiness"]["ready"] is False
    assert "chunk_token_buckets" in status["readiness"]
    assert "vector_coverage" in status["readiness"]


def test_ragflow_compatible_retrieval_status_warns_when_qdrant_points_mismatch():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    doc_store = RagflowDocStore(repository, EmbeddingClient.demo())
    service = RagflowCompatibleRetrievalService(
        repository=repository,
        fulltext_retriever=RagflowFulltextRetriever(
            doc_store=doc_store,
            rerank_client=RerankClient.demo(),
        ),
        kg_search=RagflowKgSearch(doc_store),
        llm_client=LlmClient.demo(),
        qdrant_stats_provider=lambda: {"points_count": 3025, "status": "green"},
    )

    status = service.status()

    assert status["qdrant"] == {
        "points_count": 3025,
        "status": "green",
        "postgres_embedded_points": 0,
        "point_count_matches_postgres": False,
    }
    assert "qdrant_point_count_mismatch" in status["readiness"]["warnings"]


def test_ragflow_compatible_retrieval_status_allows_missing_qdrant_collection_when_no_vectors():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    doc_store = RagflowDocStore(repository, EmbeddingClient.demo())
    service = RagflowCompatibleRetrievalService(
        repository=repository,
        fulltext_retriever=RagflowFulltextRetriever(
            doc_store=doc_store,
            rerank_client=RerankClient.demo(),
        ),
        kg_search=RagflowKgSearch(doc_store),
        llm_client=LlmClient.demo(),
        qdrant_stats_provider=lambda: (_ for _ in ()).throw(RuntimeError("not found")),
    )

    status = service.status()

    assert status["qdrant"]["postgres_embedded_points"] == 0
    assert status["qdrant"]["point_count_matches_postgres"] is True
    assert "qdrant_point_count_mismatch" not in status["readiness"]["warnings"]
