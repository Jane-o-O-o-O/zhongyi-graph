import httpx
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.services.model_clients import EmbeddingClient
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import (
    RetrievalChunk,
    RetrievalDocument,
    RetrievalKgEntity,
    RetrievalKgRelation,
)
from app.services.ragflow_compat.vector_sync import RagflowVectorSyncService


class FakeHttpClient:
    def __init__(self):
        self.requests = []

    def put(self, url, json):
        self.requests.append(("PUT", url, json))
        return httpx.Response(200, request=httpx.Request("PUT", url), json={"result": "ok"})

    def delete(self, url):
        self.requests.append(("DELETE", url, None))
        return httpx.Response(200, request=httpx.Request("DELETE", url), json={"result": "ok"})


def test_vector_sync_embeds_chunks_into_ragflow_collection_and_marks_embedded():
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
                chunk_id="chunk:1",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=1,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为心脾两虚。",
                content_with_weight="不寐 失眠可辨为心脾两虚。",
                token_count=12,
            )
        ]
    )
    http_client = FakeHttpClient()

    summary = RagflowVectorSyncService(
        repository=repository,
        embedding_client=EmbeddingClient.demo(dimensions=8),
        qdrant_url="http://qdrant:6333",
        collection="tcm_ragflow_retrieval",
        http_client=http_client,
    ).sync_missing(batch_size=10)

    assert summary["embedded"] == 1
    assert summary["failed"] == 0
    assert summary["embedded_chunks"] == 1
    assert summary["embedded_kg_entities"] == 0
    assert summary["embedded_kg_relations"] == 0
    assert http_client.requests[0][1].endswith("/collections/tcm_ragflow_retrieval")
    assert http_client.requests[1][1].endswith("/collections/tcm_ragflow_retrieval/points")
    point = http_client.requests[1][2]["points"][0]
    assert point["payload"]["content_type"] == "chunk"
    assert point["payload"]["chunk_id"] == "chunk:1"
    assert repository.audit().chunks_with_vectors == 1


def test_vector_sync_can_reset_qdrant_collection_before_rebuild():
    repository = RagflowRetrievalRepository(
        create_engine(
            "sqlite+pysqlite://",
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    )
    http_client = FakeHttpClient()

    summary = RagflowVectorSyncService(
        repository=repository,
        embedding_client=EmbeddingClient.demo(dimensions=8),
        qdrant_url="http://qdrant:6333",
        collection="tcm_ragflow_retrieval",
        http_client=http_client,
    ).reset_collection()

    assert summary == {"reset_collection": "tcm_ragflow_retrieval", "deleted": 1}
    assert http_client.requests == [
        ("DELETE", "http://qdrant:6333/collections/tcm_ragflow_retrieval", None)
    ]


def test_vector_sync_marks_failed_records_and_skips_them_in_missing_queries():
    class FailingEmbeddingClient:
        def embed(self, texts):
            raise RuntimeError("embedding failed")

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
                chunk_id="chunk:1",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=1,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为心脾两虚。",
                content_with_weight="不寐 失眠可辨为心脾两虚。",
                token_count=12,
            )
        ]
    )

    summary = RagflowVectorSyncService(
        repository=repository,
        embedding_client=FailingEmbeddingClient(),
        qdrant_url="http://qdrant:6333",
        collection="tcm_ragflow_retrieval",
        http_client=FakeHttpClient(),
    ).sync_missing(batch_size=1, limit=1, content_types=["chunk"])

    assert summary["embedded"] == 0
    assert summary["failed"] == 1
    assert summary["failed_chunks"] == 1
    assert summary["failure_error_RuntimeError"] == 1
    assert summary["failure_stage_embedding"] == 1
    assert summary["last_failure_error"] == "RuntimeError: embedding failed"
    assert summary["last_failure_stage"] == "embedding"
    assert repository.get_chunk("chunk:1").vector_status == "failed"
    assert repository.list_missing_vector_records(content_types=["chunk"], limit=1) == []


def test_vector_sync_retries_transient_embedding_failures_before_marking_failed():
    class TransientEmbeddingClient:
        def __init__(self):
            self.attempts = 0
            self.fallback = EmbeddingClient.demo(dimensions=8)

        def embed(self, texts):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("temporary embedding outage")
            return self.fallback.embed(texts)

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
                chunk_id="chunk:1",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=1,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为心脾两虚。",
                content_with_weight="不寐 失眠可辨为心脾两虚。",
                token_count=12,
            )
        ]
    )
    embedding_client = TransientEmbeddingClient()

    summary = RagflowVectorSyncService(
        repository=repository,
        embedding_client=embedding_client,
        qdrant_url="http://qdrant:6333",
        collection="tcm_ragflow_retrieval",
        http_client=FakeHttpClient(),
        max_attempts=2,
    ).sync_missing(batch_size=1, limit=1, content_types=["chunk"])

    assert summary["embedded"] == 1
    assert summary["failed"] == 0
    assert summary["retry_attempts"] == 1
    assert summary["retry_error_RuntimeError"] == 1
    assert summary["retry_stage_embedding"] == 1
    assert embedding_client.attempts == 2
    assert repository.get_chunk("chunk:1").vector_status == "embedded"


def test_vector_sync_does_not_mark_failed_after_qdrant_upsert_succeeds():
    class FailingStatusRepository:
        def __init__(self, repository):
            self.repository = repository

        def __getattr__(self, name):
            return getattr(self.repository, name)

        def update_chunk_vector_status(self, chunk_id, *, point_id, status):
            if status == "embedded":
                raise RuntimeError("postgres status update failed")
            return self.repository.update_chunk_vector_status(
                chunk_id,
                point_id=point_id,
                status=status,
            )

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
                chunk_id="chunk:1",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=1,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为心脾两虚。",
                content_with_weight="不寐 失眠可辨为心脾两虚。",
                token_count=12,
            )
        ]
    )
    http_client = FakeHttpClient()

    summary = RagflowVectorSyncService(
        repository=FailingStatusRepository(repository),
        embedding_client=EmbeddingClient.demo(dimensions=8),
        qdrant_url="http://qdrant:6333",
        collection="tcm_ragflow_retrieval",
        http_client=http_client,
    ).sync_missing(batch_size=1, limit=1, content_types=["chunk"])

    assert summary["embedded"] == 0
    assert summary["failed"] == 0
    assert repository.get_chunk("chunk:1").vector_status == "missing"
    assert http_client.requests[1][1].endswith("/collections/tcm_ragflow_retrieval/points")


def test_vector_sync_honors_limit_without_loading_all_missing_payloads():
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
                chunk_id=f"chunk:{index}",
                doc_id="doc",
                source_id="doc",
                parent_unit_id=f"unit:{index}",
                chunk_order_int=index,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content=f"失眠可辨为心脾两虚 {index}",
                content_with_weight=f"不寐 失眠可辨为心脾两虚 {index}",
                token_count=12,
            )
            for index in range(1, 6)
        ]
    )
    http_client = FakeHttpClient()

    summary = RagflowVectorSyncService(
        repository=repository,
        embedding_client=EmbeddingClient.demo(dimensions=8),
        qdrant_url="http://qdrant:6333",
        collection="tcm_ragflow_retrieval",
        http_client=http_client,
    ).sync_missing(batch_size=2, limit=3, content_types=["chunk"])

    assert summary["embedded"] == 3
    assert summary["failed"] == 0
    assert summary["embedded_chunks"] == 3
    assert summary["embedded_kg_entities"] == 0
    assert summary["embedded_kg_relations"] == 0
    assert repository.audit().chunks_with_vectors == 3
    upsert_payloads = [
        request[2]["points"]
        for request in http_client.requests
        if request[1].endswith("/points")
    ]
    assert [point["payload"]["chunk_id"] for batch in upsert_payloads for point in batch] == [
        "chunk:1",
        "chunk:2",
        "chunk:3",
    ]


def test_vector_sync_parallel_mode_claims_records_before_embedding():
    class ClaimTrackingRepository:
        def __init__(self, repository):
            self.repository = repository
            self.claim_calls = 0

        def __getattr__(self, name):
            return getattr(self.repository, name)

        def claim_missing_vector_records(self, *, content_types=None, limit=1000):
            self.claim_calls += 1
            return self.repository.claim_missing_vector_records(
                content_types=content_types,
                limit=limit,
            )

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
                chunk_id=f"chunk:{index}",
                doc_id="doc",
                source_id="doc",
                parent_unit_id=f"unit:{index}",
                chunk_order_int=index,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content=f"失眠可辨为心脾两虚 {index}",
                content_with_weight=f"不寐 失眠 心脾两虚 {index}",
                token_count=12,
            )
            for index in range(1, 5)
        ]
    )
    tracking_repository = ClaimTrackingRepository(repository)

    summary = RagflowVectorSyncService(
        repository=tracking_repository,
        embedding_client=EmbeddingClient.demo(dimensions=8),
        qdrant_url="http://qdrant:6333",
        collection="tcm_ragflow_retrieval",
        http_client=FakeHttpClient(),
    ).sync_missing_parallel(
        batch_size=2,
        limit=4,
        content_types=["chunk"],
        workers=2,
    )

    assert summary["embedded"] == 4
    assert summary["embedded_chunks"] == 4
    assert summary["failed"] == 0
    assert tracking_repository.claim_calls >= 2
    assert repository.audit().chunks_with_vectors == 4


def test_vector_sync_parallel_mode_does_not_claim_more_than_limit():
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
                chunk_id=f"chunk:{index}",
                doc_id="doc",
                source_id="doc",
                parent_unit_id=f"unit:{index}",
                chunk_order_int=index,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content=f"失眠可辨为心脾两虚 {index}",
                content_with_weight=f"不寐 失眠 心脾两虚 {index}",
                token_count=12,
            )
            for index in range(1, 11)
        ]
    )

    summary = RagflowVectorSyncService(
        repository=repository,
        embedding_client=EmbeddingClient.demo(dimensions=8),
        qdrant_url="http://qdrant:6333",
        collection="tcm_ragflow_retrieval",
        http_client=FakeHttpClient(),
    ).sync_missing_parallel(
        batch_size=4,
        limit=5,
        content_types=["chunk"],
        workers=4,
    )

    assert summary["embedded"] == 5
    assert repository.audit().chunks_with_vectors == 5


def test_vector_sync_can_balance_limit_across_content_types():
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
                chunk_id=f"chunk:{index}",
                doc_id="doc",
                source_id="doc",
                parent_unit_id=f"unit:{index}",
                chunk_order_int=index,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content=f"失眠可辨为心脾两虚 {index}",
                content_with_weight=f"不寐 失眠 心脾两虚 {index}",
                token_count=12,
            )
            for index in range(1, 4)
        ]
    )
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id=f"entity:{index}",
                entity_name=f"证候{index}",
                entity_type="Syndrome",
                source_node_id=f"syndrome:{index}",
                content_with_weight=f"证候{index} Syndrome",
            )
            for index in range(1, 4)
        ]
    )
    repository.replace_kg_relations(
        [
            RetrievalKgRelation(
                relation_id=f"relation:{index}",
                from_entity_kwd="失眠",
                to_entity_kwd=f"证候{index}",
                relation_type="MANIFESTS_AS",
                display="可辨为",
                content_with_weight=f"失眠 可辨为 证候{index}",
            )
            for index in range(1, 4)
        ]
    )
    http_client = FakeHttpClient()

    summary = RagflowVectorSyncService(
        repository=repository,
        embedding_client=EmbeddingClient.demo(dimensions=8),
        qdrant_url="http://qdrant:6333",
        collection="tcm_ragflow_retrieval",
        http_client=http_client,
    ).sync_missing(
        batch_size=6,
        limit=6,
        content_types=["chunk", "kg_entity", "kg_relation"],
        balanced=True,
    )

    assert summary["embedded"] == 6
    assert summary["failed"] == 0
    assert summary["embedded_chunks"] == 2
    assert summary["embedded_kg_entities"] == 2
    assert summary["embedded_kg_relations"] == 2
    audit = repository.audit()
    assert audit.chunks_with_vectors == 2
    assert audit.kg_entities_with_vectors == 2
    assert audit.kg_relations_with_vectors == 2
    upserted_types = [
        point["payload"]["content_type"]
        for request in http_client.requests
        if request[1].endswith("/points")
        for point in request[2]["points"]
    ]
    assert upserted_types == [
        "chunk",
        "chunk",
        "kg_entity",
        "kg_entity",
        "kg_relation",
        "kg_relation",
    ]


def test_vector_sync_reports_embedded_counts_by_content_type():
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
                chunk_id=f"chunk:{index}",
                doc_id="doc",
                source_id="doc",
                parent_unit_id=f"unit:{index}",
                chunk_order_int=index,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content=f"失眠可辨为心脾两虚 {index}",
                content_with_weight=f"不寐 失眠 心脾两虚 {index}",
                token_count=12,
            )
            for index in range(1, 4)
        ]
    )
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id=f"entity:{index}",
                entity_name=f"证候{index}",
                entity_type="Syndrome",
                source_node_id=f"syndrome:{index}",
                content_with_weight=f"证候{index} Syndrome",
            )
            for index in range(1, 4)
        ]
    )
    repository.replace_kg_relations(
        [
            RetrievalKgRelation(
                relation_id=f"relation:{index}",
                from_entity_kwd="失眠",
                to_entity_kwd=f"证候{index}",
                relation_type="MANIFESTS_AS",
                display="可辨为",
                content_with_weight=f"失眠 可辨为 证候{index}",
            )
            for index in range(1, 4)
        ]
    )

    summary = RagflowVectorSyncService(
        repository=repository,
        embedding_client=EmbeddingClient.demo(dimensions=8),
        qdrant_url="http://qdrant:6333",
        collection="tcm_ragflow_retrieval",
        http_client=FakeHttpClient(),
    ).sync_missing(
        batch_size=6,
        limit=6,
        content_types=["chunk", "kg_entity", "kg_relation"],
        balanced=True,
    )

    assert summary == {
        "embedded": 6,
        "failed": 0,
        "embedded_chunks": 2,
        "embedded_kg_entities": 2,
        "embedded_kg_relations": 2,
        "failed_chunks": 0,
        "failed_kg_entities": 0,
        "failed_kg_relations": 0,
        "status_update_failed": 0,
        "status_update_failed_chunks": 0,
        "status_update_failed_kg_entities": 0,
        "status_update_failed_kg_relations": 0,
        "retry_attempts": 0,
    }
