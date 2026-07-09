from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.services.model_clients import EmbeddingClient
from app.services.ragflow_compat.doc_store import RagflowDocStore
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import (
    RetrievalChunk,
    RetrievalDocument,
    RetrievalKgEntity,
    RetrievalKgRelation,
)


def test_doc_store_combines_lexical_and_vector_hits_for_chunks():
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
                section_path=["内科"],
                content="失眠可辨为心脾两虚。",
                content_with_weight="不寐 心脾两虚 失眠可辨为心脾两虚。",
                content_ltks="不寐 心脾两虚 失眠",
                token_count=18,
            ),
            RetrievalChunk(
                chunk_id="chunk:2",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:2",
                chunk_order_int=2,
                page_num_int=1,
                title="少阳病",
                section_path=["伤寒"],
                content="柴胡桂枝干姜汤主往来寒热。",
                content_with_weight="少阳病 柴胡桂枝干姜汤主往来寒热。",
                content_ltks="少阳病 柴胡桂枝干姜汤 往来寒热",
                token_count=18,
            ),
        ]
    )
    store = RagflowDocStore(repository=repository, embedding_client=EmbeddingClient.demo())

    hits = store.search_chunks("睡不着怎么辨证", ["失眠", "心脾两虚"], top_k=2)

    assert hits[0].chunk_id == "chunk:1"
    assert len(hits) == 1
    assert hits[0].keyword_score > 0
    assert hits[0].vector_score >= 0


def test_doc_store_embeds_only_prefiltered_chunk_candidates():
    class CountingEmbeddingClient:
        dimensions = 8

        def __init__(self):
            self.text_counts = []

        def embed(self, texts):
            self.text_counts.append(len(texts))
            return [[1.0] * self.dimensions for _text in texts]

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
                title="不寐" if index <= 3 else "少阳",
                section_path=[],
                content="失眠可辨为心脾两虚。" if index <= 3 else "柴胡桂枝干姜汤。",
                content_with_weight="不寐 失眠 心脾两虚" if index <= 3 else "少阳 柴胡桂枝干姜汤",
                content_ltks="失眠 心脾两虚" if index <= 3 else "柴胡桂枝干姜汤",
                token_count=18,
                vector_status="embedded",
            )
            for index in range(1, 21)
        ]
    )
    embedding_client = CountingEmbeddingClient()
    store = RagflowDocStore(repository=repository, embedding_client=embedding_client)

    hits = store.search_chunks("失眠怎么辨证", ["失眠"], top_k=2, candidates=5)

    assert [hit.chunk_id for hit in hits] == ["chunk:1", "chunk:2"]
    assert embedding_client.text_counts == [4]


def test_doc_store_uses_repository_chunk_candidate_search_without_full_scan():
    class CandidateOnlyRepository:
        def list_chunks(self, **_kwargs):
            raise AssertionError("doc store must not full-scan retrieval_chunks")

        def search_chunk_candidates(self, keywords, *, limit):
            assert keywords == ["失眠"]
            assert limit == 5
            return [
                RetrievalChunk(
                    chunk_id="chunk:matched",
                    doc_id="doc",
                    source_id="doc",
                    parent_unit_id="unit:matched",
                    chunk_order_int=1,
                    page_num_int=1,
                    title="不寐",
                    section_path=[],
                    content="失眠可辨为心脾两虚。",
                    content_with_weight="不寐 失眠 心脾两虚",
                    content_ltks="失眠 心脾两虚",
                    token_count=18,
                )
            ]

    store = RagflowDocStore(repository=CandidateOnlyRepository(), embedding_client=None)

    hits = store.search_chunks("失眠怎么辨证", ["失眠"], top_k=1, candidates=5)

    assert [hit.chunk_id for hit in hits] == ["chunk:matched"]


def test_doc_store_skips_online_vector_scoring_for_chunks_without_synced_vectors():
    class CountingEmbeddingClient:
        dimensions = 8

        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return [[1.0] * self.dimensions for _text in texts]

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
                content_with_weight="不寐 失眠 心脾两虚",
                content_ltks="失眠 心脾两虚",
                token_count=18,
                vector_status="missing",
            )
        ]
    )
    embedding_client = CountingEmbeddingClient()
    store = RagflowDocStore(repository=repository, embedding_client=embedding_client)

    hits = store.search_chunks("失眠怎么辨证", ["失眠"], top_k=1, candidates=5)

    assert [hit.chunk_id for hit in hits] == ["chunk:1"]
    assert hits[0].vector_score == 0.0
    assert embedding_client.calls == 0


def test_doc_store_skips_online_vector_scoring_for_kg_records_without_synced_vectors():
    class CountingEmbeddingClient:
        dimensions = 8

        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return [[1.0] * self.dimensions for _text in texts]

    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id="entity:失眠",
                entity_name="失眠",
                entity_type="Symptom",
                source_node_id="symptom:失眠",
                content_with_weight="失眠 Symptom",
                vector_status="missing",
            )
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
                vector_status="missing",
            )
        ]
    )
    embedding_client = CountingEmbeddingClient()
    store = RagflowDocStore(repository=repository, embedding_client=embedding_client)

    entity_hits = store.search_entities("失眠怎么辨证", ["失眠"], top_k=1)
    relation_hits = store.search_relations("失眠怎么辨证", top_k=1)

    assert [hit.entity.entity_name for hit in entity_hits] == ["失眠"]
    assert [hit.relation.relation_id for hit in relation_hits] == ["relation:1"]
    assert embedding_client.calls == 0


def test_doc_store_merges_qdrant_chunk_hits_with_lexical_candidates():
    class FakeVectorClient:
        def __init__(self):
            self.queries = []

        def search_chunks(self, query, *, top_k):
            self.queries.append((query, top_k))
            return {"chunk:vector-only": 0.91, "chunk:lexical": 0.72}

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
                chunk_id="chunk:lexical",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=1,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为心脾两虚。",
                content_with_weight="不寐 失眠 心脾两虚",
                content_ltks="失眠 心脾两虚",
                token_count=18,
                vector_status="embedded",
            ),
            RetrievalChunk(
                chunk_id="chunk:vector-only",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:2",
                chunk_order_int=2,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="不寐也可见阴虚火旺。",
                content_with_weight="不寐 阴虚火旺",
                content_ltks="不寐 阴虚火旺",
                token_count=18,
                vector_status="embedded",
            ),
        ]
    )
    vector_client = FakeVectorClient()
    store = RagflowDocStore(
        repository=repository,
        embedding_client=None,
        vector_search_client=vector_client,
    )

    hits = store.search_chunks("失眠怎么辨证", ["失眠"], top_k=2, candidates=1)

    assert vector_client.queries == [("失眠怎么辨证", 1)]
    assert [hit.chunk_id for hit in hits] == ["chunk:lexical", "chunk:vector-only"]
    assert hits[1].vector_score == 0.91


def test_doc_store_skips_qdrant_search_until_minimum_vector_coverage():
    class FakeVectorClient:
        def __init__(self):
            self.calls = 0

        def search_chunks(self, query, *, top_k):
            self.calls += 1
            return {"chunk:1": 0.9}

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
                content_with_weight="不寐 失眠 心脾两虚",
                content_ltks="失眠 心脾两虚",
                token_count=18,
                vector_status="embedded",
            )
        ]
    )
    vector_client = FakeVectorClient()
    store = RagflowDocStore(
        repository=repository,
        embedding_client=None,
        vector_search_client=vector_client,
        min_vector_chunks_for_search=2,
    )

    hits = store.search_chunks("失眠怎么辨证", ["失眠"], top_k=1, candidates=5)

    assert [hit.chunk_id for hit in hits] == ["chunk:1"]
    assert hits[0].vector_score == 0.0
    assert vector_client.calls == 0


def test_doc_store_skips_qdrant_chunk_search_when_vector_coverage_is_sparse():
    class FakeVectorClient:
        def __init__(self):
            self.calls = 0

        def search_chunks(self, query, *, top_k):
            self.calls += 1
            return {"chunk:5": 0.99}

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
                content="失眠可辨为心脾两虚。",
                content_with_weight="不寐 失眠 心脾两虚",
                content_ltks="失眠 心脾两虚",
                token_count=18,
                vector_status="embedded" if index <= 2 else "missing",
            )
            for index in range(1, 6)
        ]
    )
    vector_client = FakeVectorClient()
    store = RagflowDocStore(
        repository=repository,
        embedding_client=None,
        vector_search_client=vector_client,
        min_vector_chunks_for_search=2,
        min_vector_chunk_coverage_for_search=0.8,
    )

    hits = store.search_chunks("失眠怎么辨证", ["失眠"], top_k=1, candidates=5)

    assert [hit.chunk_id for hit in hits] == ["chunk:1"]
    assert hits[0].vector_score == 0.0
    assert vector_client.calls == 0


def test_doc_store_merges_qdrant_kg_entity_and_relation_hits():
    class FakeVectorClient:
        def __init__(self):
            self.entity_queries = []
            self.relation_queries = []

        def search_entities(self, query, *, top_k):
            self.entity_queries.append((query, top_k))
            return {"entity:阴虚火旺": 0.95, "entity:失眠": 0.8}

        def search_relations(self, query, *, top_k):
            self.relation_queries.append((query, top_k))
            return {"relation:2": 0.9, "relation:1": 0.75}

    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id="entity:失眠",
                entity_name="失眠",
                entity_type="Symptom",
                source_node_id="symptom:失眠",
                content_with_weight="失眠 Symptom",
                vector_status="embedded",
            ),
            RetrievalKgEntity(
                entity_id="entity:阴虚火旺",
                entity_name="阴虚火旺",
                entity_type="Syndrome",
                source_node_id="syndrome:阴虚火旺",
                content_with_weight="阴虚火旺 Syndrome",
                vector_status="embedded",
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
                vector_status="embedded",
            ),
            RetrievalKgRelation(
                relation_id="relation:2",
                from_entity_kwd="不寐",
                to_entity_kwd="阴虚火旺",
                relation_type="MANIFESTS_AS",
                display="可辨为",
                content_with_weight="不寐 可辨为 阴虚火旺",
                vector_status="embedded",
            ),
        ]
    )
    vector_client = FakeVectorClient()
    store = RagflowDocStore(
        repository=repository,
        embedding_client=None,
        vector_search_client=vector_client,
        min_vector_chunks_for_search=0,
        min_vector_kg_entities_for_search=1,
        min_vector_kg_relations_for_search=1,
    )

    entity_hits = store.search_entities("失眠怎么辨证", ["失眠"], top_k=2)
    relation_hits = store.search_relations("失眠怎么辨证", top_k=2)

    assert vector_client.entity_queries == [("失眠怎么辨证", 8)]
    assert vector_client.relation_queries == [("失眠怎么辨证", 8)]
    assert [hit.entity.entity_id for hit in entity_hits] == ["entity:失眠", "entity:阴虚火旺"]
    assert [hit.relation.relation_id for hit in relation_hits] == ["relation:2", "relation:1"]
    assert entity_hits[1].score > 0
    assert relation_hits[1].score > 0


def test_doc_store_skips_qdrant_kg_search_until_minimum_vector_coverage():
    class FakeVectorClient:
        def __init__(self):
            self.calls = 0

        def search_entities(self, query, *, top_k):
            self.calls += 1
            return {"entity:失眠": 0.9}

        def search_relations(self, query, *, top_k):
            self.calls += 1
            return {"relation:1": 0.9}

    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id="entity:失眠",
                entity_name="失眠",
                entity_type="Symptom",
                source_node_id="symptom:失眠",
                content_with_weight="失眠 Symptom",
                vector_status="embedded",
            )
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
                vector_status="embedded",
            )
        ]
    )
    vector_client = FakeVectorClient()
    store = RagflowDocStore(
        repository=repository,
        embedding_client=None,
        vector_search_client=vector_client,
        min_vector_kg_entities_for_search=2,
        min_vector_kg_relations_for_search=2,
    )

    store.search_entities("失眠怎么辨证", ["失眠"], top_k=1)
    store.search_relations("失眠怎么辨证", top_k=1)

    assert vector_client.calls == 0


def test_doc_store_skips_qdrant_kg_search_when_vector_coverage_is_sparse():
    class FakeVectorClient:
        def __init__(self):
            self.entity_calls = 0
            self.relation_calls = 0

        def search_entities(self, query, *, top_k):
            self.entity_calls += 1
            return {"entity:5": 0.99}

        def search_relations(self, query, *, top_k):
            self.relation_calls += 1
            return {"relation:5": 0.99}

    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id=f"entity:{index}",
                entity_name="失眠" if index == 1 else f"证候{index}",
                entity_type="Symptom" if index == 1 else "Syndrome",
                source_node_id=f"node:{index}",
                content_with_weight="失眠 Symptom" if index == 1 else f"证候{index} Syndrome",
                vector_status="embedded" if index <= 2 else "missing",
            )
            for index in range(1, 6)
        ]
    )
    repository.replace_kg_relations(
        [
            RetrievalKgRelation(
                relation_id=f"relation:{index}",
                from_entity_kwd="失眠" if index == 1 else f"症状{index}",
                to_entity_kwd="心脾两虚" if index == 1 else f"证候{index}",
                relation_type="MANIFESTS_AS",
                display="可辨为",
                content_with_weight=(
                    "失眠 可辨为 心脾两虚"
                    if index == 1
                    else f"症状{index} 可辨为 证候{index}"
                ),
                vector_status="embedded" if index <= 2 else "missing",
            )
            for index in range(1, 6)
        ]
    )
    vector_client = FakeVectorClient()
    store = RagflowDocStore(
        repository=repository,
        embedding_client=None,
        vector_search_client=vector_client,
        min_vector_kg_entities_for_search=2,
        min_vector_kg_relations_for_search=2,
        min_vector_kg_entity_coverage_for_search=0.8,
        min_vector_kg_relation_coverage_for_search=0.8,
    )

    entity_hits = store.search_entities("失眠怎么辨证", ["失眠"], top_k=1)
    relation_hits = store.search_relations("失眠怎么辨证", top_k=1)

    assert [hit.entity.entity_id for hit in entity_hits] == ["entity:1"]
    assert [hit.relation.relation_id for hit in relation_hits] == ["relation:1"]
    assert vector_client.entity_calls == 0
    assert vector_client.relation_calls == 0
