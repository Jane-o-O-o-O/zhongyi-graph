from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.models.ingestion import (
    DocumentChunk,
    DocumentPage,
    EntityCandidate,
    RelationCandidate,
    SourceManifest,
)
from app.models.graph import GraphEdge, GraphNode
from app.services.ingestion_repository import IngestionRepository
from app.services.graph_service import GraphService
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.sync_service import RagflowRetrievalSyncService


def _shared_repositories() -> tuple[IngestionRepository, RagflowRetrievalRepository]:
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return IngestionRepository(engine), RagflowRetrievalRepository(engine)


def test_sync_service_rebuilds_documents_chunks_entities_relations_from_existing_tables():
    ingestion_repository, retrieval_repository = _shared_repositories()
    source = SourceManifest(
        source_id="source:uploaded:abc",
        filename="资料.txt",
        mime_type="text/plain",
        checksum="abc123",
        status="published",
        object_key="sources/abc/资料.txt",
    )
    page = DocumentPage(
        page_id="page:source:uploaded:abc:1",
        source_id=source.source_id,
        page_number=1,
        text="失眠可辨为心脾两虚，治以补益心脾。",
    )
    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:abc:0001",
        source_id=source.source_id,
        page_id=page.page_id,
        chunk_index=1,
        content="失眠可辨为心脾两虚，治以补益心脾。",
        section_title="不寐",
        token_count=18,
        metadata={"parent_unit_id": "unit:abc:1", "section_path": ["内科", "不寐"]},
    )
    entity = EntityCandidate(
        entity_id="entity:source:abc:symptom:失眠",
        name="失眠",
        label="Symptom",
        normalized_name="失眠",
        source_chunk_ids=[chunk.chunk_id],
        confidence=0.9,
    )
    syndrome = EntityCandidate(
        entity_id="entity:source:abc:syndrome:心脾两虚",
        name="心脾两虚",
        label="Syndrome",
        normalized_name="心脾两虚",
        source_chunk_ids=[chunk.chunk_id],
        confidence=0.8,
    )
    relation = RelationCandidate(
        relation_id="relation:source:abc:失眠:心脾两虚",
        source_entity_id=entity.entity_id,
        target_entity_id=syndrome.entity_id,
        relation="MANIFESTS_AS",
        display="可辨为",
        evidence_chunk_ids=[chunk.chunk_id],
        confidence=0.85,
    )

    ingestion_repository.upsert_source(source)
    ingestion_repository.replace_pages_and_chunks(source.source_id, [page], [chunk])
    ingestion_repository.save_candidates(source.source_id, [entity, syndrome], [relation])

    summary = RagflowRetrievalSyncService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
    ).rebuild_from_ingestion()

    assert summary == {"documents": 1, "chunks": 1, "kg_entities": 2, "kg_relations": 1}
    document = retrieval_repository.list_documents()[0]
    assert document.doc_id == source.source_id
    assert document.eligible_chunk_count == 1
    synced_chunk = retrieval_repository.list_chunks()[0]
    assert synced_chunk.parent_unit_id == "unit:abc:1"
    assert synced_chunk.title == "不寐"
    assert synced_chunk.section_path == ["内科", "不寐"]
    assert "心脾两虚" in synced_chunk.content_ltks
    assert "不寐" in synced_chunk.content_with_weight
    synced_relation = retrieval_repository.list_kg_relations()[0]
    assert synced_relation.from_entity_kwd == "失眠"
    assert synced_relation.to_entity_kwd == "心脾两虚"
    assert synced_relation.weight_int == 1


def test_sync_service_streams_chunks_in_batches_without_full_chunk_list():
    ingestion_repository, retrieval_repository = _shared_repositories()
    source = SourceManifest(
        source_id="source:uploaded:batch",
        filename="batch.txt",
        mime_type="text/plain",
        checksum="batch",
        status="published",
    )
    page = DocumentPage(
        page_id="page:source:uploaded:batch:1",
        source_id=source.source_id,
        page_number=1,
        text="失眠可辨为心脾两虚。",
    )
    chunks = [
        DocumentChunk(
            chunk_id=f"chunk:source:uploaded:batch:{index:04d}",
            source_id=source.source_id,
            page_id=page.page_id,
            chunk_index=index,
            content=f"失眠可辨为心脾两虚。段落 {index}",
            section_title="不寐",
            token_count=18,
        )
        for index in range(1, 6)
    ]
    ingestion_repository.upsert_source(source)
    ingestion_repository.replace_pages_and_chunks(source.source_id, [page], chunks)

    def fail_full_chunk_load(source_id=None):
        raise AssertionError("rebuild must not load all chunks at once")

    ingestion_repository.list_chunks = fail_full_chunk_load

    summary = RagflowRetrievalSyncService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        chunk_batch_size=2,
    ).rebuild_from_ingestion()

    assert summary["chunks"] == 5
    assert [chunk.chunk_id for chunk in retrieval_repository.list_chunks()] == [
        "chunk:source:uploaded:batch:0001",
        "chunk:source:uploaded:batch:0002",
        "chunk:source:uploaded:batch:0003",
        "chunk:source:uploaded:batch:0004",
        "chunk:source:uploaded:batch:0005",
    ]


def test_sync_service_rebuilds_kg_index_from_graph_service_when_available():
    ingestion_repository, retrieval_repository = _shared_repositories()
    graph_service = GraphService(
        nodes=[
            GraphNode(id="symptom:失眠", label="Symptom", name="失眠"),
            GraphNode(id="syndrome:心脾两虚", label="Syndrome", name="心脾两虚"),
        ],
        edges=[
            GraphEdge(
                id="edge:失眠:心脾两虚",
                source="symptom:失眠",
                target="syndrome:心脾两虚",
                relation="MANIFESTS_AS",
                display="可辨为",
                evidence_ids=["chunk:1"],
            )
        ],
    )

    summary = RagflowRetrievalSyncService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_service=graph_service,
    ).rebuild_from_ingestion()

    assert summary["kg_entities"] == 2
    assert summary["kg_relations"] == 1
    entities = retrieval_repository.list_kg_entities()
    relations = retrieval_repository.list_kg_relations()
    assert {entity.entity_name for entity in entities} == {"失眠", "心脾两虚"}
    assert relations[0].from_entity_kwd == "失眠"
    assert relations[0].to_entity_kwd == "心脾两虚"
    assert relations[0].evidence_chunk_ids == ["chunk:1"]


def test_sync_service_keeps_list_source_chunks_from_graph_nodes():
    ingestion_repository, retrieval_repository = _shared_repositories()
    graph_service = GraphService(
        nodes=[
            GraphNode(
                id="formula:小柴胡汤",
                label="Formula",
                name="小柴胡汤",
                properties={"source_chunks": ["evidence:structured-tcm:1"]},
            ),
        ],
        edges=[],
    )

    RagflowRetrievalSyncService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_service=graph_service,
    ).rebuild_from_ingestion()

    entity = retrieval_repository.list_kg_entities()[0]
    assert entity.evidence_chunk_ids == ["evidence:structured-tcm:1"]


def test_sync_service_backfills_graph_kg_evidence_from_ingestion_candidates():
    ingestion_repository, retrieval_repository = _shared_repositories()
    source = SourceManifest(
        source_id="source:uploaded:abc",
        filename="资料.txt",
        mime_type="text/plain",
        checksum="abc123",
        status="published",
    )
    page = DocumentPage(
        page_id="page:source:uploaded:abc:1",
        source_id=source.source_id,
        page_number=1,
        text="失眠可辨为心脾两虚。",
    )
    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:abc:0001",
        source_id=source.source_id,
        page_id=page.page_id,
        chunk_index=1,
        content="失眠可辨为心脾两虚。",
        section_title="不寐",
        token_count=18,
    )
    symptom = EntityCandidate(
        entity_id="entity:source:abc:symptom:失眠",
        name="失眠",
        label="Symptom",
        normalized_name="失眠",
        source_chunk_ids=[chunk.chunk_id],
        confidence=0.9,
    )
    syndrome = EntityCandidate(
        entity_id="entity:source:abc:syndrome:心脾两虚",
        name="心脾两虚",
        label="Syndrome",
        normalized_name="心脾两虚",
        source_chunk_ids=[chunk.chunk_id],
        confidence=0.8,
    )
    relation = RelationCandidate(
        relation_id="relation:source:abc:失眠:心脾两虚",
        source_entity_id=symptom.entity_id,
        target_entity_id=syndrome.entity_id,
        relation="MANIFESTS_AS",
        display="可辨为",
        evidence_chunk_ids=[chunk.chunk_id],
        confidence=0.85,
    )
    graph_service = GraphService(
        nodes=[
            GraphNode(id="symptom:失眠", label="Symptom", name="失眠"),
            GraphNode(id="syndrome:心脾两虚", label="Syndrome", name="心脾两虚"),
        ],
        edges=[
            GraphEdge(
                id="edge:失眠:心脾两虚",
                source="symptom:失眠",
                target="syndrome:心脾两虚",
                relation="MANIFESTS_AS",
                display="可辨为",
                evidence_ids=[],
            )
        ],
    )

    ingestion_repository.upsert_source(source)
    ingestion_repository.replace_pages_and_chunks(source.source_id, [page], [chunk])
    ingestion_repository.save_candidates(source.source_id, [symptom, syndrome], [relation])

    RagflowRetrievalSyncService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_service=graph_service,
    ).rebuild_from_ingestion()

    entities = {
        entity.entity_name: entity.evidence_chunk_ids
        for entity in retrieval_repository.list_kg_entities()
    }
    relation = retrieval_repository.list_kg_relations()[0]
    assert entities == {
        "失眠": [chunk.chunk_id],
        "心脾两虚": [chunk.chunk_id],
    }
    assert relation.evidence_chunk_ids == [chunk.chunk_id]
