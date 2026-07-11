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
from app.services.graph_community_summary_service import (
    CommunitySummary,
    GraphCommunitySummaryResult,
)
from app.services.ingestion_repository import IngestionRepository
from app.services.graph_service import GraphService
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalGraphArtifact
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

    assert summary == {
        "documents": 1,
        "chunks": 1,
        "kg_entities": 2,
        "kg_relations": 1,
        "community_reports": 0,
        "graph_artifacts": 0,
    }
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


def test_sync_service_can_preserve_graph_artifacts_when_requested():
    ingestion_repository, retrieval_repository = _shared_repositories()
    retrieval_repository.save_graph_artifact(
        RetrievalGraphArtifact(
            artifact_id="subgraph:doc:a",
            artifact_type="subgraph",
            content_with_weight='{"nodes":[],"edges":[]}',
            source_id=["doc:a"],
            node_count=0,
            edge_count=0,
        )
    )

    summary = RagflowRetrievalSyncService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        write_graph_artifacts=False,
    ).rebuild_from_ingestion()

    assert summary["graph_artifacts"] == 0
    assert retrieval_repository.get_subgraph_artifact("doc:a") is not None


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
    assert summary["community_reports"] == 1
    assert summary["graph_artifacts"] == 2
    entities = retrieval_repository.list_kg_entities()
    relations = retrieval_repository.list_kg_relations()
    graph_artifacts = retrieval_repository.list_graph_artifacts()
    assert {entity.entity_name for entity in entities} == {"失眠", "心脾两虚"}
    assert relations[0].from_entity_kwd == "失眠"
    assert relations[0].to_entity_kwd == "心脾两虚"
    assert relations[0].evidence_chunk_ids == ["chunk:1"]
    assert {artifact.artifact_type for artifact in graph_artifacts} == {"graph", "subgraph"}
    assert graph_artifacts[0].node_count == 2
    assert graph_artifacts[0].edge_count == 1
    assert any("chunk:1" in artifact.source_id for artifact in graph_artifacts)
    assert retrieval_repository.has_graphrag_phase_marker("resolution_done") is True
    assert retrieval_repository.has_graphrag_phase_marker("community_done") is True
    community_report = retrieval_repository.list_community_reports()[0]
    assert community_report.entities_kwd == ["失眠", "心脾两虚"]
    assert "失眠" in community_report.content_with_weight
    entity_by_name = {entity.entity_name: entity for entity in entities}
    assert entity_by_name["失眠"].rank_flt > 0
    assert entity_by_name["失眠"].n_hop_with_weight == [
        {"path": ["失眠", "心脾两虚"], "weights": [1.0]}
    ]


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


def test_sync_service_uses_graph_analytics_for_rank_and_two_hop_neighbors():
    ingestion_repository, retrieval_repository = _shared_repositories()
    graph_service = GraphService(
        nodes=[
            GraphNode(id="formula:归脾汤", label="Formula", name="归脾汤"),
            GraphNode(id="prescription:归脾汤_1", label="Prescription", name="归脾汤_1"),
            GraphNode(id="herb:人参", label="Herb", name="人参"),
        ],
        edges=[
            GraphEdge(
                id="edge:formula:prescription",
                source="formula:归脾汤",
                target="prescription:归脾汤_1",
                relation="HAS_PRESCRIPTION",
                display="处方",
            ),
            GraphEdge(
                id="edge:prescription:herb",
                source="prescription:归脾汤_1",
                target="herb:人参",
                relation="COMPOSED_OF",
                display="组成",
            ),
        ],
    )

    RagflowRetrievalSyncService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_service=graph_service,
    ).rebuild_from_ingestion()

    entity_by_name = {
        entity.entity_name: entity
        for entity in retrieval_repository.list_kg_entities()
    }
    formula = entity_by_name["归脾汤"]
    relation_by_type = {
        relation.relation_type: relation
        for relation in retrieval_repository.list_kg_relations()
    }
    assert formula.rank_flt == graph_service.nodes[0].properties["pagerank"]
    assert {
        tuple(path["path"])
        for path in formula.n_hop_with_weight
    } == {
        ("归脾汤", "归脾汤_1"),
        ("归脾汤", "归脾汤_1", "人参"),
    }
    assert relation_by_type["HAS_PRESCRIPTION"].weight_int > 1


def test_sync_service_includes_resolution_and_community_metadata_in_kg_entities():
    ingestion_repository, retrieval_repository = _shared_repositories()
    graph_service = GraphService(
        nodes=[
            GraphNode(id="herb:白芍", label="Herb", name="白芍"),
            GraphNode(id="alias:白芍药", label="Alias", name="白芍药"),
        ],
        edges=[
            GraphEdge(
                id="edge:alias",
                source="herb:白芍",
                target="alias:白芍药",
                relation="HAS_ALIAS",
                display="别名",
            )
        ],
    )

    RagflowRetrievalSyncService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_service=graph_service,
    ).rebuild_from_ingestion()

    entity_by_name = {
        entity.entity_name: entity
        for entity in retrieval_repository.list_kg_entities()
    }
    herb = entity_by_name["白芍"]
    alias = entity_by_name["白芍药"]
    assert herb.aliases == ["白芍药"]
    assert herb.metadata["canonical_id"] == "herb:白芍"
    assert "community_summary" in herb.metadata
    assert '"aliases": ["白芍药"]' in herb.content_with_weight
    assert alias.metadata["canonical_id"] == "herb:白芍"


def test_sync_service_preserves_hierarchical_community_report_source_ids():
    ingestion_repository, retrieval_repository = _shared_repositories()
    graph_service = GraphService(
        nodes=[
            GraphNode(
                id="symptom:失眠",
                label="Symptom",
                name="失眠",
                properties={"source_chunks": ["chunk:source:insomnia:0001"]},
            ),
            GraphNode(
                id="syndrome:心脾两虚",
                label="Syndrome",
                name="心脾两虚",
                properties={"source_chunks": ["chunk:source:syndrome:0001"]},
            ),
        ],
        edges=[
            GraphEdge(
                id="edge:失眠:心脾两虚",
                source="symptom:失眠",
                target="syndrome:心脾两虚",
                relation="MANIFESTS_AS",
                display="可辨为",
            )
        ],
    )
    graph_service.community_summaries = GraphCommunitySummaryResult(
        {
            1_000_000: CommunitySummary(
                community_id=1_000_000,
                title="失眠、心脾两虚",
                summary="第 1 层社区报告",
                size=2,
                weight=2.0,
                entities=["失眠", "心脾两虚"],
                label_counts=["Symptom:1", "Syndrome:1"],
                level=1,
                source_node_ids=["symptom:失眠", "syndrome:心脾两虚"],
            )
        }
    )

    RagflowRetrievalSyncService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_service=graph_service,
    ).rebuild_from_ingestion()

    report = retrieval_repository.list_community_reports()[0]
    assert report.report_id == "community:1:1000000"
    assert report.source_id == ["source:insomnia", "source:syndrome"]
    assert report.metadata["level"] == 1
