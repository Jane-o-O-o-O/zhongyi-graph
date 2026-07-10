from __future__ import annotations

import json

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.models.graph import GraphNode
from app.models.ingestion import (
    DocumentChunk,
    EntityCandidate,
    RelationCandidate,
    SourceManifest,
)
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.graph_build_service import RagflowGraphBuildService
from app.services.ragflow_compat.phase_markers import PHASE_COMMUNITY, PHASE_RESOLUTION
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalGraphArtifact


class RecordingExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, chunks, hint_terms=None):
        self.calls += 1
        return [], []


def _engine():
    return create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _repositories():
    engine = _engine()
    return IngestionRepository(engine), RagflowRetrievalRepository(engine)


def _source(source_id: str = "doc:a") -> SourceManifest:
    return SourceManifest(
        source_id=source_id,
        filename=f"{source_id}.txt",
        mime_type="text/plain",
        checksum=f"checksum:{source_id}",
        status="parsed",
    )


def _chunk(source_id: str = "doc:a", chunk_id: str | None = None) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id or f"chunk:{source_id}:1",
        source_id=source_id,
        page_id=f"page:{source_id}:1",
        chunk_index=0,
        content="白芍可养血敛阴。",
        content_type="text",
        token_count=8,
    )


def test_build_skips_source_when_subgraph_checkpoint_exists():
    ingestion_repository, retrieval_repository = _repositories()
    source = _source("doc:a")
    ingestion_repository.upsert_source(source)
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    retrieval_repository.save_graph_artifact(
        RetrievalGraphArtifact(
            artifact_id="subgraph:doc:a",
            artifact_type="subgraph",
            content_with_weight=json.dumps(
                {
                    "nodes": [
                        GraphNode(id="entity:herb:白芍", label="Herb", name="白芍").model_dump()
                    ],
                    "edges": [],
                },
                ensure_ascii=False,
            ),
            source_id=["doc:a"],
            node_count=1,
            edge_count=0,
            metadata={"scope": "source"},
        )
    )
    extractor = RecordingExtractor()

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=extractor,
    ).build(["doc:a"])

    assert extractor.calls == 0
    assert summary.sources_total == 1
    assert summary.sources_skipped == 1
    assert summary.sources_built == 0
    assert summary.sources_failed == 0


class FixedExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, chunks, hint_terms=None):
        self.calls += 1
        return [
            EntityCandidate(
                entity_id="entity:herb:白芍",
                name="白芍",
                label="Herb",
                normalized_name="白芍",
                source_chunk_ids=[chunks[0].chunk_id],
                confidence=0.9,
            ),
            EntityCandidate(
                entity_id="entity:function:养血敛阴",
                name="养血敛阴",
                label="Function",
                normalized_name="养血敛阴",
                source_chunk_ids=[chunks[0].chunk_id],
                confidence=0.8,
            ),
        ], [
            RelationCandidate(
                relation_id="relation:白芍:HAS_FUNCTION:养血敛阴",
                source_entity_id="entity:herb:白芍",
                target_entity_id="entity:function:养血敛阴",
                relation="HAS_FUNCTION",
                display="功效",
                evidence_chunk_ids=[chunks[0].chunk_id],
                confidence=0.8,
            )
        ]


def test_build_generates_and_saves_subgraph_artifact_for_new_source():
    ingestion_repository, retrieval_repository = _repositories()
    source = _source("doc:a")
    ingestion_repository.upsert_source(source)
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    extractor = FixedExtractor()

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=extractor,
    ).build(["doc:a"])

    artifact = retrieval_repository.get_subgraph_artifact("doc:a")
    assert extractor.calls == 1
    assert summary.sources_total == 1
    assert summary.sources_built == 1
    assert summary.sources_failed == 0
    assert artifact is not None
    assert artifact.artifact_id == "subgraph:doc:a"
    assert artifact.node_count == 2
    assert artifact.edge_count == 1
    payload = json.loads(artifact.content_with_weight)
    assert {node["name"] for node in payload["nodes"]} == {"白芍", "养血敛阴"}
    assert payload["edges"][0]["relation"] == "HAS_FUNCTION"


def test_build_merges_available_subgraphs_into_global_graph_artifact():
    ingestion_repository, retrieval_repository = _repositories()
    for source_id in ["doc:a", "doc:b"]:
        ingestion_repository.upsert_source(_source(source_id))
        ingestion_repository.replace_pages_and_chunks(source_id, [], [_chunk(source_id)])
    extractor = FixedExtractor()

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=extractor,
    ).build(["doc:a", "doc:b"])

    global_artifact = retrieval_repository.get_graph_artifact("graph:global")
    assert global_artifact is not None
    assert summary.subgraphs_merged == 2
    assert summary.global_nodes == 2
    assert summary.global_edges == 1
    assert global_artifact.source_id == ["doc:a", "doc:b"]
    payload = json.loads(global_artifact.content_with_weight)
    assert {node["name"] for node in payload["nodes"]} == {"白芍", "养血敛阴"}
    assert len(payload["edges"]) == 1


def test_build_clears_phase_markers_when_new_subgraph_changes_graph():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    retrieval_repository.set_graphrag_phase_marker(PHASE_RESOLUTION)
    retrieval_repository.set_graphrag_phase_marker(PHASE_COMMUNITY)

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=FixedExtractor(),
    ).build(["doc:a"])

    assert summary.graph_changed is True
    assert summary.resolution_marker_cleared is True
    assert summary.community_marker_cleared is True


def test_build_keeps_phase_markers_on_pure_resume():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    retrieval_repository.save_graph_artifact(
        RetrievalGraphArtifact(
            artifact_id="subgraph:doc:a",
            artifact_type="subgraph",
            content_with_weight=json.dumps({"nodes": [], "edges": []}, ensure_ascii=False),
            source_id=["doc:a"],
            node_count=0,
            edge_count=0,
        )
    )
    retrieval_repository.set_graphrag_phase_marker(PHASE_RESOLUTION)
    retrieval_repository.set_graphrag_phase_marker(PHASE_COMMUNITY)

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=RecordingExtractor(),
    ).build(["doc:a"])

    assert summary.graph_changed is False
    assert summary.resolution_marker_cleared is False
    assert summary.community_marker_cleared is False
    assert retrieval_repository.has_graphrag_phase_marker(PHASE_RESOLUTION) is True
    assert retrieval_repository.has_graphrag_phase_marker(PHASE_COMMUNITY) is True
