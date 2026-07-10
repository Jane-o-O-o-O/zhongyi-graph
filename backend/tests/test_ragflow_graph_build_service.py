from __future__ import annotations

import json

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.models.graph import GraphNode
from app.models.ingestion import DocumentChunk, SourceManifest
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.graph_build_service import RagflowGraphBuildService
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


def _chunk(source_id: str = "doc:a", chunk_id: str = "chunk:doc:a:1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
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
