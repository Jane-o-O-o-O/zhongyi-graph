from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.models.graph import GraphEdge, GraphNode
from app.models.ingestion import (
    DocumentChunk,
    EntityCandidate,
    RelationCandidate,
    SourceManifest,
)
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.checkpoints import (
    COMMUNITY_CHECKPOINT,
    RESOLUTION_CHECKPOINT,
    community_checkpoint_key,
    resolution_checkpoint_key,
)
from app.services.ragflow_compat.community_reports import RagflowGraphCommunityReportService
from app.services.ragflow_compat.entity_resolution import RagflowGraphEntityResolutionService
from app.services.ragflow_compat.graph_build_service import (
    RagflowGraphBuildCanceledError,
    RagflowGraphBuildService,
)
from app.services.ragflow_compat.phase_markers import PHASE_COMMUNITY, PHASE_RESOLUTION
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import (
    RetrievalGraphArtifact,
    RetrievalGraphRagBuildRun,
)


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


class RecordingRunRepository(RagflowRetrievalRepository):
    def __init__(self, engine):
        super().__init__(engine)
        self.saved_runs: list[RetrievalGraphRagBuildRun] = []

    def save_graphrag_build_run(self, run: RetrievalGraphRagBuildRun) -> None:
        self.saved_runs.append(run)
        super().save_graphrag_build_run(run)


def _repositories_with_recording_runs():
    engine = _engine()
    return IngestionRepository(engine), RecordingRunRepository(engine)


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


class FlakyExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, chunks, hint_terms=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient extractor failure")
        return FixedExtractor().extract(chunks, hint_terms=hint_terms)


class TwiceFlakyExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, chunks, hint_terms=None):
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("transient extractor failure")
        return FixedExtractor().extract(chunks, hint_terms=hint_terms)


class AdvancingClock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class SlowSuccessfulExtractor:
    def __init__(self, clock: AdvancingClock, elapsed_seconds: float):
        self.clock = clock
        self.elapsed_seconds = elapsed_seconds
        self.calls = 0

    def extract(self, chunks, hint_terms=None):
        self.calls += 1
        self.clock.advance(self.elapsed_seconds)
        return FixedExtractor().extract(chunks, hint_terms=hint_terms)


class CancelAfterFirstSourceExtractor:
    def __init__(self, repository: RecordingRunRepository):
        self.repository = repository
        self.calls = 0
        self.cancel_run_id = ""

    def extract(self, chunks, hint_terms=None):
        self.calls += 1
        if self.calls == 1:
            running_runs = [
                run
                for run in self.repository.saved_runs
                if run.status == "running" and run.run_id.startswith("graphrag:build:")
            ]
            self.cancel_run_id = running_runs[-1].run_id
            self.repository.request_graphrag_build_cancel(self.cancel_run_id)
        return FixedExtractor().extract(chunks, hint_terms=hint_terms)


class AliasPairExtractor:
    def extract(self, chunks, hint_terms=None):
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
                entity_id="entity:herb:白芍药",
                name="白芍药",
                label="Herb",
                normalized_name="白芍药",
                source_chunk_ids=[chunks[0].chunk_id],
                confidence=0.85,
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
            ),
            RelationCandidate(
                relation_id="relation:白芍药:HAS_FUNCTION:养血敛阴",
                source_entity_id="entity:herb:白芍药",
                target_entity_id="entity:function:养血敛阴",
                relation="HAS_FUNCTION",
                display="功效",
                evidence_chunk_ids=[chunks[0].chunk_id],
                confidence=0.8,
            ),
        ]


class ExplodingResolutionDecider:
    def resolve_pairs(self, entity_type, pairs, nodes_by_name):
        raise AssertionError("resolution decider should not be called when checkpoint exists")


class ExplodingCommunityReportClient:
    def generate_community_report(self, *, community_id, entities, relations):
        raise AssertionError("community report client should not be called when checkpoint exists")


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


def test_build_retries_transient_source_extraction_failure():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    extractor = FlakyExtractor()

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=extractor,
        retry_attempts=2,
    ).build(["doc:a"], with_resolution=False, with_community=False)

    assert extractor.calls == 2
    assert summary.sources_built == 1
    assert summary.sources_failed == 0
    assert retrieval_repository.get_subgraph_artifact("doc:a") is not None


def test_build_waits_with_capped_exponential_backoff_between_retries():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    sleep_calls: list[float] = []
    extractor = TwiceFlakyExtractor()

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=extractor,
        retry_attempts=3,
        retry_backoff_seconds=2.0,
        retry_backoff_max_seconds=3.0,
        sleep_fn=sleep_calls.append,
    ).build(["doc:a"], with_resolution=False, with_community=False)

    assert extractor.calls == 3
    assert sleep_calls == [2.0, 3.0]
    assert summary.sources_built == 1
    assert summary.sources_failed == 0


def test_build_marks_source_failed_when_extraction_exceeds_timeout():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    clock = AdvancingClock()
    extractor = SlowSuccessfulExtractor(clock, elapsed_seconds=5.0)

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=extractor,
        retry_attempts=1,
        source_timeout_seconds=2.0,
        monotonic_fn=clock.now,
    ).build(["doc:a"], with_resolution=False, with_community=False)

    assert extractor.calls == 1
    assert summary.sources_built == 0
    assert summary.sources_failed == 1
    assert retrieval_repository.get_subgraph_artifact("doc:a") is None
    assert summary.source_events == [
        {"source_id": "doc:a", "status": "failed", "processed": 1, "failed": 1}
    ]


def test_build_records_per_source_progress_in_run_state():
    ingestion_repository, retrieval_repository = _repositories_with_recording_runs()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.upsert_source(_source("doc:b"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=FixedExtractor(),
    ).build(["doc:a", "doc:b"], with_resolution=False, with_community=False)

    progress_runs = [
        run
        for run in retrieval_repository.saved_runs
        if run.run_id == summary.run_id and run.status == "running" and run.processed > 0
    ]
    assert [(run.processed, run.failed) for run in progress_runs] == [(1, 0), (2, 1)]
    assert progress_runs[0].metadata["current_source_id"] == "doc:a"
    assert progress_runs[0].metadata["source_events"] == [
        {"source_id": "doc:a", "status": "built", "processed": 1, "failed": 0}
    ]
    assert progress_runs[1].metadata["current_source_id"] == "doc:b"
    assert progress_runs[1].metadata["source_events"][-1] == {
        "source_id": "doc:b",
        "status": "failed",
        "processed": 2,
        "failed": 1,
    }

    completed_run = retrieval_repository.get_graphrag_build_run(summary.run_id)
    assert completed_run is not None
    assert completed_run.status == "completed"
    assert completed_run.metadata["source_events"] == progress_runs[-1].metadata["source_events"]


def test_build_stops_at_source_boundary_when_cancel_requested():
    ingestion_repository, retrieval_repository = _repositories_with_recording_runs()
    for source_id in ["doc:a", "doc:b"]:
        ingestion_repository.upsert_source(_source(source_id))
        ingestion_repository.replace_pages_and_chunks(source_id, [], [_chunk(source_id)])
    extractor = CancelAfterFirstSourceExtractor(retrieval_repository)

    with pytest.raises(RagflowGraphBuildCanceledError):
        RagflowGraphBuildService(
            ingestion_repository=ingestion_repository,
            retrieval_repository=retrieval_repository,
            graph_extractor=extractor,
        ).build(["doc:a", "doc:b"], with_resolution=False, with_community=False)

    assert extractor.calls == 1
    assert retrieval_repository.get_subgraph_artifact("doc:a") is not None
    assert retrieval_repository.get_subgraph_artifact("doc:b") is None
    run = retrieval_repository.get_graphrag_build_run(extractor.cancel_run_id)
    assert run is not None
    assert run.status == "canceled"
    assert run.processed == 1
    assert run.failed == 0
    assert run.metadata["cancel_requested"] is True
    assert run.metadata["source_events"] == [
        {"source_id": "doc:a", "status": "built", "processed": 1, "failed": 0}
    ]
    assert retrieval_repository.claim_graphrag_build_lock(
        "graphrag:build:next",
        started_at="2026-07-10T00:00:00Z",
        metadata={},
    )


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
    assert {node["properties"]["rank"] for node in payload["nodes"]} == {1}
    assert len(payload["edges"]) == 1


def test_build_merges_duplicate_subgraph_node_provenance_like_ragflow_graph_merge():
    ingestion_repository, retrieval_repository = _repositories()
    for source_id in ["doc:a", "doc:b"]:
        ingestion_repository.upsert_source(_source(source_id))
        ingestion_repository.replace_pages_and_chunks(source_id, [], [_chunk(source_id)])

    retrieval_repository.save_graph_artifact(
        RetrievalGraphArtifact(
            artifact_id="subgraph:doc:a",
            artifact_type="subgraph",
            content_with_weight=json.dumps(
                {
                    "nodes": [
                        GraphNode(
                            id="entity:herb:白芍",
                            label="Herb",
                            name="白芍",
                            description="doc:a 白芍描述",
                            properties={
                                "source_id": "doc:a",
                                "source_chunk_ids": ["chunk:doc:a:1"],
                                "confidence": 0.72,
                            },
                        ).model_dump()
                    ],
                    "edges": [],
                },
                ensure_ascii=False,
            ),
            source_id=["doc:a"],
            node_count=1,
            edge_count=0,
        )
    )
    retrieval_repository.save_graph_artifact(
        RetrievalGraphArtifact(
            artifact_id="subgraph:doc:b",
            artifact_type="subgraph",
            content_with_weight=json.dumps(
                {
                    "nodes": [
                        GraphNode(
                            id="entity:herb:白芍",
                            label="Herb",
                            name="白芍",
                            description="doc:b 白芍描述",
                            properties={
                                "source_id": "doc:b",
                                "source_chunk_ids": ["chunk:doc:b:1"],
                                "confidence": 0.91,
                            },
                        ).model_dump()
                    ],
                    "edges": [],
                },
                ensure_ascii=False,
            ),
            source_id=["doc:b"],
            node_count=1,
            edge_count=0,
        )
    )

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=RecordingExtractor(),
    ).build(["doc:a", "doc:b"], with_resolution=False, with_community=False)

    global_artifact = retrieval_repository.get_graph_artifact("graph:global")
    assert global_artifact is not None
    assert summary.sources_skipped == 2
    assert summary.global_nodes == 1
    payload = json.loads(global_artifact.content_with_weight)
    node = payload["nodes"][0]
    assert "doc:a 白芍描述" in node["description"]
    assert "doc:b 白芍描述" in node["description"]
    assert node["properties"]["source_id"] == ["doc:a", "doc:b"]
    assert node["properties"]["source_chunk_ids"] == ["chunk:doc:a:1", "chunk:doc:b:1"]
    assert node["properties"]["confidence"] == 0.91


def test_build_merges_duplicate_subgraph_edges_by_relation_endpoints_like_ragflow_graph_merge():
    ingestion_repository, retrieval_repository = _repositories()
    nodes = [
        GraphNode(
            id="entity:symptom:失眠",
            label="Symptom",
            name="失眠",
            description="失眠",
            properties={"source_id": ["doc:a", "doc:b"], "source_chunk_ids": ["chunk:doc:a:1", "chunk:doc:b:1"]},
        ),
        GraphNode(
            id="entity:syndrome:心脾两虚",
            label="Syndrome",
            name="心脾两虚",
            description="心脾两虚",
            properties={"source_id": ["doc:a", "doc:b"], "source_chunk_ids": ["chunk:doc:a:1", "chunk:doc:b:1"]},
        ),
    ]
    for source_id, edge_id, evidence_id in [
        ("doc:a", "edge:doc:a:失眠:心脾两虚", "chunk:doc:a:1"),
        ("doc:b", "edge:doc:b:失眠:心脾两虚", "chunk:doc:b:1"),
    ]:
        ingestion_repository.upsert_source(_source(source_id))
        ingestion_repository.replace_pages_and_chunks(source_id, [], [_chunk(source_id)])
        retrieval_repository.save_graph_artifact(
            RetrievalGraphArtifact(
                artifact_id=f"subgraph:{source_id}",
                artifact_type="subgraph",
                content_with_weight=json.dumps(
                    {
                        "nodes": [node.model_dump() for node in nodes],
                        "edges": [
                            GraphEdge(
                                id=edge_id,
                                source="entity:symptom:失眠",
                                target="entity:syndrome:心脾两虚",
                                relation="MANIFESTS_AS",
                                display="可辨为",
                                evidence_ids=[evidence_id],
                            ).model_dump()
                        ],
                    },
                    ensure_ascii=False,
                ),
                source_id=[source_id],
                node_count=2,
                edge_count=1,
            )
        )

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=RecordingExtractor(),
    ).build(["doc:a", "doc:b"], with_resolution=False, with_community=False)

    global_artifact = retrieval_repository.get_graph_artifact("graph:global")
    assert global_artifact is not None
    assert summary.global_edges == 1
    payload = json.loads(global_artifact.content_with_weight)
    assert len(payload["edges"]) == 1
    assert payload["edges"][0]["evidence_ids"] == ["chunk:doc:a:1", "chunk:doc:b:1"]


def test_build_tidy_graph_removes_nodes_and_edges_missing_essential_attributes():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    retrieval_repository.save_graph_artifact(
        RetrievalGraphArtifact(
            artifact_id="subgraph:doc:a",
            artifact_type="subgraph",
            content_with_weight=json.dumps(
                {
                    "nodes": [
                        GraphNode(
                            id="entity:symptom:失眠",
                            label="Symptom",
                            name="失眠",
                            description="失眠描述",
                            properties={
                                "source_id": "doc:a",
                                "source_chunk_ids": ["chunk:doc:a:1"],
                            },
                        ).model_dump(),
                        GraphNode(
                            id="entity:treatment:养血安神",
                            label="Treatment",
                            name="养血安神",
                            description="养血安神描述",
                            properties={
                                "source_id": "doc:a",
                                "source_chunk_ids": ["chunk:doc:a:1"],
                            },
                        ).model_dump(),
                        GraphNode(
                            id="entity:syndrome:心脾两虚",
                            label="Syndrome",
                            name="心脾两虚",
                            description="",
                            properties={
                                "source_id": "doc:a",
                                "source_chunk_ids": ["chunk:doc:a:1"],
                            },
                        ).model_dump(),
                        GraphNode(
                            id="entity:formula:归脾汤",
                            label="Formula",
                            name="归脾汤",
                            description="归脾汤描述",
                            properties={},
                        ).model_dump(),
                    ],
                    "edges": [
                        GraphEdge(
                            id="edge:valid",
                            source="entity:symptom:失眠",
                            target="entity:treatment:养血安神",
                            relation="RECOMMENDS_TREATMENT",
                            display="治法",
                            evidence_ids=["chunk:doc:a:1"],
                        ).model_dump(),
                        GraphEdge(
                            id="edge:missing:evidence",
                            source="entity:symptom:失眠",
                            target="entity:treatment:养血安神",
                            relation="RELATED_TO",
                            display="相关",
                            evidence_ids=[],
                        ).model_dump(),
                        GraphEdge(
                            id="edge:dirty:target",
                            source="entity:symptom:失眠",
                            target="entity:formula:归脾汤",
                            relation="RECOMMENDS_FORMULA",
                            display="推荐方剂",
                            evidence_ids=["chunk:doc:a:1"],
                        ).model_dump(),
                    ],
                },
                ensure_ascii=False,
            ),
            source_id=["doc:a"],
            node_count=4,
            edge_count=3,
        )
    )

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=RecordingExtractor(),
    ).build(["doc:a"], with_resolution=False, with_community=False)

    global_artifact = retrieval_repository.get_graph_artifact("graph:global")
    assert global_artifact is not None
    assert summary.global_nodes == 2
    assert summary.global_edges == 1
    payload = json.loads(global_artifact.content_with_weight)
    assert {node["name"] for node in payload["nodes"]} == {"失眠", "养血安神"}
    assert [edge["id"] for edge in payload["edges"]] == ["edge:valid"]


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


def test_build_sets_phase_markers_after_postprocessing_global_graph():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=FixedExtractor(),
    ).build(["doc:a"])

    assert summary.resolution_marker_set is True
    assert summary.community_marker_set is True
    assert retrieval_repository.has_graphrag_phase_marker(PHASE_RESOLUTION) is True
    assert retrieval_repository.has_graphrag_phase_marker(PHASE_COMMUNITY) is True


def test_build_can_skip_postprocessing_phase_markers():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=FixedExtractor(),
    ).build(["doc:a"], with_resolution=False, with_community=False)

    assert summary.resolution_marker_set is False
    assert summary.community_marker_set is False
    assert retrieval_repository.has_graphrag_phase_marker(PHASE_RESOLUTION) is False
    assert retrieval_repository.has_graphrag_phase_marker(PHASE_COMMUNITY) is False


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


def test_build_syncs_retrieval_kg_index_from_global_graph():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=FixedExtractor(),
    ).build(["doc:a"])

    entities = retrieval_repository.list_kg_entities()
    relations = retrieval_repository.list_kg_relations()
    reports = retrieval_repository.list_community_reports()
    assert summary.global_nodes == 2
    assert {entity.entity_name for entity in entities} == {"白芍", "养血敛阴"}
    assert len(relations) == 1
    assert reports
    assert retrieval_repository.get_subgraph_artifact("doc:a") is not None
    assert retrieval_repository.get_graph_artifact("graph:global") is not None


def test_build_replays_community_report_checkpoint_into_retrieval_reports():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    checkpoint_key = community_checkpoint_key("0", "0", ["白芍", "养血敛阴"])
    retrieval_repository.save_graphrag_checkpoint(
        COMMUNITY_CHECKPOINT,
        checkpoint_key,
        {
            "title": "白芍养血社区",
            "summary": "白芍与养血敛阴构成一个功效主题社区。",
            "findings": [
                {"summary": "功效关联", "explanation": "白芍连接养血敛阴。"}
            ],
            "rating": 1.0,
            "rating_explanation": "测试报告",
        },
    )

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=FixedExtractor(),
        community_report_service=RagflowGraphCommunityReportService(
            report_client=ExplodingCommunityReportClient()
        ),
    ).build(["doc:a"])

    reports = retrieval_repository.list_community_reports()
    assert summary.community_reports_replayed == 1
    assert summary.community_reports_generated == 0
    assert len(reports) == 1
    assert reports[0].title == "白芍养血社区"
    assert reports[0].summary == "白芍与养血敛阴构成一个功效主题社区。"
    assert reports[0].entities_kwd == ["白芍", "养血敛阴"]


def test_build_replays_resolution_checkpoint_and_merges_entities():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    checkpoint_key = resolution_checkpoint_key("Herb", [("白芍", "白芍药")])
    retrieval_repository.save_graphrag_checkpoint(
        RESOLUTION_CHECKPOINT,
        checkpoint_key,
        [["白芍", "白芍药"]],
    )

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=AliasPairExtractor(),
        entity_resolution_service=RagflowGraphEntityResolutionService(
            decider=ExplodingResolutionDecider()
        ),
    ).build(["doc:a"])

    global_artifact = retrieval_repository.get_graph_artifact("graph:global")
    assert global_artifact is not None
    assert summary.resolution_pairs_replayed == 1
    assert summary.resolution_pairs_merged == 1
    assert summary.global_nodes == 2
    payload = json.loads(global_artifact.content_with_weight)
    herb_nodes = [node for node in payload["nodes"] if node["label"] == "Herb"]
    assert len(herb_nodes) == 1
    assert herb_nodes[0]["name"] == "白芍"
    assert herb_nodes[0]["properties"]["aliases"] == ["白芍药"]
    assert len(payload["edges"]) == 1
    assert payload["edges"][0]["source"] == "entity:herb:白芍"


class FailingForDocBExtractor(FixedExtractor):
    def extract(self, chunks, hint_terms=None):
        if chunks[0].source_id == "doc:b":
            raise RuntimeError("extract failed")
        return super().extract(chunks, hint_terms=hint_terms)


def test_build_records_source_failure_and_continues_other_sources():
    ingestion_repository, retrieval_repository = _repositories()
    for source_id in ["doc:a", "doc:b"]:
        ingestion_repository.upsert_source(_source(source_id))
        ingestion_repository.replace_pages_and_chunks(source_id, [], [_chunk(source_id)])

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=FailingForDocBExtractor(),
    ).build(["doc:a", "doc:b"])

    assert summary.sources_total == 2
    assert summary.sources_built == 1
    assert summary.sources_failed == 1
    assert retrieval_repository.get_subgraph_artifact("doc:a") is not None
    assert retrieval_repository.get_subgraph_artifact("doc:b") is None
