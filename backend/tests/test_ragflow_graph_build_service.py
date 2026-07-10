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
from app.services.ragflow_compat.checkpoints import (
    COMMUNITY_CHECKPOINT,
    RESOLUTION_CHECKPOINT,
    community_checkpoint_key,
    resolution_checkpoint_key,
)
from app.services.ragflow_compat.community_reports import RagflowGraphCommunityReportService
from app.services.ragflow_compat.entity_resolution import RagflowGraphEntityResolutionService
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
