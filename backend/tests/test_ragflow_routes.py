from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.api import routes
from app.main import app
from app.models.ingestion import DocumentChunk, EntityCandidate, RelationCandidate, SourceManifest
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.phase_markers import PHASE_COMMUNITY, PHASE_RESOLUTION
from app.services.ragflow_compat.repository import RagflowRetrievalRepository


def test_retrieval_status_endpoint_reports_engine_and_audit_counts():
    response = routes.retrieval_status()

    assert response["retrieval_engine"] in {"legacy", "ragflow_compat"}
    assert "ragflow_compat" in response
    assert "chunks" in response["ragflow_compat"]


def test_question_service_delegates_to_ragflow_service_when_configured():
    captured = {}

    class FakeRagflowService:
        def answer(self, question):
            captured["question"] = question
            return "ragflow-response"

    service = routes.QuestionService.demo()
    service.retrieval_engine = "ragflow_compat"
    service.ragflow_retrieval_service = FakeRagflowService()

    assert service.answer("失眠怎么辨证？") == "ragflow-response"
    assert captured["question"] == "失眠怎么辨证？"


class FixedRouteExtractor:
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


class AliasRouteExtractor:
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
        ], []


class RecordingResolutionClient:
    def __init__(self):
        self.calls = []

    def resolve_entity_pairs(self, *, entity_type, pairs):
        self.calls.append((entity_type, pairs))
        return [("白芍", "白芍药")]


def test_graphrag_build_endpoint_builds_global_graph_and_refreshes_overview(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ingestion_repository = IngestionRepository(engine)
    retrieval_repository = RagflowRetrievalRepository(engine)
    ingestion_repository.upsert_source(
        SourceManifest(
            source_id="doc:a",
            filename="doc:a.txt",
            mime_type="text/plain",
            checksum="checksum:doc:a",
            status="parsed",
        )
    )
    ingestion_repository.replace_pages_and_chunks(
        "doc:a",
        [],
        [
            DocumentChunk(
                chunk_id="chunk:doc:a:1",
                source_id="doc:a",
                page_id="page:doc:a:1",
                chunk_index=0,
                content="白芍可养血敛阴。",
                content_type="text",
                token_count=8,
            )
        ],
    )
    question_service = routes.QuestionService.demo()
    monkeypatch.setattr(routes, "ingestion_repository", ingestion_repository)
    monkeypatch.setattr(routes, "ragflow_repository", retrieval_repository)
    monkeypatch.setattr(routes, "question_service", question_service)
    monkeypatch.setattr(routes, "GraphExtractor", lambda llm_extractor=None: FixedRouteExtractor())

    response = TestClient(app).post(
        "/api/retrieval/graphrag/build",
        json={"source_ids": ["doc:a"], "with_resolution": False, "with_community": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["sources_total"] == 1
    assert body["sources_built"] == 1
    assert body["sources_failed"] == 0
    assert body["global_nodes"] == 2
    assert body["resolution_marker_set"] is False
    assert body["community_marker_set"] is False
    assert body["resolution_pairs_replayed"] == 0
    assert body["resolution_pairs_resolved"] == 0
    assert body["resolution_pairs_merged"] == 0
    assert body["graph_refreshed"] is True
    assert retrieval_repository.has_graphrag_phase_marker(PHASE_RESOLUTION) is False
    assert retrieval_repository.has_graphrag_phase_marker(PHASE_COMMUNITY) is False

    overview = TestClient(app).get("/api/graph/overview", params={"limit": 10})
    assert overview.status_code == 200
    assert {node["name"] for node in overview.json()["graph_nodes"]} == {"白芍", "养血敛阴"}


def test_graphrag_build_endpoint_uses_llm_resolution_client(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ingestion_repository = IngestionRepository(engine)
    retrieval_repository = RagflowRetrievalRepository(engine)
    ingestion_repository.upsert_source(
        SourceManifest(
            source_id="doc:a",
            filename="doc:a.txt",
            mime_type="text/plain",
            checksum="checksum:doc:a",
            status="parsed",
        )
    )
    ingestion_repository.replace_pages_and_chunks(
        "doc:a",
        [],
        [
            DocumentChunk(
                chunk_id="chunk:doc:a:1",
                source_id="doc:a",
                page_id="page:doc:a:1",
                chunk_index=0,
                content="白芍又名白芍药。",
                content_type="text",
                token_count=8,
            )
        ],
    )
    resolution_client = RecordingResolutionClient()
    question_service = routes.QuestionService.demo()
    monkeypatch.setattr(routes, "ingestion_repository", ingestion_repository)
    monkeypatch.setattr(routes, "ragflow_repository", retrieval_repository)
    monkeypatch.setattr(routes, "question_service", question_service)
    monkeypatch.setattr(routes, "structured_extractor", resolution_client)
    monkeypatch.setattr(routes, "GraphExtractor", lambda llm_extractor=None: AliasRouteExtractor())

    response = TestClient(app).post(
        "/api/retrieval/graphrag/build",
        json={"source_ids": ["doc:a"], "with_resolution": True, "with_community": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["resolution_pairs_resolved"] == 1
    assert body["resolution_pairs_merged"] == 1
    assert resolution_client.calls == [("Herb", [("白芍", "白芍药")])]
