from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.api import routes
from app.main import app
from app.models.graph import GraphEdge, GraphNode
from app.models.ingestion import DocumentChunk, EntityCandidate, RelationCandidate, SourceManifest
from app.services.graph_service import GraphService
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.phase_markers import PHASE_COMMUNITY, PHASE_RESOLUTION
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalGraphRagBuildRun


def test_retrieval_status_endpoint_reports_engine_and_audit_counts():
    response = routes.retrieval_status()

    assert response["retrieval_engine"] in {"legacy", "ragflow_compat"}
    assert "ragflow_compat" in response
    assert "chunks" in response["ragflow_compat"]


def test_question_service_delegates_to_ragflow_service_when_configured():
    captured = {}

    class FakeRagflowService:
        def answer(
            self,
            question,
            *,
            comm_topn=1,
            max_token=8196,
            ent_topn=6,
            rel_topn=6,
            ent_sim_threshold=0.0,
            rel_sim_threshold=0.0,
        ):
            captured["question"] = question
            captured["comm_topn"] = comm_topn
            captured["max_token"] = max_token
            captured["ent_topn"] = ent_topn
            captured["rel_topn"] = rel_topn
            captured["ent_sim_threshold"] = ent_sim_threshold
            captured["rel_sim_threshold"] = rel_sim_threshold
            return "ragflow-response"

    service = routes.QuestionService.demo()
    service.retrieval_engine = "ragflow_compat"
    service.ragflow_retrieval_service = FakeRagflowService()

    assert service.answer(
        "失眠怎么辨证？",
        comm_topn=2,
        max_token=128,
        ent_topn=8,
        rel_topn=9,
        ent_sim_threshold=0.25,
        rel_sim_threshold=0.35,
    ) == "ragflow-response"
    assert captured["question"] == "失眠怎么辨证？"
    assert captured["comm_topn"] == 2
    assert captured["max_token"] == 128
    assert captured["ent_topn"] == 8
    assert captured["rel_topn"] == 9
    assert captured["ent_sim_threshold"] == 0.25
    assert captured["rel_sim_threshold"] == 0.35


def test_seed_ragflow_retrieval_from_graph_populates_empty_repository():
    ingestion_repository = IngestionRepository.in_memory()
    retrieval_repository = RagflowRetrievalRepository(ingestion_repository.engine)
    graph_service = GraphService(
        nodes=[
            GraphNode(id="symptom:失眠", label="Indication", name="失眠"),
            GraphNode(id="formula:归脾汤", label="Formula", name="归脾汤"),
        ],
        edges=[
            GraphEdge(
                id="edge:失眠:归脾汤",
                source="symptom:失眠",
                target="formula:归脾汤",
                relation="TREATED_BY",
                display="治以",
            )
        ],
    )

    summary = routes._seed_ragflow_retrieval_from_graph_if_empty(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_service=graph_service,
    )

    assert summary["kg_entities"] == 2
    assert summary["kg_relations"] == 1
    assert summary["graph_artifacts"] == 1
    assert len(retrieval_repository.list_kg_entities(available_only=True)) == 2
    assert len(retrieval_repository.list_kg_relations(available_only=True)) == 1


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


class RecordingMethodRouteExtractor(FixedRouteExtractor):
    methods = []

    def __init__(self, llm_extractor=None, method="light"):
        self.llm_extractor = llm_extractor
        self.method = method
        self.methods.append(method)


class RecordingBatchRouteExtractor(FixedRouteExtractor):
    configs = []

    def __init__(self, llm_extractor=None, method="light", batch_token_limit=4096):
        self.llm_extractor = llm_extractor
        self.method = method
        self.batch_token_limit = batch_token_limit
        self.configs.append(
            {
                "method": method,
                "batch_token_limit": batch_token_limit,
            }
        )


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


class RecordingCommunityClient:
    def __init__(self):
        self.calls = []

    def generate_community_report(self, *, community_id, entities, relations):
        self.calls.append((community_id, entities, relations))
        return {
            "title": "白芍养血社区",
            "summary": "围绕白芍和养血敛阴形成的功效社区。",
            "findings": [],
            "rating": 1.0,
            "rating_explanation": "测试",
        }


class RecordingBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, fn, *args, **kwargs):
        self.tasks.append((fn, args, kwargs))


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
        json={
            "source_ids": ["doc:a"],
            "with_resolution": False,
            "with_community": False,
            "retry_attempts": 3,
            "retry_backoff_seconds": 1.5,
            "retry_backoff_max_seconds": 4.0,
            "source_timeout_seconds": 10.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["run_id"].startswith("graphrag:build:")
    assert body["sources_total"] == 1
    assert body["sources_built"] == 1
    assert body["sources_failed"] == 0
    assert body["global_nodes"] == 2
    assert body["resolution_marker_set"] is False
    assert body["community_marker_set"] is False
    assert body["resolution_pairs_replayed"] == 0
    assert body["resolution_pairs_resolved"] == 0
    assert body["resolution_pairs_merged"] == 0
    assert body["community_reports_replayed"] == 0
    assert body["community_reports_generated"] == 0
    assert body["graph_refreshed"] is True
    assert retrieval_repository.has_graphrag_phase_marker(PHASE_RESOLUTION) is False
    assert retrieval_repository.has_graphrag_phase_marker(PHASE_COMMUNITY) is False

    overview = TestClient(app).get("/api/graph/overview", params={"limit": 10})
    assert overview.status_code == 200
    assert {node["name"] for node in overview.json()["graph_nodes"]} == {"白芍", "养血敛阴"}

    run_response = TestClient(app).get(
        f"/api/retrieval/graphrag/runs/{body['run_id']}"
    )
    assert run_response.status_code == 200
    run_body = run_response.json()
    assert run_body["run_id"] == body["run_id"]
    assert run_body["status"] == "completed"
    assert run_body["total"] == 1
    assert run_body["processed"] == 1
    assert run_body["failed"] == 0
    assert run_body["metadata"]["source_ids"] == ["doc:a"]
    assert run_body["metadata"]["retry_attempts"] == 3
    assert run_body["metadata"]["retry_backoff_seconds"] == 1.5
    assert run_body["metadata"]["retry_backoff_max_seconds"] == 4.0
    assert run_body["metadata"]["source_timeout_seconds"] == 10.0
    assert run_body["metadata"]["summary"]["global_nodes"] == 2
    assert retrieval_repository.claim_graphrag_build_lock(
        "graphrag:build:next",
        started_at="2026-07-10T00:00:00Z",
        metadata={},
    )


def test_graphrag_build_endpoint_passes_extractor_method_to_factory(monkeypatch):
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
    RecordingMethodRouteExtractor.methods = []
    question_service = routes.QuestionService.demo()
    monkeypatch.setattr(routes, "ingestion_repository", ingestion_repository)
    monkeypatch.setattr(routes, "ragflow_repository", retrieval_repository)
    monkeypatch.setattr(routes, "question_service", question_service)
    monkeypatch.setattr(routes, "GraphExtractor", RecordingMethodRouteExtractor)

    response = TestClient(app).post(
        "/api/retrieval/graphrag/build",
        json={
            "source_ids": ["doc:a"],
            "method": "ner",
            "with_resolution": False,
            "with_community": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    run = retrieval_repository.get_graphrag_build_run(body["run_id"])
    assert RecordingMethodRouteExtractor.methods == ["ner"]
    assert run is not None
    assert run.metadata["method"] == "ner"


def test_graphrag_build_endpoint_passes_batch_token_size_to_general_extractor(monkeypatch):
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
    RecordingBatchRouteExtractor.configs = []
    question_service = routes.QuestionService.demo()
    monkeypatch.setattr(routes, "ingestion_repository", ingestion_repository)
    monkeypatch.setattr(routes, "ragflow_repository", retrieval_repository)
    monkeypatch.setattr(routes, "question_service", question_service)
    monkeypatch.setattr(routes, "GraphExtractor", RecordingBatchRouteExtractor)

    response = TestClient(app).post(
        "/api/retrieval/graphrag/build",
        json={
            "source_ids": ["doc:a"],
            "method": "general",
            "batch_chunk_token_size": 1024,
            "with_resolution": False,
            "with_community": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    run = retrieval_repository.get_graphrag_build_run(body["run_id"])
    assert RecordingBatchRouteExtractor.configs == [
        {"method": "general", "batch_token_limit": 1024}
    ]
    assert run is not None
    assert run.metadata["method"] == "general"
    assert run.metadata["batch_chunk_token_size"] == 1024


def test_graphrag_build_async_endpoint_submits_background_run(monkeypatch):
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
    background_tasks = RecordingBackgroundTasks()

    response = routes.build_ragflow_graphrag_async(
        background_tasks,
        routes.GraphBuildRequest(
            source_ids=["doc:a"],
            with_resolution=False,
            with_community=False,
        ),
    )

    assert response.status == "running"
    assert response.processed == 0
    assert response.metadata["execution_mode"] == "background"
    assert retrieval_repository.get_subgraph_artifact("doc:a") is None
    assert len(background_tasks.tasks) == 1

    task_fn, args, kwargs = background_tasks.tasks[0]
    task_fn(*args, **kwargs)

    completed = retrieval_repository.get_graphrag_build_run(response.run_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.processed == 1
    assert completed.metadata["execution_mode"] == "background"
    assert retrieval_repository.get_subgraph_artifact("doc:a") is not None
    assert retrieval_repository.claim_graphrag_build_lock(
        "graphrag:build:next",
        started_at="2026-07-10T00:00:00Z",
        metadata={},
    )


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


def test_graphrag_build_endpoint_rejects_concurrent_build(monkeypatch):
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
    retrieval_repository.claim_graphrag_build_lock(
        "graphrag:build:already-running",
        started_at="2026-07-10T00:00:00Z",
        metadata={"source_ids": ["doc:other"]},
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

    assert response.status_code == 409
    assert response.json()["detail"] == "GraphRAG build is already running"


def test_graphrag_build_cancel_endpoint_marks_running_run(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    retrieval_repository = RagflowRetrievalRepository(engine)
    retrieval_repository.save_graphrag_build_run(
        RetrievalGraphRagBuildRun(
            run_id="graphrag:build:cancel",
            status="running",
            started_at="2026-07-10T00:00:00Z",
            total=2,
            processed=1,
            metadata={"source_ids": ["doc:a", "doc:b"]},
        )
    )
    monkeypatch.setattr(routes, "ragflow_repository", retrieval_repository)

    response = TestClient(app).post(
        "/api/retrieval/graphrag/runs/graphrag:build:cancel/cancel"
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "run_id": "graphrag:build:cancel",
        "cancel_requested": True,
    }
    run_response = TestClient(app).get(
        "/api/retrieval/graphrag/runs/graphrag:build:cancel"
    )
    assert run_response.status_code == 200
    assert run_response.json()["metadata"]["cancel_requested"] is True


def test_graphrag_build_runs_endpoint_lists_recent_runs(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    retrieval_repository = RagflowRetrievalRepository(engine)
    retrieval_repository.save_graphrag_build_run(
        RetrievalGraphRagBuildRun(
            run_id="graphrag:build:old",
            status="completed",
            started_at="2026-07-10T00:00:00Z",
            total=1,
            processed=1,
            metadata={"source_ids": ["doc:old"]},
        )
    )
    retrieval_repository.save_graphrag_build_run(
        RetrievalGraphRagBuildRun(
            run_id="graphrag:build:new",
            status="running",
            started_at="2026-07-10T00:01:00Z",
            total=2,
            processed=1,
            metadata={"source_ids": ["doc:new"]},
        )
    )
    monkeypatch.setattr(routes, "ragflow_repository", retrieval_repository)

    response = TestClient(app).get("/api/retrieval/graphrag/runs", params={"limit": 1})

    assert response.status_code == 200
    body = response.json()
    assert [run["run_id"] for run in body["runs"]] == ["graphrag:build:new"]
    assert body["runs"][0]["status"] == "running"
    assert body["runs"][0]["metadata"]["source_ids"] == ["doc:new"]


def test_graphrag_build_endpoint_uses_llm_community_report_client(monkeypatch):
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
    community_client = RecordingCommunityClient()
    question_service = routes.QuestionService.demo()
    monkeypatch.setattr(routes, "ingestion_repository", ingestion_repository)
    monkeypatch.setattr(routes, "ragflow_repository", retrieval_repository)
    monkeypatch.setattr(routes, "question_service", question_service)
    monkeypatch.setattr(routes, "structured_extractor", community_client)
    monkeypatch.setattr(routes, "GraphExtractor", lambda llm_extractor=None: FixedRouteExtractor())

    response = TestClient(app).post(
        "/api/retrieval/graphrag/build",
        json={"source_ids": ["doc:a"], "with_resolution": False, "with_community": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["community_reports_generated"] == 1
    reports = retrieval_repository.list_community_reports()
    assert reports[0].title == "白芍养血社区"
    assert community_client.calls
    assert community_client.calls[0][1] == ["白芍", "养血敛阴"]
