import json
from dataclasses import asdict
from pathlib import Path
import tempfile

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

from app.core.config import get_settings
from app.models.graph import GraphEdge, GraphNode
from app.models.ingestion import IngestionJob, SourceManifest
from app.models.query import (
    GraphBuildRequest,
    GraphBuildRunResponse,
    GraphBuildRunsResponse,
    GraphBuildResponse,
    GraphOverviewResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.ingestion_service import IngestionService
from app.services.document_parser import DocumentParser
from app.services.ingestion_repository import IngestionRepository
from app.services.object_storage import LocalObjectStorage, MinioObjectStorage
from app.services.ocr_client import OcrClient
from app.services.neo4j_publisher import Neo4jPublisher
from app.services.chunk_retriever import ChunkRetriever
from app.services.graph_service import GraphService
from app.services.question_service import QuestionService
from app.services.graph_extractor import GraphExtractor
from app.services.model_clients import StructuredExtractionClient
from app.services.ragflow_compat.doc_store import RagflowDocStore, RagflowVectorSearchClient
from app.services.ragflow_compat.community_reports import RagflowGraphCommunityReportService
from app.services.ragflow_compat.entity_resolution import (
    LlmEntityResolutionDecider,
    RagflowGraphEntityResolutionService,
)
from app.services.ragflow_compat.fulltext import RagflowFulltextRetriever
from app.services.ragflow_compat.graph_build_service import (
    RagflowGraphBuildAlreadyRunningError,
    RagflowGraphBuildService,
)
from app.services.ragflow_compat.kg_search import RagflowKgSearch
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.retrieval_service import RagflowCompatibleRetrievalService
from app.services.ragflow_compat.sync_service import RagflowRetrievalSyncService

router = APIRouter(prefix="/api")
settings = get_settings()
question_service = QuestionService.from_settings(
    llm_base_url=settings.llm_base_url,
    llm_api_key=settings.llm_api_key,
    llm_model=settings.llm_model,
    embedding_model=settings.embedding_model,
    rerank_model=settings.rerank_model,
    qdrant_url=settings.qdrant_url,
    qdrant_collection=settings.qdrant_collection,
    neo4j_uri=settings.neo4j_uri,
    neo4j_user=settings.neo4j_user,
    neo4j_password=settings.neo4j_password,
)
try:
    ingestion_repository = IngestionRepository.from_dsn(settings.postgres_dsn)
except Exception:
    ingestion_repository = IngestionRepository.in_memory()
ragflow_repository = RagflowRetrievalRepository(ingestion_repository.engine)

try:
    object_storage = MinioObjectStorage(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
        secure=settings.minio_secure,
    )
except Exception:
    object_storage = LocalObjectStorage(Path(tempfile.gettempdir()) / "tcm-kg-objects")

try:
    neo4j_publisher = Neo4jPublisher.from_config(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
except Exception:
    neo4j_publisher = None

ocr_client = (
    OcrClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.ocr_model,
    )
    if settings.llm_api_key and settings.llm_api_key != "replace-with-your-key"
    else None
)
structured_extractor = (
    StructuredExtractionClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
    if settings.llm_api_key and settings.llm_api_key != "replace-with-your-key"
    else StructuredExtractionClient.demo()
)
ingestion_service = IngestionService(
    repository=ingestion_repository,
    storage=object_storage,
    parser=DocumentParser(ocr_client=ocr_client),
    extractor=GraphExtractor(llm_extractor=structured_extractor),
)
question_service.chunk_retriever = ChunkRetriever(
    repository=ingestion_repository,
    vector_index=question_service.vector_index,
)
question_service.query_extractor = structured_extractor
ragflow_doc_store = RagflowDocStore(
    repository=ragflow_repository,
    embedding_client=question_service.vector_index.embedding_client,
    vector_weight=settings.ragflow_vector_weight,
    token_weight=settings.ragflow_token_weight,
    vector_search_client=RagflowVectorSearchClient(
        embedding_client=question_service.vector_index.embedding_client,
        qdrant_url=settings.qdrant_url,
        collection=settings.ragflow_qdrant_collection,
    ),
    min_vector_chunks_for_search=settings.ragflow_vector_min_indexed_chunks,
    min_vector_kg_entities_for_search=settings.ragflow_vector_min_indexed_kg_entities,
    min_vector_kg_relations_for_search=settings.ragflow_vector_min_indexed_kg_relations,
    min_vector_chunk_coverage_for_search=(
        settings.ragflow_vector_min_chunk_coverage_for_search
    ),
    min_vector_kg_entity_coverage_for_search=(
        settings.ragflow_vector_min_kg_entity_coverage_for_search
    ),
    min_vector_kg_relation_coverage_for_search=(
        settings.ragflow_vector_min_kg_relation_coverage_for_search
    ),
)
ragflow_retrieval_service = RagflowCompatibleRetrievalService(
    repository=ragflow_repository,
    fulltext_retriever=RagflowFulltextRetriever(
        doc_store=ragflow_doc_store,
        rerank_client=question_service.hybrid_retriever.rerank_client,
        rerank_weight=settings.ragflow_rerank_weight,
    ),
    kg_search=RagflowKgSearch(ragflow_doc_store),
    llm_client=question_service.llm_client,
    query_rewriter=structured_extractor,
    qdrant_stats_provider=lambda: _qdrant_collection_stats(
        settings.qdrant_url,
        settings.ragflow_qdrant_collection,
    ),
)
question_service.ragflow_retrieval_service = ragflow_retrieval_service
question_service.retrieval_engine = settings.retrieval_engine

try:
    restored_artifact = ingestion_service.restore_published_artifact()
    if restored_artifact.nodes or restored_artifact.edges or restored_artifact.evidence:
        question_service.publish_artifact(restored_artifact)
except Exception:
    pass


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "services": {
            "graph": "ready",
            "documents": "ready",
            "vector": "ready",
            "llm": "configured",
            "ingestion": "ready",
        },
    }


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return question_service.answer(
        request.question,
        comm_topn=request.comm_topn,
        max_token=request.max_token,
    )


@router.get("/graph/overview", response_model=GraphOverviewResponse)
def graph_overview(limit: int = 3000) -> GraphOverviewResponse:
    bounded_limit = min(max(limit, 1), 3000)
    nodes, edges = question_service.graph_service.overview(
        max_nodes=bounded_limit,
        max_edges=bounded_limit * 3,
    )
    return GraphOverviewResponse(
        graph_nodes=nodes,
        graph_edges=edges,
        highlighted_path=[],
    )


@router.post("/vector/sync")
def sync_vector_index() -> dict:
    question_service.vector_index.upsert_qdrant()
    return {
        "status": "ok",
        "collection": question_service.vector_index.collection,
        "documents": len(question_service.vector_index.documents),
    }


@router.get("/retrieval/status")
def retrieval_status() -> dict:
    return {
        "retrieval_engine": question_service.retrieval_engine,
        "ragflow_compat": ragflow_retrieval_service.status(),
    }


def _qdrant_collection_stats(qdrant_url: str, collection: str) -> dict:
    response = httpx.get(
        f"{qdrant_url.rstrip('/')}/collections/{collection}",
        timeout=5,
    )
    response.raise_for_status()
    result = response.json().get("result") or {}
    return {
        "available": True,
        "status": result.get("status"),
        "points_count": int(result.get("points_count", 0)),
        "indexed_vectors_count": int(result.get("indexed_vectors_count", 0)),
    }


@router.post("/retrieval/rebuild")
def rebuild_ragflow_retrieval_index() -> dict:
    summary = RagflowRetrievalSyncService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=ragflow_repository,
        graph_service=question_service.graph_service,
    ).rebuild_from_ingestion()
    return {"status": "ok", **summary}


@router.post("/retrieval/graphrag/build", response_model=GraphBuildResponse)
def build_ragflow_graphrag(request: GraphBuildRequest | None = None) -> GraphBuildResponse:
    request = request or GraphBuildRequest()
    try:
        summary = _graphrag_build_service(request).build(
            request.source_ids,
            with_resolution=request.with_resolution,
            with_community=request.with_community,
        )
    except RagflowGraphBuildAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    graph_refreshed = _refresh_question_graph_from_ragflow_global_artifact()
    return GraphBuildResponse(
        status="ok",
        graph_refreshed=graph_refreshed,
        **asdict(summary),
    )


@router.post("/retrieval/graphrag/build/async", response_model=GraphBuildRunResponse)
def build_ragflow_graphrag_async(
    background_tasks: BackgroundTasks,
    request: GraphBuildRequest | None = None,
) -> GraphBuildRunResponse:
    request = request or GraphBuildRequest()
    service = _graphrag_build_service(request)
    try:
        submission = service.submit(
            request.source_ids,
            with_resolution=request.with_resolution,
            with_community=request.with_community,
            execution_mode="background",
        )
    except RagflowGraphBuildAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(_run_ragflow_graphrag_background, service, submission)
    return GraphBuildRunResponse(**asdict(submission.run))


def _graphrag_build_service(request: GraphBuildRequest) -> RagflowGraphBuildService:
    return RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=ragflow_repository,
        graph_extractor=_graph_extractor_for_method(
            request.method,
            request.batch_chunk_token_size,
        ),
        entity_resolution_service=RagflowGraphEntityResolutionService(
            decider=LlmEntityResolutionDecider(structured_extractor)
        ),
        community_report_service=RagflowGraphCommunityReportService(
            report_client=structured_extractor
        ),
        retry_attempts=request.retry_attempts,
        retry_backoff_seconds=request.retry_backoff_seconds,
        retry_backoff_max_seconds=request.retry_backoff_max_seconds,
        source_timeout_seconds=request.source_timeout_seconds,
        batch_chunk_token_size=request.batch_chunk_token_size,
        method=request.method,
    )


def _graph_extractor_for_method(method: str, batch_chunk_token_size: int = 4096):
    try:
        return GraphExtractor(
            llm_extractor=structured_extractor,
            method=method,
            batch_token_limit=batch_chunk_token_size,
        )
    except TypeError:
        try:
            return GraphExtractor(llm_extractor=structured_extractor, method=method)
        except TypeError:
            return GraphExtractor(llm_extractor=structured_extractor)


def _run_ragflow_graphrag_background(service, submission) -> None:
    try:
        service.run_submitted(submission)
        _refresh_question_graph_from_ragflow_global_artifact()
    except Exception:
        return


@router.get("/retrieval/graphrag/runs/{run_id}", response_model=GraphBuildRunResponse)
def get_ragflow_graphrag_run(run_id: str) -> GraphBuildRunResponse:
    run = ragflow_repository.get_graphrag_build_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="GraphRAG build run not found")
    return GraphBuildRunResponse(**asdict(run))


@router.get("/retrieval/graphrag/runs", response_model=GraphBuildRunsResponse)
def list_ragflow_graphrag_runs(limit: int = 20) -> GraphBuildRunsResponse:
    runs = ragflow_repository.list_graphrag_build_runs(limit=limit)
    return GraphBuildRunsResponse(
        runs=[GraphBuildRunResponse(**asdict(run)) for run in runs]
    )


@router.post("/retrieval/graphrag/runs/{run_id}/cancel")
def cancel_ragflow_graphrag_run(run_id: str) -> dict:
    if not ragflow_repository.request_graphrag_build_cancel(run_id):
        raise HTTPException(status_code=404, detail="GraphRAG build run not running")
    return {
        "status": "ok",
        "run_id": run_id,
        "cancel_requested": True,
    }


def _refresh_question_graph_from_ragflow_global_artifact() -> bool:
    artifact = ragflow_repository.get_graph_artifact("graph:global")
    if not artifact:
        return False
    payload = json.loads(artifact.content_with_weight)
    nodes = [GraphNode.model_validate(node) for node in payload.get("nodes", [])]
    edges = [GraphEdge.model_validate(edge) for edge in payload.get("edges", [])]
    if not nodes:
        return False
    graph_service = GraphService(nodes, edges)
    question_service.graph_service = graph_service
    question_service.hybrid_retriever.graph_service = graph_service
    return True


@router.get("/retrieval/audit")
def audit_ragflow_retrieval_index() -> dict:
    return ragflow_retrieval_service.status()


@router.post("/ingestion/sources", response_model=SourceManifest)
def register_source(manifest: SourceManifest) -> SourceManifest:
    return ingestion_service.register_source(manifest)


@router.post("/ingestion/jobs", response_model=IngestionJob)
def create_ingestion_job(source_ids: list[str]) -> IngestionJob:
    return ingestion_service.create_job(source_ids)


@router.post("/ingestion/upload", response_model=SourceManifest)
async def upload_source(file: UploadFile) -> SourceManifest:
    content = await file.read()
    return ingestion_service.upload_source(
        filename=file.filename or "uploaded-document",
        mime_type=file.content_type or "application/octet-stream",
        content=content,
    )


@router.post("/ingestion/jobs/{job_id}/run")
def run_ingestion_job(job_id: str) -> dict:
    return ingestion_service.run_job(job_id)


@router.post("/ingestion/publish")
def publish_ingestion_sources(source_ids: list[str]) -> dict:
    artifact, batch = ingestion_service.publish_sources(source_ids)
    question_service.publish_artifact(artifact)
    graph_persisted = False
    if neo4j_publisher:
        try:
            neo4j_publisher.publish(artifact)
            graph_persisted = True
        except Exception:
            graph_persisted = False
    try:
        question_service.vector_index.upsert_payloads_qdrant(artifact.vector_payloads)
    except Exception:
        pass
    return {
        "status": "published",
        "batch_id": batch["batch_id"],
        "node_count": len(artifact.nodes),
        "edge_count": len(artifact.edges),
        "evidence_count": len(artifact.evidence),
        "chunk_count": batch["chunk_count"],
        "graph_persisted": graph_persisted,
    }


@router.get("/ingestion/sources/{source_id}/chunks")
def list_source_chunks(source_id: str) -> list[dict]:
    return [chunk.model_dump() for chunk in ingestion_service.list_chunks(source_id)]
