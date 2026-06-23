from fastapi import APIRouter, UploadFile

from app.core.config import get_settings
from app.models.ingestion import IngestionJob, SourceManifest
from app.models.query import QueryRequest, QueryResponse
from app.services.ingestion_service import IngestionService
from app.services.document_parser import DocumentParser
from app.services.ingestion_repository import IngestionRepository
from app.services.object_storage import LocalObjectStorage, MinioObjectStorage
from app.services.ocr_client import OcrClient
from app.services.neo4j_publisher import Neo4jPublisher
from app.services.chunk_retriever import ChunkRetriever
from app.services.question_service import QuestionService
from app.services.graph_extractor import GraphExtractor
from app.services.model_clients import StructuredExtractionClient
from pathlib import Path
import tempfile

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
)
try:
    ingestion_repository = IngestionRepository.from_dsn(settings.postgres_dsn)
except Exception:
    ingestion_repository = IngestionRepository.in_memory()

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
    return question_service.answer(request.question)


@router.post("/vector/sync")
def sync_vector_index() -> dict:
    question_service.vector_index.upsert_qdrant()
    return {
        "status": "ok",
        "collection": question_service.vector_index.collection,
        "documents": len(question_service.vector_index.documents),
    }


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
