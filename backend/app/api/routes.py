from fastapi import APIRouter

from app.core.config import get_settings
from app.models.ingestion import IngestionJob, SourceManifest
from app.models.query import QueryRequest, QueryResponse
from app.services.ingestion_service import IngestionService
from app.services.question_service import QuestionService

router = APIRouter(prefix="/api")
settings = get_settings()
question_service = QuestionService.from_settings(
    llm_base_url=settings.llm_base_url,
    llm_api_key=settings.llm_api_key,
    llm_model=settings.llm_model,
)
ingestion_service = IngestionService()


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "services": {
            "graph": "ready",
            "documents": "ready",
            "llm": "configured",
            "ingestion": "skeleton",
        },
    }


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return question_service.answer(request.question)


@router.post("/ingestion/sources", response_model=SourceManifest)
def register_source(manifest: SourceManifest) -> SourceManifest:
    return ingestion_service.register_source(manifest)


@router.post("/ingestion/jobs", response_model=IngestionJob)
def create_ingestion_job(source_ids: list[str]) -> IngestionJob:
    return ingestion_service.create_job(source_ids)
