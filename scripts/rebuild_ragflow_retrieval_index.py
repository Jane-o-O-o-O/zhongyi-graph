from __future__ import annotations

from pathlib import Path
import sys

PROJECT_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(PROJECT_BACKEND) not in sys.path:
    sys.path.insert(0, str(PROJECT_BACKEND))


def build_summary_message(summary: dict) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(summary.items()))


def main() -> None:
    from app.core.config import get_settings
    from app.services.ingestion_repository import IngestionRepository
    from app.services.neo4j_graph_loader import graph_service_from_neo4j
    from app.services.ragflow_compat.repository import RagflowRetrievalRepository
    from app.services.ragflow_compat.sync_service import RagflowRetrievalSyncService

    settings = get_settings()
    ingestion_repository = IngestionRepository.from_dsn(settings.postgres_dsn)
    retrieval_repository = RagflowRetrievalRepository(ingestion_repository.engine)
    graph_service = None
    if settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password:
        try:
            graph_service = graph_service_from_neo4j(
                settings.neo4j_uri,
                settings.neo4j_user,
                settings.neo4j_password,
            )
        except Exception:
            graph_service = None
    summary = RagflowRetrievalSyncService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_service=graph_service,
    ).rebuild_from_ingestion()
    print(build_summary_message(summary))


if __name__ == "__main__":
    main()
