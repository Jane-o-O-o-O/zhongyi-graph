from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(PROJECT_BACKEND) not in sys.path:
    sys.path.insert(0, str(PROJECT_BACKEND))


def format_sync_summary(summary: dict) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(summary.items()))


def build_dry_run_summary(records: list) -> dict[str, int]:
    return {
        "missing_preview": len(records),
        "missing_chunks_preview": sum(1 for record in records if record.content_type == "chunk"),
        "missing_kg_entities_preview": sum(
            1 for record in records if record.content_type == "kg_entity"
        ),
        "missing_kg_relations_preview": sum(
            1 for record in records if record.content_type == "kg_relation"
        ),
    }


def dry_run_missing_records(
    repository,
    *,
    content_types: list[str],
    limit: int,
    balanced: bool,
) -> list:
    from app.services.model_clients import EmbeddingClient
    from app.services.ragflow_compat.vector_sync import RagflowVectorSyncService

    return RagflowVectorSyncService(
        repository=repository,
        embedding_client=EmbeddingClient.demo(),
        qdrant_url="",
        collection="",
    ).preview_missing(
        content_types=content_types,
        limit=limit,
        balanced=balanced,
    )


def reset_failed_records(repository, *, content_types: list[str]) -> dict[str, int]:
    return repository.reset_failed_vector_statuses(content_types=content_types)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync RAGFlow-compatible retrieval vectors.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel vector sync workers. Uses queued claims when > 1.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum attempts per vector batch before marking it failed.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=0.2,
        help="Base sleep seconds between retry attempts. Multiplied by the attempt number.",
    )
    parser.add_argument(
        "--content-types",
        default="chunk,kg_entity,kg_relation",
        help="Comma-separated subset of chunk,kg_entity,kg_relation.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Distribute --limit across selected content types before falling back to leftovers.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete the RAGFlow-compatible Qdrant collection before syncing.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reset failed vector rows back to missing before previewing or syncing.",
    )
    return parser.parse_args()


def main() -> None:
    from app.core.config import get_settings
    from app.services.ingestion_repository import IngestionRepository
    from app.services.model_clients import EmbeddingClient
    from app.services.ragflow_compat.repository import RagflowRetrievalRepository
    from app.services.ragflow_compat.vector_sync import RagflowVectorSyncService

    args = parse_args()
    settings = get_settings()
    ingestion_repository = IngestionRepository.from_dsn(settings.postgres_dsn)
    repository = RagflowRetrievalRepository(ingestion_repository.engine)
    content_types = [item.strip() for item in args.content_types.split(",") if item.strip()]
    vector_sync_service = RagflowVectorSyncService(
        repository=repository,
        embedding_client=EmbeddingClient.demo() if args.dry_run else EmbeddingClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.embedding_model,
        ),
        qdrant_url=settings.qdrant_url,
        collection=settings.ragflow_qdrant_collection,
        max_attempts=args.max_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    if args.reset_collection:
        print(format_sync_summary(vector_sync_service.reset_collection()))
        if args.dry_run:
            return
    if args.retry_failed:
        print(format_sync_summary(reset_failed_records(repository, content_types=content_types)))
    if args.dry_run:
        records = dry_run_missing_records(
            repository,
            content_types=content_types,
            limit=args.limit or args.batch_size,
            balanced=args.balanced,
        )
        print(format_sync_summary(build_dry_run_summary(records)))
        return
    if args.workers > 1:
        summary = vector_sync_service.sync_missing_parallel(
            batch_size=args.batch_size,
            limit=args.limit or None,
            content_types=content_types,
            workers=args.workers,
        )
    else:
        summary = vector_sync_service.sync_missing(
            batch_size=args.batch_size,
            limit=args.limit or None,
            content_types=content_types,
            balanced=args.balanced,
        )
    print(format_sync_summary(summary))


if __name__ == "__main__":
    main()
