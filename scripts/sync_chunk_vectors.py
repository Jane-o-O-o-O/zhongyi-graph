from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import asdict
import os
import time
from uuid import NAMESPACE_URL, uuid5

import httpx
from sqlalchemy import create_engine

from typing import TypeVar

from app.models.ingestion import DocumentChunk
from app.services.ingestion_repository import IngestionRepository
from app.services.model_clients import EmbeddingClient
from app.services.vector_service import VectorPayload

DEFAULT_POSTGRES_DSN = "postgresql+psycopg://tcm:tcm@127.0.0.1:5432/tcm_kg"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION = "tcm_knowledge"
DEFAULT_LLM_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_DIMENSIONS = 1024
T = TypeVar("T")


def build_chunk_payload(chunk: DocumentChunk) -> VectorPayload:
    return VectorPayload(
        id=f"chunk:{chunk.chunk_id}",
        text=chunk.content,
        content_type="chunk",
        chunk_id=chunk.chunk_id,
    )


def qdrant_point_id(document_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"tcm-kg-platform:{document_id}"))


def iter_batches(items: Sequence[T], batch_size: int) -> Iterable[list[T]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def ensure_collection(
    client: httpx.Client,
    *,
    qdrant_url: str,
    collection: str,
    dimensions: int,
) -> None:
    response = client.put(
        f"{qdrant_url.rstrip('/')}/collections/{collection}",
        json={"vectors": {"size": dimensions, "distance": "Cosine"}},
    )
    if response.status_code == 409:
        return
    response.raise_for_status()


def upsert_payload_batch(
    client: httpx.Client,
    *,
    qdrant_url: str,
    collection: str,
    payloads: list[VectorPayload],
    vectors: list[list[float]],
) -> None:
    points = []
    for payload, vector in zip(payloads, vectors, strict=True):
        points.append(
            {
                "id": qdrant_point_id(payload.id),
                "vector": vector,
                "payload": asdict(payload) | {"text": payload.text},
            }
        )
    response = client.put(
        f"{qdrant_url.rstrip('/')}/collections/{collection}/points",
        json={"points": points},
    )
    response.raise_for_status()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync PostgreSQL document chunks into Qdrant.")
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN", DEFAULT_POSTGRES_DSN))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--published-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key and not args.dry_run:
        raise SystemExit("LLM_API_KEY is required unless --dry-run is used.")

    repository = IngestionRepository(create_engine(args.postgres_dsn, future=True, pool_pre_ping=True))
    chunks = repository.list_chunks()
    if args.published_only:
        published_source_ids = set(repository.list_published_source_ids())
        chunks = [chunk for chunk in chunks if chunk.source_id in published_source_ids]
    if args.offset:
        chunks = chunks[args.offset :]
    if args.limit:
        chunks = chunks[: args.limit]
    payloads = [build_chunk_payload(chunk) for chunk in chunks]
    print(
        f"chunks={len(chunks)} qdrant={args.qdrant_url.rstrip('/')}/{args.collection} "
        f"model={args.embedding_model} batch_size={args.batch_size}",
        flush=True,
    )
    if args.dry_run:
        return

    embedding_client = EmbeddingClient(
        base_url=args.llm_base_url,
        api_key=args.api_key,
        model=args.embedding_model,
        dimensions=args.dimensions,
        http_client=httpx.Client(timeout=300, trust_env=False),
    )
    with httpx.Client(timeout=300, trust_env=False) as qdrant_client:
        ensure_collection(
            qdrant_client,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            dimensions=args.dimensions,
        )
        total = len(payloads)
        for batch_index, batch in enumerate(iter_batches(payloads, args.batch_size), start=1):
            vectors = run_with_retries(
                lambda: embedding_client.embed([payload.text for payload in batch]),
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
                label="embedding",
            )
            run_with_retries(
                lambda: upsert_payload_batch(
                    qdrant_client,
                    qdrant_url=args.qdrant_url,
                    collection=args.collection,
                    payloads=batch,
                    vectors=vectors,
                ),
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
                label="qdrant upsert",
            )
            synced = min(batch_index * args.batch_size, total)
            print(f"[{synced}/{total}] chunk vectors synced", flush=True)


def run_with_retries(fn, *, retries: int, sleep_seconds: float, label: str):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(f"{label} failed on attempt {attempt}/{retries}: {exc}; retrying", flush=True)
            time.sleep(sleep_seconds * attempt)
    raise last_error


if __name__ == "__main__":
    main()
