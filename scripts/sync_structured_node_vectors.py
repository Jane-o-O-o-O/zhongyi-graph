from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import asdict
import json
import os
import time
from pathlib import Path
from typing import TypeVar
from uuid import NAMESPACE_URL, uuid5

import httpx

from app.services.model_clients import EmbeddingClient
from app.services.vector_service import VectorPayload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_PATH = ROOT / "data" / "imports" / "tcm_structured_graph.json"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION = "tcm_knowledge"
DEFAULT_LLM_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_DIMENSIONS = 1024
T = TypeVar("T")


def build_node_payloads(graph_path: str | Path) -> list[VectorPayload]:
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    payloads = []
    for node in graph["nodes"]:
        payloads.append(
            VectorPayload(
                id=f"entity:{node['id']}",
                text=_node_text(node),
                content_type="entity",
                node_id=node["id"],
                label=node["label"],
            )
        )
    return payloads


def existing_entity_node_ids(
    client: httpx.Client,
    *,
    qdrant_url: str,
    collection: str,
    page_size: int = 1000,
) -> set[str]:
    qdrant_url = qdrant_url.rstrip("/")
    node_ids: set[str] = set()
    offset = None
    while True:
        body = {
            "limit": page_size,
            "with_payload": True,
            "with_vector": False,
            "filter": {"must": [{"key": "content_type", "match": {"value": "entity"}}]},
        }
        if offset is not None:
            body["offset"] = offset
        response = client.post(
            f"{qdrant_url}/collections/{collection}/points/scroll",
            json=body,
        )
        response.raise_for_status()
        result = response.json()["result"]
        for point in result.get("points", []):
            payload = point.get("payload") or {}
            node_id = payload.get("node_id")
            if node_id:
                node_ids.add(node_id)
        offset = result.get("next_page_offset")
        if offset is None:
            return node_ids


def missing_payloads(
    payloads: Sequence[VectorPayload],
    *,
    existing_node_ids: set[str],
) -> list[VectorPayload]:
    return [payload for payload in payloads if payload.node_id not in existing_node_ids]


def sync_payloads(
    payloads: Sequence[VectorPayload],
    *,
    embedding_client: EmbeddingClient,
    qdrant_client: httpx.Client,
    qdrant_url: str,
    collection: str,
    dimensions: int,
    batch_size: int,
    retries: int = 4,
    sleep_seconds: float = 3.0,
) -> None:
    if not payloads:
        return
    ensure_collection(
        qdrant_client,
        qdrant_url=qdrant_url,
        collection=collection,
        dimensions=dimensions,
    )
    total = len(payloads)
    for batch_index, batch in enumerate(iter_batches(payloads, batch_size), start=1):
        vectors = run_with_retries(
            lambda: embedding_client.embed([payload.text for payload in batch]),
            retries=retries,
            sleep_seconds=sleep_seconds,
            label="embedding",
        )
        run_with_retries(
            lambda: upsert_payload_batch(
                qdrant_client,
                qdrant_url=qdrant_url,
                collection=collection,
                payloads=batch,
                vectors=vectors,
            ),
            retries=retries,
            sleep_seconds=sleep_seconds,
            label="qdrant upsert",
        )
        synced = min(batch_index * batch_size, total)
        print(f"[{synced}/{total}] structured node vectors synced", flush=True)


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
    payloads: Sequence[VectorPayload],
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


def qdrant_point_id(document_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"tcm-kg-platform:{document_id}"))


def iter_batches(items: Sequence[T], batch_size: int) -> Iterable[list[T]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


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


def _node_text(node: dict) -> str:
    return " ".join(
        part
        for part in [
            str(node.get("name", "")).strip(),
            str(node.get("label", "")).strip(),
            str(node.get("description", "")).strip(),
        ]
        if part
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync structured graph nodes into Qdrant.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key and not args.dry_run:
        raise SystemExit("LLM_API_KEY is required unless --dry-run is used.")

    payloads = build_node_payloads(args.graph)
    with httpx.Client(timeout=300, trust_env=False) as qdrant_client:
        existing_node_ids = existing_entity_node_ids(
            qdrant_client,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
        )
        pending = missing_payloads(payloads, existing_node_ids=existing_node_ids)
        if args.limit:
            pending = pending[: args.limit]
        print(
            f"graph_nodes={len(payloads)} existing_entity_nodes={len(existing_node_ids)} "
            f"pending={len(pending)} qdrant={args.qdrant_url.rstrip('/')}/{args.collection} "
            f"model={args.embedding_model} batch_size={args.batch_size}",
            flush=True,
        )
        if args.dry_run or not pending:
            return

        embedding_client = EmbeddingClient(
            base_url=args.llm_base_url,
            api_key=args.api_key,
            model=args.embedding_model,
            dimensions=args.dimensions,
            http_client=httpx.Client(timeout=300, trust_env=False),
        )
        sync_payloads(
            pending,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            dimensions=args.dimensions,
            batch_size=args.batch_size,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
        )


if __name__ == "__main__":
    main()
