from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from threading import Lock
from uuid import NAMESPACE_URL, uuid5

import httpx

from app.services.model_clients import EmbeddingClient
from app.services.ragflow_compat.repository import MissingVectorRecord, RagflowRetrievalRepository


class RagflowVectorSyncService:
    def __init__(
        self,
        *,
        repository: RagflowRetrievalRepository,
        embedding_client: EmbeddingClient,
        qdrant_url: str,
        collection: str,
        http_client: httpx.Client | None = None,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.2,
    ):
        self.repository = repository
        self.embedding_client = embedding_client
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection = collection
        self.http_client = http_client or httpx.Client(timeout=30)
        self.max_attempts = max(1, max_attempts)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    def preview_missing(
        self,
        *,
        content_types: list[str] | None = None,
        limit: int = 64,
        balanced: bool = False,
    ) -> list[MissingVectorRecord]:
        if balanced and content_types:
            return self._balanced_missing_records(
                content_types=content_types,
                limit=limit,
            )
        return self.repository.list_missing_vector_records(
            content_types=content_types,
            limit=limit,
        )

    def reset_collection(self) -> dict[str, int | str]:
        response = self.http_client.delete(
            f"{self.qdrant_url}/collections/{self.collection}",
        )
        if response.status_code == 404:
            return {"reset_collection": self.collection, "deleted": 0}
        response.raise_for_status()
        return {"reset_collection": self.collection, "deleted": 1}

    def sync_missing(
        self,
        *,
        batch_size: int = 64,
        limit: int | None = None,
        content_types: list[str] | None = None,
        balanced: bool = False,
    ) -> dict[str, int]:
        if balanced and limit is not None and content_types:
            return self._sync_balanced_missing(
                batch_size=batch_size,
                limit=limit,
                content_types=content_types,
            )
        summary = _new_sync_summary()
        while limit is None or _processed_count(summary) < limit:
            remaining = None if limit is None else limit - _processed_count(summary)
            current_batch_size = min(batch_size, remaining) if remaining else batch_size
            batch = self.repository.list_missing_vector_records(
                content_types=content_types,
                limit=current_batch_size,
            )
            if not batch:
                break
            if not self._sync_batch(batch, summary):
                break
        return summary

    def sync_missing_parallel(
        self,
        *,
        batch_size: int = 64,
        limit: int | None = None,
        content_types: list[str] | None = None,
        workers: int = 4,
    ) -> dict[str, int]:
        if workers <= 1:
            return self.sync_missing(
                batch_size=batch_size,
                limit=limit,
                content_types=content_types,
            )
        summary = _new_sync_summary()
        lock = Lock()
        claimed_count = 0

        def worker() -> dict[str, int]:
            nonlocal claimed_count
            worker_summary = _new_sync_summary()
            while True:
                with lock:
                    remaining = None if limit is None else limit - claimed_count
                    if remaining is not None and remaining <= 0:
                        break
                    claim_size = min(batch_size, remaining) if remaining else batch_size
                    if limit is not None:
                        claimed_count += claim_size
                batch = self.repository.claim_missing_vector_records(
                    content_types=content_types,
                    limit=claim_size,
                )
                if limit is not None and len(batch) < claim_size:
                    with lock:
                        claimed_count -= claim_size - len(batch)
                if not batch:
                    break
                keep_going = self._sync_batch(batch, worker_summary)
                with lock:
                    _merge_sync_summary(summary, worker_summary)
                    _reset_sync_summary(worker_summary)
                if not keep_going:
                    break
            return worker_summary

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker) for _ in range(workers)]
            for future in as_completed(futures):
                remainder = future.result()
                with lock:
                    _merge_sync_summary(summary, remainder)
        return summary

    def _sync_balanced_missing(
        self,
        *,
        batch_size: int,
        limit: int,
        content_types: list[str],
    ) -> dict[str, int]:
        summary = _new_sync_summary()
        while _processed_count(summary) < limit:
            remaining = limit - _processed_count(summary)
            current_batch_size = min(batch_size, remaining)
            batch = self._balanced_missing_records(
                content_types=content_types,
                limit=current_batch_size,
            )
            if not batch:
                break
            if not self._sync_batch(batch, summary):
                break
        return summary

    def _sync_batch(self, batch: list[MissingVectorRecord], summary: dict[str, int]) -> bool:
        vectors: list[list[float]] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                vectors = self.embedding_client.embed([payload.text for payload in batch])
                self._ensure_collection(len(vectors[0]))
                self._upsert(batch, vectors)
                break
            except Exception as error:
                stage = _failure_stage(vectors)
                if attempt < self.max_attempts:
                    _record_retry(summary, error, stage)
                    if self.retry_backoff_seconds > 0:
                        sleep(self.retry_backoff_seconds * attempt)
                    continue
                self._mark_failed(batch)
                _add_sync_counts(summary, "failed", batch)
                _record_failure(summary, error, stage)
                return True
        try:
            self._mark_embedded(batch)
        except Exception:
            _add_sync_counts(summary, "status_update_failed", batch)
            return False
        _add_sync_counts(summary, "embedded", batch)
        return True

    def _balanced_missing_records(
        self,
        *,
        content_types: list[str],
        limit: int,
    ) -> list[MissingVectorRecord]:
        quotas = _balanced_quotas(limit, content_types)
        records: list[MissingVectorRecord] = []
        for content_type in content_types:
            quota = quotas.get(content_type, 0)
            if quota <= 0:
                continue
            records.extend(
                self.repository.list_missing_vector_records(
                    content_types=[content_type],
                    limit=quota,
                )
            )
        if len(records) >= limit:
            return records[:limit]
        remaining = limit - len(records)
        existing_ids = {record.id for record in records}
        fallback = self.repository.list_missing_vector_records(
            content_types=content_types,
            limit=limit,
        )
        for record in fallback:
            if record.id in existing_ids:
                continue
            records.append(record)
            existing_ids.add(record.id)
            remaining -= 1
            if remaining <= 0:
                break
        return records

    def _mark_embedded(self, batch: list[MissingVectorRecord]) -> None:
        for payload in batch:
            point_id = qdrant_point_id(payload.id)
            if payload.content_type == "chunk":
                self.repository.update_chunk_vector_status(
                    payload.chunk_id,
                    point_id=point_id,
                    status="embedded",
                )
            elif payload.content_type == "kg_entity":
                self.repository.update_entity_vector_status(
                    payload.entity_id,
                    point_id=point_id,
                    status="embedded",
                )
            elif payload.content_type == "kg_relation":
                self.repository.update_relation_vector_status(
                    payload.relation_id,
                    point_id=point_id,
                    status="embedded",
                )

    def _mark_failed(self, batch: list[MissingVectorRecord]) -> None:
        for payload in batch:
            if payload.content_type == "chunk":
                self.repository.update_chunk_vector_status(
                    payload.chunk_id,
                    point_id="",
                    status="failed",
                )
            elif payload.content_type == "kg_entity":
                self.repository.update_entity_vector_status(
                    payload.entity_id,
                    point_id="",
                    status="failed",
                )
            elif payload.content_type == "kg_relation":
                self.repository.update_relation_vector_status(
                    payload.relation_id,
                    point_id="",
                    status="failed",
                )

    def _ensure_collection(self, vector_size: int) -> None:
        response = self.http_client.put(
            f"{self.qdrant_url}/collections/{self.collection}",
            json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        )
        if response.status_code == 409:
            return
        response.raise_for_status()

    def _upsert(
        self,
        payloads: list[MissingVectorRecord],
        vectors: list[list[float]],
    ) -> None:
        points = [
            {
                "id": qdrant_point_id(payload.id),
                "vector": vector,
                "payload": {
                    "id": payload.id,
                    "text": payload.text,
                    "content_type": payload.content_type,
                    "chunk_id": payload.chunk_id,
                    "entity_id": payload.entity_id,
                    "relation_id": payload.relation_id,
                    "label": payload.label,
                },
            }
            for payload, vector in zip(payloads, vectors, strict=True)
        ]
        self.http_client.put(
            f"{self.qdrant_url}/collections/{self.collection}/points",
            json={"points": points},
        ).raise_for_status()


def qdrant_point_id(value: str) -> str:
    return str(uuid5(NAMESPACE_URL, value))


def _balanced_quotas(limit: int, content_types: list[str]) -> dict[str, int]:
    unique_types = []
    for content_type in content_types:
        if content_type and content_type not in unique_types:
            unique_types.append(content_type)
    if not unique_types or limit <= 0:
        return {}
    base = limit // len(unique_types)
    remainder = limit % len(unique_types)
    return {
        content_type: base + (1 if index < remainder else 0)
        for index, content_type in enumerate(unique_types)
    }


def _new_sync_summary() -> dict[str, int]:
    return {
        "embedded": 0,
        "failed": 0,
        "embedded_chunks": 0,
        "embedded_kg_entities": 0,
        "embedded_kg_relations": 0,
        "failed_chunks": 0,
        "failed_kg_entities": 0,
        "failed_kg_relations": 0,
        "status_update_failed": 0,
        "status_update_failed_chunks": 0,
        "status_update_failed_kg_entities": 0,
        "status_update_failed_kg_relations": 0,
        "retry_attempts": 0,
    }


def _processed_count(summary: dict[str, int]) -> int:
    return (
        summary["embedded"]
        + summary["failed"]
        + summary["status_update_failed"]
    )


def _merge_sync_summary(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value
        elif value:
            target[key] = value


def _reset_sync_summary(summary: dict[str, int]) -> None:
    for key in summary:
        if isinstance(summary[key], int):
            summary[key] = 0
        else:
            summary[key] = ""


def _add_sync_counts(
    summary: dict[str, int],
    status: str,
    records: list[MissingVectorRecord],
) -> None:
    summary[status] += len(records)
    for record in records:
        key = _summary_key(status, record.content_type)
        if key:
            summary[key] += 1


def _summary_key(status: str, content_type: str) -> str:
    suffix_by_type = {
        "chunk": "chunks",
        "kg_entity": "kg_entities",
        "kg_relation": "kg_relations",
    }
    suffix = suffix_by_type.get(content_type)
    return f"{status}_{suffix}" if suffix else ""


def _record_retry(summary: dict[str, int], error: Exception, stage: str) -> None:
    summary["retry_attempts"] = summary.get("retry_attempts", 0) + 1
    _increment_dynamic_count(summary, "retry_error", type(error).__name__)
    _increment_dynamic_count(summary, "retry_stage", stage)
    _record_last_error(summary, "last_retry", error, stage)


def _record_failure(summary: dict[str, int], error: Exception, stage: str) -> None:
    _increment_dynamic_count(summary, "failure_error", type(error).__name__)
    _increment_dynamic_count(summary, "failure_stage", stage)
    _record_last_error(summary, "last_failure", error, stage)


def _increment_dynamic_count(summary: dict[str, int], prefix: str, value: str) -> None:
    key = f"{prefix}_{value}"
    summary[key] = summary.get(key, 0) + 1


def _record_last_error(
    summary: dict[str, int],
    prefix: str,
    error: Exception,
    stage: str,
) -> None:
    summary[f"{prefix}_error"] = f"{type(error).__name__}: {error}"
    summary[f"{prefix}_stage"] = stage


def _failure_stage(vectors: list[list[float]]) -> str:
    return "qdrant" if vectors else "embedding"
