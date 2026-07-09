from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, TypeVar

import httpx
from sqlalchemy import create_engine, select

PROJECT_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(PROJECT_BACKEND) not in sys.path:
    sys.path.insert(0, str(PROJECT_BACKEND))

from app.core.config import get_settings
from app.models.ingestion import DocumentChunk, ExtractionUnit
from app.models.ingestion import EntityCandidate, RelationCandidate
from app.services.graph_extractor import GraphExtractor
from app.services.ingestion_repository import (
    IngestionRepository,
    chunks_table,
    extraction_units_table,
    sources_table,
)
from app.services.model_clients import StructuredExtractionClient

DEFAULT_PROGRESS_PATH = Path("/tmp/tcm_graphrag_reextract_progress.jsonl")
DEFAULT_API_BASE = "http://127.0.0.1:8000/api"
DEFAULT_UNIT_BATCH_SIZE = 8
DEFAULT_UNIT_BATCH_MIN = 5
DEFAULT_UNIT_BATCH_MAX = 10
DEFAULT_UNIT_BATCH_MAX_CHARS = 12_000
COMPLETED_STATUSES = {"extracted", "published"}
MEDICAL_KEYWORDS = (
    "汤",
    "方",
    "丸",
    "散",
    "证",
    "治",
    "病",
    "痛",
    "不寐",
    "失眠",
    "头痛",
    "眩晕",
    "归脾汤",
    "温胆汤",
    "肝",
    "脾",
    "心",
    "肾",
    "气",
    "血",
    "阴",
    "阳",
)
T = TypeVar("T")


class ExtractionProgress:
    def __init__(self, path: Path):
        self.path = path
        self.completed_source_ids: set[str] = set()
        self.counts: Counter[str] = Counter()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = str(row.get("status", "unknown"))
            source_id = str(row.get("source_id", ""))
            self.counts[status] += 1
            if source_id and status in COMPLETED_STATUSES:
                self.completed_source_ids.add(source_id)

    def append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        status = str(row.get("status", "unknown"))
        source_id = str(row.get("source_id", ""))
        self.counts[status] += 1
        if source_id and status in COMPLETED_STATUSES:
            self.completed_source_ids.add(source_id)


@dataclass(frozen=True)
class ReextractStats:
    source_id: str
    unit_count: int
    selected_unit_count: int
    chunk_count: int
    selected_chunk_count: int
    entity_count: int
    relation_count: int


@dataclass(frozen=True)
class ExtractorWorker:
    api_key: str
    model: str
    extractor: StructuredExtractionClient


def chunk_batches(items: Sequence[T], batch_size: int) -> Iterable[list[T]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def adaptive_extraction_batches(
    chunks: Sequence[DocumentChunk],
    *,
    min_items: int,
    max_items: int,
    max_chars: int,
) -> Iterable[list[DocumentChunk]]:
    if not chunks:
        return
    min_items = max(1, min_items)
    max_items = max(min_items, max_items)
    max_chars = max(1, max_chars)
    batches: list[list[DocumentChunk]] = []
    current: list[DocumentChunk] = []
    current_chars = 0

    for chunk in chunks:
        chunk_chars = len(chunk.content)
        would_exceed_count = len(current) >= max_items
        would_exceed_chars = current and current_chars + chunk_chars > max_chars
        if would_exceed_count or would_exceed_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += chunk_chars

    if current:
        batches.append(current)

    if len(batches) >= 2 and len(batches[-1]) < min_items:
        previous = batches[-2]
        tail = batches[-1]
        previous_chars = sum(len(chunk.content) for chunk in previous)
        tail_chars = sum(len(chunk.content) for chunk in tail)
        if len(previous) + len(tail) <= max_items and previous_chars + tail_chars <= max_chars:
            batches[-2] = previous + tail
            batches.pop()
        else:
            needed = min_items - len(tail)
            movable = min(needed, max(0, len(previous) - min_items))
            if movable:
                batches[-2] = previous[:-movable]
                batches[-1] = previous[-movable:] + tail

    yield from batches


def limited_chunks(chunks: list[DocumentChunk], max_chunks: int) -> list[DocumentChunk]:
    if max_chunks <= 0 or len(chunks) <= max_chunks:
        return chunks
    return sorted(chunks, key=_chunk_priority)[:max_chunks]


def limited_units(units: list[ExtractionUnit], max_units: int) -> list[ExtractionUnit]:
    if max_units <= 0 or len(units) <= max_units:
        return units
    return sorted(units, key=_unit_priority)[:max_units]


def units_as_extraction_chunks(units: list[ExtractionUnit]) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=unit.unit_id,
            source_id=unit.source_id,
            page_id=unit.page_id,
            chunk_index=unit.unit_index,
            content=unit.content,
            parent_unit_id=unit.unit_id,
            unit_index=unit.unit_index,
            content_type="text" if unit.unit_type in {"article", "section"} else unit.unit_type,
            section_title=unit.title,
            token_count=unit.token_count,
            char_start=unit.char_start,
            char_end=unit.char_end,
            metadata={
                **unit.metadata,
                "section_path": unit.section_path,
                "extraction_unit": True,
            },
        )
        for unit in units
    ]


def build_extractor_workers(
    *,
    api_keys: list[str],
    per_key_concurrency: int,
    base_url: str,
    model: str,
    timeout: float,
) -> list[ExtractorWorker]:
    workers: list[ExtractorWorker] = []
    for api_key in api_keys:
        for _index in range(max(per_key_concurrency, 1)):
            workers.append(
                ExtractorWorker(
                    api_key=api_key,
                    model=model,
                    extractor=StructuredExtractionClient(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        http_client=httpx.Client(timeout=timeout, trust_env=False),
                    ),
                )
            )
    return workers


def extract_batch_with_llm(
    llm_extractor,
    chunks: list[DocumentChunk],
    *,
    retries: int = 1,
    sleep_seconds: float = 0,
) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
    payload = run_with_retries(
        lambda: llm_extractor.extract_chunks_batch(
            [{"unit_id": chunk.chunk_id, "text": chunk.content} for chunk in chunks]
        ),
        retries=retries,
        sleep_seconds=sleep_seconds,
        label="extract_batch",
    )
    entities: dict[str, EntityCandidate] = {}
    relations: dict[str, RelationCandidate] = {}
    labels_by_unit: dict[str, dict[str, str]] = {}
    canonical_by_unit: dict[str, dict[str, str]] = {}

    for item in payload.get("items", []):
        unit_id = str(item.get("unit_id", "")).strip()
        if not unit_id:
            continue
        entity_labels: dict[str, str] = {}
        canonical_names: dict[str, str] = {}
        for entity_item in item.get("entities", []):
            raw_name = _first_text(entity_item, "name", "text", "value", "entity", "entity_id")
            label = _normalize_label(_first_text(entity_item, "label", "type", "category"))
            name = _canonical_name(raw_name, label)
            if not name or not label:
                continue
            entity_id = _entity_id(label, name)
            canonical_names[raw_name] = name
            entity_labels[name] = label
            existing = entities.get(entity_id)
            source_ids = [unit_id]
            if existing:
                source_ids = sorted(set(existing.source_chunk_ids + source_ids))
            entities[entity_id] = EntityCandidate(
                entity_id=entity_id,
                name=name,
                label=label,
                normalized_name=name,
                source_chunk_ids=source_ids,
                confidence=float(entity_item.get("confidence") or 0.75),
            )
        labels_by_unit[unit_id] = entity_labels
        canonical_by_unit[unit_id] = canonical_names

    for item in payload.get("items", []):
        unit_id = str(item.get("unit_id", "")).strip()
        entity_labels = labels_by_unit.get(unit_id, {})
        canonical_names = canonical_by_unit.get(unit_id, {})
        for relation_item in item.get("relations", []):
            raw_source_name = _first_text(relation_item, "source", "subject", "head", "from")
            raw_target_name = _first_text(relation_item, "target", "object", "tail", "to")
            source_name = canonical_names.get(raw_source_name, raw_source_name)
            target_name = canonical_names.get(raw_target_name, raw_target_name)
            relation = _normalize_relation(str(relation_item.get("relation", "")))
            display = str(relation_item.get("display", "")).strip() or _display_for_relation(relation)
            source_label = entity_labels.get(source_name)
            target_label = entity_labels.get(target_name)
            if not source_name or not target_name or not relation or not source_label or not target_label:
                continue
            source_id = _entity_id(source_label, source_name)
            target_id = _entity_id(target_label, target_name)
            relation_id = f"relation:{source_id}:{relation}:{target_id}"
            existing = relations.get(relation_id)
            evidence_ids = [unit_id]
            if existing:
                evidence_ids = sorted(set(existing.evidence_chunk_ids + evidence_ids))
            relations[relation_id] = RelationCandidate(
                relation_id=relation_id,
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation=relation,
                display=display,
                evidence_chunk_ids=evidence_ids,
                confidence=float(relation_item.get("confidence") or 0.72),
            )
    return list(entities.values()), list(relations.values())


def should_replace_candidates(entities: list[Any], relations: list[Any], *, replace_empty: bool) -> bool:
    return replace_empty or bool(entities or relations)


def _first_text(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _merge_extraction_results(
    entities_by_id: dict[str, EntityCandidate],
    relations_by_id: dict[str, RelationCandidate],
    entities: list[EntityCandidate],
    relations: list[RelationCandidate],
) -> None:
    for entity in entities:
        existing = entities_by_id.get(entity.entity_id)
        if existing:
            entities_by_id[entity.entity_id] = existing.model_copy(
                update={
                    "source_chunk_ids": sorted(set(existing.source_chunk_ids + entity.source_chunk_ids)),
                    "confidence": max(existing.confidence, entity.confidence),
                }
            )
        else:
            entities_by_id[entity.entity_id] = entity
    for relation in relations:
        existing = relations_by_id.get(relation.relation_id)
        if existing:
            relations_by_id[relation.relation_id] = existing.model_copy(
                update={
                    "evidence_chunk_ids": sorted(
                        set(existing.evidence_chunk_ids + relation.evidence_chunk_ids)
                    ),
                    "confidence": max(existing.confidence, relation.confidence),
                }
            )
        else:
            relations_by_id[relation.relation_id] = relation


def _chunk_priority(chunk: DocumentChunk) -> tuple[int, int, int]:
    keyword_hits = sum(1 for keyword in MEDICAL_KEYWORDS if keyword in chunk.content)
    return (-keyword_hits, len(chunk.content), chunk.chunk_index)


def _unit_priority(unit: ExtractionUnit) -> tuple[int, int, int]:
    keyword_hits = sum(1 for keyword in MEDICAL_KEYWORDS if keyword in unit.content)
    return (-keyword_hits, len(unit.content), unit.unit_index)


def _entity_id(label: str, name: str) -> str:
    label_prefix = {
        "Symptom": "symptom",
        "Syndrome": "syndrome",
        "Treatment": "treatment",
        "Formula": "formula",
        "Herb": "herb",
    }.get(label, label.lower())
    return f"entity:{label_prefix}:{name}"


def _normalize_label(label: str) -> str:
    aliases = {
        "symptom": "Symptom",
        "syndrome": "Syndrome",
        "treatment": "Treatment",
        "formula": "Formula",
        "herb": "Herb",
        "indication": "Indication",
        "function": "Function",
        "症状": "Symptom",
        "证候": "Syndrome",
        "病名": "Syndrome",
        "病机": "Syndrome",
        "治法": "Treatment",
        "方剂": "Formula",
        "药方": "Formula",
        "处方": "Formula",
        "medicine": "Formula",
        "prescription": "Formula",
        "中药": "Herb",
        "药物": "Herb",
        "药材": "Herb",
        "herbal": "Herb",
        "病因": "Syndrome",
        "舌象": "Indication",
        "脉象": "Indication",
        "诊法": "Indication",
        "体征": "Indication",
        "主治": "Indication",
        "功效": "Function",
    }
    normalized = aliases.get(label.strip(), aliases.get(label.strip().lower(), label.strip()))
    allowed = {"Symptom", "Syndrome", "Treatment", "Formula", "Herb", "Indication", "Function"}
    return normalized if normalized in allowed else ""


def _canonical_name(name: str, label: str) -> str:
    if label == "Symptom":
        aliases = {
            "头疼": "头痛",
            "偏头疼": "头痛",
            "偏头痛": "头痛",
            "发热头痛": "头痛",
            "头痛发热": "头痛",
        }
        if name in aliases:
            return aliases[name]
        if name.endswith("头痛") and name != "头痛":
            return "头痛"
    return name


def _normalize_relation(relation: str) -> str:
    normalized = relation.strip().upper()
    aliases = {
        "可辨为": "MANIFESTS_AS",
        "证候": "MANIFESTS_AS",
        "治法": "RECOMMENDS_TREATMENT",
        "推荐方剂": "RECOMMENDS_FORMULA",
        "组成": "COMPOSED_OF",
        "主治": "TREATS",
        "相关": "RELATED_TO",
    }
    normalized = aliases.get(relation.strip(), normalized)
    allowed = {
        "MANIFESTS_AS",
        "RECOMMENDS_TREATMENT",
        "RECOMMENDS_FORMULA",
        "COMPOSED_OF",
        "TREATS",
        "RELATED_TO",
    }
    return normalized if normalized in allowed else "RELATED_TO"


def _display_for_relation(relation: str) -> str:
    return {
        "MANIFESTS_AS": "可辨为",
        "RECOMMENDS_TREATMENT": "治法",
        "RECOMMENDS_FORMULA": "推荐方剂",
        "COMPOSED_OF": "组成",
        "TREATS": "主治",
        "RELATED_TO": "相关",
    }.get(relation, "相关")


def list_source_ids(repository: IngestionRepository) -> list[str]:
    statement = (
        select(sources_table.c.source_id)
        .join(chunks_table, chunks_table.c.source_id == sources_table.c.source_id)
        .group_by(sources_table.c.source_id)
        .order_by(sources_table.c.source_id)
    )
    with repository.engine.begin() as connection:
        return [row.source_id for row in connection.execute(statement)]


def list_extraction_units(repository: IngestionRepository, source_id: str) -> list[ExtractionUnit]:
    statement = (
        select(extraction_units_table)
        .where(extraction_units_table.c.source_id == source_id)
        .order_by(extraction_units_table.c.unit_index)
    )
    with repository.engine.begin() as connection:
        units = []
        for row in connection.execute(statement):
            data = dict(row._mapping)
            units.append(ExtractionUnit(**data))
        return units


class ExistingGraphRagReextractor:
    def __init__(
        self,
        *,
        repository: IngestionRepository,
        extractor: GraphExtractor,
        progress: ExtractionProgress,
        api_base: str,
        publish_batch_size: int,
        unit_batch_min: int,
        unit_batch_max: int,
        unit_batch_max_chars: int,
        max_chunks_per_source: int,
        extractor_workers: list[ExtractorWorker],
        replace_empty: bool,
        retries: int,
        sleep_seconds: float,
        dry_run: bool,
    ):
        self.repository = repository
        self.extractor = extractor
        self.progress = progress
        self.api_base = api_base.rstrip("/")
        self.publish_batch_size = publish_batch_size
        self.unit_batch_min = unit_batch_min
        self.unit_batch_max = unit_batch_max
        self.unit_batch_max_chars = unit_batch_max_chars
        self.max_chunks_per_source = max_chunks_per_source
        self.extractor_workers = extractor_workers
        self.replace_empty = replace_empty
        self.retries = retries
        self.sleep_seconds = sleep_seconds
        self.dry_run = dry_run
        self.pending_publish_source_ids: list[str] = []
        self.summary: Counter[str] = Counter()

    def run(self, source_ids: list[str]) -> None:
        with httpx.Client(timeout=300, trust_env=False) as client:
            for index, source_id in enumerate(source_ids, start=1):
                if source_id in self.progress.completed_source_ids:
                    self.summary["already_completed"] += 1
                    continue
                self._run_source(client, source_id, index=index, total=len(source_ids))
            self._publish_pending(client)

    def _run_source(
        self,
        client: httpx.Client,
        source_id: str,
        *,
        index: int,
        total: int,
    ) -> None:
        try:
            chunks = self.repository.list_chunks(source_id)
            units = list_extraction_units(self.repository, source_id)
            if units:
                selected_units = limited_units(units, self.max_chunks_per_source)
                selected_chunks = units_as_extraction_chunks(selected_units)
            else:
                selected_units = []
                selected_chunks = limited_chunks(chunks, self.max_chunks_per_source)
            if self.dry_run:
                self.summary["would_extract"] += 1
                print(
                    f"[{index}/{total}] dry-run source={source_id} "
                    f"units={len(units)} selected_units={len(selected_units)} "
                    f"chunks={len(chunks)} selected_chunks={len(selected_chunks)}",
                    flush=True,
                )
                return

            entities_by_id, relations_by_id = self._extract_selected_chunks(
                source_id=source_id,
                selected_chunks=selected_chunks,
            )

            entities = list(entities_by_id.values())
            relations = list(relations_by_id.values())
            if should_replace_candidates(entities, relations, replace_empty=self.replace_empty):
                self.repository.save_candidates(source_id, entities, relations)
                status = "extracted"
            else:
                status = "empty_skipped"
            stats = ReextractStats(
                source_id=source_id,
                unit_count=len(units),
                selected_unit_count=len(selected_units),
                chunk_count=len(chunks),
                selected_chunk_count=len(selected_chunks),
                entity_count=len(entities),
                relation_count=len(relations),
            )
            self.progress.append({"status": status, **stats.__dict__})
            self.summary[status] += 1
            if relations:
                self.pending_publish_source_ids.append(source_id)
            if len(self.pending_publish_source_ids) >= self.publish_batch_size:
                self._publish_pending(client)
            print(
                f"[{index}/{total}] {status} source={source_id} chunks={stats.selected_chunk_count}/"
                f"{stats.chunk_count} units={stats.selected_unit_count}/{stats.unit_count} "
                f"entities={stats.entity_count} relations={stats.relation_count}",
                flush=True,
            )
        except Exception as exc:
            self.progress.append(
                {
                    "status": "failed",
                    "source_id": source_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self.summary["failed"] += 1
            print(f"[{index}/{total}] failed source={source_id}: {exc}", flush=True)

    def _extract_selected_chunks(
        self,
        *,
        source_id: str,
        selected_chunks: list[DocumentChunk],
    ) -> tuple[dict[str, EntityCandidate], dict[str, RelationCandidate]]:
        entities_by_id: dict[str, EntityCandidate] = {}
        relations_by_id: dict[str, RelationCandidate] = {}
        batches = list(
            adaptive_extraction_batches(
                selected_chunks,
                min_items=self.unit_batch_min,
                max_items=self.unit_batch_max,
                max_chars=self.unit_batch_max_chars,
            )
        )
        if not batches:
            return entities_by_id, relations_by_id
        if self.extractor_workers:
            with ThreadPoolExecutor(max_workers=len(self.extractor_workers)) as executor:
                future_to_batch = {}
                for batch_index, batch in enumerate(batches, start=1):
                    worker = self.extractor_workers[(batch_index - 1) % len(self.extractor_workers)]
                    print(
                        f"  source={source_id} extracting batch={batch_index}/{len(batches)} "
                        f"items={len(batch)} worker_model={worker.model}",
                        flush=True,
                    )
                    future = executor.submit(
                        extract_batch_with_llm,
                        worker.extractor,
                        batch,
                        retries=self.retries,
                        sleep_seconds=self.sleep_seconds,
                    )
                    future_to_batch[future] = batch_index
                for future in as_completed(future_to_batch):
                    entities, relations = future.result()
                    _merge_extraction_results(entities_by_id, relations_by_id, entities, relations)
            return entities_by_id, relations_by_id

        for batch_index, batch in enumerate(batches, start=1):
            print(
                f"  source={source_id} extracting batch={batch_index}/{len(batches)} "
                f"items={len(batch)}",
                flush=True,
            )
            entities, relations = self.extractor.extract(batch)
            _merge_extraction_results(entities_by_id, relations_by_id, entities, relations)
        return entities_by_id, relations_by_id

    def _publish_pending(self, client: httpx.Client) -> None:
        if self.dry_run or not self.pending_publish_source_ids:
            return
        source_ids = self.pending_publish_source_ids
        self.pending_publish_source_ids = []
        result = run_with_retries(
            lambda: self._request(client, "POST", "/ingestion/publish", json=source_ids),
            retries=self.retries,
            sleep_seconds=self.sleep_seconds,
            label="publish",
        )
        self.summary["publish_batches"] += 1
        self.summary["published_sources"] += len(source_ids)
        self.progress.append(
            {
                "status": "publish_batch",
                "source_ids": source_ids,
                "batch_id": result.get("batch_id"),
                "node_count": result.get("node_count"),
                "edge_count": result.get("edge_count"),
                "evidence_count": result.get("evidence_count"),
                "chunk_count": result.get("chunk_count"),
                "graph_persisted": result.get("graph_persisted"),
            }
        )
        print(
            f"[publish] sources={len(source_ids)} nodes={result.get('node_count')} "
            f"edges={result.get('edge_count')} graph_persisted={result.get('graph_persisted')}",
            flush=True,
        )

    def _request(
        self,
        client: httpx.Client,
        method: str,
        path: str,
        **kwargs,
    ) -> dict[str, Any]:
        response = client.request(method, f"{self.api_base}{path}", **kwargs)
        response.raise_for_status()
        return response.json()


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


def parse_api_keys(cli_keys: list[str], env_value: str, fallback_key: str) -> list[str]:
    keys: list[str] = []
    for raw in cli_keys:
        keys.extend(part.strip() for part in raw.split(",") if part.strip())
    keys.extend(part.strip() for part in env_value.split(",") if part.strip())
    if not keys and fallback_key:
        keys.append(fallback_key)
    return _unique_strings(keys)


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-extract LLM graph candidates from structural extraction units."
    )
    parser.add_argument("--postgres-dsn", default="")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS_PATH)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--limit-sources", type=int, default=0)
    parser.add_argument("--offset-sources", type=int, default=0)
    parser.add_argument("--max-chunks-per-source", type=int, default=0)
    parser.add_argument("--chunk-batch-size", type=int, default=DEFAULT_UNIT_BATCH_SIZE)
    parser.add_argument("--unit-batch-size", type=int, default=0)
    parser.add_argument("--unit-batch-min", type=int, default=DEFAULT_UNIT_BATCH_MIN)
    parser.add_argument("--unit-batch-max", type=int, default=DEFAULT_UNIT_BATCH_MAX)
    parser.add_argument("--unit-batch-max-chars", type=int, default=DEFAULT_UNIT_BATCH_MAX_CHARS)
    parser.add_argument("--publish-batch-size", type=int, default=5)
    parser.add_argument("--replace-empty", action="store_true")
    parser.add_argument("--llm-api-key", action="append", default=[])
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--per-key-concurrency", type=int, default=1)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    dsn = args.postgres_dsn or settings.postgres_dsn
    repository = IngestionRepository(create_engine(dsn, future=True, pool_pre_ping=True))
    source_ids = args.source_id or list_source_ids(repository)
    if args.offset_sources:
        source_ids = source_ids[args.offset_sources :]
    if args.limit_sources:
        source_ids = source_ids[: args.limit_sources]

    progress = ExtractionProgress(args.progress)
    api_keys = parse_api_keys(
        args.llm_api_key,
        os.getenv("LLM_API_KEYS", ""),
        settings.llm_api_key,
    )
    llm_model = args.llm_model or settings.llm_model
    preferred_unit_batch_size = args.unit_batch_size or args.chunk_batch_size
    unit_batch_min = min(args.unit_batch_min, preferred_unit_batch_size)
    unit_batch_max = max(args.unit_batch_max, preferred_unit_batch_size)
    extractor_workers = (
        build_extractor_workers(
            api_keys=api_keys,
            per_key_concurrency=args.per_key_concurrency,
            base_url=settings.llm_base_url,
            model=llm_model,
            timeout=args.llm_timeout,
        )
        if args.unit_batch_size or len(api_keys) > 1 or args.per_key_concurrency > 1
        else []
    )
    extractor = GraphExtractor(
        llm_extractor=StructuredExtractionClient(
            base_url=settings.llm_base_url,
            api_key=api_keys[0] if api_keys else settings.llm_api_key,
            model=llm_model,
            http_client=httpx.Client(timeout=args.llm_timeout, trust_env=False),
        )
    )
    runner = ExistingGraphRagReextractor(
        repository=repository,
        extractor=extractor,
        progress=progress,
        api_base=args.api_base,
        publish_batch_size=args.publish_batch_size,
        unit_batch_min=unit_batch_min,
        unit_batch_max=unit_batch_max,
        unit_batch_max_chars=args.unit_batch_max_chars,
        max_chunks_per_source=args.max_chunks_per_source,
        extractor_workers=extractor_workers,
        replace_empty=args.replace_empty,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        dry_run=args.dry_run,
    )
    print(
        f"sources={len(source_ids)} progress={args.progress} "
        f"unit_batch_min={unit_batch_min} unit_batch_max={unit_batch_max} "
        f"unit_batch_max_chars={args.unit_batch_max_chars} "
        f"max_chunks_per_source={args.max_chunks_per_source} "
        f"workers={len(extractor_workers) or 1} model={llm_model}",
        flush=True,
    )
    runner.run(source_ids)
    print(json.dumps(dict(runner.summary), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
