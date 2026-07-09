from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

from sqlalchemy import create_engine, select

PROJECT_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(PROJECT_BACKEND) not in sys.path:
    sys.path.insert(0, str(PROJECT_BACKEND))

from app.core.config import get_settings
from app.models.ingestion import SourceManifest
from app.services.document_parser import DocumentParser
from app.services.ingestion_repository import IngestionRepository, sources_table
from app.services.object_storage import LocalObjectStorage, MinioObjectStorage

DEFAULT_PROGRESS_PATH = Path("/tmp/tcm_structural_chunk_rebuild_progress.jsonl")
COMPLETED_STATUSES = {"rebuilt", "requires_ocr", "empty"}


class RebuildProgress:
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
class RebuildStats:
    source_id: str
    filename: str
    unit_count: int
    chunk_count: int
    page_count: int


def list_rebuildable_sources(
    repository: IngestionRepository,
    source_ids: list[str] | None = None,
) -> list[SourceManifest]:
    statement = select(sources_table).where(sources_table.c.object_key != "").order_by(
        sources_table.c.source_id
    )
    if source_ids:
        statement = statement.where(sources_table.c.source_id.in_(source_ids))
    with repository.engine.begin() as connection:
        return [SourceManifest(**dict(row._mapping)) for row in connection.execute(statement)]


class StructuralChunkRebuilder:
    def __init__(
        self,
        *,
        repository: IngestionRepository,
        storage,
        parser: DocumentParser,
        progress: RebuildProgress,
        dry_run: bool,
    ):
        self.repository = repository
        self.storage = storage
        self.parser = parser
        self.progress = progress
        self.dry_run = dry_run
        self.summary: Counter[str] = Counter()

    def run(self, sources: list[SourceManifest]) -> None:
        for index, source in enumerate(sources, start=1):
            if source.source_id in self.progress.completed_source_ids:
                self.summary["already_completed"] += 1
                continue
            self._run_source(source, index=index, total=len(sources))

    def _run_source(self, source: SourceManifest, *, index: int, total: int) -> None:
        try:
            content = self.storage.get_bytes(source.object_key)
            pages, units, chunks, status = self.parser.parse(source, content)
            row = {
                "source_id": source.source_id,
                "filename": source.filename,
                "status": _row_status(status, units, chunks),
                "page_count": len(pages),
                "unit_count": len(units),
                "chunk_count": len(chunks),
            }
            if not self.dry_run:
                source.status = _next_source_status(source.status, status)
                self.repository.upsert_source(source)
                self.repository.replace_pages_units_and_chunks(source.source_id, pages, units, chunks)
                self.progress.append(row)
            self.summary[row["status"]] += 1
            self.summary["total_units"] += len(units)
            self.summary["total_chunks"] += len(chunks)
            print(
                f"[{index}/{total}] {row['status']} source={source.source_id} "
                f"units={len(units)} chunks={len(chunks)} {source.filename}",
                flush=True,
            )
        except Exception as exc:
            row = {
                "source_id": source.source_id,
                "filename": source.filename,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if not self.dry_run:
                self.progress.append(row)
            self.summary["failed"] += 1
            print(f"[{index}/{total}] failed source={source.source_id}: {row['error']}", flush=True)


def _row_status(parser_status: str, units, chunks) -> str:
    if parser_status == "requires_ocr":
        return "requires_ocr"
    if not units and not chunks:
        return "empty"
    return "rebuilt"


def _next_source_status(current_status: str, parser_status: str) -> str:
    if current_status == "published" and parser_status == "parsed":
        return "published"
    return parser_status


def build_storage(settings, *, local_storage_root: Path | None = None):
    if local_storage_root:
        return LocalObjectStorage(local_storage_root)
    return MinioObjectStorage(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
        secure=settings.minio_secure,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild structural extraction units and retrieval chunks from stored source files."
    )
    parser.add_argument("--postgres-dsn", default="")
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS_PATH)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--limit-sources", type=int, default=0)
    parser.add_argument("--offset-sources", type=int, default=0)
    parser.add_argument("--local-storage-root", type=Path, default=None)
    parser.add_argument("--parent-target-chars", type=int, default=2400)
    parser.add_argument("--parent-max-chars", type=int, default=4000)
    parser.add_argument("--child-target-chars", type=int, default=360)
    parser.add_argument("--child-max-chars", type=int, default=520)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    dsn = args.postgres_dsn or settings.postgres_dsn
    repository = IngestionRepository(create_engine(dsn, future=True, pool_pre_ping=True))
    sources = list_rebuildable_sources(repository, args.source_id or None)
    if args.offset_sources:
        sources = sources[args.offset_sources :]
    if args.limit_sources:
        sources = sources[: args.limit_sources]
    parser = DocumentParser(
        parent_target_chars=args.parent_target_chars,
        parent_max_chars=args.parent_max_chars,
        child_target_chars=args.child_target_chars,
        child_max_chars=args.child_max_chars,
    )
    rebuilder = StructuralChunkRebuilder(
        repository=repository,
        storage=build_storage(settings, local_storage_root=args.local_storage_root),
        parser=parser,
        progress=RebuildProgress(args.progress),
        dry_run=args.dry_run,
    )
    print(
        f"sources={len(sources)} progress={args.progress} "
        f"parent={args.parent_target_chars}/{args.parent_max_chars} "
        f"child={args.child_target_chars}/{args.child_max_chars} dry_run={args.dry_run}",
        flush=True,
    )
    rebuilder.run(sources)
    print(json.dumps(dict(rebuilder.summary), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
