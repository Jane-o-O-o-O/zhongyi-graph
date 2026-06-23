from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
import time
from typing import Any

import httpx

DEFAULT_API_BASE = "http://127.0.0.1:8000/api"
DEFAULT_PROGRESS_PATH = Path("/tmp/tcm_literature_import_progress_v2.jsonl")
DEFAULT_SUMMARY_PATH = Path("/tmp/tcm_literature_import_summary_v2.json")
DEFAULT_MAX_JSON_BYTES = 20 * 1024 * 1024
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
COMPLETED_STATUSES = {"parsed", "published", "requires_ocr", "skipped"}


@dataclass(frozen=True)
class FileClassification:
    path: Path
    importable: bool
    reason: str


class ImportProgress:
    def __init__(self, path: Path):
        self.path = path
        self.completed_paths: set[str] = set()
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
            self.counts[status] += 1
            if status in COMPLETED_STATUSES and row.get("path"):
                self.completed_paths.add(str(row["path"]))

    def append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        status = str(row.get("status", "unknown"))
        self.counts[status] += 1
        if status in COMPLETED_STATUSES and row.get("path"):
            self.completed_paths.add(str(row["path"]))


def classify_file(
    path: Path,
    *,
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
    include_images: bool = True,
) -> FileClassification:
    name = path.name
    suffix = path.suffix.lower()
    if name == ".DS_Store" or name.startswith("._"):
        return FileClassification(path=path, importable=False, reason="hidden_or_macos_metadata")
    if suffix == ".json" and path.stat().st_size > max_json_bytes:
        return FileClassification(path=path, importable=False, reason="large_json")
    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        return FileClassification(path=path, importable=True, reason="supported")
    if include_images and suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return FileClassification(path=path, importable=True, reason="supported")
    return FileClassification(path=path, importable=False, reason="unsupported")


def iter_importable_files(
    root: Path,
    *,
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
    include_images: bool = True,
) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        if not path.is_file():
            continue
        classification = classify_file(
            path,
            max_json_bytes=max_json_bytes,
            include_images=include_images,
        )
        if classification.importable:
            files.append(path)
    return files


def count_skips(
    root: Path,
    *,
    max_json_bytes: int,
    include_images: bool,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        classification = classify_file(
            path,
            max_json_bytes=max_json_bytes,
            include_images=include_images,
        )
        if not classification.importable:
            counts[classification.reason] += 1
    return counts


def should_publish_run(run_result: dict[str, Any], *, publish_all: bool) -> bool:
    if publish_all:
        return True
    return int(run_result.get("relation_count") or 0) > 0


class GraphRagImporter:
    def __init__(
        self,
        *,
        api_base: str,
        progress: ImportProgress,
        publish_batch_size: int,
        publish_all: bool,
        dry_run: bool,
        retries: int,
        sleep_seconds: float,
    ):
        self.api_base = api_base.rstrip("/")
        self.progress = progress
        self.publish_batch_size = publish_batch_size
        self.publish_all = publish_all
        self.dry_run = dry_run
        self.retries = retries
        self.sleep_seconds = sleep_seconds
        self.pending_publish_source_ids: list[str] = []
        self.summary: Counter[str] = Counter()

    def import_files(self, files: list[Path]) -> None:
        if self.dry_run:
            for path in files:
                if str(path) in self.progress.completed_paths:
                    self.summary["already_completed"] += 1
                else:
                    self.summary["would_import"] += 1
            return

        with httpx.Client(timeout=None, trust_env=False) as client:
            for index, path in enumerate(files, start=1):
                if str(path) in self.progress.completed_paths:
                    self.summary["already_completed"] += 1
                    continue
                self._import_one(client, path, index=index, total=len(files))
            self._publish_pending(client)

    def _import_one(self, client: httpx.Client, path: Path, *, index: int, total: int) -> None:
        try:
            source = self._upload(client, path)
            job = self._request(client, "POST", "/ingestion/jobs", json=[source["source_id"]])
            run_result = self._request(client, "POST", f"/ingestion/jobs/{job['job_id']}/run")
            row = {
                "path": str(path),
                "status": run_result_status(run_result),
                "source_id": source["source_id"],
                "job_id": job["job_id"],
                "chunk_count": int(run_result.get("chunk_count") or 0),
                "entity_count": int(run_result.get("entity_count") or 0),
                "relation_count": int(run_result.get("relation_count") or 0),
            }
            self.progress.append(row)
            self.summary[row["status"]] += 1
            if should_publish_run(run_result, publish_all=self.publish_all):
                self.pending_publish_source_ids.append(source["source_id"])
            if len(self.pending_publish_source_ids) >= self.publish_batch_size:
                self._publish_pending(client)
            print(
                f"[{index}/{total}] {row['status']} chunks={row['chunk_count']} "
                f"entities={row['entity_count']} relations={row['relation_count']} {path.name}",
                flush=True,
            )
        except Exception as exc:
            row = {
                "path": str(path),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            self.progress.append(row)
            self.summary["failed"] += 1
            print(f"[{index}/{total}] failed {path.name}: {row['error']}", flush=True)

    def _upload(self, client: httpx.Client, path: Path) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(path.name)[0] or mime_type_for_suffix(path.suffix)
        with path.open("rb") as handle:
            return self._request(
                client,
                "POST",
                "/ingestion/upload",
                files={"file": (path.name, handle, mime_type)},
            )

    def _publish_pending(self, client: httpx.Client) -> None:
        if not self.pending_publish_source_ids:
            return
        source_ids = self.pending_publish_source_ids
        self.pending_publish_source_ids = []
        try:
            result = self._request(client, "POST", "/ingestion/publish", json=source_ids)
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
        except Exception:
            self.pending_publish_source_ids = source_ids + self.pending_publish_source_ids
            raise

    def _request(
        self,
        client: httpx.Client,
        method: str,
        path: str,
        **kwargs,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = client.request(method, f"{self.api_base}{path}", **kwargs)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(self.sleep_seconds * attempt)
        assert last_error is not None
        raise last_error


def run_result_status(run_result: dict[str, Any]) -> str:
    if int(run_result.get("chunk_count") or 0) == 0:
        return "requires_ocr"
    return "parsed"


def mime_type_for_suffix(suffix: str) -> str:
    return {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(suffix.lower(), "application/octet-stream")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import local TCM literature through GraphRAG ingestion API.")
    parser.add_argument("root", type=Path, help="Literature folder to import.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--max-json-bytes", type=int, default=DEFAULT_MAX_JSON_BYTES)
    parser.add_argument("--include-images", action="store_true", help="Also import png/jpg/jpeg with OCR.")
    parser.add_argument("--publish-all", action="store_true", help="Publish every parsed source, even without relations.")
    parser.add_argument("--publish-batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    files = iter_importable_files(
        root,
        max_json_bytes=args.max_json_bytes,
        include_images=args.include_images,
    )
    if args.limit:
        files = files[: args.limit]
    skipped = count_skips(
        root,
        max_json_bytes=args.max_json_bytes,
        include_images=args.include_images,
    )
    progress = ImportProgress(args.progress)
    importer = GraphRagImporter(
        api_base=args.api_base,
        progress=progress,
        publish_batch_size=args.publish_batch_size,
        publish_all=args.publish_all,
        dry_run=args.dry_run,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
    )
    print(
        f"root={root} selected={len(files)} skipped={sum(skipped.values())} "
        f"progress={args.progress}",
        flush=True,
    )
    importer.import_files(files)
    summary = {
        "root": str(root),
        "selected": len(files),
        "skipped": dict(skipped),
        "progress_path": str(args.progress),
        "summary": dict(importer.summary),
    }
    write_summary(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
