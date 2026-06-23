from __future__ import annotations

import csv
import json
from io import BytesIO, StringIO
from pathlib import Path

from app.models.ingestion import DocumentChunk, DocumentPage, SourceManifest

TEXT_MIME_TYPES = {"text/plain", "text/markdown", "application/x-markdown"}
CSV_MIME_TYPES = {"text/csv", "application/csv"}
JSON_MIME_TYPES = {"application/json"}
IMAGE_MIME_PREFIX = "image/"


class DocumentParser:
    def __init__(self, ocr_client=None, chunk_chars: int = 900):
        self.ocr_client = ocr_client
        self.chunk_chars = chunk_chars

    def parse(
        self,
        source: SourceManifest,
        content: bytes,
    ) -> tuple[list[DocumentPage], list[DocumentChunk], str]:
        text = ""
        content_type = "text"
        status = "parsed"
        if source.mime_type in TEXT_MIME_TYPES or _suffix(source.filename) in {".txt", ".md"}:
            text = _decode(content)
        elif source.mime_type in CSV_MIME_TYPES or _suffix(source.filename) == ".csv":
            text = _csv_to_markdown(content)
            content_type = "table"
        elif source.mime_type in JSON_MIME_TYPES or _suffix(source.filename) == ".json":
            text = json.dumps(json.loads(_decode(content)), ensure_ascii=False, indent=2)
            content_type = "json"
        elif source.mime_type.startswith(IMAGE_MIME_PREFIX):
            if not self.ocr_client:
                return [], [], "requires_ocr"
            text = self.ocr_client.recognize_image(content, mime_type=source.mime_type)
            content_type = "ocr"
        elif _suffix(source.filename) == ".docx":
            text = _docx_text(content)
        elif _suffix(source.filename) == ".pdf":
            text = _pdf_text(content)
            if not text.strip():
                status = "requires_ocr"
        else:
            text = _decode(content)

        if not text.strip():
            return [], [], status

        page = DocumentPage(
            page_id=f"page:{source.source_id}:1",
            source_id=source.source_id,
            page_number=1,
            text=text,
        )
        chunks = _chunk_text(
            source_id=source.source_id,
            page_id=page.page_id,
            text=text,
            content_type=content_type,
            chunk_chars=self.chunk_chars,
        )
        return [page], chunks, status


def _chunk_text(
    source_id: str,
    page_id: str,
    text: str,
    content_type: str,
    chunk_chars: int,
) -> list[DocumentChunk]:
    normalized = text.strip()
    chunks: list[DocumentChunk] = []
    start = 0
    index = 1
    while start < len(normalized):
        end = min(start + chunk_chars, len(normalized))
        content = normalized[start:end].strip()
        if content:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk:{source_id}:{index:04d}",
                    source_id=source_id,
                    page_id=page_id,
                    chunk_index=index,
                    content=content,
                    content_type=content_type,
                    token_count=len(content),
                    char_start=start,
                    char_end=end,
                )
            )
            index += 1
        start = end
    return chunks


def _decode(content: bytes) -> str:
    for encoding in ("utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _csv_to_markdown(content: bytes) -> str:
    text = _decode(content)
    rows = list(csv.reader(StringIO(text)))
    if not rows:
        return ""
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)


def _docx_text(content: bytes) -> str:
    from docx import Document

    document = Document(BytesIO(content))
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


def _pdf_text(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _suffix(filename: str) -> str:
    return Path(filename).suffix.lower()
