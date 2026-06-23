from typing import Literal

from pydantic import BaseModel, Field


class SourceManifest(BaseModel):
    source_id: str
    filename: str
    mime_type: str
    checksum: str
    status: Literal["registered", "parsed", "requires_ocr", "failed", "published"]
    version: int = 1
    object_key: str = ""


class IngestionJob(BaseModel):
    job_id: str
    source_ids: list[str] = Field(default_factory=list)
    status: Literal["queued", "running", "completed", "failed"] = "queued"


class DocumentPage(BaseModel):
    page_id: str
    source_id: str
    page_number: int
    text: str
    layout_json: dict = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    source_id: str
    page_id: str
    chunk_index: int
    content: str
    content_type: Literal["text", "table", "json", "ocr"] = "text"
    section_title: str = ""
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0
    metadata: dict = Field(default_factory=dict)


class EntityCandidate(BaseModel):
    entity_id: str
    name: str
    label: str
    normalized_name: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class RelationCandidate(BaseModel):
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation: str
    display: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class PublishBatch(BaseModel):
    batch_id: str
    source_ids: list[str] = Field(default_factory=list)
    status: Literal["queued", "published", "failed"] = "queued"
    node_count: int = 0
    edge_count: int = 0
    chunk_count: int = 0


class KnowledgeBundle(BaseModel):
    source: SourceManifest
    pages: list[DocumentPage] = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)
    entities: list[EntityCandidate] = Field(default_factory=list)
    relations: list[RelationCandidate] = Field(default_factory=list)
    publish_batch: PublishBatch | None = None
