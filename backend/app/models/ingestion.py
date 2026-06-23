from typing import Literal

from pydantic import BaseModel, Field


class SourceManifest(BaseModel):
    source_id: str
    filename: str
    mime_type: str
    checksum: str
    status: Literal["registered", "parsed", "requires_ocr", "failed", "published"]
    version: int = 1


class IngestionJob(BaseModel):
    job_id: str
    source_ids: list[str] = Field(default_factory=list)
    status: Literal["queued", "running", "completed", "failed"] = "queued"
