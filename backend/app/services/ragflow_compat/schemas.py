from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

VectorStatus = Literal["missing", "queued", "embedded", "failed"]


@dataclass(frozen=True)
class RetrievalDocument:
    doc_id: str
    source_id: str
    filename: str
    mime_type: str = ""
    checksum: str = ""
    status: str = "parsed"
    object_key: str = ""
    source_version: int = 1
    chunk_count: int = 0
    eligible_chunk_count: int = 0
    created_from: str = "document_chunks"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalChunk:
    chunk_id: str
    doc_id: str
    source_id: str
    parent_unit_id: str
    chunk_order_int: int
    page_num_int: int
    title: str
    section_path: list[str]
    content: str
    content_with_weight: str
    content_ltks: str = ""
    title_tks: str = ""
    important_kwd: list[str] = field(default_factory=list)
    question_tks: str = ""
    content_type: str = "text"
    token_count: int = 0
    available_int: int = 1
    vector_point_id: str = ""
    vector_status: VectorStatus = "missing"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalChunkTerm:
    chunk_id: str
    term: str
    term_type: str
    weight: float = 1.0


@dataclass(frozen=True)
class RetrievalKgEntity:
    entity_id: str
    entity_name: str
    entity_type: str
    source_node_id: str
    content_with_weight: str
    description: str = ""
    rank_flt: float = 1.0
    n_hop_with_weight: list[dict[str, Any]] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    evidence_chunk_ids: list[str] = field(default_factory=list)
    available_int: int = 1
    vector_point_id: str = ""
    vector_status: VectorStatus = "missing"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalKgRelation:
    relation_id: str
    from_entity_kwd: str
    to_entity_kwd: str
    relation_type: str
    display: str
    content_with_weight: str
    weight_int: int = 1
    evidence_chunk_ids: list[str] = field(default_factory=list)
    source_edge_id: str = ""
    available_int: int = 1
    vector_point_id: str = ""
    vector_status: VectorStatus = "missing"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalCommunityReport:
    report_id: str
    title: str
    content_with_weight: str
    summary: str = ""
    evidences: str = ""
    entities_kwd: list[str] = field(default_factory=list)
    weight_flt: float = 0.0
    source_id: list[str] = field(default_factory=list)
    available_int: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalGraphArtifact:
    artifact_id: str
    artifact_type: str
    content_with_weight: str
    source_id: list[str] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    available_int: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalTypeSamples:
    entity_type: str
    sample_entities: list[str]
    sample_count: int
    updated_at: str


@dataclass(frozen=True)
class RetrievalAudit:
    documents: int
    chunks: int
    chunks_with_vectors: int
    chunks_failed_vectors: int
    kg_entities: int
    kg_entities_with_vectors: int
    kg_entities_failed_vectors: int
    kg_entities_with_evidence: int
    kg_relations: int
    community_reports: int
    graph_artifacts: int
    kg_relations_with_vectors: int
    kg_relations_failed_vectors: int
    kg_relations_with_evidence: int
    short_chunks: int
    long_chunks: int
