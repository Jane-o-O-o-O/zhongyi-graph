from __future__ import annotations

from sqlalchemy import JSON, Column, Float, Integer, MetaData, String, Table, Text

retrieval_metadata = MetaData()

retrieval_documents_table = Table(
    "retrieval_documents",
    retrieval_metadata,
    Column("doc_id", String, primary_key=True),
    Column("source_id", String, nullable=False, unique=True, index=True),
    Column("filename", String, nullable=False),
    Column("mime_type", String, nullable=False, default=""),
    Column("checksum", String, nullable=False, default=""),
    Column("status", String, nullable=False, default="parsed"),
    Column("object_key", String, nullable=False, default=""),
    Column("source_version", Integer, nullable=False, default=1),
    Column("chunk_count", Integer, nullable=False, default=0),
    Column("eligible_chunk_count", Integer, nullable=False, default=0),
    Column("created_from", String, nullable=False, default="document_chunks"),
    Column("metadata", JSON, nullable=False, default=dict),
)

retrieval_chunks_table = Table(
    "retrieval_chunks",
    retrieval_metadata,
    Column("chunk_id", String, primary_key=True),
    Column("doc_id", String, nullable=False, index=True),
    Column("source_id", String, nullable=False, index=True),
    Column("parent_unit_id", String, nullable=False, default="", index=True),
    Column("chunk_order_int", Integer, nullable=False, default=0),
    Column("page_num_int", Integer, nullable=False, default=1),
    Column("title", String, nullable=False, default=""),
    Column("section_path", JSON, nullable=False, default=list),
    Column("content", Text, nullable=False),
    Column("content_with_weight", Text, nullable=False),
    Column("content_ltks", Text, nullable=False, default=""),
    Column("title_tks", Text, nullable=False, default=""),
    Column("important_kwd", JSON, nullable=False, default=list),
    Column("question_tks", Text, nullable=False, default=""),
    Column("content_type", String, nullable=False, default="text"),
    Column("token_count", Integer, nullable=False, default=0),
    Column("available_int", Integer, nullable=False, default=1, index=True),
    Column("vector_point_id", String, nullable=False, default=""),
    Column("vector_status", String, nullable=False, default="missing", index=True),
    Column("metadata", JSON, nullable=False, default=dict),
)

retrieval_chunk_terms_table = Table(
    "retrieval_chunk_terms",
    retrieval_metadata,
    Column("chunk_id", String, primary_key=True),
    Column("term", String, primary_key=True),
    Column("term_type", String, primary_key=True),
    Column("weight", Float, nullable=False, default=1.0),
)

retrieval_kg_entities_table = Table(
    "retrieval_kg_entities",
    retrieval_metadata,
    Column("entity_id", String, primary_key=True),
    Column("entity_name", String, nullable=False, index=True),
    Column("entity_type", String, nullable=False, index=True),
    Column("source_node_id", String, nullable=False, default=""),
    Column("content_with_weight", Text, nullable=False),
    Column("description", Text, nullable=False, default=""),
    Column("rank_flt", Float, nullable=False, default=1.0),
    Column("n_hop_with_weight", JSON, nullable=False, default=list),
    Column("aliases", JSON, nullable=False, default=list),
    Column("evidence_chunk_ids", JSON, nullable=False, default=list),
    Column("available_int", Integer, nullable=False, default=1, index=True),
    Column("vector_point_id", String, nullable=False, default=""),
    Column("vector_status", String, nullable=False, default="missing", index=True),
    Column("metadata", JSON, nullable=False, default=dict),
)

retrieval_kg_relations_table = Table(
    "retrieval_kg_relations",
    retrieval_metadata,
    Column("relation_id", String, primary_key=True),
    Column("from_entity_kwd", String, nullable=False, index=True),
    Column("to_entity_kwd", String, nullable=False, index=True),
    Column("relation_type", String, nullable=False, index=True),
    Column("display", String, nullable=False, default=""),
    Column("content_with_weight", Text, nullable=False),
    Column("weight_int", Integer, nullable=False, default=1),
    Column("evidence_chunk_ids", JSON, nullable=False, default=list),
    Column("source_edge_id", String, nullable=False, default=""),
    Column("available_int", Integer, nullable=False, default=1, index=True),
    Column("vector_point_id", String, nullable=False, default=""),
    Column("vector_status", String, nullable=False, default="missing", index=True),
    Column("metadata", JSON, nullable=False, default=dict),
)

retrieval_kg_type_samples_table = Table(
    "retrieval_kg_type_samples",
    retrieval_metadata,
    Column("entity_type", String, primary_key=True),
    Column("sample_entities", JSON, nullable=False, default=list),
    Column("sample_count", Integer, nullable=False, default=0),
    Column("updated_at", String, nullable=False, default=""),
)

retrieval_sync_state_table = Table(
    "retrieval_sync_state",
    retrieval_metadata,
    Column("sync_key", String, primary_key=True),
    Column("status", String, nullable=False),
    Column("started_at", String, nullable=False, default=""),
    Column("finished_at", String, nullable=False, default=""),
    Column("cursor", String, nullable=False, default=""),
    Column("total", Integer, nullable=False, default=0),
    Column("processed", Integer, nullable=False, default=0),
    Column("failed", Integer, nullable=False, default=0),
    Column("metadata", JSON, nullable=False, default=dict),
)

retrieval_query_logs_table = Table(
    "retrieval_query_logs",
    retrieval_metadata,
    Column("query_id", String, primary_key=True),
    Column("question", Text, nullable=False),
    Column("rewrite_result", JSON, nullable=False, default=dict),
    Column("chunk_candidates", JSON, nullable=False, default=list),
    Column("kg_entities", JSON, nullable=False, default=list),
    Column("kg_relations", JSON, nullable=False, default=list),
    Column("final_evidence", JSON, nullable=False, default=list),
    Column("latency_ms", Integer, nullable=False, default=0),
    Column("created_at", String, nullable=False, default=""),
)
