from __future__ import annotations

import json
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from app.models.ingestion import (
    DocumentChunk,
    DocumentPage,
    EntityCandidate,
    KnowledgeBundle,
    PublishBatch,
    RelationCandidate,
    SourceManifest,
)

metadata = MetaData()

sources_table = Table(
    "sources",
    metadata,
    Column("source_id", String, primary_key=True),
    Column("filename", String, nullable=False),
    Column("mime_type", String, nullable=False),
    Column("checksum", String, nullable=False),
    Column("status", String, nullable=False),
    Column("version", Integer, nullable=False, default=1),
    Column("object_key", String, nullable=False, default=""),
)

pages_table = Table(
    "document_pages",
    metadata,
    Column("page_id", String, primary_key=True),
    Column("source_id", String, nullable=False, index=True),
    Column("page_number", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("layout_json", JSON, nullable=False, default=dict),
)

chunks_table = Table(
    "document_chunks",
    metadata,
    Column("chunk_id", String, primary_key=True),
    Column("source_id", String, nullable=False, index=True),
    Column("page_id", String, nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("content_type", String, nullable=False),
    Column("section_title", String, nullable=False, default=""),
    Column("token_count", Integer, nullable=False, default=0),
    Column("char_start", Integer, nullable=False, default=0),
    Column("char_end", Integer, nullable=False, default=0),
    Column("metadata", JSON, nullable=False, default=dict),
)

entities_table = Table(
    "entity_candidates",
    metadata,
    Column("source_id", String, primary_key=True),
    Column("entity_id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("label", String, nullable=False),
    Column("normalized_name", String, nullable=False),
    Column("source_chunk_ids", JSON, nullable=False, default=list),
    Column("confidence", Float, nullable=False, default=0.0),
)

relations_table = Table(
    "relation_candidates",
    metadata,
    Column("source_id", String, primary_key=True),
    Column("relation_id", String, primary_key=True),
    Column("source_entity_id", String, nullable=False),
    Column("target_entity_id", String, nullable=False),
    Column("relation", String, nullable=False),
    Column("display", String, nullable=False),
    Column("evidence_chunk_ids", JSON, nullable=False, default=list),
    Column("confidence", Float, nullable=False, default=0.0),
)

publish_batches_table = Table(
    "publish_batches",
    metadata,
    Column("batch_id", String, primary_key=True),
    Column("source_ids", JSON, nullable=False, default=list),
    Column("status", String, nullable=False),
    Column("node_count", Integer, nullable=False, default=0),
    Column("edge_count", Integer, nullable=False, default=0),
    Column("chunk_count", Integer, nullable=False, default=0),
)

class IngestionRepository:
    def __init__(self, engine: Engine):
        self.engine = engine
        _migrate_candidate_primary_keys(engine)
        metadata.create_all(engine)

    @classmethod
    def in_memory(cls) -> "IngestionRepository":
        return cls(
            create_engine(
                "sqlite+pysqlite://",
                future=True,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        )

    @classmethod
    def from_dsn(cls, dsn: str) -> "IngestionRepository":
        return cls(create_engine(dsn, future=True, pool_pre_ping=True))

    def upsert_source(self, source: SourceManifest) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(sources_table).where(sources_table.c.source_id == source.source_id))
            connection.execute(sources_table.insert().values(**source.model_dump()))

    def replace_pages_and_chunks(
        self,
        source_id: str,
        pages: list[DocumentPage],
        chunks: list[DocumentChunk],
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(pages_table).where(pages_table.c.source_id == source_id))
            connection.execute(delete(chunks_table).where(chunks_table.c.source_id == source_id))
            if pages:
                connection.execute(pages_table.insert(), [page.model_dump() for page in pages])
            if chunks:
                connection.execute(chunks_table.insert(), [_chunk_row(chunk) for chunk in chunks])

    def save_candidates(
        self,
        source_id: str,
        entities: list[EntityCandidate],
        relations: list[RelationCandidate],
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(entities_table).where(entities_table.c.source_id == source_id))
            connection.execute(delete(relations_table).where(relations_table.c.source_id == source_id))
            if entities:
                connection.execute(
                    entities_table.insert(),
                    [entity.model_dump() | {"source_id": source_id} for entity in entities],
                )
            if relations:
                connection.execute(
                    relations_table.insert(),
                    [relation.model_dump() | {"source_id": source_id} for relation in relations],
                )

    def merge_candidates(
        self,
        source_id: str,
        entities: list[EntityCandidate],
        relations: list[RelationCandidate],
    ) -> None:
        bundle = self.get_bundle(source_id)
        merged_entities = {entity.entity_id: entity for entity in bundle.entities}
        merged_relations = {relation.relation_id: relation for relation in bundle.relations}

        for entity in entities:
            existing = merged_entities.get(entity.entity_id)
            merged_entities[entity.entity_id] = _merge_entity(existing, entity)

        for relation in relations:
            existing = merged_relations.get(relation.relation_id)
            merged_relations[relation.relation_id] = _merge_relation(existing, relation)

        self.save_candidates(source_id, list(merged_entities.values()), list(merged_relations.values()))

    def list_chunks(self, source_id: str | None = None) -> list[DocumentChunk]:
        statement = select(chunks_table).order_by(chunks_table.c.source_id, chunks_table.c.chunk_index)
        if source_id:
            statement = statement.where(chunks_table.c.source_id == source_id)
        with self.engine.begin() as connection:
            return [_chunk_from_row(row._mapping) for row in connection.execute(statement)]

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(chunks_table).where(chunks_table.c.chunk_id == chunk_id)
            ).first()
            return _chunk_from_row(row._mapping) if row else None

    def list_published_source_ids(self) -> list[str]:
        with self.engine.begin() as connection:
            return [
                row.source_id
                for row in connection.execute(
                    select(sources_table.c.source_id)
                    .where(sources_table.c.status == "published")
                    .order_by(sources_table.c.source_id)
                )
            ]

    def get_bundle(self, source_id: str) -> KnowledgeBundle:
        with self.engine.begin() as connection:
            source_row = connection.execute(
                select(sources_table).where(sources_table.c.source_id == source_id)
            ).first()
            if source_row is None:
                raise KeyError(source_id)
            pages = [
                DocumentPage(**dict(row._mapping))
                for row in connection.execute(
                    select(pages_table)
                    .where(pages_table.c.source_id == source_id)
                    .order_by(pages_table.c.page_number)
                )
            ]
            chunks = [
                _chunk_from_row(row._mapping)
                for row in connection.execute(
                    select(chunks_table)
                    .where(chunks_table.c.source_id == source_id)
                    .order_by(chunks_table.c.chunk_index)
                )
            ]
            entities = [
                EntityCandidate(**_candidate_mapping(row._mapping, "source_chunk_ids"))
                for row in connection.execute(
                    select(entities_table).where(entities_table.c.source_id == source_id)
                )
            ]
            relations = [
                RelationCandidate(**_candidate_mapping(row._mapping, "evidence_chunk_ids"))
                for row in connection.execute(
                    select(relations_table).where(relations_table.c.source_id == source_id)
                )
            ]
        return KnowledgeBundle(
            source=SourceManifest(**dict(source_row._mapping)),
            pages=pages,
            chunks=chunks,
            entities=entities,
            relations=relations,
        )

    def record_publish_batch(
        self,
        source_ids: list[str],
        node_count: int,
        edge_count: int,
        chunk_count: int,
    ) -> PublishBatch:
        batch = PublishBatch(
            batch_id=f"publish:{uuid4().hex}",
            source_ids=source_ids,
            status="published",
            node_count=node_count,
            edge_count=edge_count,
            chunk_count=chunk_count,
        )
        with self.engine.begin() as connection:
            connection.execute(publish_batches_table.insert().values(**batch.model_dump()))
        return batch


def _chunk_row(chunk: DocumentChunk) -> dict:
    data = chunk.model_dump()
    data["metadata"] = _jsonable(data["metadata"])
    return data


def _chunk_from_row(row) -> DocumentChunk:
    data = dict(row)
    data["metadata"] = _ensure_json(data.get("metadata", {}))
    return DocumentChunk(**data)


def _candidate_mapping(row, json_field: str) -> dict:
    data = dict(row)
    data.pop("source_id", None)
    data[json_field] = _ensure_json(data.get(json_field, []))
    return data


def _merge_entity(
    existing: EntityCandidate | None,
    incoming: EntityCandidate,
) -> EntityCandidate:
    if existing is None:
        return incoming
    return existing.model_copy(
        update={
            "source_chunk_ids": _unique_sorted(existing.source_chunk_ids + incoming.source_chunk_ids),
            "confidence": max(existing.confidence, incoming.confidence),
        }
    )


def _merge_relation(
    existing: RelationCandidate | None,
    incoming: RelationCandidate,
) -> RelationCandidate:
    if existing is None:
        return incoming
    return existing.model_copy(
        update={
            "evidence_chunk_ids": _unique_sorted(
                existing.evidence_chunk_ids + incoming.evidence_chunk_ids
            ),
            "confidence": max(existing.confidence, incoming.confidence),
        }
    )


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted(set(values))


def _ensure_json(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _jsonable(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _migrate_candidate_primary_keys(engine: Engine) -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    migrations = [
        ("entity_candidates", ["source_id", "entity_id"], "entity_candidates_pkey"),
        ("relation_candidates", ["source_id", "relation_id"], "relation_candidates_pkey"),
    ]
    for table_name, expected_columns, postgres_constraint in migrations:
        if table_name not in existing_tables:
            continue
        primary_key = inspector.get_pk_constraint(table_name).get("constrained_columns") or []
        if primary_key == expected_columns:
            continue
        if engine.dialect.name == "postgresql":
            with engine.begin() as connection:
                connection.execute(text(f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {postgres_constraint}"))
                connection.execute(text(f"ALTER TABLE {table_name} ADD PRIMARY KEY ({', '.join(expected_columns)})"))
            continue
        metadata.tables[table_name].drop(engine, checkfirst=True)
