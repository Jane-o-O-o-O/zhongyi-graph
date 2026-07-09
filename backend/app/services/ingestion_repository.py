from __future__ import annotations

import json
from uuid import uuid4

from collections.abc import Iterable

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
    func,
)
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from app.models.ingestion import (
    DocumentChunk,
    DocumentPage,
    EntityCandidate,
    ExtractionUnit,
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

extraction_units_table = Table(
    "extraction_units",
    metadata,
    Column("unit_id", String, primary_key=True),
    Column("source_id", String, nullable=False, index=True),
    Column("page_id", String, nullable=False),
    Column("unit_index", Integer, nullable=False),
    Column("title", String, nullable=False, default=""),
    Column("content", Text, nullable=False),
    Column("unit_type", String, nullable=False, default="text"),
    Column("section_path", JSON, nullable=False, default=list),
    Column("token_count", Integer, nullable=False, default=0),
    Column("char_start", Integer, nullable=False, default=0),
    Column("char_end", Integer, nullable=False, default=0),
    Column("metadata", JSON, nullable=False, default=dict),
)

chunks_table = Table(
    "document_chunks",
    metadata,
    Column("chunk_id", String, primary_key=True),
    Column("source_id", String, nullable=False, index=True),
    Column("page_id", String, nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("parent_unit_id", String, nullable=False, default=""),
    Column("unit_index", Integer, nullable=False, default=0),
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
        _migrate_structural_chunking_columns(engine)

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
        fallback_units = _fallback_units_from_chunks(source_id, pages, chunks)
        self.replace_pages_units_and_chunks(source_id, pages, fallback_units, chunks)

    def replace_pages_units_and_chunks(
        self,
        source_id: str,
        pages: list[DocumentPage],
        units: list[ExtractionUnit],
        chunks: list[DocumentChunk],
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(pages_table).where(pages_table.c.source_id == source_id))
            connection.execute(
                delete(extraction_units_table).where(extraction_units_table.c.source_id == source_id)
            )
            connection.execute(delete(chunks_table).where(chunks_table.c.source_id == source_id))
            if pages:
                connection.execute(pages_table.insert(), [page.model_dump() for page in pages])
            if units:
                connection.execute(extraction_units_table.insert(), [_unit_row(unit) for unit in units])
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

    def iter_chunk_batches(self, batch_size: int = 1000) -> Iterable[list[DocumentChunk]]:
        offset = 0
        while True:
            statement = (
                select(chunks_table)
                .order_by(chunks_table.c.source_id, chunks_table.c.chunk_index, chunks_table.c.chunk_id)
                .limit(batch_size)
                .offset(offset)
            )
            with self.engine.begin() as connection:
                batch = [_chunk_from_row(row._mapping) for row in connection.execute(statement)]
            if not batch:
                break
            yield batch
            offset += batch_size

    def chunk_counts_by_source(self, *, min_tokens: int, max_tokens: int) -> dict[str, tuple[int, int]]:
        with self.engine.begin() as connection:
            rows = connection.execute(
                select(
                    chunks_table.c.source_id,
                    func.count().label("chunk_count"),
                    func.sum(
                        (
                            chunks_table.c.token_count.between(min_tokens, max_tokens)
                            & (chunks_table.c.content != "")
                        ).cast(Integer)
                    ).label("eligible_chunk_count"),
                )
                .group_by(chunks_table.c.source_id)
            )
            return {
                row.source_id: (
                    int(row.chunk_count or 0),
                    int(row.eligible_chunk_count or 0),
                )
                for row in rows
            }

    def list_sources(self) -> list[SourceManifest]:
        with self.engine.begin() as connection:
            return [
                SourceManifest(**dict(row._mapping))
                for row in connection.execute(select(sources_table).order_by(sources_table.c.source_id))
            ]

    def list_pages(self) -> list[DocumentPage]:
        with self.engine.begin() as connection:
            return [
                DocumentPage(**dict(row._mapping))
                for row in connection.execute(
                    select(pages_table).order_by(pages_table.c.source_id, pages_table.c.page_number)
                )
            ]

    def list_entities(self) -> list[tuple[str, EntityCandidate]]:
        with self.engine.begin() as connection:
            return [
                (
                    row._mapping["source_id"],
                    EntityCandidate(**_candidate_mapping(row._mapping, "source_chunk_ids")),
                )
                for row in connection.execute(
                    select(entities_table).order_by(
                        entities_table.c.source_id,
                        entities_table.c.name,
                        entities_table.c.entity_id,
                    )
                )
            ]

    def list_relations(self) -> list[tuple[str, RelationCandidate]]:
        with self.engine.begin() as connection:
            return [
                (
                    row._mapping["source_id"],
                    RelationCandidate(**_candidate_mapping(row._mapping, "evidence_chunk_ids")),
                )
                for row in connection.execute(
                    select(relations_table).order_by(
                        relations_table.c.source_id,
                        relations_table.c.source_entity_id,
                        relations_table.c.target_entity_id,
                        relations_table.c.relation_id,
                    )
                )
            ]

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
            units = [
                _unit_from_row(row._mapping)
                for row in connection.execute(
                    select(extraction_units_table)
                    .where(extraction_units_table.c.source_id == source_id)
                    .order_by(extraction_units_table.c.unit_index)
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
            extraction_units=units,
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


def _unit_row(unit: ExtractionUnit) -> dict:
    data = unit.model_dump()
    data["section_path"] = _jsonable(data["section_path"])
    data["metadata"] = _jsonable(data["metadata"])
    return data


def _chunk_from_row(row) -> DocumentChunk:
    data = dict(row)
    data["metadata"] = _ensure_json(data.get("metadata", {}))
    return DocumentChunk(**data)


def _unit_from_row(row) -> ExtractionUnit:
    data = dict(row)
    data["section_path"] = _ensure_json(data.get("section_path", []))
    data["metadata"] = _ensure_json(data.get("metadata", {}))
    return ExtractionUnit(**data)


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


def _migrate_structural_chunking_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    if "document_chunks" not in existing_tables:
        return
    columns = {column["name"] for column in inspector.get_columns("document_chunks")}
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            if "parent_unit_id" not in columns:
                connection.execute(
                    text("ALTER TABLE document_chunks ADD COLUMN parent_unit_id VARCHAR NOT NULL DEFAULT ''")
                )
            if "unit_index" not in columns:
                connection.execute(
                    text("ALTER TABLE document_chunks ADD COLUMN unit_index INTEGER NOT NULL DEFAULT 0")
                )
        return
    with engine.begin() as connection:
        if "parent_unit_id" not in columns:
            connection.execute(
                text("ALTER TABLE document_chunks ADD COLUMN parent_unit_id VARCHAR NOT NULL DEFAULT ''")
            )
        if "unit_index" not in columns:
            connection.execute(
                text("ALTER TABLE document_chunks ADD COLUMN unit_index INTEGER NOT NULL DEFAULT 0")
            )


def _fallback_units_from_chunks(
    source_id: str,
    pages: list[DocumentPage],
    chunks: list[DocumentChunk],
) -> list[ExtractionUnit]:
    page_id = pages[0].page_id if pages else (chunks[0].page_id if chunks else f"page:{source_id}:1")
    units: list[ExtractionUnit] = []
    for index, chunk in enumerate(chunks, start=1):
        if chunk.parent_unit_id:
            continue
        units.append(
            ExtractionUnit(
                unit_id=f"unit:{source_id}:{index:04d}",
                source_id=source_id,
                page_id=page_id,
                unit_index=index,
                title=chunk.section_title,
                content=chunk.content,
                unit_type=chunk.content_type,
                section_path=chunk.metadata.get("section_path", []) if chunk.metadata else [],
                token_count=chunk.token_count,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                metadata={"generated_from_legacy_chunk": chunk.chunk_id},
            )
        )
        chunk.parent_unit_id = units[-1].unit_id
        chunk.unit_index = index
    return units
