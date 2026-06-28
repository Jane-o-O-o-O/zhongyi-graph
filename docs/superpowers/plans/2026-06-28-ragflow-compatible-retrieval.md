# RAGFlow-Compatible Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an internal RAGFlow-compatible retrieval engine that reuses RAGFlow retrieval algorithms, rebuilds compatible retrieval data from the existing PostgreSQL/Neo4j/Qdrant/MinIO data, and switches `/api/query` through a configurable retrieval engine without deploying the full RAGFlow service.

**Architecture:** Add a focused `backend/app/services/ragflow_compat/` package with RAGFlow-shaped schemas, repository tables, tokenizer/query utilities, document retrieval, KG retrieval, vector sync, auditing, and a query-service adapter. Existing source tables remain authoritative; new retrieval tables and a new Qdrant collection are derived indexes that can be rebuilt and rolled back independently.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2, PostgreSQL, SQLite test fixtures, Qdrant HTTP API, Neo4j driver, existing `EmbeddingClient`/`RerankClient`/`StructuredExtractionClient`, pytest.

---

## Files

Create:

- `backend/app/services/ragflow_compat/__init__.py` exports the retrieval package.
- `backend/app/services/ragflow_compat/schemas.py` defines dataclasses and table row models used by the compatibility layer.
- `backend/app/services/ragflow_compat/tables.py` defines SQLAlchemy table metadata for retrieval indexes.
- `backend/app/services/ragflow_compat/repository.py` creates, upserts, lists, and audits retrieval tables.
- `backend/app/services/ragflow_compat/query.py` ports RAGFlow query rewrite prompt parsing and lightweight Chinese token utilities.
- `backend/app/services/ragflow_compat/scoring.py` ports KG scoring functions from RAGFlow `internal/service/kg/scoring.go`.
- `backend/app/services/ragflow_compat/doc_store.py` implements a PostgreSQL/Qdrant adapter with RAGFlow-shaped search results.
- `backend/app/services/ragflow_compat/fulltext.py` ports the RAGFlow `Dealer.retrieval` flow for document chunks.
- `backend/app/services/ragflow_compat/kg_search.py` ports the RAGFlow KG retrieval flow.
- `backend/app/services/ragflow_compat/context.py` expands child chunks to parent-unit context.
- `backend/app/services/ragflow_compat/evidence.py` assembles evidence cards and diagnostics.
- `backend/app/services/ragflow_compat/sync_service.py` rebuilds retrieval tables from current data.
- `backend/app/services/ragflow_compat/vector_sync.py` embeds retrieval records into Qdrant.
- `backend/app/services/ragflow_compat/retrieval_service.py` is the `/api/query` facade.
- `scripts/rebuild_ragflow_retrieval_index.py` runs table rebuilds.
- `scripts/sync_ragflow_retrieval_vectors.py` syncs missing retrieval vectors.
- `scripts/audit_ragflow_retrieval_index.py` audits data coverage and smoke inputs.
- `backend/tests/test_ragflow_query.py`
- `backend/tests/test_ragflow_scoring.py`
- `backend/tests/test_ragflow_repository.py`
- `backend/tests/test_ragflow_sync_service.py`
- `backend/tests/test_ragflow_doc_store.py`
- `backend/tests/test_ragflow_fulltext.py`
- `backend/tests/test_ragflow_kg_search.py`
- `backend/tests/test_ragflow_context.py`
- `backend/tests/test_ragflow_vector_sync.py`
- `backend/tests/test_ragflow_retrieval_service.py`
- `backend/tests/test_ragflow_scripts.py`

Modify:

- `backend/app/core/config.py` adds retrieval engine, new Qdrant collection, and RAGFlow weights.
- `backend/app/api/routes.py` wires retrieval debug/rebuild/sync/audit endpoints.
- `backend/app/services/question_service.py` delegates to `RagflowCompatibleRetrievalService` when configured.
- `backend/app/models/query.py` adds optional diagnostics metadata without breaking existing frontend fields.
- `backend/pyproject.toml` adds `jieba` only if lightweight tokenization cannot meet acceptance tests. Default plan avoids a new dependency.

Do not modify or revert unrelated existing dirty files unless a task explicitly requires it.

---

## Task 1: Retrieval Table Schema

**Files:**

- Create: `backend/app/services/ragflow_compat/__init__.py`
- Create: `backend/app/services/ragflow_compat/tables.py`
- Test: `backend/tests/test_ragflow_repository.py`

- [ ] **Step 1: Write failing table-creation test**

Create `backend/tests/test_ragflow_repository.py`:

```python
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

from app.services.ragflow_compat.repository import RagflowRetrievalRepository


def test_repository_creates_ragflow_compatible_tables():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    RagflowRetrievalRepository(engine)

    tables = set(inspect(engine).get_table_names())
    assert {
        "retrieval_documents",
        "retrieval_chunks",
        "retrieval_chunk_terms",
        "retrieval_kg_entities",
        "retrieval_kg_relations",
        "retrieval_kg_type_samples",
        "retrieval_sync_state",
        "retrieval_query_logs",
    }.issubset(tables)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_repository.py::test_repository_creates_ragflow_compatible_tables -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.ragflow_compat'`.

- [ ] **Step 3: Add package marker**

Create `backend/app/services/ragflow_compat/__init__.py`:

```python
"""RAGFlow-compatible retrieval index and search services."""
```

- [ ] **Step 4: Add SQLAlchemy retrieval tables**

Create `backend/app/services/ragflow_compat/tables.py`:

```python
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
```

- [ ] **Step 5: Add minimal repository constructor**

Create `backend/app/services/ragflow_compat/repository.py`:

```python
from __future__ import annotations

from sqlalchemy.engine import Engine

from app.services.ragflow_compat.tables import retrieval_metadata


class RagflowRetrievalRepository:
    def __init__(self, engine: Engine):
        self.engine = engine
        retrieval_metadata.create_all(engine)
```

- [ ] **Step 6: Run table test**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_repository.py::test_repository_creates_ragflow_compatible_tables -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/ragflow_compat/__init__.py \
  backend/app/services/ragflow_compat/tables.py \
  backend/app/services/ragflow_compat/repository.py \
  backend/tests/test_ragflow_repository.py
git commit -m "feat: add ragflow retrieval schema"
```

---

## Task 2: Retrieval Schemas and Repository Upserts

**Files:**

- Create: `backend/app/services/ragflow_compat/schemas.py`
- Modify: `backend/app/services/ragflow_compat/repository.py`
- Test: `backend/tests/test_ragflow_repository.py`

- [ ] **Step 1: Add failing repository round-trip tests**

Append to `backend/tests/test_ragflow_repository.py`:

```python
from app.services.ragflow_compat.schemas import (
    RetrievalChunk,
    RetrievalDocument,
    RetrievalKgEntity,
    RetrievalKgRelation,
    RetrievalTypeSamples,
)


def test_repository_round_trips_retrieval_rows():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    document = RetrievalDocument(
        doc_id="source:uploaded:abc",
        source_id="source:uploaded:abc",
        filename="abc.txt",
        mime_type="text/plain",
        checksum="abc",
        status="parsed",
        object_key="sources/abc/abc.txt",
        source_version=1,
        chunk_count=1,
        eligible_chunk_count=1,
        metadata={"kind": "fixture"},
    )
    chunk = RetrievalChunk(
        chunk_id="chunk:source:uploaded:abc:0001",
        doc_id=document.doc_id,
        source_id=document.source_id,
        parent_unit_id="unit:source:uploaded:abc:0001",
        chunk_order_int=1,
        page_num_int=1,
        title="失眠",
        section_path=["资料", "失眠"],
        content="失眠可辨为心脾两虚。",
        content_with_weight="失眠 资料 失眠 失眠可辨为心脾两虚。",
        content_ltks="失眠 心脾两虚",
        title_tks="失眠",
        important_kwd=["失眠", "心脾两虚"],
        token_count=12,
        metadata={"short_chunk": True},
    )
    entity = RetrievalKgEntity(
        entity_id="entity:syndrome:心脾两虚",
        entity_name="心脾两虚",
        entity_type="Syndrome",
        source_node_id="syndrome:心脾两虚",
        content_with_weight='{"description":"心脾两虚 Syndrome"}',
        description="心脾两虚 Syndrome",
        rank_flt=2.0,
        n_hop_with_weight=[{"path": ["失眠", "心脾两虚"], "weights": [1.0]}],
        aliases=["心脾不足"],
        evidence_chunk_ids=[chunk.chunk_id],
    )
    relation = RetrievalKgRelation(
        relation_id="relation:失眠:心脾两虚",
        from_entity_kwd="失眠",
        to_entity_kwd="心脾两虚",
        relation_type="MANIFESTS_AS",
        display="可辨为",
        content_with_weight="失眠 可辨为 心脾两虚",
        weight_int=3,
        evidence_chunk_ids=[chunk.chunk_id],
        source_edge_id="edge:1",
    )
    samples = RetrievalTypeSamples(
        entity_type="Syndrome",
        sample_entities=["心脾两虚", "肝郁化火"],
        sample_count=2,
        updated_at="2026-06-28T00:00:00Z",
    )

    repository.replace_documents([document])
    repository.replace_chunks([chunk])
    repository.replace_kg_entities([entity])
    repository.replace_kg_relations([relation])
    repository.replace_type_samples([samples])

    assert repository.list_documents() == [document]
    assert repository.list_chunks() == [chunk]
    assert repository.list_kg_entities() == [entity]
    assert repository.list_kg_relations() == [relation]
    assert repository.list_type_samples() == [samples]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_repository.py::test_repository_round_trips_retrieval_rows -q
```

Expected: FAIL because `schemas.py` and repository methods do not exist.

- [ ] **Step 3: Add dataclass schemas**

Create `backend/app/services/ragflow_compat/schemas.py`:

```python
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
    kg_entities: int
    kg_entities_with_vectors: int
    kg_relations: int
    kg_relations_with_vectors: int
    short_chunks: int
    long_chunks: int
```

- [ ] **Step 4: Add repository replace/list methods**

Replace `backend/app/services/ragflow_compat/repository.py` with:

```python
from __future__ import annotations

from dataclasses import asdict

from sqlalchemy import delete, func, select
from sqlalchemy.engine import Engine

from app.services.ragflow_compat.schemas import (
    RetrievalAudit,
    RetrievalChunk,
    RetrievalDocument,
    RetrievalKgEntity,
    RetrievalKgRelation,
    RetrievalTypeSamples,
)
from app.services.ragflow_compat.tables import (
    retrieval_chunks_table,
    retrieval_documents_table,
    retrieval_kg_entities_table,
    retrieval_kg_relations_table,
    retrieval_kg_type_samples_table,
    retrieval_metadata,
)


class RagflowRetrievalRepository:
    def __init__(self, engine: Engine):
        self.engine = engine
        retrieval_metadata.create_all(engine)

    def replace_documents(self, documents: list[RetrievalDocument]) -> None:
        self._replace(retrieval_documents_table, documents)

    def replace_chunks(self, chunks: list[RetrievalChunk]) -> None:
        self._replace(retrieval_chunks_table, chunks)

    def replace_kg_entities(self, entities: list[RetrievalKgEntity]) -> None:
        self._replace(retrieval_kg_entities_table, entities)

    def replace_kg_relations(self, relations: list[RetrievalKgRelation]) -> None:
        self._replace(retrieval_kg_relations_table, relations)

    def replace_type_samples(self, samples: list[RetrievalTypeSamples]) -> None:
        self._replace(retrieval_kg_type_samples_table, samples)

    def list_documents(self) -> list[RetrievalDocument]:
        return self._list(retrieval_documents_table, RetrievalDocument, retrieval_documents_table.c.doc_id)

    def list_chunks(self) -> list[RetrievalChunk]:
        return self._list(retrieval_chunks_table, RetrievalChunk, retrieval_chunks_table.c.chunk_id)

    def list_kg_entities(self) -> list[RetrievalKgEntity]:
        return self._list(
            retrieval_kg_entities_table,
            RetrievalKgEntity,
            retrieval_kg_entities_table.c.entity_id,
        )

    def list_kg_relations(self) -> list[RetrievalKgRelation]:
        return self._list(
            retrieval_kg_relations_table,
            RetrievalKgRelation,
            retrieval_kg_relations_table.c.relation_id,
        )

    def list_type_samples(self) -> list[RetrievalTypeSamples]:
        return self._list(
            retrieval_kg_type_samples_table,
            RetrievalTypeSamples,
            retrieval_kg_type_samples_table.c.entity_type,
        )

    def audit(self) -> RetrievalAudit:
        with self.engine.begin() as connection:
            chunks = _count(connection, retrieval_chunks_table)
            return RetrievalAudit(
                documents=_count(connection, retrieval_documents_table),
                chunks=chunks,
                chunks_with_vectors=_count_where_vector_embedded(connection, retrieval_chunks_table),
                kg_entities=_count(connection, retrieval_kg_entities_table),
                kg_entities_with_vectors=_count_where_vector_embedded(
                    connection, retrieval_kg_entities_table
                ),
                kg_relations=_count(connection, retrieval_kg_relations_table),
                kg_relations_with_vectors=_count_where_vector_embedded(
                    connection, retrieval_kg_relations_table
                ),
                short_chunks=_count_metadata_flag(connection, "short_chunk"),
                long_chunks=_count_metadata_flag(connection, "long_chunk"),
            )

    def _replace(self, table, rows) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(table))
            if rows:
                connection.execute(table.insert(), [_row(row) for row in rows])

    def _list(self, table, cls, order_column):
        with self.engine.begin() as connection:
            return [
                cls(**dict(row._mapping))
                for row in connection.execute(select(table).order_by(order_column))
            ]


def _row(item) -> dict:
    return asdict(item)


def _count(connection, table) -> int:
    return int(connection.execute(select(func.count()).select_from(table)).scalar_one())


def _count_where_vector_embedded(connection, table) -> int:
    return int(
        connection.execute(
            select(func.count()).select_from(table).where(table.c.vector_status == "embedded")
        ).scalar_one()
    )


def _count_metadata_flag(connection, flag: str) -> int:
    rows = connection.execute(select(retrieval_chunks_table.c.metadata))
    return sum(1 for row in rows if isinstance(row.metadata, dict) and row.metadata.get(flag))
```

- [ ] **Step 5: Run repository tests**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_repository.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/ragflow_compat/schemas.py \
  backend/app/services/ragflow_compat/repository.py \
  backend/tests/test_ragflow_repository.py
git commit -m "feat: add ragflow retrieval repository"
```

---

## Task 3: RAGFlow Query Rewrite and Token Utilities

**Files:**

- Create: `backend/app/services/ragflow_compat/query.py`
- Test: `backend/tests/test_ragflow_query.py`

- [ ] **Step 1: Write failing query tests**

Create `backend/tests/test_ragflow_query.py`:

```python
from app.services.ragflow_compat.query import (
    build_query_rewrite_prompt,
    parse_query_rewrite_response,
    tokenize_query,
    token_similarity,
)


def test_parse_query_rewrite_response_handles_json_and_code_fence():
    assert parse_query_rewrite_response(
        '{"answer_type_keywords":["Syndrome"],"entities_from_query":["失眠"]}'
    ) == (["Syndrome"], ["失眠"])
    assert parse_query_rewrite_response(
        '```json\n{"answer_type_keywords":["Formula"],"entities_from_query":["归脾汤"]}\n```'
    ) == (["Formula"], ["归脾汤"])


def test_parse_query_rewrite_response_falls_back_to_raw_question():
    assert parse_query_rewrite_response("not json", fallback_question="党参功效") == (
        [],
        ["党参功效"],
    )


def test_prompt_contains_question_and_type_pool():
    prompt = build_query_rewrite_prompt(
        question="失眠怎么辨证",
        type_pool={"Syndrome": ["心脾两虚"], "Formula": ["归脾汤"]},
    )

    assert "失眠怎么辨证" in prompt
    assert "Syndrome" in prompt
    assert "心脾两虚" in prompt


def test_tokenize_and_similarity_support_chinese_terms():
    query_tokens = tokenize_query("柴胡桂枝干姜汤适合什么情况")
    doc_tokens = tokenize_query("柴胡 桂枝 干姜 方剂 主治 寒热")

    assert "柴胡" in query_tokens
    assert "桂枝" in query_tokens
    assert token_similarity(query_tokens, doc_tokens) > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_query.py -q
```

Expected: FAIL because `query.py` does not exist.

- [ ] **Step 3: Implement query utilities**

Create `backend/app/services/ragflow_compat/query.py`:

```python
from __future__ import annotations

import json
import re
from collections import defaultdict

QUERY_REWRITE_PROMPT = """---Role---

You are a helpful assistant tasked with identifying both answer-type and low-level keywords in the user's query.

---Goal---

Given the query, list both answer-type and low-level keywords.
answer_type_keywords focus on the type of the answer to the certain query, while low-level keywords focus on specific entities, details, or concrete terms.
The answer_type_keywords must be selected from Answer type pool.
This pool is in the form of a dictionary, where the key represents the Type you should choose from and the value represents the example samples.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "answer_type_keywords" for the types of the answer. In this list, the types with the highest likelihood should be placed at the forefront. No more than 3.
  - "entities_from_query" for specific entities or details. It must be extracted from the query.

-Real Data-
######################
Query: {query}
Answer type pool:{TYPE_POOL}
######################
Output:
"""

DOMAIN_TERMS = (
    "柴胡桂枝干姜汤",
    "心脾两虚",
    "肝郁化火",
    "阴虚火旺",
    "补益心脾",
    "归脾汤",
    "不寐",
    "失眠",
    "党参",
    "柴胡",
    "桂枝",
    "干姜",
    "普济方",
    "外台秘要",
)


def build_query_rewrite_prompt(question: str, type_pool: dict[str, list[str]]) -> str:
    return QUERY_REWRITE_PROMPT.format(
        query=question,
        TYPE_POOL=json.dumps(type_pool, ensure_ascii=False, indent=2),
    )


def parse_query_rewrite_response(
    response: str,
    fallback_question: str = "",
) -> tuple[list[str], list[str]]:
    for candidate in _json_candidates(response):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return (
            _clean_terms(parsed.get("answer_type_keywords", []))[:3],
            _clean_terms(parsed.get("entities_from_query", []))[:5],
        )
    return ([], [fallback_question] if fallback_question else [])


def tokenize_query(text: str) -> list[str]:
    normalized = re.sub(r"[\s,，。？?！!；;：:（）()《》<>\"'`]+", " ", text.lower())
    tokens: list[str] = []
    for term in DOMAIN_TERMS:
        if term.lower() in normalized or term in text:
            tokens.append(term)
    for part in normalized.split():
        if not part:
            continue
        tokens.append(part)
        if _contains_chinese(part):
            tokens.extend(_chinese_ngrams(part, 2))
            tokens.extend(_chinese_ngrams(part, 3))
    return _unique(tokens)


def token_similarity(query_tokens: list[str], document_tokens: list[str]) -> float:
    query_weights = _token_weights(query_tokens)
    document_set = set(document_tokens)
    if not query_weights:
        return 0.0
    matched = sum(weight for token, weight in query_weights.items() if token in document_set)
    total = sum(query_weights.values())
    return matched / total if total else 0.0


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                candidates.append(part)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    return candidates


def _clean_terms(values) -> list[str]:
    if not isinstance(values, list):
        return []
    return _unique(str(value).strip() for value in values if str(value).strip())


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _chinese_ngrams(text: str, size: int) -> list[str]:
    chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    return ["".join(chars[index : index + size]) for index in range(0, len(chars) - size + 1)]


def _token_weights(tokens: list[str]) -> dict[str, float]:
    weights: dict[str, float] = defaultdict(float)
    for token in tokens:
        if len(token) >= 4:
            weights[token] += 2.0
        elif len(token) >= 2:
            weights[token] += 1.0
        else:
            weights[token] += 0.2
    return dict(weights)


def _unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
```

- [ ] **Step 4: Run query tests**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_query.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ragflow_compat/query.py backend/tests/test_ragflow_query.py
git commit -m "feat: add ragflow query utilities"
```

---

## Task 4: KG Scoring Port From RAGFlow

**Files:**

- Create: `backend/app/services/ragflow_compat/scoring.py`
- Test: `backend/tests/test_ragflow_scoring.py`

- [ ] **Step 1: Write failing scoring tests**

Create `backend/tests/test_ragflow_scoring.py`:

```python
from app.services.ragflow_compat.scoring import (
    analyze_nhop_paths,
    double_hit_boost,
    fuse_relation_scores,
    sort_and_trim_entities,
    sort_and_trim_relations,
)


def test_analyze_nhop_paths_uses_distance_decay_and_weights():
    entities = {
        "失眠": {
            "sim": 0.8,
            "pagerank": 2.0,
            "n_hop_ents": [
                {"path": ["失眠", "心脾两虚", "归脾汤"], "weights": [3.0, 2.0]}
            ],
        }
    }

    paths = analyze_nhop_paths(entities)

    assert paths[("失眠", "心脾两虚")]["sim"] == 0.4
    assert paths[("失眠", "心脾两虚")]["pagerank"] == 3.0
    assert round(paths[("心脾两虚", "归脾汤")]["sim"], 4) == round(0.8 / 3.0, 4)
    assert paths[("心脾两虚", "归脾汤")]["pagerank"] == 2.0


def test_double_hit_boost_only_boosts_shared_entities():
    entities = {"失眠": {"sim": 0.5}, "归脾汤": {"sim": 0.4}}

    double_hit_boost(entities, {"失眠"})

    assert entities["失眠"]["sim"] == 1.0
    assert entities["归脾汤"]["sim"] == 0.4


def test_fuse_relation_scores_adds_nhop_relations_and_type_boosts():
    relations = {("失眠", "心脾两虚"): {"sim": 0.5, "pagerank": 3.0}}
    nhop = {
        ("失眠", "心脾两虚"): {"sim": 0.4, "pagerank": 3.0},
        ("心脾两虚", "归脾汤"): {"sim": 0.2, "pagerank": 2.0},
    }

    fuse_relation_scores(relations, {"心脾两虚"}, nhop)

    assert relations[("失眠", "心脾两虚")]["sim"] == 1.2
    assert relations[("心脾两虚", "归脾汤")]["sim"] == 0.4
    assert relations[("心脾两虚", "归脾汤")]["pagerank"] == 2.0


def test_sort_and_trim_scores_by_sim_times_pagerank():
    entities = {
        "低分": {"sim": 0.9, "pagerank": 1.0, "description": ""},
        "高分": {"sim": 0.5, "pagerank": 3.0, "description": "desc"},
    }
    relations = {
        ("a", "b"): {"sim": 0.2, "pagerank": 10.0, "description": "ab"},
        ("c", "d"): {"sim": 0.9, "pagerank": 1.0, "description": "cd"},
    }

    assert sort_and_trim_entities(entities, topn=1)[0]["Entity"] == "高分"
    assert sort_and_trim_relations(relations, topn=1)[0]["From Entity"] == "a"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_scoring.py -q
```

Expected: FAIL because `scoring.py` does not exist.

- [ ] **Step 3: Implement scoring functions**

Create `backend/app/services/ragflow_compat/scoring.py`:

```python
from __future__ import annotations

from copy import deepcopy
from typing import Any

Edge = tuple[str, str]


def analyze_nhop_paths(ents_from_query: dict[str, dict[str, Any]]) -> dict[Edge, dict[str, float]]:
    nhop_paths: dict[Edge, dict[str, float]] = {}
    for entity in ents_from_query.values():
        for neighbor in entity.get("n_hop_ents", []) or []:
            path = neighbor.get("path", [])
            weights = neighbor.get("weights", [])
            if not isinstance(path, list):
                continue
            for index in range(len(path) - 1):
                edge = (path[index], path[index + 1])
                current = nhop_paths.setdefault(edge, {"sim": 0.0, "pagerank": 0.0})
                current["sim"] += float(entity.get("sim", 0.0)) / (2.0 + index)
                if index < len(weights):
                    current["pagerank"] = max(current["pagerank"], float(weights[index]))
    return nhop_paths


def double_hit_boost(ents_from_query: dict[str, dict[str, Any]], ents_from_types: set[str]) -> None:
    for entity_name in list(ents_from_query):
        if entity_name in ents_from_types:
            ents_from_query[entity_name]["sim"] = float(ents_from_query[entity_name].get("sim", 0)) * 2


def fuse_relation_scores(
    rels_from_text: dict[Edge, dict[str, Any]],
    ents_from_types: set[str],
    nhop_paths: dict[Edge, dict[str, float]],
) -> None:
    remaining_nhop = deepcopy(nhop_paths)
    for edge, relation in list(rels_from_text.items()):
        score = 0.0
        if edge in remaining_nhop:
            score += float(remaining_nhop[edge]["sim"])
            del remaining_nhop[edge]
        if edge[0] in ents_from_types:
            score += 1
        if edge[1] in ents_from_types:
            score += 1
        relation["sim"] = float(relation.get("sim", 0.0)) * (score + 1)

    for edge, path_score in remaining_nhop.items():
        score = 0.0
        if edge[0] in ents_from_types:
            score += 1
        if edge[1] in ents_from_types:
            score += 1
        rels_from_text[edge] = {
            "sim": float(path_score.get("sim", 0.0)) * (score + 1),
            "pagerank": float(path_score.get("pagerank", 0.0)),
            "description": "",
        }


def sort_and_trim_entities(entities: dict[str, dict[str, Any]], topn: int) -> list[dict[str, Any]]:
    scored = [
        {
            "Entity": name,
            "Score": float(item.get("sim", 0.0)) * float(item.get("pagerank", 0.0)),
            "Description": item.get("description", ""),
        }
        for name, item in entities.items()
    ]
    return sorted(scored, key=lambda item: item["Score"], reverse=True)[:topn]


def sort_and_trim_relations(relations: dict[Edge, dict[str, Any]], topn: int) -> list[dict[str, Any]]:
    scored = [
        {
            "From Entity": edge[0],
            "To Entity": edge[1],
            "Score": float(item.get("sim", 0.0)) * float(item.get("pagerank", 0.0)),
            "Description": item.get("description", ""),
        }
        for edge, item in relations.items()
    ]
    return sorted(scored, key=lambda item: item["Score"], reverse=True)[:topn]
```

- [ ] **Step 4: Run scoring tests**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_scoring.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ragflow_compat/scoring.py backend/tests/test_ragflow_scoring.py
git commit -m "feat: port ragflow kg scoring"
```

---

## Task 5: Build Retrieval Index From Existing PostgreSQL Data

**Files:**

- Create: `backend/app/services/ragflow_compat/sync_service.py`
- Modify: `backend/app/services/ragflow_compat/repository.py`
- Test: `backend/tests/test_ragflow_sync_service.py`

- [ ] **Step 1: Write failing sync-service test**

Create `backend/tests/test_ragflow_sync_service.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.models.ingestion import DocumentChunk, DocumentPage, ExtractionUnit, SourceManifest
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.sync_service import RagflowIndexSyncService


def _engine():
    return create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def test_sync_service_rebuilds_documents_and_chunks_from_ingestion_tables():
    engine = _engine()
    ingestion = IngestionRepository(engine)
    retrieval = RagflowRetrievalRepository(engine)
    source = SourceManifest(
        source_id="source:uploaded:abc",
        filename="abc.txt",
        mime_type="text/plain",
        checksum="abc",
        status="parsed",
        object_key="sources/abc/abc.txt",
    )
    page = DocumentPage(
        page_id="page:source:uploaded:abc:1",
        source_id=source.source_id,
        page_number=1,
        text="失眠可辨为心脾两虚。",
    )
    unit = ExtractionUnit(
        unit_id="unit:source:uploaded:abc:0001",
        source_id=source.source_id,
        page_id=page.page_id,
        unit_index=1,
        title="失眠条",
        content="失眠可辨为心脾两虚。",
        section_path=["资料", "失眠条"],
        token_count=12,
    )
    short_chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:abc:0001",
        source_id=source.source_id,
        page_id=page.page_id,
        chunk_index=1,
        content="失眠可辨为心脾两虚。",
        parent_unit_id=unit.unit_id,
        unit_index=unit.unit_index,
        section_title="失眠条",
        token_count=12,
    )
    empty_chunk = short_chunk.model_copy(
        update={"chunk_id": "chunk:source:uploaded:abc:0002", "chunk_index": 2, "content": ""}
    )
    ingestion.upsert_source(source)
    ingestion.replace_pages_units_and_chunks(source.source_id, [page], [unit], [short_chunk, empty_chunk])

    report = RagflowIndexSyncService(ingestion, retrieval).rebuild_chunks()

    assert report["raw_chunks"] == 2
    assert report["retrieval_chunks"] == 1
    assert report["skipped_empty_chunks"] == 1
    assert retrieval.list_documents()[0].eligible_chunk_count == 1
    chunk = retrieval.list_chunks()[0]
    assert chunk.content_with_weight.startswith("失眠条 资料 失眠条")
    assert chunk.metadata["short_chunk"] is True
    assert "失眠" in chunk.important_kwd
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_sync_service.py -q
```

Expected: FAIL because `sync_service.py` does not exist.

- [ ] **Step 3: Add source listing to ingestion repository if missing**

If `IngestionRepository` lacks a `list_sources()` method, add this method to `backend/app/services/ingestion_repository.py`:

```python
    def list_sources(self) -> list[SourceManifest]:
        with self.engine.begin() as connection:
            return [
                SourceManifest(**dict(row._mapping))
                for row in connection.execute(select(sources_table).order_by(sources_table.c.source_id))
            ]
```

- [ ] **Step 4: Implement chunk index rebuild**

Create `backend/app/services/ragflow_compat/sync_service.py`:

```python
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from app.models.ingestion import DocumentChunk, SourceManifest
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.query import tokenize_query
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalChunk, RetrievalDocument


class RagflowIndexSyncService:
    def __init__(
        self,
        ingestion_repository: IngestionRepository,
        retrieval_repository: RagflowRetrievalRepository,
    ):
        self.ingestion_repository = ingestion_repository
        self.retrieval_repository = retrieval_repository

    def rebuild_chunks(self) -> dict[str, int | str]:
        sources = self.ingestion_repository.list_sources()
        documents: list[RetrievalDocument] = []
        retrieval_chunks: list[RetrievalChunk] = []
        raw_chunks = 0
        skipped_empty = 0
        chunks_by_source: Counter[str] = Counter()
        eligible_by_source: Counter[str] = Counter()

        for source in sources:
            chunks = self.ingestion_repository.list_chunks(source.source_id)
            raw_chunks += len(chunks)
            chunks_by_source[source.source_id] = len(chunks)
            for chunk in chunks:
                retrieval_chunk = build_retrieval_chunk(source, chunk)
                if retrieval_chunk is None:
                    skipped_empty += 1
                    continue
                retrieval_chunks.append(retrieval_chunk)
                eligible_by_source[source.source_id] += 1

        for source in sources:
            documents.append(
                RetrievalDocument(
                    doc_id=source.source_id,
                    source_id=source.source_id,
                    filename=source.filename,
                    mime_type=source.mime_type,
                    checksum=source.checksum,
                    status=source.status,
                    object_key=source.object_key,
                    source_version=source.version,
                    chunk_count=chunks_by_source[source.source_id],
                    eligible_chunk_count=eligible_by_source[source.source_id],
                    metadata={"rebuilt_at": _utc_now()},
                )
            )

        self.retrieval_repository.replace_documents(documents)
        self.retrieval_repository.replace_chunks(retrieval_chunks)
        return {
            "raw_sources": len(sources),
            "raw_chunks": raw_chunks,
            "retrieval_documents": len(documents),
            "retrieval_chunks": len(retrieval_chunks),
            "skipped_empty_chunks": skipped_empty,
        }


def build_retrieval_chunk(source: SourceManifest, chunk: DocumentChunk) -> RetrievalChunk | None:
    content = chunk.content.strip()
    if not content:
        return None
    section_path = [str(item) for item in chunk.metadata.get("section_path", [])] if chunk.metadata else []
    title = chunk.section_title or (section_path[-1] if section_path else source.filename)
    weighted_parts = [title, " ".join(section_path), content]
    content_with_weight = " ".join(part for part in weighted_parts if part).strip()
    title_tokens = tokenize_query(title)
    content_tokens = tokenize_query(content_with_weight)
    important = _important_keywords(title_tokens + content_tokens)
    metadata = dict(chunk.metadata or {})
    metadata["short_chunk"] = chunk.token_count < 20
    metadata["long_chunk"] = chunk.token_count >= 1000
    metadata["rebuilt_from"] = "document_chunks"
    return RetrievalChunk(
        chunk_id=chunk.chunk_id,
        doc_id=source.source_id,
        source_id=source.source_id,
        parent_unit_id=chunk.parent_unit_id or chunk.chunk_id,
        chunk_order_int=chunk.chunk_index,
        page_num_int=1,
        title=title,
        section_path=section_path,
        content=content,
        content_with_weight=content_with_weight,
        content_ltks=" ".join(content_tokens),
        title_tks=" ".join(title_tokens),
        important_kwd=important,
        content_type=chunk.content_type,
        token_count=chunk.token_count,
        metadata=metadata,
    )


def _important_keywords(tokens: list[str]) -> list[str]:
    selected = [token for token in tokens if len(token) >= 2]
    return list(dict.fromkeys(selected[:30]))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
```

- [ ] **Step 5: Run sync-service tests**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_sync_service.py tests/test_ingestion_repository.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/ragflow_compat/sync_service.py \
  backend/app/services/ingestion_repository.py \
  backend/tests/test_ragflow_sync_service.py
git commit -m "feat: rebuild ragflow chunk index"
```

---

## Task 6: Audit Script and Repository Audit

**Files:**

- Create: `scripts/audit_ragflow_retrieval_index.py`
- Test: `backend/tests/test_ragflow_scripts.py`

- [ ] **Step 1: Write failing script test**

Create `backend/tests/test_ragflow_scripts.py`:

```python
import subprocess
import sys
from pathlib import Path


def test_audit_script_supports_dry_run_help():
    script = Path(__file__).parents[2] / "scripts" / "audit_ragflow_retrieval_index.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--postgres-dsn" in result.stdout
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_scripts.py::test_audit_script_supports_dry_run_help -q
```

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Add audit script**

Create `scripts/audit_ragflow_retrieval_index.py`:

```python
from __future__ import annotations

import argparse
import json

from sqlalchemy import create_engine

from app.services.ragflow_compat.repository import RagflowRetrievalRepository

DEFAULT_POSTGRES_DSN = "postgresql+psycopg://tcm:tcm@127.0.0.1:5432/tcm_kg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RAGFlow-compatible retrieval index.")
    parser.add_argument("--postgres-dsn", default=DEFAULT_POSTGRES_DSN)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = RagflowRetrievalRepository(
        create_engine(args.postgres_dsn, future=True, pool_pre_ping=True)
    )
    audit = repository.audit()
    payload = audit.__dict__
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run script test**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_scripts.py::test_audit_script_supports_dry_run_help -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_ragflow_retrieval_index.py backend/tests/test_ragflow_scripts.py
git commit -m "feat: add ragflow retrieval audit script"
```

---

## Task 7: Rebuild Script

**Files:**

- Create: `scripts/rebuild_ragflow_retrieval_index.py`
- Modify: `backend/tests/test_ragflow_scripts.py`

- [ ] **Step 1: Add failing help test**

Append to `backend/tests/test_ragflow_scripts.py`:

```python
def test_rebuild_script_supports_dry_run_help():
    script = Path(__file__).parents[2] / "scripts" / "rebuild_ragflow_retrieval_index.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--dry-run" in result.stdout
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_scripts.py::test_rebuild_script_supports_dry_run_help -q
```

Expected: FAIL because rebuild script does not exist.

- [ ] **Step 3: Add rebuild script**

Create `scripts/rebuild_ragflow_retrieval_index.py`:

```python
from __future__ import annotations

import argparse
import json

from sqlalchemy import create_engine

from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.sync_service import RagflowIndexSyncService

DEFAULT_POSTGRES_DSN = "postgresql+psycopg://tcm:tcm@127.0.0.1:5432/tcm_kg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild RAGFlow-compatible retrieval index.")
    parser.add_argument("--postgres-dsn", default=DEFAULT_POSTGRES_DSN)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = create_engine(args.postgres_dsn, future=True, pool_pre_ping=True)
    ingestion = IngestionRepository(engine)
    retrieval = RagflowRetrievalRepository(engine)
    if args.dry_run:
        sources = ingestion.list_sources()
        raw_chunks = sum(len(ingestion.list_chunks(source.source_id)) for source in sources)
        print(json.dumps({"raw_sources": len(sources), "raw_chunks": raw_chunks}, ensure_ascii=False))
        return
    report = RagflowIndexSyncService(ingestion, retrieval).rebuild_chunks()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run script tests**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_scripts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/rebuild_ragflow_retrieval_index.py backend/tests/test_ragflow_scripts.py
git commit -m "feat: add ragflow retrieval rebuild script"
```

---

## Task 8: Qdrant Vector Sync

**Files:**

- Create: `backend/app/services/ragflow_compat/vector_sync.py`
- Create: `scripts/sync_ragflow_retrieval_vectors.py`
- Test: `backend/tests/test_ragflow_vector_sync.py`

- [ ] **Step 1: Write failing vector-sync test**

Create `backend/tests/test_ragflow_vector_sync.py`:

```python
import httpx
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.services.model_clients import EmbeddingClient
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalChunk, RetrievalKgEntity, RetrievalKgRelation
from app.services.ragflow_compat.vector_sync import RagflowVectorSyncService


class FakeQdrant:
    def __init__(self):
        self.requests = []

    def put(self, url, json):
        self.requests.append(("PUT", url, json))
        return httpx.Response(200, request=httpx.Request("PUT", url), json={"status": "ok"})


def test_vector_sync_embeds_missing_chunks_and_updates_payloads():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_chunks([
        RetrievalChunk(
            chunk_id="chunk:1",
            doc_id="source:1",
            source_id="source:1",
            parent_unit_id="unit:1",
            chunk_order_int=1,
            page_num_int=1,
            title="失眠",
            section_path=[],
            content="失眠可辨为心脾两虚。",
            content_with_weight="失眠可辨为心脾两虚。",
        )
    ])
    qdrant = FakeQdrant()
    service = RagflowVectorSyncService(
        repository=repository,
        embedding_client=EmbeddingClient.demo(dimensions=8),
        qdrant_url="http://qdrant:6333",
        collection="tcm_ragflow_retrieval",
        http_client=qdrant,
    )

    report = service.sync_chunks(limit=10)

    assert report["embedded"] == 1
    assert qdrant.requests[0][2]["vectors"]["size"] == 8
    assert qdrant.requests[1][2]["points"][0]["payload"]["content_type"] == "retrieval_chunk"


def test_vector_sync_embeds_missing_entities_and_relations():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_kg_entities([
        RetrievalKgEntity(
            entity_id="entity:syndrome:心脾两虚",
            entity_name="心脾两虚",
            entity_type="Syndrome",
            source_node_id="syndrome:心脾两虚",
            content_with_weight='{"description":"心脾两虚 Syndrome"}',
            description="心脾两虚 Syndrome",
        )
    ])
    repository.replace_kg_relations([
        RetrievalKgRelation(
            relation_id="relation:失眠:心脾两虚",
            from_entity_kwd="失眠",
            to_entity_kwd="心脾两虚",
            relation_type="MANIFESTS_AS",
            display="可辨为",
            content_with_weight="失眠 可辨为 心脾两虚",
        )
    ])
    qdrant = FakeQdrant()
    service = RagflowVectorSyncService(
        repository=repository,
        embedding_client=EmbeddingClient.demo(dimensions=8),
        qdrant_url="http://qdrant:6333",
        collection="tcm_ragflow_retrieval",
        http_client=qdrant,
    )

    entity_report = service.sync_entities(limit=10)
    relation_report = service.sync_relations(limit=10)

    assert entity_report["embedded"] == 1
    assert relation_report["embedded"] == 1
    payload_types = [
        request[2]["points"][0]["payload"]["content_type"]
        for request in qdrant.requests
        if request[0] == "PUT" and request[1].endswith("/points")
    ]
    assert payload_types == ["retrieval_kg_entity", "retrieval_kg_relation"]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_vector_sync.py -q
```

Expected: FAIL because `vector_sync.py` does not exist.

- [ ] **Step 3: Add repository methods for missing vectors**

Add methods to `backend/app/services/ragflow_compat/repository.py`:

```python
    def list_chunks_missing_vectors(self, limit: int = 0) -> list[RetrievalChunk]:
        statement = (
            select(retrieval_chunks_table)
            .where(retrieval_chunks_table.c.available_int == 1)
            .where(retrieval_chunks_table.c.vector_status != "embedded")
            .order_by(retrieval_chunks_table.c.chunk_id)
        )
        if limit:
            statement = statement.limit(limit)
        with self.engine.begin() as connection:
            return [RetrievalChunk(**dict(row._mapping)) for row in connection.execute(statement)]

    def mark_chunks_embedded(self, chunk_ids_to_point_ids: dict[str, str]) -> None:
        with self.engine.begin() as connection:
            for chunk_id, point_id in chunk_ids_to_point_ids.items():
                connection.execute(
                    retrieval_chunks_table.update()
                    .where(retrieval_chunks_table.c.chunk_id == chunk_id)
                    .values(vector_status="embedded", vector_point_id=point_id)
                )

    def list_entities_missing_vectors(self, limit: int = 0) -> list[RetrievalKgEntity]:
        statement = (
            select(retrieval_kg_entities_table)
            .where(retrieval_kg_entities_table.c.available_int == 1)
            .where(retrieval_kg_entities_table.c.vector_status != "embedded")
            .order_by(retrieval_kg_entities_table.c.entity_id)
        )
        if limit:
            statement = statement.limit(limit)
        with self.engine.begin() as connection:
            return [RetrievalKgEntity(**dict(row._mapping)) for row in connection.execute(statement)]

    def list_relations_missing_vectors(self, limit: int = 0) -> list[RetrievalKgRelation]:
        statement = (
            select(retrieval_kg_relations_table)
            .where(retrieval_kg_relations_table.c.available_int == 1)
            .where(retrieval_kg_relations_table.c.vector_status != "embedded")
            .order_by(retrieval_kg_relations_table.c.relation_id)
        )
        if limit:
            statement = statement.limit(limit)
        with self.engine.begin() as connection:
            return [RetrievalKgRelation(**dict(row._mapping)) for row in connection.execute(statement)]

    def mark_entities_embedded(self, entity_ids_to_point_ids: dict[str, str]) -> None:
        with self.engine.begin() as connection:
            for entity_id, point_id in entity_ids_to_point_ids.items():
                connection.execute(
                    retrieval_kg_entities_table.update()
                    .where(retrieval_kg_entities_table.c.entity_id == entity_id)
                    .values(vector_status="embedded", vector_point_id=point_id)
                )

    def mark_relations_embedded(self, relation_ids_to_point_ids: dict[str, str]) -> None:
        with self.engine.begin() as connection:
            for relation_id, point_id in relation_ids_to_point_ids.items():
                connection.execute(
                    retrieval_kg_relations_table.update()
                    .where(retrieval_kg_relations_table.c.relation_id == relation_id)
                    .values(vector_status="embedded", vector_point_id=point_id)
                )
```

- [ ] **Step 4: Implement vector sync service**

Create `backend/app/services/ragflow_compat/vector_sync.py`:

```python
from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

import httpx

from app.services.model_clients import EmbeddingClient
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalChunk, RetrievalKgEntity, RetrievalKgRelation


class RagflowVectorSyncService:
    def __init__(
        self,
        repository: RagflowRetrievalRepository,
        embedding_client: EmbeddingClient,
        qdrant_url: str,
        collection: str,
        http_client=None,
    ):
        self.repository = repository
        self.embedding_client = embedding_client
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection = collection
        self.http_client = http_client or httpx.Client(timeout=300)

    def sync_chunks(self, limit: int = 0, batch_size: int = 64) -> dict[str, int]:
        chunks = self.repository.list_chunks_missing_vectors(limit=limit)
        if not chunks:
            return {"requested": 0, "embedded": 0}
        embedded = 0
        for batch in _batches(chunks, batch_size):
            vectors = self.embedding_client.embed([chunk.content_with_weight for chunk in batch])
            self._ensure_collection(len(vectors[0]))
            point_ids = self._upsert_chunks(batch, vectors)
            self.repository.mark_chunks_embedded(point_ids)
            embedded += len(batch)
        return {"requested": len(chunks), "embedded": embedded}

    def sync_entities(self, limit: int = 0, batch_size: int = 64) -> dict[str, int]:
        entities = self.repository.list_entities_missing_vectors(limit=limit)
        if not entities:
            return {"requested": 0, "embedded": 0}
        embedded = 0
        for batch in _batches(entities, batch_size):
            vectors = self.embedding_client.embed([entity.content_with_weight for entity in batch])
            self._ensure_collection(len(vectors[0]))
            point_ids = self._upsert_entities(batch, vectors)
            self.repository.mark_entities_embedded(point_ids)
            embedded += len(batch)
        return {"requested": len(entities), "embedded": embedded}

    def sync_relations(self, limit: int = 0, batch_size: int = 64) -> dict[str, int]:
        relations = self.repository.list_relations_missing_vectors(limit=limit)
        if not relations:
            return {"requested": 0, "embedded": 0}
        embedded = 0
        for batch in _batches(relations, batch_size):
            vectors = self.embedding_client.embed([relation.content_with_weight for relation in batch])
            self._ensure_collection(len(vectors[0]))
            point_ids = self._upsert_relations(batch, vectors)
            self.repository.mark_relations_embedded(point_ids)
            embedded += len(batch)
        return {"requested": len(relations), "embedded": embedded}

    def _ensure_collection(self, dimensions: int) -> None:
        response = self.http_client.put(
            f"{self.qdrant_url}/collections/{self.collection}",
            json={"vectors": {"size": dimensions, "distance": "Cosine"}},
        )
        if response.status_code not in {200, 409}:
            response.raise_for_status()

    def _upsert_chunks(
        self,
        chunks: list[RetrievalChunk],
        vectors: list[list[float]],
    ) -> dict[str, str]:
        points = []
        point_ids: dict[str, str] = {}
        for chunk, vector in zip(chunks, vectors, strict=True):
            point_id = qdrant_point_id(f"retrieval_chunk:{chunk.chunk_id}")
            point_ids[chunk.chunk_id] = point_id
            points.append(
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "id": f"retrieval_chunk:{chunk.chunk_id}",
                        "content_type": "retrieval_chunk",
                        "text": chunk.content_with_weight,
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "source_id": chunk.source_id,
                    },
                }
            )
        self.http_client.put(
            f"{self.qdrant_url}/collections/{self.collection}/points",
            json={"points": points},
        ).raise_for_status()
        return point_ids

    def _upsert_entities(
        self,
        entities: list[RetrievalKgEntity],
        vectors: list[list[float]],
    ) -> dict[str, str]:
        points = []
        point_ids: dict[str, str] = {}
        for entity, vector in zip(entities, vectors, strict=True):
            point_id = qdrant_point_id(f"retrieval_kg_entity:{entity.entity_id}")
            point_ids[entity.entity_id] = point_id
            points.append(
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "id": f"retrieval_kg_entity:{entity.entity_id}",
                        "content_type": "retrieval_kg_entity",
                        "text": entity.content_with_weight,
                        "entity_id": entity.entity_id,
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                    },
                }
            )
        self.http_client.put(
            f"{self.qdrant_url}/collections/{self.collection}/points",
            json={"points": points},
        ).raise_for_status()
        return point_ids

    def _upsert_relations(
        self,
        relations: list[RetrievalKgRelation],
        vectors: list[list[float]],
    ) -> dict[str, str]:
        points = []
        point_ids: dict[str, str] = {}
        for relation, vector in zip(relations, vectors, strict=True):
            point_id = qdrant_point_id(f"retrieval_kg_relation:{relation.relation_id}")
            point_ids[relation.relation_id] = point_id
            points.append(
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "id": f"retrieval_kg_relation:{relation.relation_id}",
                        "content_type": "retrieval_kg_relation",
                        "text": relation.content_with_weight,
                        "relation_id": relation.relation_id,
                        "from_entity_kwd": relation.from_entity_kwd,
                        "to_entity_kwd": relation.to_entity_kwd,
                    },
                }
            )
        self.http_client.put(
            f"{self.qdrant_url}/collections/{self.collection}/points",
            json={"points": points},
        ).raise_for_status()
        return point_ids


def qdrant_point_id(document_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"tcm-kg-platform:{document_id}"))


def _batches(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])
```

- [ ] **Step 5: Add CLI script**

Create `scripts/sync_ragflow_retrieval_vectors.py`:

```python
from __future__ import annotations

import argparse
import json

import httpx
from sqlalchemy import create_engine

from app.services.model_clients import EmbeddingClient
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.vector_sync import RagflowVectorSyncService

DEFAULT_POSTGRES_DSN = "postgresql+psycopg://tcm:tcm@127.0.0.1:5432/tcm_kg"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION = "tcm_ragflow_retrieval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync RAGFlow-compatible retrieval vectors.")
    parser.add_argument("--postgres-dsn", default=DEFAULT_POSTGRES_DSN)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = RagflowRetrievalRepository(
        create_engine(args.postgres_dsn, future=True, pool_pre_ping=True)
    )
    embedding = EmbeddingClient(
        base_url=args.llm_base_url,
        api_key=args.api_key,
        model=args.embedding_model,
        dimensions=args.dimensions,
        http_client=httpx.Client(timeout=300, trust_env=False),
    )
    service = RagflowVectorSyncService(
        repository=repository,
        embedding_client=embedding,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        http_client=httpx.Client(timeout=300, trust_env=False),
    )
    report = {
        "chunks": service.sync_chunks(limit=args.limit, batch_size=args.batch_size),
        "entities": service.sync_entities(limit=args.limit, batch_size=args.batch_size),
        "relations": service.sync_relations(limit=args.limit, batch_size=args.batch_size),
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run vector sync tests**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_vector_sync.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/ragflow_compat/vector_sync.py \
  backend/app/services/ragflow_compat/repository.py \
  scripts/sync_ragflow_retrieval_vectors.py \
  backend/tests/test_ragflow_vector_sync.py
git commit -m "feat: sync ragflow retrieval vectors"
```

---

## Task 9: RAGFlow-Shaped Doc Store Adapter

**Files:**

- Create: `backend/app/services/ragflow_compat/doc_store.py`
- Test: `backend/tests/test_ragflow_doc_store.py`

- [ ] **Step 1: Write failing doc-store test**

Create `backend/tests/test_ragflow_doc_store.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.services.ragflow_compat.doc_store import (
    RagflowDocStore,
    RagflowSearchResult,
    search_chunks_by_terms,
)
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalChunk


def test_search_chunks_by_terms_returns_ragflow_shaped_result():
    chunks = [
        RetrievalChunk(
            chunk_id="chunk:1",
            doc_id="source:1",
            source_id="source:1",
            parent_unit_id="unit:1",
            chunk_order_int=1,
            page_num_int=1,
            title="失眠",
            section_path=[],
            content="失眠可辨为心脾两虚。",
            content_with_weight="失眠可辨为心脾两虚。",
            content_ltks="失眠 心脾两虚",
            important_kwd=["失眠"],
        )
    ]

    result = search_chunks_by_terms(chunks, ["失眠"])

    assert isinstance(result, RagflowSearchResult)
    assert result.total == 1
    assert result.ids == ["chunk:1"]
    assert result.field["chunk:1"]["content_with_weight"] == "失眠可辨为心脾两虚。"


def test_doc_store_loads_candidates_from_repository():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_chunks([
        RetrievalChunk(
            chunk_id="chunk:1",
            doc_id="source:1",
            source_id="source:1",
            parent_unit_id="unit:1",
            chunk_order_int=1,
            page_num_int=1,
            title="失眠",
            section_path=[],
            content="失眠可辨为心脾两虚。",
            content_with_weight="失眠可辨为心脾两虚。",
            content_ltks="失眠 心脾两虚",
            important_kwd=["失眠"],
        )
    ])

    result = RagflowDocStore(repository).search_chunks(["失眠"])

    assert result.ids == ["chunk:1"]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_doc_store.py -q
```

Expected: FAIL because `doc_store.py` does not exist.

- [ ] **Step 3: Add repository method for available chunks**

Add this method to `backend/app/services/ragflow_compat/repository.py`:

```python
    def list_available_chunks(self, limit: int = 0) -> list[RetrievalChunk]:
        statement = (
            select(retrieval_chunks_table)
            .where(retrieval_chunks_table.c.available_int == 1)
            .order_by(retrieval_chunks_table.c.doc_id, retrieval_chunks_table.c.chunk_order_int)
        )
        if limit:
            statement = statement.limit(limit)
        with self.engine.begin() as connection:
            return [RetrievalChunk(**dict(row._mapping)) for row in connection.execute(statement)]
```

- [ ] **Step 4: Implement repository-backed doc search primitive**

Create `backend/app/services/ragflow_compat/doc_store.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from app.services.ragflow_compat.query import token_similarity
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalChunk


@dataclass(frozen=True)
class RagflowSearchResult:
    total: int
    ids: list[str]
    query_vector: list[float]
    field: dict[str, dict]
    keywords: list[str]


class RagflowDocStore:
    def __init__(self, repository: RagflowRetrievalRepository):
        self.repository = repository

    def search_chunks(self, keywords: list[str], limit: int = 64) -> RagflowSearchResult:
        return search_chunks_by_terms(self.repository.list_available_chunks(), keywords, limit=limit)


def search_chunks_by_terms(
    chunks: list[RetrievalChunk],
    keywords: list[str],
    limit: int = 64,
) -> RagflowSearchResult:
    scored: list[tuple[float, RetrievalChunk]] = []
    for chunk in chunks:
        doc_tokens = (chunk.content_ltks or chunk.content_with_weight).split()
        score = token_similarity(keywords, doc_tokens)
        if score <= 0 and not any(keyword in chunk.content_with_weight for keyword in keywords):
            continue
        scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    selected = [chunk for _score, chunk in scored[:limit]]
    return RagflowSearchResult(
        total=len(scored),
        ids=[chunk.chunk_id for chunk in selected],
        query_vector=[],
        field={chunk.chunk_id: _chunk_field(chunk) for chunk in selected},
        keywords=keywords,
    )


def _chunk_field(chunk: RetrievalChunk) -> dict:
    return {
        "content_ltks": chunk.content_ltks,
        "content_with_weight": chunk.content_with_weight,
        "doc_id": chunk.doc_id,
        "docnm_kwd": chunk.title,
        "kb_id": chunk.source_id,
        "important_kwd": chunk.important_kwd,
        "title_tks": chunk.title_tks,
        "question_tks": chunk.question_tks,
        "chunk_order_int": chunk.chunk_order_int,
        "page_num_int": chunk.page_num_int,
        "mom_id": chunk.parent_unit_id,
        "_score": 0.0,
    }
```

- [ ] **Step 5: Run doc-store test**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_doc_store.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/ragflow_compat/doc_store.py \
  backend/app/services/ragflow_compat/repository.py \
  backend/tests/test_ragflow_doc_store.py
git commit -m "feat: add ragflow doc store adapter"
```

---

## Task 10: RAGFlow Document Retrieval Flow

**Files:**

- Create: `backend/app/services/ragflow_compat/fulltext.py`
- Test: `backend/tests/test_ragflow_fulltext.py`

- [ ] **Step 1: Write failing fulltext retrieval test**

Create `backend/tests/test_ragflow_fulltext.py`:

```python
from app.services.model_clients import RerankClient
from app.services.ragflow_compat.fulltext import RagflowDocumentRetriever
from app.services.ragflow_compat.schemas import RetrievalChunk


def test_document_retriever_blends_token_score_and_rerank_score():
    chunks = [
        RetrievalChunk(
            chunk_id="chunk:1",
            doc_id="source:1",
            source_id="source:1",
            parent_unit_id="unit:1",
            chunk_order_int=1,
            page_num_int=1,
            title="失眠",
            section_path=[],
            content="失眠可辨为心脾两虚。",
            content_with_weight="失眠可辨为心脾两虚。",
            content_ltks="失眠 心脾两虚",
            important_kwd=["失眠"],
        ),
        RetrievalChunk(
            chunk_id="chunk:2",
            doc_id="source:1",
            source_id="source:1",
            parent_unit_id="unit:1",
            chunk_order_int=2,
            page_num_int=1,
            title="饮食",
            section_path=[],
            content="饮食宜节。",
            content_with_weight="饮食宜节。",
            content_ltks="饮食",
        ),
    ]

    result = RagflowDocumentRetriever(rerank_client=RerankClient.demo()).retrieve(
        question="失眠怎么辨证",
        chunks=chunks,
        topn=1,
    )

    assert result[0].chunk_id == "chunk:1"
    assert result[0].similarity > 0
    assert result[0].term_similarity > 0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_fulltext.py -q
```

Expected: FAIL because `fulltext.py` does not exist.

- [ ] **Step 3: Implement document retrieval**

Create `backend/app/services/ragflow_compat/fulltext.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from app.services.model_clients import RerankClient
from app.services.ragflow_compat.query import tokenize_query, token_similarity
from app.services.ragflow_compat.schemas import RetrievalChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    chunk: RetrievalChunk
    similarity: float
    term_similarity: float
    vector_similarity: float
    rerank_similarity: float


class RagflowDocumentRetriever:
    def __init__(
        self,
        rerank_client: RerankClient | None = None,
        term_weight: float = 0.3,
        vector_weight: float = 0.7,
        similarity_threshold: float = 0.0,
    ):
        self.rerank_client = rerank_client
        self.term_weight = term_weight
        self.vector_weight = vector_weight
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        question: str,
        chunks: list[RetrievalChunk],
        topn: int = 8,
    ) -> list[RetrievedChunk]:
        query_tokens = tokenize_query(question)
        candidates = []
        for chunk in chunks:
            term_score = token_similarity(query_tokens, chunk.content_ltks.split())
            lexical_hit = any(token in chunk.content_with_weight for token in query_tokens)
            if term_score <= 0 and not lexical_hit:
                continue
            candidates.append((chunk, term_score))
        if not candidates:
            return []

        rerank_scores = self._rerank(question, [chunk.content_with_weight for chunk, _ in candidates])
        retrieved: list[RetrievedChunk] = []
        for index, (chunk, term_score) in enumerate(candidates):
            rerank_score = rerank_scores.get(index, 0.0)
            score = self.term_weight * term_score + self.vector_weight * rerank_score
            if score < self.similarity_threshold:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    chunk=chunk,
                    similarity=score,
                    term_similarity=term_score,
                    vector_similarity=rerank_score,
                    rerank_similarity=rerank_score,
                )
            )
        retrieved.sort(key=lambda item: (-item.similarity, item.chunk_id))
        return retrieved[:topn]

    def _rerank(self, question: str, documents: list[str]) -> dict[int, float]:
        if not self.rerank_client:
            return {}
        ranked = self.rerank_client.rerank(question, documents)
        if not ranked:
            return {}
        max_score = max(score for _index, score in ranked) or 1.0
        return {index: score / max_score for index, score in ranked}
```

- [ ] **Step 4: Run fulltext tests**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_fulltext.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ragflow_compat/fulltext.py backend/tests/test_ragflow_fulltext.py
git commit -m "feat: add ragflow document retrieval flow"
```

---

## Task 11: KG Index From Graph Service

**Files:**

- Modify: `backend/app/services/ragflow_compat/sync_service.py`
- Test: `backend/tests/test_ragflow_sync_service.py`

- [ ] **Step 1: Add failing KG sync test**

Append to `backend/tests/test_ragflow_sync_service.py`:

```python
from app.models.graph import GraphEdge, GraphNode
from app.services.graph_service import GraphService


def test_sync_service_rebuilds_kg_entities_relations_and_type_samples():
    engine = _engine()
    ingestion = IngestionRepository(engine)
    retrieval = RagflowRetrievalRepository(engine)
    graph = GraphService(
        nodes=[
            GraphNode(id="symptom:失眠", name="失眠", label="Symptom", description="不寐"),
            GraphNode(id="syndrome:心脾两虚", name="心脾两虚", label="Syndrome", description="心脾不足"),
        ],
        edges=[
            GraphEdge(
                id="edge:1",
                source="symptom:失眠",
                target="syndrome:心脾两虚",
                relation="MANIFESTS_AS",
                display="可辨为",
                evidence_ids=["evidence:source:uploaded:abc:0001"],
            )
        ],
    )

    report = RagflowIndexSyncService(ingestion, retrieval).rebuild_kg(graph)

    assert report["kg_entities"] == 2
    assert report["kg_relations"] == 1
    assert retrieval.list_kg_entities()[0].entity_name == "失眠"
    assert retrieval.list_kg_relations()[0].from_entity_kwd == "失眠"
    assert retrieval.list_type_samples()[0].sample_entities
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_sync_service.py::test_sync_service_rebuilds_kg_entities_relations_and_type_samples -q
```

Expected: FAIL because `rebuild_kg` does not exist.

- [ ] **Step 3: Add KG rebuild implementation**

Extend `backend/app/services/ragflow_compat/sync_service.py`:

```python
from collections import defaultdict
from app.models.graph import GraphEdge, GraphNode
from app.services.graph_service import GraphService
from app.services.ragflow_compat.schemas import RetrievalKgEntity, RetrievalKgRelation, RetrievalTypeSamples
```

Add method inside `RagflowIndexSyncService`:

```python
    def rebuild_kg(self, graph_service: GraphService) -> dict[str, int]:
        nodes_by_id = {node.id: node for node in graph_service.nodes}
        adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in graph_service.edges:
            adjacency[edge.source].append(edge)
            adjacency[edge.target].append(edge)

        entities = [
            _entity_from_node(node, adjacency.get(node.id, []), nodes_by_id)
            for node in graph_service.nodes
        ]
        relations = [
            _relation_from_edge(edge, nodes_by_id)
            for edge in graph_service.edges
            if edge.source in nodes_by_id and edge.target in nodes_by_id
        ]
        samples = _type_samples(graph_service.nodes)
        self.retrieval_repository.replace_kg_entities(entities)
        self.retrieval_repository.replace_kg_relations(relations)
        self.retrieval_repository.replace_type_samples(samples)
        return {
            "kg_entities": len(entities),
            "kg_relations": len(relations),
            "kg_type_samples": len(samples),
        }
```

Add helpers:

```python
def _entity_from_node(
    node: GraphNode,
    edges: list[GraphEdge],
    nodes_by_id: dict[str, GraphNode],
) -> RetrievalKgEntity:
    neighbors = []
    for edge in edges[:20]:
        other_id = edge.target if edge.source == node.id else edge.source
        other = nodes_by_id.get(other_id)
        if other:
            neighbors.append(other.name)
    description = " ".join(part for part in [node.name, node.label, node.description, " ".join(neighbors)] if part)
    return RetrievalKgEntity(
        entity_id=f"entity:{node.id}",
        entity_name=node.name,
        entity_type=node.label,
        source_node_id=node.id,
        content_with_weight=json.dumps({"description": description}, ensure_ascii=False),
        description=description,
        rank_flt=max(1.0, float(len(edges))),
        n_hop_with_weight=_nhop_paths(node.id, nodes_by_id, edges),
        aliases=[],
        evidence_chunk_ids=[],
    )


def _relation_from_edge(edge: GraphEdge, nodes_by_id: dict[str, GraphNode]) -> RetrievalKgRelation:
    source = nodes_by_id[edge.source]
    target = nodes_by_id[edge.target]
    content = " ".join([source.name, edge.display or edge.relation, target.name])
    return RetrievalKgRelation(
        relation_id=f"relation:{edge.id}",
        from_entity_kwd=source.name,
        to_entity_kwd=target.name,
        relation_type=edge.relation,
        display=edge.display,
        content_with_weight=content,
        weight_int=max(1, len(edge.evidence_ids)),
        evidence_chunk_ids=[_chunk_id_from_evidence_id(evidence_id) for evidence_id in edge.evidence_ids],
        source_edge_id=edge.id,
    )


def _type_samples(nodes: list[GraphNode]) -> list[RetrievalTypeSamples]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        if node.name not in grouped[node.label]:
            grouped[node.label].append(node.name)
    return [
        RetrievalTypeSamples(
            entity_type=label,
            sample_entities=names[:50],
            sample_count=len(names),
            updated_at=_utc_now(),
        )
        for label, names in sorted(grouped.items())
    ]


def _nhop_paths(node_id: str, nodes_by_id: dict[str, GraphNode], edges: list[GraphEdge]) -> list[dict]:
    paths = []
    for edge in edges[:20]:
        other_id = edge.target if edge.source == node_id else edge.source
        node = nodes_by_id.get(node_id)
        other = nodes_by_id.get(other_id)
        if node and other:
            paths.append({"path": [node.name, other.name], "weights": [max(1.0, float(len(edge.evidence_ids) or 1))]})
    return paths


def _chunk_id_from_evidence_id(evidence_id: str) -> str:
    return evidence_id.replace("evidence:", "chunk:", 1) if evidence_id.startswith("evidence:") else evidence_id
```

- [ ] **Step 4: Run KG sync test**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_sync_service.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ragflow_compat/sync_service.py backend/tests/test_ragflow_sync_service.py
git commit -m "feat: rebuild ragflow kg index"
```

---

## Task 12: RAGFlow KG Search Flow

**Files:**

- Create: `backend/app/services/ragflow_compat/kg_search.py`
- Test: `backend/tests/test_ragflow_kg_search.py`

- [ ] **Step 1: Write failing KG search test**

Create `backend/tests/test_ragflow_kg_search.py`:

```python
from app.services.ragflow_compat.kg_search import RagflowKgSearch
from app.services.ragflow_compat.schemas import RetrievalKgEntity, RetrievalKgRelation, RetrievalTypeSamples


def test_kg_search_uses_entities_types_relations_and_nhop_scores():
    entities = [
        RetrievalKgEntity(
            entity_id="entity:symptom:失眠",
            entity_name="失眠",
            entity_type="Symptom",
            source_node_id="symptom:失眠",
            content_with_weight='{"description":"失眠 Symptom"}',
            description="失眠 Symptom",
            rank_flt=2,
            n_hop_with_weight=[{"path": ["失眠", "心脾两虚"], "weights": [3]}],
        ),
        RetrievalKgEntity(
            entity_id="entity:syndrome:心脾两虚",
            entity_name="心脾两虚",
            entity_type="Syndrome",
            source_node_id="syndrome:心脾两虚",
            content_with_weight='{"description":"心脾两虚 Syndrome"}',
            description="心脾两虚 Syndrome",
            rank_flt=3,
        ),
    ]
    relations = [
        RetrievalKgRelation(
            relation_id="relation:1",
            from_entity_kwd="失眠",
            to_entity_kwd="心脾两虚",
            relation_type="MANIFESTS_AS",
            display="可辨为",
            content_with_weight="失眠 可辨为 心脾两虚",
            weight_int=3,
        )
    ]
    samples = [RetrievalTypeSamples("Syndrome", ["心脾两虚"], 1, "now")]

    result = RagflowKgSearch().retrieve(
        question="失眠可以从哪些证候分析",
        entities=entities,
        relations=relations,
        type_samples=samples,
        type_keywords=["Syndrome"],
        query_entities=["失眠"],
    )

    assert result.entities[0]["Entity"] == "失眠"
    assert result.relations[0]["From Entity"] == "失眠"
    assert "---- Entities ----" in result.content
    assert "---- Relations ----" in result.content
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_kg_search.py -q
```

Expected: FAIL because `kg_search.py` does not exist.

- [ ] **Step 3: Implement KG search**

Create `backend/app/services/ragflow_compat/kg_search.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from app.services.ragflow_compat.query import token_similarity, tokenize_query
from app.services.ragflow_compat.schemas import RetrievalKgEntity, RetrievalKgRelation, RetrievalTypeSamples
from app.services.ragflow_compat.scoring import (
    analyze_nhop_paths,
    double_hit_boost,
    fuse_relation_scores,
    sort_and_trim_entities,
    sort_and_trim_relations,
)


@dataclass(frozen=True)
class KgSearchResult:
    entities: list[dict]
    relations: list[dict]
    content: str


class RagflowKgSearch:
    def retrieve(
        self,
        question: str,
        entities: list[RetrievalKgEntity],
        relations: list[RetrievalKgRelation],
        type_samples: list[RetrievalTypeSamples],
        type_keywords: list[str],
        query_entities: list[str],
        ent_topn: int = 6,
        rel_topn: int = 6,
    ) -> KgSearchResult:
        ents_from_query = self._relevant_entities_by_keywords(query_entities or [question], entities)
        ents_from_types = self._relevant_entities_by_types(type_keywords, entities)
        rels_from_text = self._relevant_relations_by_text(question, relations)
        nhop_paths = analyze_nhop_paths(ents_from_query)
        double_hit_boost(ents_from_query, set(ents_from_types))
        fuse_relation_scores(rels_from_text, set(ents_from_types), nhop_paths)
        scored_entities = sort_and_trim_entities(ents_from_query, ent_topn)
        scored_relations = sort_and_trim_relations(rels_from_text, rel_topn)
        return KgSearchResult(
            entities=scored_entities,
            relations=scored_relations,
            content=_format_content(scored_entities, scored_relations),
        )

    def _relevant_entities_by_keywords(
        self,
        keywords: list[str],
        entities: list[RetrievalKgEntity],
    ) -> dict[str, dict]:
        query_tokens = tokenize_query(" ".join(keywords))
        result = {}
        for entity in entities:
            tokens = tokenize_query(entity.content_with_weight)
            sim = token_similarity(query_tokens, tokens)
            if sim <= 0 and not any(keyword in entity.entity_name for keyword in keywords):
                continue
            result[entity.entity_name] = {
                "sim": max(sim, 0.1),
                "pagerank": entity.rank_flt,
                "n_hop_ents": entity.n_hop_with_weight,
                "description": entity.content_with_weight,
            }
        return result

    def _relevant_entities_by_types(
        self,
        type_keywords: list[str],
        entities: list[RetrievalKgEntity],
    ) -> dict[str, dict]:
        allowed = set(type_keywords)
        if not allowed:
            return {}
        return {
            entity.entity_name: {
                "sim": 1.0,
                "pagerank": entity.rank_flt,
                "description": entity.content_with_weight,
            }
            for entity in entities
            if entity.entity_type in allowed
        }

    def _relevant_relations_by_text(
        self,
        question: str,
        relations: list[RetrievalKgRelation],
    ) -> dict[tuple[str, str], dict]:
        query_tokens = tokenize_query(question)
        result = {}
        for relation in relations:
            tokens = tokenize_query(relation.content_with_weight)
            sim = token_similarity(query_tokens, tokens)
            if sim <= 0 and relation.from_entity_kwd not in question and relation.to_entity_kwd not in question:
                continue
            result[(relation.from_entity_kwd, relation.to_entity_kwd)] = {
                "sim": max(sim, 0.1),
                "pagerank": float(relation.weight_int),
                "description": relation.content_with_weight,
            }
        return result


def _format_content(entities: list[dict], relations: list[dict]) -> str:
    lines = ["---- Entities ----", "Entity,Score,Description"]
    lines.extend(f"{item['Entity']},{item['Score']:.2f},{item['Description']}" for item in entities)
    lines.extend(["---- Relations ----", "From Entity,To Entity,Score,Description"])
    lines.extend(
        f"{item['From Entity']},{item['To Entity']},{item['Score']:.2f},{item['Description']}"
        for item in relations
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run KG search tests**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_kg_search.py tests/test_ragflow_scoring.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ragflow_compat/kg_search.py backend/tests/test_ragflow_kg_search.py
git commit -m "feat: add ragflow kg search flow"
```

---

## Task 13: Parent Context Expansion

**Files:**

- Create: `backend/app/services/ragflow_compat/context.py`
- Test: `backend/tests/test_ragflow_context.py`

- [ ] **Step 1: Write failing context test**

Create `backend/tests/test_ragflow_context.py`:

```python
from app.services.ragflow_compat.context import expand_parent_context
from app.services.ragflow_compat.schemas import RetrievalChunk


def _chunk(chunk_id, order, content):
    return RetrievalChunk(
        chunk_id=chunk_id,
        doc_id="source:1",
        source_id="source:1",
        parent_unit_id="unit:1",
        chunk_order_int=order,
        page_num_int=1,
        title="失眠",
        section_path=[],
        content=content,
        content_with_weight=content,
        token_count=len(content),
    )


def test_expand_parent_context_returns_neighbor_chunks_in_order():
    chunks = [_chunk("chunk:1", 1, "前文"), _chunk("chunk:2", 2, "命中"), _chunk("chunk:3", 3, "后文")]

    expanded = expand_parent_context(chunks[1], chunks, window=1, max_tokens=100)

    assert [chunk.chunk_id for chunk in expanded] == ["chunk:1", "chunk:2", "chunk:3"]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_context.py -q
```

Expected: FAIL because `context.py` does not exist.

- [ ] **Step 3: Implement context expansion**

Create `backend/app/services/ragflow_compat/context.py`:

```python
from __future__ import annotations

from app.services.ragflow_compat.schemas import RetrievalChunk


def expand_parent_context(
    hit: RetrievalChunk,
    all_chunks: list[RetrievalChunk],
    window: int = 1,
    max_tokens: int = 1200,
) -> list[RetrievalChunk]:
    siblings = sorted(
        [
            chunk
            for chunk in all_chunks
            if chunk.doc_id == hit.doc_id and chunk.parent_unit_id == hit.parent_unit_id
        ],
        key=lambda chunk: chunk.chunk_order_int,
    )
    index = next((idx for idx, chunk in enumerate(siblings) if chunk.chunk_id == hit.chunk_id), -1)
    if index < 0:
        return [hit]
    selected = siblings[max(0, index - window) : index + window + 1]
    total = 0
    output = []
    for chunk in selected:
        total += chunk.token_count
        if total > max_tokens and output:
            break
        output.append(chunk)
    return output
```

- [ ] **Step 4: Run context tests**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_context.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ragflow_compat/context.py backend/tests/test_ragflow_context.py
git commit -m "feat: add ragflow parent context expansion"
```

---

## Task 14: Evidence Assembly

**Files:**

- Create: `backend/app/services/ragflow_compat/evidence.py`
- Test: `backend/tests/test_ragflow_retrieval_service.py`

- [ ] **Step 1: Write failing evidence assembly test**

Create `backend/tests/test_ragflow_retrieval_service.py`:

```python
from app.models.graph import GraphEdge, GraphNode
from app.services.ragflow_compat.evidence import build_evidence_cards
from app.services.ragflow_compat.fulltext import RetrievedChunk
from app.services.ragflow_compat.schemas import RetrievalChunk


def test_build_evidence_cards_preserves_source_score_and_snippet():
    chunk = RetrievalChunk(
        chunk_id="chunk:1",
        doc_id="source:1",
        source_id="source:1",
        parent_unit_id="unit:1",
        chunk_order_int=1,
        page_num_int=1,
        title="失眠",
        section_path=["卷一"],
        content="失眠可辨为心脾两虚。",
        content_with_weight="失眠可辨为心脾两虚。",
    )
    hit = RetrievedChunk("chunk:1", chunk, 0.9, 0.3, 0.8, 0.8)

    cards = build_evidence_cards([hit])

    assert cards[0].id == "evidence:1"
    assert cards[0].title == "失眠"
    assert cards[0].source == "source:1"
    assert "失眠可辨为心脾两虚" in cards[0].snippet
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_retrieval_service.py::test_build_evidence_cards_preserves_source_score_and_snippet -q
```

Expected: FAIL because `evidence.py` does not exist.

- [ ] **Step 3: Implement evidence assembly**

Create `backend/app/services/ragflow_compat/evidence.py`:

```python
from __future__ import annotations

from app.models.graph import EvidenceCard
from app.services.ragflow_compat.fulltext import RetrievedChunk


def build_evidence_cards(hits: list[RetrievedChunk]) -> list[EvidenceCard]:
    cards: list[EvidenceCard] = []
    for hit in hits:
        chunk = hit.chunk
        cards.append(
            EvidenceCard(
                id=chunk.chunk_id.replace("chunk:", "evidence:", 1),
                title=chunk.title or chunk.doc_id,
                source=chunk.source_id,
                snippet=chunk.content,
                source_type="local",
                location=f"{chunk.doc_id}:{chunk.chunk_order_int}",
            )
        )
    return cards
```

- [ ] **Step 4: Run evidence test**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_retrieval_service.py::test_build_evidence_cards_preserves_source_score_and_snippet -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ragflow_compat/evidence.py backend/tests/test_ragflow_retrieval_service.py
git commit -m "feat: add ragflow evidence assembly"
```

---

## Task 15: Retrieval Service Facade

**Files:**

- Create: `backend/app/services/ragflow_compat/retrieval_service.py`
- Test: `backend/tests/test_ragflow_retrieval_service.py`

- [ ] **Step 1: Add failing facade test**

Append to `backend/tests/test_ragflow_retrieval_service.py`:

```python
from app.services.llm import LlmClient
from app.services.model_clients import RerankClient
from app.services.ragflow_compat.kg_search import RagflowKgSearch
from app.services.ragflow_compat.retrieval_service import RagflowCompatibleRetrievalService


def test_retrieval_service_returns_query_response_with_ragflow_evidence():
    chunk = RetrievalChunk(
        chunk_id="chunk:source:1:0001",
        doc_id="source:1",
        source_id="source:1",
        parent_unit_id="unit:1",
        chunk_order_int=1,
        page_num_int=1,
        title="失眠",
        section_path=[],
        content="失眠可辨为心脾两虚。",
        content_with_weight="失眠可辨为心脾两虚。",
        content_ltks="失眠 心脾两虚",
        important_kwd=["失眠"],
    )
    service = RagflowCompatibleRetrievalService(
        chunks=[chunk],
        kg_entities=[],
        kg_relations=[],
        type_samples=[],
        llm_client=LlmClient.demo(),
        rerank_client=RerankClient.demo(),
    )

    response = service.answer("失眠怎么辨证")

    assert response.question == "失眠怎么辨证"
    assert response.evidence
    assert response.evidence[0].source == "source:1"
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_retrieval_service.py::test_retrieval_service_returns_query_response_with_ragflow_evidence -q
```

Expected: FAIL because `retrieval_service.py` does not exist.

- [ ] **Step 3: Implement facade**

Create `backend/app/services/ragflow_compat/retrieval_service.py`:

```python
from __future__ import annotations

from app.models.query import QueryResponse
from app.services.llm import LlmClient
from app.services.model_clients import RerankClient
from app.services.ragflow_compat.evidence import build_evidence_cards
from app.services.ragflow_compat.fulltext import RagflowDocumentRetriever
from app.services.ragflow_compat.kg_search import RagflowKgSearch
from app.services.ragflow_compat.schemas import RetrievalChunk, RetrievalKgEntity, RetrievalKgRelation, RetrievalTypeSamples


class RagflowCompatibleRetrievalService:
    def __init__(
        self,
        chunks: list[RetrievalChunk],
        kg_entities: list[RetrievalKgEntity],
        kg_relations: list[RetrievalKgRelation],
        type_samples: list[RetrievalTypeSamples],
        llm_client: LlmClient,
        rerank_client: RerankClient | None = None,
    ):
        self.chunks = chunks
        self.kg_entities = kg_entities
        self.kg_relations = kg_relations
        self.type_samples = type_samples
        self.llm_client = llm_client
        self.document_retriever = RagflowDocumentRetriever(rerank_client=rerank_client)
        self.kg_search = RagflowKgSearch()

    def answer(self, question: str) -> QueryResponse:
        chunk_hits = self.document_retriever.retrieve(question, self.chunks, topn=8)
        evidence = build_evidence_cards(chunk_hits)
        kg = self.kg_search.retrieve(
            question=question,
            entities=self.kg_entities,
            relations=self.kg_relations,
            type_samples=self.type_samples,
            type_keywords=[],
            query_entities=[question],
        )
        answer = self.llm_client.synthesize(
            question=question,
            entities=[item["Entity"] for item in kg.entities[:5]],
            evidence=[card.snippet for card in evidence],
            graph_paths=[
                f"{item['From Entity']} -> {item['To Entity']}"
                for item in kg.relations[:8]
            ],
        )
        return QueryResponse(
            question=question,
            answer=answer,
            intent="ragflow_compat",
            entities=[item["Entity"] for item in kg.entities[:5]],
            graph_nodes=[],
            graph_edges=[],
            highlighted_path=[],
            evidence=evidence,
        )
```

- [ ] **Step 4: Run facade tests**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_retrieval_service.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ragflow_compat/retrieval_service.py backend/tests/test_ragflow_retrieval_service.py
git commit -m "feat: add ragflow retrieval facade"
```

---

## Task 16: Config and `/api/query` Engine Switch

**Files:**

- Modify: `backend/app/core/config.py`
- Modify: `backend/app/services/question_service.py`
- Test: `backend/tests/test_question_service.py`

- [ ] **Step 1: Add failing config test**

Append to `backend/tests/test_question_service.py`:

```python
def test_question_service_can_delegate_to_ragflow_compatible_engine():
    class FakeRagflowService:
        def __init__(self):
            self.called = False

        def answer(self, question):
            self.called = True
            response = QuestionService.demo().answer(question)
            return response.model_copy(update={"intent": "ragflow_compat"})

    service = QuestionService.demo()
    fake = FakeRagflowService()
    service.ragflow_retrieval_service = fake
    service.retrieval_engine = "ragflow_compat"

    response = service.answer("失眠怎么辨证")

    assert fake.called is True
    assert response.intent == "ragflow_compat"
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_question_service.py::test_question_service_can_delegate_to_ragflow_compatible_engine -q
```

Expected: FAIL because `QuestionService` has no switch fields.

- [ ] **Step 3: Add settings**

Modify `backend/app/core/config.py` to include:

```python
    retrieval_engine: str = "legacy"
    ragflow_qdrant_collection: str = "tcm_ragflow_retrieval"
    ragflow_term_weight: float = 0.3
    ragflow_vector_weight: float = 0.7
    ragflow_similarity_threshold: float = 0.2
    ragflow_topk: int = 1024
    ragflow_rerank_topn: int = 64
```

- [ ] **Step 4: Add delegation fields**

Modify `QuestionService.__init__` in `backend/app/services/question_service.py`:

```python
        ragflow_retrieval_service=None,
        retrieval_engine: str = "legacy",
```

Inside `__init__`, set:

```python
        self.ragflow_retrieval_service = ragflow_retrieval_service
        self.retrieval_engine = retrieval_engine
```

At the top of `answer()` add:

```python
        if self.retrieval_engine == "ragflow_compat" and self.ragflow_retrieval_service:
            return self.ragflow_retrieval_service.answer(question)
```

- [ ] **Step 5: Run question-service test**

Run:

```bash
cd backend && python -m pytest tests/test_question_service.py::test_question_service_can_delegate_to_ragflow_compatible_engine -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/core/config.py backend/app/services/question_service.py backend/tests/test_question_service.py
git commit -m "feat: switch query service to ragflow retrieval"
```

---

## Task 17: Retrieval API Endpoints

**Files:**

- Modify: `backend/app/api/routes.py`
- Test: `backend/tests/test_api.py`

- [ ] **Step 1: Add failing API tests**

Append to `backend/tests/test_api.py`:

```python
def test_retrieval_status_endpoint(client):
    response = client.get("/api/retrieval/status")

    assert response.status_code == 200
    assert "engine" in response.json()
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_api.py::test_retrieval_status_endpoint -q
```

Expected: FAIL with 404.

- [ ] **Step 3: Add minimal status endpoint**

Add to `backend/app/api/routes.py`:

```python
@router.get("/retrieval/status")
def retrieval_status() -> dict:
    return {
        "engine": getattr(question_service, "retrieval_engine", "legacy"),
        "ragflow_collection": settings.ragflow_qdrant_collection,
    }
```

- [ ] **Step 4: Run API test**

Run:

```bash
cd backend && python -m pytest tests/test_api.py::test_retrieval_status_endpoint -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes.py backend/tests/test_api.py
git commit -m "feat: expose ragflow retrieval status"
```

---

## Task 18: End-to-End Local Verification

**Files:**

- No new source files.

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
cd backend && python -m pytest tests/test_ragflow_query.py \
  tests/test_ragflow_scoring.py \
  tests/test_ragflow_repository.py \
  tests/test_ragflow_sync_service.py \
  tests/test_ragflow_doc_store.py \
  tests/test_ragflow_fulltext.py \
  tests/test_ragflow_kg_search.py \
  tests/test_ragflow_context.py \
  tests/test_ragflow_vector_sync.py \
  tests/test_ragflow_retrieval_service.py \
  tests/test_ragflow_scripts.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 2: Run full backend tests**

Run:

```bash
cd backend && python -m pytest -q
```

Expected: all tests PASS. If existing unrelated tests fail, record exact failures and confirm whether they predate this plan by running the same failing tests on a clean baseline or by inspecting git diff.

- [ ] **Step 3: Run rebuild dry run against local Docker PostgreSQL**

Run:

```bash
python scripts/rebuild_ragflow_retrieval_index.py --dry-run
```

Expected: prints JSON with nonzero `raw_sources` and `raw_chunks` when Docker services are running.

- [ ] **Step 4: Run audit against local Docker PostgreSQL**

Run:

```bash
python scripts/audit_ragflow_retrieval_index.py --json
```

Expected: prints JSON with retrieval table counts. Counts may be zero before rebuild.

- [ ] **Step 5: Commit verification-only fixes if any**

If verification reveals a bug from these tasks, fix it in the smallest aligned change and commit:

```bash
git add <changed files>
git commit -m "fix: stabilize ragflow retrieval verification"
```

---

## Task 19: Full Data Rebuild and Vector Coverage

**Files:**

- No code files unless verification reveals implementation defects.

- [ ] **Step 1: Rebuild retrieval index**

Run:

```bash
python scripts/rebuild_ragflow_retrieval_index.py
```

Expected:

- `raw_sources` equals current `sources` count.
- `retrieval_chunks` is close to current non-empty `document_chunks` count.
- `skipped_empty_chunks` is reported.

- [ ] **Step 2: Audit rebuilt index**

Run:

```bash
python scripts/audit_ragflow_retrieval_index.py --json
```

Expected:

- `documents > 0`
- `chunks > 0`
- `chunks_with_vectors == 0` before vector sync unless vectors were already synced.

- [ ] **Step 3: Sync vectors in controlled batches**

Run first:

```bash
python scripts/sync_ragflow_retrieval_vectors.py \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --embedding-model "$EMBEDDING_MODEL" \
  --limit 128
```

Expected: `embedded` equals 128 or the number of missing chunks if fewer than 128 remain.

Run full sync only when API key, quota, and time are acceptable:

```bash
python scripts/sync_ragflow_retrieval_vectors.py \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --embedding-model "$EMBEDDING_MODEL"
```

Expected: all eligible retrieval chunks reach `vector_status=embedded`.

- [ ] **Step 4: Audit vector coverage**

Run:

```bash
python scripts/audit_ragflow_retrieval_index.py --json
```

Expected:

- `chunks_with_vectors == chunks`
- `kg_entities_with_vectors == kg_entities`
- `kg_relations_with_vectors == kg_relations`

---

## Task 20: Final Acceptance Verification

**Files:**

- No source files unless acceptance checks reveal implementation defects.

- [ ] **Step 1: Confirm no full RAGFlow service is deployed**

Run:

```bash
docker compose ps
```

Expected: no service named `ragflow`, no RAGFlow web/API container added by this work.

- [ ] **Step 2: Confirm retrieval tables exist**

Run:

```bash
docker exec tcm-postgres psql -U tcm -d tcm_kg -c "select table_name from information_schema.tables where table_schema='public' and table_name like 'retrieval_%' order by table_name;"
```

Expected: lists all retrieval tables from Task 1.

- [ ] **Step 3: Confirm Qdrant collection exists**

Run:

```bash
curl -sS http://localhost:6333/collections/tcm_ragflow_retrieval
```

Expected: JSON status ok with collection configuration.

- [ ] **Step 4: Confirm `/api/query` legacy still works**

Run:

```bash
curl -sS http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"失眠可以从哪些证候分析？"}'
```

Expected: JSON response with `answer` and `evidence`.

- [ ] **Step 5: Confirm ragflow-compatible query works**

Run with environment configured and API restarted:

```bash
RETRIEVAL_ENGINE=ragflow_compat docker compose up -d tcm-api
curl -sS http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"失眠可以从哪些证候分析？"}'
```

Expected: JSON response has `intent` equal to `ragflow_compat` or includes ragflow diagnostics; evidence sources point to PostgreSQL chunks.

- [ ] **Step 6: Run smoke query set**

Run each question:

```text
失眠可以从哪些证候分析？
柴胡桂枝干姜汤适合什么情况？
党参有什么功效？
普济方里关于中风有哪些治法？
外台秘要里针灸相关内容有哪些？
```

Expected for every response:

- non-empty answer
- at least one evidence card
- source identifier present
- score available in diagnostics or debug endpoint
- KG query has entity/relation hit in diagnostics

- [ ] **Step 7: Commit acceptance fixes**

If acceptance checks require fixes, commit them:

```bash
git add <changed files>
git commit -m "fix: complete ragflow retrieval acceptance"
```

---

## Self-Review

Spec coverage:

- No full RAGFlow deployment: Task 20 checks compose services.
- RAGFlow-compatible tables: Tasks 1, 2, 20.
- Existing data rebuild: Tasks 5, 7, 11, 19.
- Qdrant new collection and vector sync: Tasks 8, 19, 20.
- RAGFlow document retrieval flow: Tasks 3, 9, 10.
- RAGFlow KG scoring and retrieval flow: Tasks 4, 11, 12.
- Parent context and evidence assembly: Tasks 13, 14.
- `/api/query` switch: Tasks 15, 16, 17, 20.
- Auditing and smoke verification: Tasks 6, 18, 19, 20.
