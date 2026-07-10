# RAGFlow GraphRAG Build Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first DB-native RAGFlow-style GraphRAG construction pipeline: source subgraph resume, subgraph generation, global graph merge, phase marker invalidation, and retrieval index sync without destroying graph artifacts.

**Architecture:** Add `RagflowGraphBuildService` under `backend/app/services/ragflow_compat/`. The service uses existing `IngestionRepository`, `GraphExtractor`, `RagflowRetrievalRepository`, `GraphService`, and `RagflowRetrievalSyncService`; it does not copy RAGFlow runtime infrastructure. Graph artifacts remain the source of truth for per-source subgraph checkpoints and the global graph artifact.

**Tech Stack:** Python 3.13 in current local environment, SQLAlchemy 2, SQLite test repositories, pytest, existing Pydantic `GraphNode` / `GraphEdge` models, existing RAGFlow compatibility repository.

---

## Files

Create:

- `backend/app/services/ragflow_compat/graph_build_service.py`
  - Owns `RagflowGraphBuildService`, `RagflowGraphBuildSummary`, candidate-to-graph conversion, subgraph artifact creation, global merge, marker invalidation, and sync orchestration.
- `backend/tests/test_ragflow_graph_build_service.py`
  - Covers resume, subgraph build, merge, phase marker invalidation, sync preserving artifacts, and source-level failures.

Modify:

- `backend/app/services/ragflow_compat/repository.py`
  - Add `clear_rebuild_tables(include_graph_artifacts: bool = True)` so sync can preserve graph artifacts.
- `backend/app/services/ragflow_compat/sync_service.py`
  - Add constructor parameter `write_graph_artifacts: bool = True`.
  - Pass `include_graph_artifacts=write_graph_artifacts` into repository clearing.
  - Skip `_graph_artifacts_from_graph` append when `write_graph_artifacts=False`.
- `backend/tests/test_ragflow_sync_service.py`
  - Add regression test that sync can preserve graph artifacts.

Do not modify frontend files in this plan.

---

## Task 1: Build Service Skeleton And Resume Behavior

**Files:**

- Create: `backend/app/services/ragflow_compat/graph_build_service.py`
- Create: `backend/tests/test_ragflow_graph_build_service.py`

- [ ] **Step 1: Write the failing resume test**

Create `backend/tests/test_ragflow_graph_build_service.py` with this initial content:

```python
from __future__ import annotations

import json

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.models.graph import GraphNode
from app.models.ingestion import DocumentChunk, SourceManifest
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.graph_build_service import RagflowGraphBuildService
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalGraphArtifact


class RecordingExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, chunks, hint_terms=None):
        self.calls += 1
        return [], []


def _engine():
    return create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _repositories():
    engine = _engine()
    return IngestionRepository(engine), RagflowRetrievalRepository(engine)


def _source(source_id: str = "doc:a") -> SourceManifest:
    return SourceManifest(
        source_id=source_id,
        filename=f"{source_id}.txt",
        mime_type="text/plain",
        checksum=f"checksum:{source_id}",
        status="parsed",
    )


def _chunk(source_id: str = "doc:a", chunk_id: str = "chunk:doc:a:1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        page_id=f"page:{source_id}:1",
        chunk_index=0,
        content="白芍可养血敛阴。",
        content_type="text",
        token_count=8,
    )


def test_build_skips_source_when_subgraph_checkpoint_exists():
    ingestion_repository, retrieval_repository = _repositories()
    source = _source("doc:a")
    ingestion_repository.upsert_source(source)
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    retrieval_repository.save_graph_artifact(
        RetrievalGraphArtifact(
            artifact_id="subgraph:doc:a",
            artifact_type="subgraph",
            content_with_weight=json.dumps(
                {
                    "nodes": [
                        GraphNode(id="entity:herb:白芍", label="Herb", name="白芍").model_dump()
                    ],
                    "edges": [],
                },
                ensure_ascii=False,
            ),
            source_id=["doc:a"],
            node_count=1,
            edge_count=0,
            metadata={"scope": "source"},
        )
    )
    extractor = RecordingExtractor()

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=extractor,
    ).build(["doc:a"])

    assert extractor.calls == 0
    assert summary.sources_total == 1
    assert summary.sources_skipped == 1
    assert summary.sources_built == 0
    assert summary.sources_failed == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py::test_build_skips_source_when_subgraph_checkpoint_exists
```

Expected: FAIL with `ModuleNotFoundError` for `app.services.ragflow_compat.graph_build_service`.

- [ ] **Step 3: Implement the minimal skeleton**

Create `backend/app/services/ragflow_compat/graph_build_service.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from app.services.graph_extractor import GraphExtractor
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.repository import RagflowRetrievalRepository


@dataclass(frozen=True)
class RagflowGraphBuildSummary:
    sources_total: int
    sources_skipped: int
    sources_built: int
    sources_failed: int
    subgraphs_merged: int = 0
    global_nodes: int = 0
    global_edges: int = 0
    graph_changed: bool = False
    resolution_marker_cleared: bool = False
    community_marker_cleared: bool = False


class RagflowGraphBuildService:
    def __init__(
        self,
        *,
        ingestion_repository: IngestionRepository,
        retrieval_repository: RagflowRetrievalRepository,
        graph_extractor: GraphExtractor,
        chunk_batch_size: int = 1000,
    ):
        self.ingestion_repository = ingestion_repository
        self.retrieval_repository = retrieval_repository
        self.graph_extractor = graph_extractor
        self.chunk_batch_size = chunk_batch_size

    def build(self, source_ids: list[str] | None = None) -> RagflowGraphBuildSummary:
        selected_source_ids = self._source_ids(source_ids)
        skipped = 0
        failed = 0
        for source_id in selected_source_ids:
            if self.retrieval_repository.get_subgraph_artifact(source_id):
                skipped += 1
                continue
            failed += 1
        return RagflowGraphBuildSummary(
            sources_total=len(selected_source_ids),
            sources_skipped=skipped,
            sources_built=0,
            sources_failed=failed,
        )

    def _source_ids(self, source_ids: list[str] | None) -> list[str]:
        if source_ids is not None:
            return list(dict.fromkeys(source_ids))
        return [source.source_id for source in self.ingestion_repository.list_sources()]
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py::test_build_skips_source_when_subgraph_checkpoint_exists
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add backend/app/services/ragflow_compat/graph_build_service.py backend/tests/test_ragflow_graph_build_service.py
git commit -m "feat: add graphrag build service skeleton"
```

---

## Task 2: Generate And Save Per-Source Subgraph Artifacts

**Files:**

- Modify: `backend/app/services/ragflow_compat/graph_build_service.py`
- Modify: `backend/tests/test_ragflow_graph_build_service.py`

- [ ] **Step 1: Add the failing subgraph generation test**

Append to `backend/tests/test_ragflow_graph_build_service.py`:

```python
from app.models.ingestion import EntityCandidate, RelationCandidate


class FixedExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, chunks, hint_terms=None):
        self.calls += 1
        return [
            EntityCandidate(
                entity_id="entity:herb:白芍",
                name="白芍",
                label="Herb",
                normalized_name="白芍",
                source_chunk_ids=[chunks[0].chunk_id],
                confidence=0.9,
            ),
            EntityCandidate(
                entity_id="entity:function:养血敛阴",
                name="养血敛阴",
                label="Function",
                normalized_name="养血敛阴",
                source_chunk_ids=[chunks[0].chunk_id],
                confidence=0.8,
            ),
        ], [
            RelationCandidate(
                relation_id="relation:白芍:HAS_FUNCTION:养血敛阴",
                source_entity_id="entity:herb:白芍",
                target_entity_id="entity:function:养血敛阴",
                relation="HAS_FUNCTION",
                display="功效",
                evidence_chunk_ids=[chunks[0].chunk_id],
                confidence=0.8,
            )
        ]


def test_build_generates_and_saves_subgraph_artifact_for_new_source():
    ingestion_repository, retrieval_repository = _repositories()
    source = _source("doc:a")
    ingestion_repository.upsert_source(source)
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    extractor = FixedExtractor()

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=extractor,
    ).build(["doc:a"])

    artifact = retrieval_repository.get_subgraph_artifact("doc:a")
    assert extractor.calls == 1
    assert summary.sources_total == 1
    assert summary.sources_built == 1
    assert summary.sources_failed == 0
    assert artifact is not None
    assert artifact.artifact_id == "subgraph:doc:a"
    assert artifact.node_count == 2
    assert artifact.edge_count == 1
    payload = json.loads(artifact.content_with_weight)
    assert {node["name"] for node in payload["nodes"]} == {"白芍", "养血敛阴"}
    assert payload["edges"][0]["relation"] == "HAS_FUNCTION"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py::test_build_generates_and_saves_subgraph_artifact_for_new_source
```

Expected: FAIL because `artifact is None`.

- [ ] **Step 3: Implement subgraph artifact generation**

Replace `backend/app/services/ragflow_compat/graph_build_service.py` with:

```python
from __future__ import annotations

import json
from dataclasses import dataclass

from app.models.graph import GraphEdge, GraphNode
from app.models.ingestion import EntityCandidate, RelationCandidate
from app.services.graph_extractor import GraphExtractor
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalGraphArtifact


@dataclass(frozen=True)
class RagflowGraphBuildSummary:
    sources_total: int
    sources_skipped: int
    sources_built: int
    sources_failed: int
    subgraphs_merged: int = 0
    global_nodes: int = 0
    global_edges: int = 0
    graph_changed: bool = False
    resolution_marker_cleared: bool = False
    community_marker_cleared: bool = False


class RagflowGraphBuildService:
    def __init__(
        self,
        *,
        ingestion_repository: IngestionRepository,
        retrieval_repository: RagflowRetrievalRepository,
        graph_extractor: GraphExtractor,
        chunk_batch_size: int = 1000,
    ):
        self.ingestion_repository = ingestion_repository
        self.retrieval_repository = retrieval_repository
        self.graph_extractor = graph_extractor
        self.chunk_batch_size = chunk_batch_size

    def build(self, source_ids: list[str] | None = None) -> RagflowGraphBuildSummary:
        selected_source_ids = self._source_ids(source_ids)
        skipped = 0
        built = 0
        failed = 0
        for source_id in selected_source_ids:
            if self.retrieval_repository.get_subgraph_artifact(source_id):
                skipped += 1
                continue
            chunks = self.ingestion_repository.list_chunks(source_id)
            if not chunks:
                failed += 1
                continue
            entities, relations = self.graph_extractor.extract(chunks)
            nodes, edges = _graph_from_candidates(source_id, entities, relations)
            if not nodes:
                failed += 1
                continue
            self.retrieval_repository.save_graph_artifact(_subgraph_artifact(source_id, nodes, edges))
            built += 1
        return RagflowGraphBuildSummary(
            sources_total=len(selected_source_ids),
            sources_skipped=skipped,
            sources_built=built,
            sources_failed=failed,
            graph_changed=built > 0,
        )

    def _source_ids(self, source_ids: list[str] | None) -> list[str]:
        if source_ids is not None:
            return list(dict.fromkeys(source_ids))
        return [source.source_id for source in self.ingestion_repository.list_sources()]


def _graph_from_candidates(
    source_id: str,
    entities: list[EntityCandidate],
    relations: list[RelationCandidate],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes_by_id = {
        entity.entity_id: GraphNode(
            id=entity.entity_id,
            label=entity.label,
            name=entity.name,
            description=f"{entity.name} {entity.label}",
            properties={
                "source_id": source_id,
                "source_chunk_ids": entity.source_chunk_ids,
                "confidence": entity.confidence,
            },
        )
        for entity in entities
    }
    edges: list[GraphEdge] = []
    for relation in relations:
        if relation.source_entity_id not in nodes_by_id or relation.target_entity_id not in nodes_by_id:
            continue
        edges.append(
            GraphEdge(
                id=relation.relation_id,
                source=relation.source_entity_id,
                target=relation.target_entity_id,
                relation=relation.relation,
                display=relation.display,
                evidence_ids=relation.evidence_chunk_ids,
            )
        )
    return list(nodes_by_id.values()), edges


def _subgraph_artifact(
    source_id: str,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> RetrievalGraphArtifact:
    return RetrievalGraphArtifact(
        artifact_id=f"subgraph:{source_id}",
        artifact_type="subgraph",
        content_with_weight=json.dumps(_graph_payload(nodes, edges), ensure_ascii=False),
        source_id=[source_id],
        node_count=len(nodes),
        edge_count=len(edges),
        metadata={"scope": "source", "built_from": "graph_extractor"},
    )


def _graph_payload(nodes: list[GraphNode], edges: list[GraphEdge]) -> dict:
    return {
        "nodes": [node.model_dump() for node in nodes],
        "edges": [edge.model_dump() for edge in edges],
    }
```

- [ ] **Step 4: Run the graph build service tests**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add backend/app/services/ragflow_compat/graph_build_service.py backend/tests/test_ragflow_graph_build_service.py
git commit -m "feat: build source subgraph artifacts"
```

---

## Task 3: Merge Subgraphs Into Global Graph Artifact

**Files:**

- Modify: `backend/app/services/ragflow_compat/graph_build_service.py`
- Modify: `backend/tests/test_ragflow_graph_build_service.py`

- [ ] **Step 1: Add the failing global merge test**

Append to `backend/tests/test_ragflow_graph_build_service.py`:

```python
def test_build_merges_available_subgraphs_into_global_graph_artifact():
    ingestion_repository, retrieval_repository = _repositories()
    for source_id in ["doc:a", "doc:b"]:
        ingestion_repository.upsert_source(_source(source_id))
        ingestion_repository.replace_pages_and_chunks(source_id, [], [_chunk(source_id)])
    extractor = FixedExtractor()

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=extractor,
    ).build(["doc:a", "doc:b"])

    global_artifact = retrieval_repository.get_graph_artifact("graph:global")
    assert global_artifact is not None
    assert summary.subgraphs_merged == 2
    assert summary.global_nodes == 2
    assert summary.global_edges == 1
    assert global_artifact.source_id == ["doc:a", "doc:b"]
    payload = json.loads(global_artifact.content_with_weight)
    assert {node["name"] for node in payload["nodes"]} == {"白芍", "养血敛阴"}
    assert len(payload["edges"]) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py::test_build_merges_available_subgraphs_into_global_graph_artifact
```

Expected: FAIL because `global_artifact is None`.

- [ ] **Step 3: Add global merge implementation**

In `RagflowGraphBuildService.build`, after processing sources and before returning summary, add:

```python
        merged_nodes, merged_edges, merged_sources = _merge_subgraph_artifacts(
            self.retrieval_repository.list_graph_artifacts(available_only=True)
        )
        if merged_nodes or merged_edges:
            self.retrieval_repository.save_graph_artifact(
                _global_graph_artifact(merged_nodes, merged_edges, merged_sources)
            )
```

Then change the return block to:

```python
        return RagflowGraphBuildSummary(
            sources_total=len(selected_source_ids),
            sources_skipped=skipped,
            sources_built=built,
            sources_failed=failed,
            subgraphs_merged=len(merged_sources),
            global_nodes=len(merged_nodes),
            global_edges=len(merged_edges),
            graph_changed=built > 0,
        )
```

Add these helper functions to the bottom of `graph_build_service.py`:

```python
def _merge_subgraph_artifacts(artifacts) -> tuple[list[GraphNode], list[GraphEdge], list[str]]:
    nodes_by_id: dict[str, GraphNode] = {}
    edges_by_id: dict[str, GraphEdge] = {}
    merged_sources: set[str] = set()
    for artifact in sorted(artifacts, key=lambda item: item.artifact_id):
        if artifact.artifact_type != "subgraph" or artifact.available_int != 1:
            continue
        payload = json.loads(artifact.content_with_weight)
        merged_sources.update(str(source_id) for source_id in artifact.source_id)
        for node_data in payload.get("nodes", []):
            node = GraphNode.model_validate(node_data)
            existing = nodes_by_id.get(node.id)
            if existing:
                node = _merge_node(existing, node)
            nodes_by_id[node.id] = node
        for edge_data in payload.get("edges", []):
            edge = GraphEdge.model_validate(edge_data)
            existing = edges_by_id.get(edge.id)
            if existing:
                edge = _merge_edge(existing, edge)
            edges_by_id[edge.id] = edge
    return (
        [nodes_by_id[node_id] for node_id in sorted(nodes_by_id)],
        [edges_by_id[edge_id] for edge_id in sorted(edges_by_id)],
        sorted(merged_sources),
    )


def _merge_node(left: GraphNode, right: GraphNode) -> GraphNode:
    properties = dict(left.properties)
    for key, value in right.properties.items():
        if key == "source_chunk_ids":
            properties[key] = sorted(
                set(str(item) for item in properties.get(key, []))
                | set(str(item) for item in value)
            )
        elif key not in properties or not properties[key]:
            properties[key] = value
    return left.model_copy(update={"properties": properties})


def _merge_edge(left: GraphEdge, right: GraphEdge) -> GraphEdge:
    evidence_ids = sorted(set(left.evidence_ids) | set(right.evidence_ids))
    return left.model_copy(update={"evidence_ids": evidence_ids})


def _global_graph_artifact(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    source_ids: list[str],
) -> RetrievalGraphArtifact:
    return RetrievalGraphArtifact(
        artifact_id="graph:global",
        artifact_type="graph",
        content_with_weight=json.dumps(_graph_payload(nodes, edges), ensure_ascii=False),
        source_id=source_ids,
        node_count=len(nodes),
        edge_count=len(edges),
        metadata={"scope": "global", "built_from": "subgraphs"},
    )
```

- [ ] **Step 4: Run graph build service tests**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add backend/app/services/ragflow_compat/graph_build_service.py backend/tests/test_ragflow_graph_build_service.py
git commit -m "feat: merge graphrag subgraphs"
```

---

## Task 4: Phase Marker Invalidation

**Files:**

- Modify: `backend/app/services/ragflow_compat/graph_build_service.py`
- Modify: `backend/tests/test_ragflow_graph_build_service.py`

- [ ] **Step 1: Add failing invalidation tests**

Append to `backend/tests/test_ragflow_graph_build_service.py`:

```python
from app.services.ragflow_compat.phase_markers import PHASE_COMMUNITY, PHASE_RESOLUTION


def test_build_clears_phase_markers_when_new_subgraph_changes_graph():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    retrieval_repository.set_graphrag_phase_marker(PHASE_RESOLUTION)
    retrieval_repository.set_graphrag_phase_marker(PHASE_COMMUNITY)

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=FixedExtractor(),
    ).build(["doc:a"])

    assert summary.graph_changed is True
    assert summary.resolution_marker_cleared is True
    assert summary.community_marker_cleared is True


def test_build_keeps_phase_markers_on_pure_resume():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])
    retrieval_repository.save_graph_artifact(
        RetrievalGraphArtifact(
            artifact_id="subgraph:doc:a",
            artifact_type="subgraph",
            content_with_weight=json.dumps({"nodes": [], "edges": []}, ensure_ascii=False),
            source_id=["doc:a"],
            node_count=0,
            edge_count=0,
        )
    )
    retrieval_repository.set_graphrag_phase_marker(PHASE_RESOLUTION)
    retrieval_repository.set_graphrag_phase_marker(PHASE_COMMUNITY)

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=RecordingExtractor(),
    ).build(["doc:a"])

    assert summary.graph_changed is False
    assert summary.resolution_marker_cleared is False
    assert summary.community_marker_cleared is False
    assert retrieval_repository.has_graphrag_phase_marker(PHASE_RESOLUTION) is True
    assert retrieval_repository.has_graphrag_phase_marker(PHASE_COMMUNITY) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py -k phase_markers
```

Expected: first test FAILS because `summary.resolution_marker_cleared` is still `False`.

- [ ] **Step 3: Implement marker invalidation**

Add imports to `graph_build_service.py`:

```python
from app.services.ragflow_compat.phase_markers import PHASE_COMMUNITY, PHASE_RESOLUTION
```

Before the summary return in `build`, add:

```python
        resolution_marker_cleared = False
        community_marker_cleared = False
        if built > 0:
            self.retrieval_repository.clear_graphrag_phase_markers(
                [PHASE_RESOLUTION, PHASE_COMMUNITY]
            )
            resolution_marker_cleared = True
            community_marker_cleared = True
```

Update the summary return fields:

```python
            resolution_marker_cleared=resolution_marker_cleared,
            community_marker_cleared=community_marker_cleared,
```

- [ ] **Step 4: Run graph build service tests**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add backend/app/services/ragflow_compat/graph_build_service.py backend/tests/test_ragflow_graph_build_service.py
git commit -m "feat: invalidate graphrag phase markers"
```

---

## Task 5: Preserve Graph Artifacts During Retrieval Sync

**Files:**

- Modify: `backend/app/services/ragflow_compat/repository.py`
- Modify: `backend/app/services/ragflow_compat/sync_service.py`
- Modify: `backend/tests/test_ragflow_sync_service.py`

- [ ] **Step 1: Add failing sync preservation test**

Append to `backend/tests/test_ragflow_sync_service.py`:

```python
def test_sync_service_can_preserve_graph_artifacts_when_requested():
    ingestion_repository, retrieval_repository = _shared_repositories()
    retrieval_repository.save_graph_artifact(
        RetrievalGraphArtifact(
            artifact_id="subgraph:doc:a",
            artifact_type="subgraph",
            content_with_weight='{"nodes":[],"edges":[]}',
            source_id=["doc:a"],
            node_count=0,
            edge_count=0,
        )
    )

    summary = RagflowRetrievalSyncService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        write_graph_artifacts=False,
    ).rebuild_from_ingestion()

    assert summary["graph_artifacts"] == 0
    assert retrieval_repository.get_subgraph_artifact("doc:a") is not None
```

Add this import near the existing RAGFlow compatibility imports:

```python
from app.services.ragflow_compat.schemas import RetrievalGraphArtifact
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest backend/tests/test_ragflow_sync_service.py::test_sync_service_can_preserve_graph_artifacts_when_requested
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'write_graph_artifacts'`.

- [ ] **Step 3: Modify repository clear behavior**

In `backend/app/services/ragflow_compat/repository.py`, change:

```python
    def clear_rebuild_tables(self) -> None:
```

to:

```python
    def clear_rebuild_tables(self, *, include_graph_artifacts: bool = True) -> None:
```

Inside the table list, replace the inline list with:

```python
            tables = [
                retrieval_chunk_terms_table,
                retrieval_chunks_table,
                retrieval_documents_table,
                retrieval_kg_entities_table,
                retrieval_kg_relations_table,
                retrieval_kg_community_reports_table,
                retrieval_kg_type_samples_table,
            ]
            if include_graph_artifacts:
                tables.append(retrieval_kg_graph_artifacts_table)
            for table in tables:
                connection.execute(delete(table))
```

- [ ] **Step 4: Modify sync service constructor and graph artifact writes**

In `backend/app/services/ragflow_compat/sync_service.py`, add the constructor parameter:

```python
        write_graph_artifacts: bool = True,
```

Set it in `__init__`:

```python
        self.write_graph_artifacts = write_graph_artifacts
```

After graph branch computes `graph_artifacts`, add:

```python
            if not self.write_graph_artifacts:
                graph_artifacts = []
```

Change:

```python
        self.retrieval_repository.clear_rebuild_tables()
```

to:

```python
        self.retrieval_repository.clear_rebuild_tables(
            include_graph_artifacts=self.write_graph_artifacts
        )
```

- [ ] **Step 5: Run sync tests**

Run:

```bash
pytest backend/tests/test_ragflow_sync_service.py backend/tests/test_ragflow_repository.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add backend/app/services/ragflow_compat/repository.py backend/app/services/ragflow_compat/sync_service.py backend/tests/test_ragflow_sync_service.py
git commit -m "feat: preserve graph artifacts during sync"
```

---

## Task 6: Sync Retrieval Index From Build Service Global Graph

**Files:**

- Modify: `backend/app/services/ragflow_compat/graph_build_service.py`
- Modify: `backend/tests/test_ragflow_graph_build_service.py`

- [ ] **Step 1: Add failing end-to-end sync test**

Append to `backend/tests/test_ragflow_graph_build_service.py`:

```python
def test_build_syncs_retrieval_kg_index_from_global_graph():
    ingestion_repository, retrieval_repository = _repositories()
    ingestion_repository.upsert_source(_source("doc:a"))
    ingestion_repository.replace_pages_and_chunks("doc:a", [], [_chunk("doc:a")])

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=FixedExtractor(),
    ).build(["doc:a"])

    entities = retrieval_repository.list_kg_entities()
    relations = retrieval_repository.list_kg_relations()
    reports = retrieval_repository.list_community_reports()
    assert summary.global_nodes == 2
    assert {entity.entity_name for entity in entities} == {"白芍", "养血敛阴"}
    assert len(relations) == 1
    assert reports
    assert retrieval_repository.get_subgraph_artifact("doc:a") is not None
    assert retrieval_repository.get_graph_artifact("graph:global") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py::test_build_syncs_retrieval_kg_index_from_global_graph
```

Expected: FAIL because `list_kg_entities()` is empty.

- [ ] **Step 3: Implement sync from global graph**

Add imports to `graph_build_service.py`:

```python
from app.services.graph_service import GraphService
from app.services.ragflow_compat.sync_service import RagflowRetrievalSyncService
```

After marker invalidation and before summary return, add:

```python
        if merged_nodes or merged_edges:
            RagflowRetrievalSyncService(
                ingestion_repository=self.ingestion_repository,
                retrieval_repository=self.retrieval_repository,
                graph_service=GraphService(merged_nodes, merged_edges),
                write_graph_artifacts=False,
            ).rebuild_from_ingestion()
```

- [ ] **Step 4: Run graph build service tests**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

Run:

```bash
git add backend/app/services/ragflow_compat/graph_build_service.py backend/tests/test_ragflow_graph_build_service.py
git commit -m "feat: sync kg index from graphrag build"
```

---

## Task 7: Source-Level Failure Handling

**Files:**

- Modify: `backend/app/services/ragflow_compat/graph_build_service.py`
- Modify: `backend/tests/test_ragflow_graph_build_service.py`

- [ ] **Step 1: Add failing failure-isolation test**

Append to `backend/tests/test_ragflow_graph_build_service.py`:

```python
class FailingForDocBExtractor(FixedExtractor):
    def extract(self, chunks, hint_terms=None):
        if chunks[0].source_id == "doc:b":
            raise RuntimeError("extract failed")
        return super().extract(chunks, hint_terms=hint_terms)


def test_build_records_source_failure_and_continues_other_sources():
    ingestion_repository, retrieval_repository = _repositories()
    for source_id in ["doc:a", "doc:b"]:
        ingestion_repository.upsert_source(_source(source_id))
        ingestion_repository.replace_pages_and_chunks(source_id, [], [_chunk(source_id)])

    summary = RagflowGraphBuildService(
        ingestion_repository=ingestion_repository,
        retrieval_repository=retrieval_repository,
        graph_extractor=FailingForDocBExtractor(),
    ).build(["doc:a", "doc:b"])

    assert summary.sources_total == 2
    assert summary.sources_built == 1
    assert summary.sources_failed == 1
    assert retrieval_repository.get_subgraph_artifact("doc:a") is not None
    assert retrieval_repository.get_subgraph_artifact("doc:b") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py::test_build_records_source_failure_and_continues_other_sources
```

Expected: FAIL with `RuntimeError: extract failed`.

- [ ] **Step 3: Catch per-source extractor failures**

In `RagflowGraphBuildService.build`, wrap the extraction and artifact save block:

```python
            try:
                entities, relations = self.graph_extractor.extract(chunks)
                nodes, edges = _graph_from_candidates(source_id, entities, relations)
                if not nodes:
                    failed += 1
                    continue
                self.retrieval_repository.save_graph_artifact(_subgraph_artifact(source_id, nodes, edges))
                built += 1
            except Exception:
                failed += 1
                continue
```

- [ ] **Step 4: Run graph build service tests**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 7**

Run:

```bash
git add backend/app/services/ragflow_compat/graph_build_service.py backend/tests/test_ragflow_graph_build_service.py
git commit -m "feat: isolate graphrag source failures"
```

---

## Task 8: Full Verification

**Files:**

- No code changes unless verification fails.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest backend/tests/test_ragflow_graph_build_service.py backend/tests/test_ragflow_sync_service.py backend/tests/test_ragflow_repository.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full backend tests**

Run:

```bash
pytest backend/tests
```

Expected: all backend tests pass. Existing deprecation warnings are acceptable only if they match the current Starlette/httpx warning already present in previous runs.

- [ ] **Step 3: Run lint**

Run:

```bash
python3 -m ruff check backend/app/services/ragflow_compat/graph_build_service.py backend/app/services/ragflow_compat/repository.py backend/app/services/ragflow_compat/sync_service.py backend/tests/test_ragflow_graph_build_service.py backend/tests/test_ragflow_sync_service.py
```

Expected: `All checks passed!`.

- [ ] **Step 4: Confirm clean worktree**

Run:

```bash
git status --short
```

Expected: no output.

---

## Self-Review Notes

Spec coverage:

- Resume behavior: Task 1.
- Per-source subgraph artifact: Task 2.
- Global graph merge: Task 3.
- Phase marker invalidation: Task 4.
- Sync preserving artifacts: Task 5.
- Retrieval KG index sync: Task 6.
- Source-level failure isolation: Task 7.
- Verification gates: Task 8.

Scope intentionally excluded from this implementation plan:

- RAGFlow general/light LLM prompt loop.
- LLM entity resolution checkpoint replay.
- Leiden hierarchical community report generation.
- Distributed task lock, retry timeout, cancellation callback.

Those items require separate specs because they touch model prompts, async task orchestration, and graph algorithms beyond this first DB-native build service.
