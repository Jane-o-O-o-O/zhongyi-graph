# RAGFlow GraphRAG Build Service Design

## Goal

把当前项目从“RAGFlow-compatible 检索索引”推进到“RAGFlow-style GraphRAG 构建流水线”。下一阶段新增一个 DB-native 的 `RagflowGraphBuildService`，复用本项目已有 ingestion、GraphExtractor、GraphService、retrieval repository，同时对齐 RAGFlow 的核心阶段语义：

```text
sources/chunks
-> per-source subgraph checkpoint
-> merge global graph
-> invalidate resolution/community phase markers when graph changes
-> graph analytics + entity resolution + community summaries
-> retrieval KG index rebuild
-> mark resolution_done/community_done
```

这不是最终完全移植。它是完整对齐 RAGFlow 前必须补上的构建主干，让后续 LLM entity resolution、Leiden community report、retry/timeout/task lock 都能接到清晰的位置。

## Non-Goals

- 不直接复制 RAGFlow 的 Redis、docStore、task queue、tenant/kb runtime。
- 不把当前 ingestion 表迁移成 RAGFlow 原表。
- 不在第一版实现 RAGFlow 的 general/light extractor prompt loop。
- 不在第一版实现真正的 LLM entity resolution 和 LLM community report。
- 不改前端 3D 图谱接口，除非后续实现需要新增诊断字段。

## Current State

已完成的兼容基础：

- `retrieval_kg_entities`、`retrieval_kg_relations`、`retrieval_kg_community_reports`、`retrieval_kg_graph_artifacts`。
- `retrieval_graphrag_checkpoints`、`retrieval_graphrag_phase_markers`。
- RAGFlow-compatible checkpoint constants、stable checkpoint key、phase marker constants。
- DB-backed checkpoint adapter: `load_checkpoints`、`save_checkpoint`、`cleanup_checkpoints`。
- graph artifact recovery: `save_graph_artifact`、`get_graph_artifact`、`get_subgraph_artifact`。
- `RagflowRetrievalSyncService` 可以从现有 `GraphService` 写入 retrieval KG index。

主要缺口：

- 没有服务按 source 构建 subgraph artifact。
- 没有服务 merge subgraph artifacts 生成 `graph:global`。
- 没有明确的 graph-changed invalidation：新 subgraph 合并后必须清理 `resolution_done` 和 `community_done`。
- 没有 resume 行为：已有 subgraph 应跳过重复抽取。
- 没有构建结果 summary，无法知道哪些 source skipped、built、failed、merged。

## RAGFlow Reference Behavior

参考文件：

- `/Users/jinzhangzheng/中医/ragflow/rag/graphrag/general/index.py`
- `/Users/jinzhangzheng/中医/ragflow/rag/graphrag/checkpoints.py`
- `/Users/jinzhangzheng/中医/ragflow/rag/graphrag/phase_markers.py`
- `/Users/jinzhangzheng/中医/ragflow/rag/graphrag/utils.py`

需要保留的行为：

- `load_subgraph_from_store` 按 `knowledge_graph_kwd=subgraph` 和 `source_id/doc_id` 命中已有子图。
- `generate_subgraph` 每个 doc/source 生成一个可恢复 subgraph。
- `merge_subgraph` 把子图合入全局图。
- 新内容合入全局图后，清理 `resolution_done` 和 `community_done`。
- resolution/community 成功后写 phase marker。
- checkpoint/phase marker 是恢复优化，失败不能破坏旧图。

## Recommended Architecture

新增模块：

```text
backend/app/services/ragflow_compat/graph_build_service.py
```

主要类：

```python
class RagflowGraphBuildService:
    def __init__(
        *,
        ingestion_repository: IngestionRepository,
        retrieval_repository: RagflowRetrievalRepository,
        graph_extractor: GraphExtractor,
        chunk_batch_size: int = 1000,
    ): ...

    def build(self, source_ids: list[str] | None = None) -> RagflowGraphBuildSummary: ...
```

主要 summary：

```python
@dataclass(frozen=True)
class RagflowGraphBuildSummary:
    sources_total: int
    sources_skipped: int
    sources_built: int
    sources_failed: int
    subgraphs_merged: int
    global_nodes: int
    global_edges: int
    graph_changed: bool
    resolution_marker_cleared: bool
    community_marker_cleared: bool
```

## Data Flow

### 1. Source Selection

`build(source_ids=None)`:

- 如果传入 `source_ids`，按该列表处理。
- 如果为空，使用 `ingestion_repository.list_sources()` 的全部 source。
- 没有 chunks 的 source 计入 `sources_failed`，原因是它没有可构建输入，不能被当作成功 resume。

### 2. Subgraph Resume

每个 source 先调用：

```python
retrieval_repository.get_subgraph_artifact(source_id)
```

命中则跳过抽取，计入 `sources_skipped`。

这对应 RAGFlow 的：

```text
doc subgraph found in store, skipping LLM extraction
```

### 3. Subgraph Generation

未命中的 source：

- 读取 `ingestion_repository.list_chunks(source_id)`。
- 调用现有 `GraphExtractor.extract(chunks)`。
- 把 `EntityCandidate` 转为 `GraphNode`。
- 把 `RelationCandidate` 转为 `GraphEdge`。
- 保存为 `RetrievalGraphArtifact`：

```text
artifact_id = subgraph:<source_id>
artifact_type = subgraph
source_id = [source_id]
content_with_weight = {"nodes":[...], "edges":[...]}
node_count = len(nodes)
edge_count = len(edges)
metadata = {"scope":"source", "built_from":"graph_extractor"}
```

第一版使用当前项目的 `GraphNode` / `GraphEdge` JSON shape，不使用 NetworkX node-link shape。原因是当前前端和 retrieval sync 已经使用该 shape；后续如要和 RagFlow node-link 完全一致，可以在 artifact metadata 中新增 `format` 并做迁移。

### 4. Global Merge

构建或跳过 source 后，读取全部 available subgraph artifacts：

```python
retrieval_repository.list_graph_artifacts(available_only=True)
```

过滤 `artifact_type == "subgraph"`，按 `artifact_id` 排序 merge：

- node id 相同则合并 properties。
- edge id 相同则合并 evidence ids。
- source_id 汇总去重。
- 不直接丢弃已有 global graph，新的 global 从 subgraphs 重新组合，保证幂等。

保存：

```text
artifact_id = graph:global
artifact_type = graph
source_id = all merged source ids
content_with_weight = {"nodes":[...], "edges":[...]}
node_count
edge_count
metadata = {"scope":"global", "built_from":"subgraphs"}
```

### 5. Phase Marker Invalidation

如果本次有新 subgraph built，或 global graph 内容变化：

- `clear_graphrag_phase_markers([PHASE_RESOLUTION, PHASE_COMMUNITY])`
- summary 中标记两个 marker 已清理。

如果只是 resume 且 global 未变化，不清理 marker。

### 6. Retrieval Index Sync

global graph 更新后：

- 从 `graph:global` payload 还原 `GraphService`。
- 调用 `RagflowRetrievalSyncService(..., graph_service=graph_service).rebuild_from_ingestion()`。
- 第一版需要给 sync service 增加参数，让 build service 调用时不清理、不重写 `retrieval_kg_graph_artifacts`。graph artifacts 由 build service 负责，sync service 只写 documents、chunks、entities、relations、community reports、type samples 和 phase marker。

原因：当前 `RagflowRetrievalSyncService.clear_rebuild_tables()` 会清理 graph artifacts。如果 build service 先保存 source subgraph，再调用 sync service，全量清理会破坏 RAGFlow 的 per-doc/per-source subgraph checkpoint 语义。

## Error Handling

- 单个 source 抽取失败：记录失败并继续处理其他 source。
- 单个 source 无 chunks：计入 failed，不保存空 subgraph。
- extractor 返回空 entities：保存空 subgraph 价值低，第一版计入 failed。
- relation 指向缺失 entity：忽略该 relation，保留 nodes。
- global merge 失败：不清理 phase marker，不覆盖旧 `graph:global`。
- retrieval sync 失败：保留 graph artifacts，summary 抛出异常；这样后续可以重跑 sync 而不用重抽 subgraph。

## Testing Plan

第一批测试：

- 已存在 subgraph artifact 时，`build()` 跳过 extractor。
- 没有 subgraph artifact 时，`build()` 调用 extractor 并保存 `subgraph:<source_id>`。
- 多个 subgraph merge 后生成 `graph:global`。
- 新 subgraph 导致 `resolution_done` / `community_done` marker 被清理。
- 纯 resume 且 global 未变化时，不清理 marker。
- extractor 单 source 失败不影响其他 source。
- build 后 retrieval KG index 有 entities、relations、community reports。

第二批测试：

- relation 指向缺失 entity 时被过滤。
- 同一 entity 跨 source 合并时 source/evidence 不丢失。
- 重复运行 build 幂等，不重复生成边和 source ids。
- graph artifacts 内容能被前端现有图谱 API 使用。

## Implementation Order

1. 增加 `RagflowGraphBuildSummary` 和 `RagflowGraphBuildService` skeleton。
2. 实现 source selection 和 subgraph resume。
3. 实现 candidate -> GraphNode/GraphEdge 转换和 subgraph save。
4. 实现 subgraph merge -> global graph artifact。
5. 实现 phase marker invalidation。
6. 调整 `RagflowRetrievalSyncService`，支持 build service 调用时保留 graph artifacts。
7. 接入 `RagflowRetrievalSyncService`。
8. 增加脚本或 API 入口，后续用于手动触发 build。

## Acceptance Criteria For This Stage

- 后端测试覆盖 build service 的 resume、build、merge、invalidation、sync。
- `pytest backend/tests` 通过。
- `ruff check` 通过。
- 每个可验证小阶段单独 commit。
- 不宣称“完全对齐 RagFlow”，直到后续 LLM extractor loop、LLM entity resolution、Leiden community report、task retry/lock/cancel 都完成并有测试。
