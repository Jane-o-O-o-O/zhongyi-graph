# RAGFlow-Compatible Retrieval Redesign

## Goal

在当前 `tcm-kg-platform` 内复用和移植 RAGFlow 的检索流程，不部署完整 RAGFlow 服务。现有 PostgreSQL、Neo4j、Qdrant、MinIO 继续作为本项目的数据底座，但要新增一层 RAGFlow-compatible 检索数据层，让 RAGFlow 的普通 RAG 检索和 KG 检索流程可以稳定运行。

最终 `/api/query` 应切换到新的检索链路：

```text
用户问题
-> RAGFlow query rewrite / query tokenize
-> 普通文档检索：全文 + 向量 + rerank
-> KG 检索：实体 + 类型 + 关系 + n-hop path scoring
-> parent context 回填
-> evidence assembly
-> LLM synthesis
```

## Non-Goals

- 不启动 RAGFlow 的完整 Web、API、任务队列、租户、权限、ES/Infinity/OceanBase 系统。
- 不把当前业务数据迁移到 RAGFlow 自带数据库。
- 不直接破坏现有 `sources`、`document_chunks`、`extraction_units`、`entity_candidates`、`relation_candidates` 表。
- 不把 RAGFlow 全仓库复制进本项目。只移植检索所需的最小代码和算法，并保留来源注释和许可证信息。

## Current Data Findings

当前 PostgreSQL 文档侧数据：

- `sources`: 1,318 条。
- `document_chunks`: 343,901 条。
- `extraction_units`: 158,857 条。
- `document_pages`: 1,318 条。
- `entity_candidates`: 266 条。
- `relation_candidates`: 192 条。
- `publish_batches`: 14 条。

当前发布状态：

- `parsed`: 1,226 个 source，261,910 个 chunks。
- `published`: 92 个 source，81,991 个 chunks。

当前 Qdrant：

- collection: `tcm_knowledge`
- total points: 42,325
- `content_type=chunk`: 35,728
- `content_type=entity`: 6,482
- `content_type=evidence`: 115
- 对全部 PostgreSQL chunks 的向量覆盖率只有 10.39%。

当前 Neo4j：

- nodes: 6,702
- relationships: 16,545
- 主要 label: `Indication`, `Herb`, `Prescription`, `Alias`, `Formula`
- 主要 relation: `COMPOSED_OF`, `TREATS`, `HAS_DOSE`, `HAS_PRESCRIPTION`

当前 MinIO：

- bucket: `tcm-documents`
- objects: 1,348
- size: 171 MiB

这些数据说明：原始文档和结构化切片已经较多，但检索用向量、全文索引、KG 实体/关系索引都不完整，不能直接支撑 RAGFlow 的检索流程。

## RAGFlow Code Sources

移植时以这些 RAGFlow 文件作为优先参考：

- `/Users/jinzhangzheng/中医/ragflow/rag/nlp/search.py`
  - 普通文档检索流程。
  - 全文检索、向量检索、融合、rerank window、score filtering。
- `/Users/jinzhangzheng/中医/ragflow/rag/nlp/query.py`
  - 中文 query 处理、token 权重、同义词扩展、token similarity。
- `/Users/jinzhangzheng/中医/ragflow/rag/graphrag/search.py`
  - KG query rewrite、实体召回、类型召回、关系召回、n-hop 路径融合。
- `/Users/jinzhangzheng/中医/ragflow/rag/graphrag/query_analyze_prompt.py`
  - KG query rewrite prompt。
- `/Users/jinzhangzheng/中医/ragflow/internal/service/kg/scoring.go`
  - KG scoring 的 Go 版清晰实现，可用来校验 Python 迁移。
- `/Users/jinzhangzheng/中医/ragflow/rag/advanced_rag/tree_structured_query_decomposition_retrieval.py`
  - 复杂问题递归拆解。第一阶段只保留接口空间，不优先实现。

原则：遇到检索行为不确定时，优先对齐 RAGFlow 源码，而不是沿用本项目现有简化逻辑。

## Architecture

新增后端包：

```text
backend/app/services/ragflow_compat/
  __init__.py
  doc_store.py
  fulltext.py
  kg_search.py
  query.py
  rerank.py
  retrieval_service.py
  scoring.py
  schemas.py
  sync_service.py
```

职责：

- `doc_store.py`: 实现 RAGFlow `Dealer` 需要的 doc store 适配器。底层访问 PostgreSQL 和 Qdrant。
- `fulltext.py`: 移植 RAGFlow 普通文档检索流程。优先复用 RAGFlow 的 query/token/scoring 代码。
- `kg_search.py`: 移植 RAGFlow `KGSearch.retrieval` 流程。
- `query.py`: query rewrite prompt、JSON parse fallback、TCM type pool 构造。
- `rerank.py`: 统一调用现有 `RerankClient`，对齐 RAGFlow `rerank_by_model` 的分数融合。
- `scoring.py`: 实体、关系、n-hop 路径融合打分，和 RAGFlow `scoring.go` 保持公式一致。
- `sync_service.py`: 从现有表、Neo4j、Qdrant 重建兼容检索层。
- `retrieval_service.py`: `/api/query` 使用的门面服务。

现有 `QuestionService` 保留，但内部改为调用 `RagflowCompatibleRetrievalService`。用环境变量控制切换：

```env
RETRIEVAL_ENGINE=ragflow_compat
```

可选值：

- `legacy`: 当前检索流程。
- `ragflow_compat`: 新流程。

默认在开发阶段先保留 `legacy` fallback，数据同步和验证完成后再改默认值。

## Database Design

新增表不替换旧表。

### retrieval_documents

文档级检索元数据。

- `doc_id` primary key。使用 `source_id`。
- `source_id` unique。
- `filename`
- `mime_type`
- `checksum`
- `status`
- `object_key`
- `source_version`
- `chunk_count`
- `eligible_chunk_count`
- `created_from`
- `metadata` jsonb

用途：对齐 RAGFlow 的 `doc_id`、`docnm_kwd`、KB 文档聚合。

### retrieval_chunks

RAGFlow-compatible chunk 主表。

- `chunk_id` primary key。沿用 `document_chunks.chunk_id`。
- `doc_id`
- `source_id`
- `parent_unit_id`
- `chunk_order_int`
- `page_num_int`
- `title`
- `section_path` jsonb
- `content`
- `content_with_weight`
- `content_ltks`
- `title_tks`
- `important_kwd` jsonb
- `question_tks`
- `content_type`
- `token_count`
- `available_int`
- `vector_point_id`
- `vector_status`: `missing`, `queued`, `embedded`, `failed`
- `metadata` jsonb

索引：

- btree: `doc_id`, `source_id`, `parent_unit_id`, `available_int`, `vector_status`
- GIN: `to_tsvector('simple', content_with_weight)`
- GIN jsonb: `important_kwd`, `section_path`
- trigram: `content_with_weight`, `title`

说明：PostgreSQL 不是 RAGFlow 默认的 ES/Infinity，但可以通过 `retrieval_chunks` 提供 RAGFlow 检索流程需要的字段形状。

### retrieval_chunk_terms

可选但建议保留，用于 RAGFlow token similarity 和诊断。

- `chunk_id`
- `term`
- `weight`
- `term_type`: `content`, `title`, `important`, `question`

索引：

- btree: `term`
- btree: `chunk_id`

### retrieval_kg_entities

KG entity 检索索引。

- `entity_id` primary key
- `entity_name`
- `entity_type`
- `source_node_id`
- `content_with_weight`
- `description`
- `rank_flt`
- `n_hop_with_weight` jsonb
- `aliases` jsonb
- `evidence_chunk_ids` jsonb
- `available_int`
- `vector_point_id`
- `vector_status`
- `metadata` jsonb

RAGFlow 对应字段：

- `knowledge_graph_kwd = entity`
- `entity_kwd = entity_name`
- `entity_type_kwd = entity_type`
- `rank_flt`
- `n_hop_with_weight`
- `content_with_weight`

数据来源：

- 优先 Neo4j 当前图谱。
- 辅助使用 `entity_candidates`。
- 对没有 description 的节点，用 label、name、aliases、相邻关系和证据 chunk 构造描述。

### retrieval_kg_relations

KG relation 检索索引。

- `relation_id` primary key
- `from_entity_kwd`
- `to_entity_kwd`
- `relation_type`
- `display`
- `content_with_weight`
- `weight_int`
- `evidence_chunk_ids` jsonb
- `source_edge_id`
- `available_int`
- `vector_point_id`
- `vector_status`
- `metadata` jsonb

RAGFlow 对应字段：

- `knowledge_graph_kwd = relation`
- `from_entity_kwd`
- `to_entity_kwd`
- `weight_int`
- `content_with_weight`

数据来源：

- 优先 Neo4j relationships。
- 辅助使用 `relation_candidates`。
- evidence 通过边上的 `evidence_ids`、`source_chunks`、candidate evidence chunk ids 解析。

### retrieval_kg_type_samples

给 RAGFlow query rewrite 构造 `TYPE_POOL`。

- `entity_type`
- `sample_entities` jsonb
- `sample_count`
- `updated_at`

### retrieval_sync_state

同步状态。

- `sync_key` primary key
- `status`
- `started_at`
- `finished_at`
- `cursor`
- `total`
- `processed`
- `failed`
- `metadata` jsonb

用途：全量重建、断点续跑、失败诊断。

### retrieval_query_logs

诊断表。

- `query_id`
- `question`
- `rewrite_result` jsonb
- `chunk_candidates` jsonb
- `kg_entities` jsonb
- `kg_relations` jsonb
- `final_evidence` jsonb
- `latency_ms`
- `created_at`

只在开发和诊断模式启用。

## Qdrant Design

新增 collection，避免污染当前 `tcm_knowledge`：

```text
tcm_ragflow_retrieval
```

payload 类型：

- `retrieval_chunk`
- `retrieval_kg_entity`
- `retrieval_kg_relation`

payload 字段：

- `id`
- `content_type`
- `text`
- `chunk_id`
- `doc_id`
- `source_id`
- `entity_id`
- `entity_name`
- `entity_type`
- `relation_id`
- `from_entity_kwd`
- `to_entity_kwd`

向量维度继续使用当前 `EMBEDDING_MODEL` 配置的 1024 维。

同步要求：

- 所有 `available_int=1` 的 `retrieval_chunks` 必须有向量。
- 所有 `available_int=1` 的 `retrieval_kg_entities` 必须有向量。
- 所有 `available_int=1` 的 `retrieval_kg_relations` 必须有向量。
- 同步脚本必须支持 `--dry-run`、`--limit`、`--offset`、`--only-missing`、`--resume`。

## Data Quality Rules

全量重建时必须处理这些问题：

1. 空内容 chunk 不进入检索层。
2. `token_count < 20` 的短 chunk 不删除，但标记 `metadata.short_chunk=true`，召回后必须 parent context 回填。
3. `token_count >= 1000` 的长 chunk 标记 `metadata.long_chunk=true`，优先在重建阶段拆分或在召回阶段降权。
4. 同一 `source_id` 下 chunk 顺序必须由 `chunk_index`、`unit_index`、`page_number` 稳定确定。
5. `parent_unit_id` 缺失的旧数据要从 `metadata.generated_from_legacy_chunk` 或相邻位置信息补齐；无法补齐时使用自身作为 parent。
6. `section_path`、`title` 缺失时使用 source filename 和 chunk index 构造 fallback。
7. `entity_candidates` 和 `relation_candidates` 数据量太少，不能作为 KG 主索引来源；Neo4j 才是主来源。
8. `published` 状态不能作为唯一检索范围；全量检索层默认覆盖 `parsed + published`。
9. 每条 KG relation 至少保留 relation type、两端实体和 display 文本；没有 evidence 时仍进入 KG 检索，但 evidence score 降权。
10. 每次重建后要输出审计报告：原始行数、进入检索层行数、跳过行数、向量覆盖率、KG entity/relation 覆盖率。

## Sync Pipeline

新增脚本：

```text
scripts/rebuild_ragflow_retrieval_index.py
scripts/sync_ragflow_retrieval_vectors.py
scripts/audit_ragflow_retrieval_index.py
```

### rebuild_ragflow_retrieval_index.py

步骤：

1. 读取 `sources`、`extraction_units`、`document_chunks`。
2. 重建 `retrieval_documents`。
3. 生成 `retrieval_chunks`：
   - `content_with_weight = title + section_path + content`
   - `content_ltks` 使用移植的 RAGFlow tokenizer。
   - `important_kwd` 从标题、目录、实体候选、方药名、病证名中提取。
4. 从 Neo4j 读取节点和关系。
5. 生成 `retrieval_kg_entities`。
6. 生成 `retrieval_kg_relations`。
7. 计算每个 entity 的 `n_hop_with_weight`：
   - 默认 2-hop。
   - 对中医路径可配置到 3-hop：病证 -> 治法 -> 方剂 -> 药物。
   - 每段 path 带 relation weight。
8. 生成 `retrieval_kg_type_samples`。
9. 写入 `retrieval_sync_state` 和审计报告。

### sync_ragflow_retrieval_vectors.py

步骤：

1. 找出 `vector_status != embedded` 的 retrieval records。
2. 按 batch 调 embedding API。
3. 写入 `tcm_ragflow_retrieval`。
4. 更新 `vector_point_id` 和 `vector_status`。
5. 失败记录写入 sync state，支持续跑。

### audit_ragflow_retrieval_index.py

输出：

- 原始 chunk count vs retrieval chunk count。
- retrieval chunk vector coverage。
- KG entity count and vector coverage。
- KG relation count and vector coverage。
- short/long chunk count。
- parent context coverage。
- sample query smoke test 结果。

## Retrieval Flow

### Query Analysis

使用 RAGFlow `minirag_query2kwd` prompt。

输入：

- 用户问题。
- `retrieval_kg_type_samples` 生成的 type pool。

输出：

- `answer_type_keywords`
- `entities_from_query`

失败 fallback：

- entities = literal terms + raw question。
- answer types = intent heuristic。

### Document Retrieval

对齐 RAGFlow `Dealer.retrieval`：

1. 构造 fulltext query。
2. 构造 dense vector query。
3. 从 PostgreSQL/Qdrant 拉候选。
4. 计算：
   - token similarity
   - vector similarity
   - rank feature
5. 如果配置 rerank model，则走 RAGFlow `rerank_by_model` 方式融合。
6. threshold 过滤。
7. 稳定排序。
8. 返回 RAGFlow-shaped chunks。

默认权重：

```text
term_similarity_weight = 0.3
vector_similarity_weight = 0.7
```

可配置：

```env
RAGFLOW_TERM_WEIGHT=0.3
RAGFLOW_VECTOR_WEIGHT=0.7
RAGFLOW_SIMILARITY_THRESHOLD=0.2
RAGFLOW_TOPK=1024
RAGFLOW_RERANK_TOPN=64
```

### KG Retrieval

对齐 RAGFlow `KGSearch.retrieval`：

1. query rewrite 得到 `type keywords` 和 `entities`。
2. `get_relevant_ents_by_keywords`
   - 用 entities 拼接文本做 entity dense search。
3. `get_relevant_ents_by_types`
   - 用 answer type keywords 查 entity type pool。
4. `get_relevant_relations_by_txt`
   - 用原始问题做 relation dense search。
5. `AnalyzeNHopPaths`
   - 从 matched entity 的 `n_hop_with_weight` 拆出 path edge。
   - score 按距离衰减：`ent.sim / (2 + hop_index)`。
6. `DoubleHitBoost`
   - 同时命中 query entity 和 type entity 的实体 sim 翻倍。
7. `FuseRelationScores`
   - relation text hit、type hit、n-hop hit 融合。
8. entity 和 relation 都按 `sim * pagerank` 排序。
9. 输出 KG synthetic chunk，作为证据候选插入普通 chunks 前面。

这部分公式必须用单元测试对齐 RAGFlow `internal/service/kg/scoring.go`。

### Context Expansion

普通 chunk 和 KG evidence 都要做 small-to-big：

1. 命中 chunk。
2. 根据 `parent_unit_id` 取 parent unit。
3. 取同 parent 下相邻 chunk。
4. 控制总 token。
5. evidence 展示仍指向原始命中 chunk，但 LLM 上下文包含 parent/context。

### Answer Assembly

新 `EvidenceAssembler` 输出：

- evidence cards
- graph paths
- retrieved chunks
- KG entities
- KG relations
- citations
- diagnostics

`QueryResponse` 保持前端兼容。新增字段可以先放到可选 `metadata` 或诊断接口，避免破坏前端。

## API Changes

新增接口：

```text
POST /api/retrieval/rebuild-index
POST /api/retrieval/sync-vectors
GET  /api/retrieval/status
POST /api/retrieval/audit
POST /api/retrieval/debug-query
```

开发阶段这些接口可直接触发脚本；生产阶段可以改为后台任务。

`/api/query`：

- `RETRIEVAL_ENGINE=legacy` 时保持当前行为。
- `RETRIEVAL_ENGINE=ragflow_compat` 时走新链路。

## Testing

单元测试：

- RAGFlow query rewrite JSON parse fallback。
- token query 构造和中文 token similarity。
- KG `AnalyzeNHopPaths`。
- KG `DoubleHitBoost`。
- KG `FuseRelationScores`。
- entity/relation sorting by `sim * pagerank`。
- PostgreSQL doc store adapter filter/search behavior。
- Qdrant payload mapping。
- parent context expansion。

集成测试：

- 小型 fixture 重建 retrieval tables。
- fixture 向量同步到 fake Qdrant。
- `/api/query` 在 `ragflow_compat` 下返回 evidence、chunks、graph nodes。

数据审计测试：

- retrieval chunks 覆盖所有 eligible document chunks。
- retrieval vectors 覆盖所有 eligible retrieval chunks。
- KG entities 覆盖 Neo4j nodes。
- KG relations 覆盖 Neo4j relationships。

手动 smoke queries：

```text
失眠可以从哪些证候分析？
柴胡桂枝干姜汤适合什么情况？
党参有什么功效？
普济方里关于中风有哪些治法？
外台秘要里针灸相关内容有哪些？
```

每个问题至少验证：

- 有 chunk evidence。
- 有 source filename。
- 有 similarity score。
- KG 问题有 entity/relation 命中。
- 回答引用的证据能追溯到 PostgreSQL chunk。

## Migration and Rollback

迁移顺序：

1. 新增 retrieval tables。
2. 全量 rebuild retrieval index。
3. 全量 sync retrieval vectors。
4. 跑 audit。
5. 跑 debug queries。
6. 开启 `RETRIEVAL_ENGINE=ragflow_compat`。

回滚：

- 环境变量改回 `RETRIEVAL_ENGINE=legacy`。
- 新表和新 Qdrant collection 保留，不影响旧流程。
- 如需清理，只删除 retrieval tables 和 `tcm_ragflow_retrieval` collection。

## Implementation Phases

### Phase 1: Schema and Audit

- 新增 retrieval schema。
- 新增 repository。
- 新增 audit 脚本。
- 不改变 `/api/query`。

### Phase 2: Chunk Retrieval Index

- 从旧 chunks 重建 `retrieval_chunks`。
- 移植 RAGFlow query/token/fulltext 逻辑。
- 加 PostgreSQL fulltext/trigram index。

### Phase 3: Full Vector Coverage

- 新建 Qdrant collection。
- 全量补齐 chunk vectors。
- 断点续跑。
- audit 覆盖率。

### Phase 4: RAGFlow Document Retrieval

- 移植 `Dealer.retrieval`。
- 接入 rerank。
- 输出 RAGFlow-shaped chunks。

### Phase 5: KG Retrieval Index

- 从 Neo4j 重建 entity/relation/type/n-hop 索引。
- 补齐 entity/relation vectors。

### Phase 6: RAGFlow KG Retrieval

- 移植 `KGSearch.retrieval`。
- 对齐 scoring tests。
- 输出 KG synthetic chunk 和 graph paths。

### Phase 7: Query Integration

- `QuestionService.answer()` 接入新 service。
- 增加 debug query endpoint。
- 保留 legacy fallback。

### Phase 8: Verification and Default Switch

- 跑单元测试、集成测试、审计脚本、smoke queries。
- 确认后将默认检索引擎切到 `ragflow_compat`。

## Acceptance Criteria

完成状态必须同时满足：

1. 没有部署完整 RAGFlow 服务。
2. 本项目内存在 RAGFlow-compatible retrieval tables。
3. 检索流程使用移植后的 RAGFlow 普通 RAG 和 KG 检索代码。
4. 所有 eligible PostgreSQL chunks 都进入 retrieval index。
5. 所有 eligible retrieval chunks 都有 Qdrant 向量。
6. Neo4j nodes/relationships 进入 KG retrieval index。
7. KG entity/relation/n-hop scoring 与 RAGFlow 公式一致，有测试覆盖。
8. `/api/query` 可通过环境变量切换 legacy 和 ragflow_compat。
9. `ragflow_compat` 查询返回 chunk evidence、KG evidence、source、score、graph paths。
10. 审计脚本能证明数据覆盖率和向量覆盖率。
11. smoke queries 通过。

## Risks

- 全量 343,901 chunks embedding 会消耗时间和外部 API 额度，必须支持续跑和审计。
- PostgreSQL fulltext/trigram 不等价于 RAGFlow 的 ES/Infinity，需要 doc store adapter 尽量模拟 RAGFlow search expression 行为。
- RAGFlow tokenizer 和 synonym 依赖较重，第一版需要最小移植，并在中医语料上补 domain terms。
- 旧数据里 `entity_candidates` 和 `relation_candidates` 质量不足，KG index 必须以 Neo4j 为主。
- 长短 chunk 分布会影响召回，必须启用 parent context expansion。
- 直接复制 RAGFlow 文件要保留许可证说明，避免来源不清。
