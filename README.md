<div align="center">
  <img src="docs/assets/readme-hero.svg" alt="中医知识图谱与 GraphRAG 智能问答平台 - Traditional Chinese Medicine Knowledge Graph Platform" width="100%" />

  <h1>TCM Knowledge Graph Platform</h1>

  <p><strong>中医知识图谱 · Traditional Chinese Medicine Knowledge Graph · GraphRAG</strong></p>
  <p>面向中医药文献、症状证候、治法方药与典籍证据的可视化知识工程平台。</p>
  <p><em>An end-to-end TCM knowledge graph and GraphRAG platform for traceable question answering, hybrid retrieval, document ingestion, entity-relation extraction, and 3D graph exploration.</em></p>

  <p>
    <img src="https://img.shields.io/badge/React-18-20232a?style=flat-square&logo=react&logoColor=61dafb" alt="React 18" />
    <img src="https://img.shields.io/badge/FastAPI-Python%203.11-0b8f70?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Neo4j-5-4581c3?style=flat-square&logo=neo4j&logoColor=white" alt="Neo4j 5" />
    <img src="https://img.shields.io/badge/Qdrant-Vector%20Search-dc244c?style=flat-square&logo=qdrant&logoColor=white" alt="Qdrant Vector Search" />
    <img src="https://img.shields.io/badge/GraphRAG-RAGFlow--compatible-b34235?style=flat-square" alt="GraphRAG" />
    <img src="https://img.shields.io/badge/LLM-OpenAI--compatible-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI-compatible LLM" />
    <img src="https://img.shields.io/badge/Docker-Compose-2496ed?style=flat-square&logo=docker&logoColor=white" alt="Docker Compose" />
  </p>

  <p>
    <a href="#about">项目介绍</a> ·
    <a href="#why">为什么选择</a> ·
    <a href="#use-cases">应用场景</a> ·
    <a href="#features">核心能力</a> ·
    <a href="#architecture">系统架构</a> ·
    <a href="#quick-start">快速开始</a> ·
    <a href="#faq">FAQ</a>
  </p>
</div>

---

<a id="about"></a>

## 项目介绍 / Project Overview

**TCM Knowledge Graph Platform** 是一个端到端的中医知识图谱（Traditional Chinese Medicine Knowledge Graph）与 GraphRAG 智能问答平台。项目使用 **Neo4j、Qdrant、PostgreSQL、MinIO、FastAPI 和 React**，将分散在 PDF、Word、结构化数据和中医典籍中的症状、证候、治法、方剂、中药、功效与出处组织为可检索、可探索、可追溯的知识网络。

Unlike a generic TCM chatbot or vector-only RAG demo, this project combines a medical knowledge graph, RAGFlow-compatible hybrid retrieval, evidence-grounded generation, resumable GraphRAG indexing, and an interactive 3D graph workbench.

它不是一个只返回文字的通用聊天界面，而是将一次中医问答还原为一条可解释的知识路径：

```text
问题 -> 实体识别 -> 图谱路径 -> 混合检索 -> 证据溯源 -> 答案生成
```

项目同时包含两条相互独立、又能协同工作的主线：

- **图谱问答工作台**：面向演示、查询与知识探索，展示答案、关系路径和原始证据。
- **知识摄取与构建系统**：面向知识管理，负责文件解析、切片、抽取、校验、子图构建与发布。

> [!NOTE]
> 项目定位为中医知识工程、检索研究与产品演示平台，不替代专业医疗诊断或治疗建议。

> [!TIP]
> 如果你正在研究中医知识图谱、中医大模型、医疗 GraphRAG 或可追溯 RAG，可以 Star 本仓库以便持续关注后续构建。

<a id="why"></a>

## 为什么选择这个项目

中医知识天然具有多层次、多跳关系和强证据依赖：同一症状可能关联多个证候，同一方剂又会关联组成、功效、主治、剂量与典籍来源。只使用向量相似度很难稳定表达这些结构。

| 对比项 | 普通向量 RAG | TCM Knowledge Graph Platform |
| --- | --- | --- |
| 知识组织 | 文本切片与向量相似度 | 症状、证候、治法、方剂、中药、典籍关系图 |
| 检索方式 | 以向量召回为主 | 全文 + 向量 + 图谱 + 重排 + 社区摘要 |
| 回答展示 | 文字答案与参考片段 | 答案 + 3D 图谱 + 高亮路径 + 证据卡片 |
| 知识构建 | 文档分块后一次性索引 | 结构化切片、实体关系抽取、subgraph checkpoint 与全局图合并 |
| 可解释性 | 依赖模型文本解释 | 结论可回溯到图谱关系、文档切片和原始来源 |
| 无模型演示 | 通常不可用 | 保留本地稳定演示路径 |

这使它同时适合作为 **中医知识库、医疗知识图谱、GraphRAG 研究原型、中医智能问答平台** 与可视化产品演示基座。

<a id="use-cases"></a>

## 应用场景

| 场景 | 可以做什么 |
| --- | --- |
| 中医文献数字化 | 将古籍、现代文献、方剂表格和本草资料转换为统一知识产物。 |
| 中医知识图谱建设 | 构建症状、证候、治法、方剂、中药、功效、归经和典籍关系。 |
| 方剂与中药研究 | 探索方剂组成、药味配伍、功效主治、相关证候与文献出处。 |
| 证据型中医问答 | 让 LLM 答案携带图谱推理路径、来源片段与典籍证据。 |
| GraphRAG 与 RAG 研究 | 评估图检索、混合检索、实体消歧、社区摘要和证据组织。 |
| 医疗 AI 产品演示 | 以图谱为视觉中心，展示不同于通用聊天机器人的知识工作台。 |

<a id="features"></a>

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 图谱驱动问答 | 从自然语言问题中识别意图与实体，组织症状 -> 证候 -> 治法 -> 方药 -> 典籍路径。 |
| 3D 知识探索 | 基于 `Three.js` 与 `3d-force-graph` 展示全局图谱、高亮推理路径并支持节点交互。 |
| GraphRAG 检索 | 融合图谱检索、全文检索、向量检索、重排与社区摘要。 |
| 证据可追溯 | 答案与原始切片、典籍、本地文档和图谱关系建立关联。 |
| 多格式文档入库 | 支持 PDF、DOCX、Markdown、TXT、CSV、JSON 和图片类来源，可配置 OCR。 |
| 可恢复图谱构建 | 按数据源保存 subgraph checkpoint，支持全局图合并、阶段标记和异步构建任务。 |
| 模型接入可替换 | 通过 OpenAI-compatible API 分别配置 LLM、Embedding、Rerank 和 OCR 模型。 |
| 本地稳定演示 | 未配置外部模型时使用内置演示路径，便于快速启动与展示。 |

## 知识图谱数据模型

当前图谱面向中医辨证、方剂和本草知识组织，核心实体包括：

| 知识类型 | 典型实体 | 典型关系 |
| --- | --- | --- |
| 辨证知识 | 症状 `Symptom`、证候 `Syndrome` | 可见于、辨证为、伴随 |
| 治疗知识 | 治法 `Treatment`、功效 `Function` | 治以、具有功效 |
| 方剂知识 | 方剂 `Formula`、组成 `Prescription`、剂量 `Dosage` | 方剂、组成、剂量、主治 |
| 本草知识 | 中药 `Herb`、归经 `Channel`、主治 `Indication` | 归经、配伍、相关方剂 |
| 证据知识 | 典籍、文档、原文切片 `Evidence` | 出典、引用、支持关系 |

典型推理路径：

```text
症状 -> 证候 -> 治法 -> 方剂 -> 中药 -> 功效 / 归经 -> 典籍与文献证据
```

<a id="architecture"></a>

## 系统架构

<div align="center">
  <img src="docs/assets/architecture.svg" alt="TCM Knowledge Graph Platform architecture" width="100%" />
</div>

| 层级 | 主要技术 | 职责 |
| --- | --- | --- |
| 交互层 | React 18, TypeScript, Vite, Ant Design, Three.js | 问题输入、3D 图谱、答案、证据和运行状态 |
| 服务层 | FastAPI, Pydantic, SQLAlchemy | 问答编排、GraphRAG、文档摄取、图谱发布与任务管理 |
| 数据层 | Neo4j, Qdrant, PostgreSQL, MinIO | 图数据、向量索引、元数据与原始文件存储 |
| 模型层 | OpenAI-compatible APIs | 回答生成、结构化抽取、嵌入、重排与 OCR |

<a id="quick-start"></a>

## 快速开始

### 1. 环境要求

- Docker 24+
- Docker Compose v2
- 可选：一组 OpenAI-compatible 模型 API 凭据

### 2. 启动全部服务

```bash
cp .env.example .env
docker compose up -d
```

### 3. 打开服务

| 服务 | 地址 |
| --- | --- |
| Web 工作台 | <http://localhost:3000> |
| API 健康检查 | <http://localhost:8000/api/health> |
| Neo4j Browser | <http://localhost:7474> |
| Qdrant | <http://localhost:6333> |
| MinIO Console | <http://localhost:9001> |
| PostgreSQL | `localhost:5432` |

### 4. 体验演示问题

```text
失眠可以从哪些证候分析？
柴胡桂枝干姜汤适合什么情况？
党参有什么功效？
```

首次启动会下载容器镜像并安装前后端依赖。即使没有配置真实 LLM Key，项目也会保留可操作的本地演示结果。

### 配置模型

编辑 `.env`：

```env
LLM_BASE_URL=https://your-llm-provider.example/v1
LLM_API_KEY=your-real-api-key
LLM_MODEL=your-model-name

EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
RERANK_MODEL=Qwen/Qwen3-Reranker-8B
OCR_MODEL=deepseek-ai/DeepSeek-OCR
```

重启 API 使配置生效：

```bash
docker compose restart tcm-api
```

<a id="ingestion"></a>

## 知识入库

上游入库链路将原始资料统一转换为文档切片、候选实体、候选关系、证据与可发布图谱：

```text
原始文件
  -> MinIO 对象存储
  -> 文档解析 / OCR / 结构化切片
  -> PostgreSQL 来源、任务、切片与候选知识
  -> 按来源构建 subgraph checkpoint
  -> 合并全局图并生成社区摘要
  -> Neo4j 关系图 + Qdrant 向量索引
  -> 问答系统检索与引用
```

上传并发布一份文档：

```bash
SOURCE_ID=$(
  curl -s http://localhost:8000/api/ingestion/upload \
    -F "file=@./data/import/demo.txt;type=text/plain" \
  | python -c "import json,sys; print(json.load(sys.stdin)['source_id'])"
)

JOB_ID=$(
  curl -s http://localhost:8000/api/ingestion/jobs \
    -H 'Content-Type: application/json' \
    -d "[\"$SOURCE_ID\"]" \
  | python -c "import json,sys; print(json.load(sys.stdin)['job_id'])"
)

curl -s -X POST "http://localhost:8000/api/ingestion/jobs/$JOB_ID/run"
curl -s http://localhost:8000/api/ingestion/publish \
  -H 'Content-Type: application/json' \
  -d "[\"$SOURCE_ID\"]"
```

构建或恢复 GraphRAG 全局图：

```bash
curl -s -X POST http://localhost:8000/api/retrieval/graphrag/build \
  -H 'Content-Type: application/json' \
  -d '{}'
```

<a id="development"></a>

## 开发指南

### 项目结构

```text
.
├── frontend/               # React + TypeScript 图谱工作台
├── backend/                # FastAPI API 与核心服务
│   ├── app/api/            # 问答、图谱、检索和入库 API
│   └── app/services/       # GraphRAG、抽取、发布与存储适配
├── data/                   # 种子数据、导入文件与中间产物
├── scripts/                # 导入、重建、同步与审计脚本
├── docs/                   # 设计文档与 README 视觉素材
├── compose.yml             # 本地全栈编排
└── Makefile                # 测试快捷命令
```

### 测试

```bash
make test
```

也可分别执行：

```bash
make test-backend
make test-frontend
```

### 初始化种子图谱

```bash
python scripts/build_seed_artifacts.py
python scripts/import_seed_graph.py
```

## API 概览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/api/query` | 提交中医问题并获取答案、图谱与证据 |
| `GET` | `/api/graph/overview` | 读取图谱总览 |
| `GET` | `/api/retrieval/status` | 查看检索引擎状态 |
| `POST` | `/api/retrieval/graphrag/build` | 同步构建 GraphRAG 全局图 |
| `POST` | `/api/retrieval/graphrag/build/async` | 创建异步构建任务 |
| `POST` | `/api/ingestion/upload` | 上传知识源文件 |
| `POST` | `/api/ingestion/jobs` | 创建解析与抽取任务 |
| `POST` | `/api/ingestion/publish` | 发布验证后的知识产物 |

API 启动后可访问 <http://localhost:8000/docs> 查看完整 OpenAPI 文档。

<a id="faq"></a>

## FAQ

### 这是一个中医聊天机器人吗？

不只是。项目的主体是中医知识图谱工作台和 GraphRAG 数据管线。问答只是入口，结果会同时展示关联实体、推理路径和可追溯证据。

### 项目与 RAGFlow 是什么关系？

项目实现了 **RAGFlow-compatible retrieval** 和 RAGFlow-style GraphRAG 构建语义，包括查询改写、全文/向量融合、KG 检索、子图 checkpoint、全局图合并与阶段标记；它不是对完整 RAGFlow 服务的直接复制或必需部署。

### 必须使用 OpenAI 吗？

不必。只要供应商提供 OpenAI-compatible API，就可以配置 LLM、Embedding、Rerank 和 OCR 模型。没有配置外部模型时，系统仍可使用内置演示数据。

### 支持哪些中医资料？

当前支持 PDF、DOCX、Markdown、TXT、CSV、JSON 和图片类来源。扫描图片可调用 OCR，文本内容会进入结构化切片、实体关系抽取、图谱发布和向量索引流程。

### 为什么中医问答需要知识图谱？

中医知识常以症状、证候、治法、方剂、药味和典籍出处组成多跳网络。知识图谱可以保留关系方向与路径，帮助检索系统组织比单个相似文本片段更稳定的上下文。

### 可以直接用于医疗诊断吗？

不可以。当前项目用于知识工程、技术研究、资料检索和产品演示，不构成医疗诊断或治疗建议。

## 相关主题与技术关键词

- **中文主题**：中医知识图谱、中医药知识库、中医大模型、中医智能问答、医疗知识图谱、方剂知识图谱、本草知识库、中医文献数字化。
- **English topics**: Traditional Chinese Medicine, TCM knowledge graph, Chinese medicine AI, medical knowledge graph, GraphRAG, retrieval-augmented generation, evidence-grounded question answering.
- **AI 与检索**：RAGFlow-compatible retrieval、hybrid search、vector search、entity resolution、community summary、document ingestion、OCR、LLM、Embedding、Rerank。
- **技术栈**：Neo4j、Qdrant、PostgreSQL、MinIO、FastAPI、React、TypeScript、Three.js、Docker Compose。

## Roadmap

- [x] 图谱驱动的问答与证据展示
- [x] 3D 知识图谱总览与路径高亮
- [x] PDF / DOCX / Markdown / TXT / CSV / JSON 文档摄取
- [x] RAGFlow-compatible 混合检索与 GraphRAG 构建主干
- [x] 子图 checkpoint、全局图合并与社区摘要
- [ ] 知识审核、冲突处理与版本对比界面
- [ ] 更完整的中医术语规范化与同义实体消歧
- [ ] 面向知识库的评测集、可观测性与持续集成

## 贡献

建议在提交修改前先运行 `make test`。新增图谱实体或关系时，请同步提供来源、证据切片与对应测试，保持知识链路可追溯。

---

<div align="center">
  <sub>Built for traceable TCM knowledge exploration and GraphRAG research.</sub>
</div>
