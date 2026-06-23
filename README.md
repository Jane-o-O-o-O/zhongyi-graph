# TCM Knowledge Graph Platform

中医知识图谱智能展示平台。第一版优先完成面向领导演示的主系统：领导可以自由输入问题，系统基于本地知识库、知识图谱和可配置的大模型 API 生成回答，并以图谱路径作为主视觉展开证候、治法、方药和证据来源。

知识摄取系统作为独立上游模块存在：负责文件上传、文档解析、切片持久化、候选实体/关系抽取、发布到图谱和向量索引。

## Local Development

```bash
cp .env.example .env
docker compose up -d
```

Frontend: http://localhost:3000  
API: http://localhost:8000/api/health  
Neo4j: http://localhost:7474
Qdrant: http://localhost:6333
PostgreSQL: localhost:5432
MinIO: http://localhost:9001

## Demo Flow

1. 打开 http://localhost:3000
2. 在顶部输入框提交演示问题，例如：

```text
失眠可以从哪些证候分析？
柴胡桂枝干姜汤适合什么情况？
党参有什么功效？
```

3. 中间区域以知识图谱为主展示实体路径，左右两侧展示回答摘要、证据卡片和本地资源状态。

## LLM Configuration

默认 `.env.example` 使用占位 key，系统会走本地稳定演示模式，方便没有外部 API 时直接展示。

接入真实 OpenAI-compatible 大模型 API 时，编辑 `.env`：

```env
LLM_BASE_URL=https://your-llm-provider.example/v1
LLM_API_KEY=your-real-api-key
LLM_MODEL=your-model-name
```

然后重启 API：

```bash
docker compose restart tcm-api
```

## Tests

```bash
make test-backend
make test-frontend
```

## Seed Graph

```bash
python scripts/build_seed_artifacts.py
python scripts/import_seed_graph.py
```

## GraphRAG Ingestion Flow

上传一个文档并发布到当前问答系统：

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

完整存储链路：

```text
原始文件 -> MinIO
文档元数据/页面/切片/候选实体关系 -> PostgreSQL
切片/实体/证据向量 -> Qdrant
发布后的实体关系 -> 当前图谱服务，后续可扩展持久写 Neo4j
问答 -> 图谱召回 + 向量召回 + PostgreSQL 切片证据
```

OCR 使用硅基流动模型：

```env
OCR_MODEL=deepseek-ai/DeepSeek-OCR
```

## System Boundary

主系统负责演示、问答、图谱路径和证据呈现。上游文档解析和知识入库负责把 PDF、Word、CSV、TXT、图片等资料转成稳定的 chunks、候选实体/关系、证据和图谱发布结果。
