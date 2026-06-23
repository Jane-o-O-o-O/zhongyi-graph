# TCM Knowledge Graph Platform

中医知识图谱智能展示平台。第一版优先完成面向领导演示的主系统：领导可以自由输入问题，系统基于本地知识库、知识图谱和可配置的大模型 API 生成回答，并以图谱路径作为主视觉展开证候、治法、方药和证据来源。

知识摄取系统作为独立上游模块保留接口骨架，不和主展示系统混在一起。

## Local Development

```bash
cp .env.example .env
docker compose up -d
```

Frontend: http://localhost:3000  
API: http://localhost:8000/api/health  
Neo4j: http://localhost:7474

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

## System Boundary

主系统负责演示、问答、图谱路径和证据呈现。上游文档解析和知识入库用于后续用户新增 PDF/Word 等资料，目前保留 ingestion API 和 artifact contract，后续可以独立接入解析服务、RAGFlow 或自建文档处理流水线。
