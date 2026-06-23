# TCM Knowledge Graph Platform

中医知识图谱智能展示平台。第一版优先主展示/问答系统，知识摄取系统作为独立上游模块保留接口骨架。

## Local Development

```bash
cp .env.example .env
docker compose up -d
```

Frontend: http://localhost:3000
API: http://localhost:8000/api/health
Neo4j: http://localhost:7474

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
