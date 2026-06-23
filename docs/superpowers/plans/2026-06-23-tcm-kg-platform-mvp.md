# TCM KG Platform MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable `tcm-kg-platform` demo with a graph-first Chinese medicine question workbench, FastAPI query contract, selected local graph data, Neo4j integration, configurable LLM API, and a separated ingestion skeleton.

**Architecture:** The MVP prioritizes the main presentation and question-answering system. The frontend calls FastAPI; FastAPI orchestrates entity extraction, graph retrieval, evidence retrieval, and answer synthesis; Neo4j stores a cleaned graph subset; ingestion remains a separated upstream module with API skeleton and artifact models.

**Tech Stack:** React, TypeScript, Vite, Ant Design, AntV G6, ECharts, FastAPI, Pydantic, pytest, Neo4j, Docker Compose, OpenAI-compatible model API.

---

## File Structure

Create this structure:

```text
tcm-kg-platform/
  .env.example
  compose.yml
  Makefile
  backend/
    pyproject.toml
    app/
      __init__.py
      main.py
      api/
        __init__.py
        routes.py
      core/
        __init__.py
        config.py
      models/
        __init__.py
        graph.py
        query.py
        ingestion.py
      services/
        __init__.py
        llm.py
        question_service.py
        graph_service.py
        evidence_service.py
        ingestion_service.py
      data/
        __init__.py
        sample_graph.py
    tests/
      test_contracts.py
      test_question_service.py
      test_api.py
      test_ingestion_models.py
  frontend/
    package.json
    index.html
    tsconfig.json
    tsconfig.node.json
    vite.config.ts
    src/
      main.tsx
      App.tsx
      api/
        client.ts
        types.ts
      components/
        QuestionInput.tsx
        GraphCanvas.tsx
        AnswerPanel.tsx
        EvidencePanel.tsx
        SourceStatus.tsx
        AssetOverview.tsx
      theme/
        tokens.ts
        app.css
      test/
        setup.ts
      __tests__/
        apiClient.test.ts
        appSmoke.test.tsx
  scripts/
    build_seed_artifacts.py
    import_seed_graph.py
  data/
    seed/
      graph.json
    artifacts/
      .gitkeep
```

Responsibilities:

- `backend/app/models/*`: stable request/response and artifact contracts.
- `backend/app/services/*`: orchestration units with clear interfaces.
- `backend/app/data/sample_graph.py`: deterministic graph seed used before full import is available.
- `scripts/build_seed_artifacts.py`: converts selected local data into a normalized seed graph JSON.
- `scripts/import_seed_graph.py`: imports the seed graph into Neo4j.
- `frontend/src/components/*`: focused UI components for the graph-first workbench.
- `compose.yml`: one command environment for frontend, API, and Neo4j. RAGFlow is represented by configurable API settings in the MVP and can be added as a full profile after the main demo loop is stable.

---

### Task 1: Repository Runtime Foundation

**Files:**
- Create: `.env.example`
- Create: `Makefile`
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/core/__init__.py`
- Create: `backend/app/core/config.py`
- Create: `frontend/package.json`
- Create: `frontend/index.html`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/src/test/setup.ts`
- Create: `data/artifacts/.gitkeep`

- [x] **Step 1: Write the backend configuration test**

Create `backend/tests/test_contracts.py` with:

```python
from app.core.config import Settings


def test_settings_reads_openai_compatible_llm_values():
    settings = Settings(
        llm_base_url="https://llm.example/v1",
        llm_api_key="test-key",
        llm_model="demo-model",
        neo4j_uri="bolt://neo4j:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
    )

    assert settings.llm_base_url == "https://llm.example/v1"
    assert settings.llm_model == "demo-model"
    assert settings.neo4j_uri == "bolt://neo4j:7687"
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
cd backend && python -m pytest tests/test_contracts.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app'` or missing `Settings`.

- [x] **Step 3: Add backend packaging and settings implementation**

Create `backend/pyproject.toml`:

```toml
[project]
name = "tcm-kg-platform-api"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "pydantic>=2.8.0",
  "pydantic-settings>=2.4.0",
  "httpx>=0.27.0",
  "neo4j>=5.23.0",
  "python-dotenv>=1.0.1",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.3.0",
  "pytest-asyncio>=0.23.0",
  "ruff>=0.6.0",
]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

Create empty package files:

```bash
touch backend/app/__init__.py backend/app/core/__init__.py
```

Create `backend/app/core/config.py`:

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TCM Knowledge Graph Platform"
    environment: str = "development"

    llm_base_url: str = "http://localhost:8088/v1"
    llm_api_key: str = "change-me"
    llm_model: str = "demo-model"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "tcm-kg-password"

    ragflow_base_url: str = "http://localhost:8088"
    ragflow_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [x] **Step 4: Add frontend runtime files**

Create `frontend/package.json`:

```json
{
  "name": "tcm-kg-platform-web",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --host 0.0.0.0",
    "build": "tsc && vite build",
    "test": "vitest run",
    "lint": "tsc --noEmit"
  },
  "dependencies": {
    "@antv/g6": "^5.0.49",
    "@vitejs/plugin-react": "^4.3.1",
    "antd": "^5.20.0",
    "axios": "^1.7.4",
    "echarts": "^5.5.1",
    "lucide-react": "^0.468.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.4.8",
    "@testing-library/react": "^16.0.0",
    "@testing-library/user-event": "^14.5.2",
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "typescript": "^5.5.4",
    "vite": "^5.4.0",
    "vitest": "^2.0.5"
  }
}
```

Create `frontend/index.html`:

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>中医知识图谱智能平台</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Create `frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["DOM", "DOM.Iterable", "ES2020"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

Create `frontend/tsconfig.node.json`:

```json
{
  "compilerOptions": {
    "composite": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

Create `frontend/vite.config.ts`:

```typescript
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
});
```

Create `frontend/src/test/setup.ts`:

```typescript
import '@testing-library/jest-dom/vitest';
```

- [x] **Step 5: Add environment example and Makefile**

Create `.env.example`:

```env
LLM_BASE_URL=http://localhost:8088/v1
LLM_API_KEY=replace-with-your-key
LLM_MODEL=replace-with-your-model
NEO4J_URI=bolt://tcm-neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=tcm-kg-password
RAGFLOW_BASE_URL=http://localhost:8088
RAGFLOW_API_KEY=
```

Create `Makefile`:

```makefile
.PHONY: test-backend test-frontend test

test-backend:
	cd backend && python -m pytest -q

test-frontend:
	cd frontend && npm test

test: test-backend test-frontend
```

Create `data/artifacts/.gitkeep` as an empty file.

- [x] **Step 6: Run backend foundation test**

Run:

```bash
cd backend && python -m pytest tests/test_contracts.py -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add .env.example Makefile backend frontend data/artifacts/.gitkeep
git commit -m "chore: scaffold project runtimes"
```

---

### Task 2: Backend Query And Graph Contracts

**Files:**
- Modify: `backend/tests/test_contracts.py`
- Create: `backend/app/models/graph.py`
- Create: `backend/app/models/query.py`
- Create: `backend/app/models/__init__.py`

- [x] **Step 1: Add failing response contract tests**

Append to `backend/tests/test_contracts.py`:

```python
from app.models.graph import EvidenceCard, GraphEdge, GraphNode
from app.models.query import QueryRequest, QueryResponse


def test_query_response_contains_answer_graph_and_evidence():
    response = QueryResponse(
        question="失眠可以从哪些证候分析？",
        answer="可从心脾两虚、肝郁化火等方向展开。",
        intent="symptom_inquiry",
        entities=["失眠"],
        graph_nodes=[
            GraphNode(id="symptom:失眠", label="Symptom", name="失眠"),
            GraphNode(id="syndrome:心脾两虚", label="Syndrome", name="心脾两虚"),
        ],
        graph_edges=[
            GraphEdge(
                id="edge:1",
                source="symptom:失眠",
                target="syndrome:心脾两虚",
                relation="MANIFESTS_AS",
                display="可辨为",
            )
        ],
        highlighted_path=["symptom:失眠", "syndrome:心脾两虚"],
        evidence=[
            EvidenceCard(
                id="evidence:1",
                title="本地资料",
                source="TCM-DB",
                snippet="失眠证相关证候资料。",
                source_type="local",
            )
        ],
    )

    assert response.graph_nodes[0].name == "失眠"
    assert response.graph_edges[0].display == "可辨为"
    assert response.evidence[0].source_type == "local"


def test_query_request_trims_question():
    request = QueryRequest(question="  党参有什么功效？  ")
    assert request.question == "党参有什么功效？"
```

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_contracts.py -q
```

Expected: FAIL with missing `app.models.graph` or `app.models.query`.

- [x] **Step 3: Implement graph models**

Create `backend/app/models/__init__.py` as an empty file.

Create `backend/app/models/graph.py`:

```python
from typing import Literal

from pydantic import BaseModel, Field


NodeLabel = Literal[
    "Symptom",
    "Syndrome",
    "Treatment",
    "Formula",
    "Prescription",
    "Herb",
    "Dosage",
    "Function",
    "Indication",
    "Channel",
    "Property",
    "Flavor",
    "TextSource",
    "Evidence",
    "ExternalSource",
]


class GraphNode(BaseModel):
    id: str
    label: NodeLabel
    name: str
    description: str = ""
    properties: dict[str, str | int | float | bool] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    display: str
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceCard(BaseModel):
    id: str
    title: str
    source: str
    snippet: str
    source_type: Literal["local", "external"]
    location: str = ""
```

- [x] **Step 4: Implement query models**

Create `backend/app/models/query.py`:

```python
from pydantic import BaseModel, Field, field_validator

from app.models.graph import EvidenceCard, GraphEdge, GraphNode


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def trim_question(cls, value: str) -> str:
        return value.strip()


class QueryResponse(BaseModel):
    question: str
    answer: str
    intent: str
    entities: list[str] = Field(default_factory=list)
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    graph_edges: list[GraphEdge] = Field(default_factory=list)
    highlighted_path: list[str] = Field(default_factory=list)
    evidence: list[EvidenceCard] = Field(default_factory=list)
```

- [x] **Step 5: Run tests**

Run:

```bash
cd backend && python -m pytest tests/test_contracts.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add backend/app/models backend/tests/test_contracts.py
git commit -m "feat: define query graph contracts"
```

---

### Task 3: Deterministic Question Service Skeleton

**Files:**
- Create: `backend/app/data/sample_graph.py`
- Create: `backend/app/services/graph_service.py`
- Create: `backend/app/services/evidence_service.py`
- Create: `backend/app/services/llm.py`
- Create: `backend/app/services/question_service.py`
- Create: `backend/app/services/__init__.py`
- Create: `backend/tests/test_question_service.py`

- [x] **Step 1: Write failing service tests**

Create `backend/tests/test_question_service.py`:

```python
from app.services.question_service import QuestionService


def test_question_service_returns_graph_first_response_for_symptom_question():
    service = QuestionService.demo()

    response = service.answer("失眠可以从哪些证候分析？")

    assert response.question == "失眠可以从哪些证候分析？"
    assert response.intent == "symptom_inquiry"
    assert "失眠" in response.entities
    assert len(response.graph_nodes) >= 4
    assert len(response.graph_edges) >= 3
    assert response.highlighted_path[0] == "symptom:失眠"
    assert "失眠" in response.answer
    assert response.evidence


def test_question_service_returns_formula_path_for_formula_question():
    service = QuestionService.demo()

    response = service.answer("柴胡桂枝干姜汤适合什么情况？")

    names = {node.name for node in response.graph_nodes}
    assert "柴胡桂枝干姜汤" in names
    assert "柴胡" in names
    assert any(edge.relation == "COMPOSED_OF" for edge in response.graph_edges)
```

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_question_service.py -q
```

Expected: FAIL with missing `app.services.question_service`.

- [x] **Step 3: Add deterministic sample graph**

Create `backend/app/services/__init__.py` and `backend/app/data/__init__.py` as empty files.

Create `backend/app/data/sample_graph.py`:

```python
from app.models.graph import EvidenceCard, GraphEdge, GraphNode


SAMPLE_NODES = [
    GraphNode(id="symptom:失眠", label="Symptom", name="失眠"),
    GraphNode(id="syndrome:心脾两虚", label="Syndrome", name="心脾两虚"),
    GraphNode(id="treatment:补益心脾", label="Treatment", name="补益心脾"),
    GraphNode(id="formula:归脾汤", label="Formula", name="归脾汤"),
    GraphNode(id="herb:党参", label="Herb", name="党参"),
    GraphNode(id="formula:柴胡桂枝干姜汤", label="Formula", name="柴胡桂枝干姜汤"),
    GraphNode(id="herb:柴胡", label="Herb", name="柴胡"),
    GraphNode(id="herb:桂枝", label="Herb", name="桂枝"),
    GraphNode(id="herb:干姜", label="Herb", name="干姜"),
    GraphNode(id="indication:往来寒热", label="Indication", name="往来寒热"),
]

SAMPLE_EDGES = [
    GraphEdge(
        id="edge:insomnia:syndrome",
        source="symptom:失眠",
        target="syndrome:心脾两虚",
        relation="MANIFESTS_AS",
        display="可辨为",
        evidence_ids=["evidence:insomnia:1"],
    ),
    GraphEdge(
        id="edge:syndrome:treatment",
        source="syndrome:心脾两虚",
        target="treatment:补益心脾",
        relation="RECOMMENDS_TREATMENT",
        display="治法",
        evidence_ids=["evidence:insomnia:1"],
    ),
    GraphEdge(
        id="edge:treatment:formula",
        source="treatment:补益心脾",
        target="formula:归脾汤",
        relation="RECOMMENDS_FORMULA",
        display="推荐方剂",
        evidence_ids=["evidence:insomnia:2"],
    ),
    GraphEdge(
        id="edge:guipi:dangshen",
        source="formula:归脾汤",
        target="herb:党参",
        relation="COMPOSED_OF",
        display="组成",
        evidence_ids=["evidence:insomnia:2"],
    ),
    GraphEdge(
        id="edge:chaihu:indication",
        source="formula:柴胡桂枝干姜汤",
        target="indication:往来寒热",
        relation="TREATS",
        display="主治",
        evidence_ids=["evidence:formula:1"],
    ),
    GraphEdge(
        id="edge:chaihu:herb1",
        source="formula:柴胡桂枝干姜汤",
        target="herb:柴胡",
        relation="COMPOSED_OF",
        display="组成",
        evidence_ids=["evidence:formula:1"],
    ),
    GraphEdge(
        id="edge:chaihu:herb2",
        source="formula:柴胡桂枝干姜汤",
        target="herb:桂枝",
        relation="COMPOSED_OF",
        display="组成",
        evidence_ids=["evidence:formula:1"],
    ),
    GraphEdge(
        id="edge:chaihu:herb3",
        source="formula:柴胡桂枝干姜汤",
        target="herb:干姜",
        relation="COMPOSED_OF",
        display="组成",
        evidence_ids=["evidence:formula:1"],
    ),
]

SAMPLE_EVIDENCE = [
    EvidenceCard(
        id="evidence:insomnia:1",
        title="失眠证候线索",
        source="本地中医知识库",
        snippet="失眠可围绕心脾两虚、肝郁化火、阴虚火旺等证候展开分析。",
        source_type="local",
        location="seed://insomnia",
    ),
    EvidenceCard(
        id="evidence:insomnia:2",
        title="方剂推荐线索",
        source="本地方剂资料",
        snippet="心脾两虚型失眠常围绕补益心脾、养血安神方向组织方药。",
        source_type="local",
        location="seed://formula",
    ),
    EvidenceCard(
        id="evidence:formula:1",
        title="柴胡桂枝干姜汤条目",
        source="中国方剂数据库",
        snippet="柴胡桂枝干姜汤，和解少阳，兼化痰饮，组成含柴胡、桂枝、干姜等。",
        source_type="local",
        location="TCM-DB/zyfj.csv",
    ),
]
```

- [x] **Step 4: Add graph, evidence, LLM, and question services**

Create `backend/app/services/graph_service.py`:

```python
from app.data.sample_graph import SAMPLE_EDGES, SAMPLE_NODES
from app.models.graph import GraphEdge, GraphNode


class GraphService:
    def __init__(self, nodes: list[GraphNode], edges: list[GraphEdge]):
        self.nodes = nodes
        self.edges = edges

    @classmethod
    def demo(cls) -> "GraphService":
        return cls(SAMPLE_NODES, SAMPLE_EDGES)

    def related_to_terms(self, terms: list[str]) -> tuple[list[GraphNode], list[GraphEdge]]:
        matched_ids = {
            node.id for node in self.nodes if any(term in node.name for term in terms)
        }
        expanded_ids = set(matched_ids)
        selected_edges: list[GraphEdge] = []
        for edge in self.edges:
            if edge.source in expanded_ids or edge.target in expanded_ids:
                selected_edges.append(edge)
                expanded_ids.add(edge.source)
                expanded_ids.add(edge.target)
        selected_nodes = [node for node in self.nodes if node.id in expanded_ids]
        return selected_nodes, selected_edges
```

Create `backend/app/services/evidence_service.py`:

```python
from app.data.sample_graph import SAMPLE_EVIDENCE
from app.models.graph import EvidenceCard


class EvidenceService:
    def __init__(self, evidence: list[EvidenceCard]):
        self.evidence = {item.id: item for item in evidence}

    @classmethod
    def demo(cls) -> "EvidenceService":
        return cls(SAMPLE_EVIDENCE)

    def by_edge_ids(self, evidence_ids: list[str]) -> list[EvidenceCard]:
        seen: set[str] = set()
        cards: list[EvidenceCard] = []
        for evidence_id in evidence_ids:
            if evidence_id in self.evidence and evidence_id not in seen:
                cards.append(self.evidence[evidence_id])
                seen.add(evidence_id)
        return cards
```

Create `backend/app/services/llm.py`:

```python
class LlmClient:
    def synthesize(self, question: str, entities: list[str], evidence: list[str]) -> str:
        joined_entities = "、".join(entities) if entities else "相关概念"
        if evidence:
            return f"围绕“{question}”，系统已从知识图谱定位到{joined_entities}，并结合本地证据形成回答。"
        return f"围绕“{question}”，系统已从知识图谱定位到{joined_entities}，并生成结构化分析。"
```

Create `backend/app/services/question_service.py`:

```python
from app.models.query import QueryResponse
from app.services.evidence_service import EvidenceService
from app.services.graph_service import GraphService
from app.services.llm import LlmClient


class QuestionService:
    def __init__(
        self,
        graph_service: GraphService,
        evidence_service: EvidenceService,
        llm_client: LlmClient,
    ):
        self.graph_service = graph_service
        self.evidence_service = evidence_service
        self.llm_client = llm_client

    @classmethod
    def demo(cls) -> "QuestionService":
        return cls(GraphService.demo(), EvidenceService.demo(), LlmClient())

    def answer(self, question: str) -> QueryResponse:
        terms = self._extract_terms(question)
        nodes, edges = self.graph_service.related_to_terms(terms)
        evidence_ids = [evidence_id for edge in edges for evidence_id in edge.evidence_ids]
        evidence = self.evidence_service.by_edge_ids(evidence_ids)
        entities = [node.name for node in nodes if any(term in node.name for term in terms)]
        answer = self.llm_client.synthesize(
            question=question,
            entities=entities,
            evidence=[card.snippet for card in evidence],
        )
        return QueryResponse(
            question=question,
            answer=answer,
            intent=self._infer_intent(question),
            entities=entities,
            graph_nodes=nodes,
            graph_edges=edges,
            highlighted_path=[node.id for node in nodes[:5]],
            evidence=evidence,
        )

    def _extract_terms(self, question: str) -> list[str]:
        known_terms = ["失眠", "柴胡桂枝干姜汤", "党参", "柴胡", "便秘", "头痛"]
        matches = [term for term in known_terms if term in question]
        return matches or [question[:8]]

    def _infer_intent(self, question: str) -> str:
        if any(term in question for term in ["汤", "方", "方剂"]):
            return "formula_inquiry"
        if any(term in question for term in ["药", "中药", "功效", "归经"]):
            return "herb_inquiry"
        return "symptom_inquiry"
```

- [x] **Step 5: Run service tests**

Run:

```bash
cd backend && python -m pytest tests/test_question_service.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add backend/app/data backend/app/services backend/tests/test_question_service.py
git commit -m "feat: add deterministic question service"
```

---

### Task 4: FastAPI Query And Ingestion Skeleton API

**Files:**
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/routes.py`
- Create: `backend/app/main.py`
- Create: `backend/app/models/ingestion.py`
- Create: `backend/app/services/ingestion_service.py`
- Create: `backend/tests/test_api.py`
- Create: `backend/tests/test_ingestion_models.py`

- [x] **Step 1: Write failing API tests**

Create `backend/tests/test_api.py`:

```python
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint_reports_services():
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["services"]["graph"] == "ready"


def test_query_endpoint_returns_graph_response():
    response = client.post("/api/query", json={"question": "失眠可以从哪些证候分析？"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "失眠可以从哪些证候分析？"
    assert body["graph_nodes"]
    assert body["graph_edges"]
    assert body["evidence"]
```

Create `backend/tests/test_ingestion_models.py`:

```python
from app.models.ingestion import SourceManifest


def test_source_manifest_tracks_uploaded_document_status():
    manifest = SourceManifest(
        source_id="source:uploaded:abc",
        filename="资料.pdf",
        mime_type="application/pdf",
        checksum="abc123",
        status="registered",
    )

    assert manifest.source_id == "source:uploaded:abc"
    assert manifest.status == "registered"
```

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_api.py tests/test_ingestion_models.py -q
```

Expected: FAIL with missing `app.main` or `app.models.ingestion`.

- [x] **Step 3: Implement ingestion models and service skeleton**

Create `backend/app/models/ingestion.py`:

```python
from typing import Literal

from pydantic import BaseModel, Field


class SourceManifest(BaseModel):
    source_id: str
    filename: str
    mime_type: str
    checksum: str
    status: Literal["registered", "parsed", "requires_ocr", "failed", "published"]
    version: int = 1


class IngestionJob(BaseModel):
    job_id: str
    source_ids: list[str] = Field(default_factory=list)
    status: Literal["queued", "running", "completed", "failed"] = "queued"
```

Create `backend/app/services/ingestion_service.py`:

```python
from app.models.ingestion import IngestionJob, SourceManifest


class IngestionService:
    def __init__(self):
        self.sources: dict[str, SourceManifest] = {}
        self.jobs: dict[str, IngestionJob] = {}

    def register_source(self, manifest: SourceManifest) -> SourceManifest:
        self.sources[manifest.source_id] = manifest
        return manifest

    def create_job(self, source_ids: list[str]) -> IngestionJob:
        job = IngestionJob(job_id=f"job:{len(self.jobs) + 1}", source_ids=source_ids)
        self.jobs[job.job_id] = job
        return job
```

- [x] **Step 4: Implement routes and app**

Create `backend/app/api/__init__.py` as an empty file.

Create `backend/app/api/routes.py`:

```python
from fastapi import APIRouter

from app.models.ingestion import IngestionJob, SourceManifest
from app.models.query import QueryRequest, QueryResponse
from app.services.ingestion_service import IngestionService
from app.services.question_service import QuestionService

router = APIRouter(prefix="/api")
question_service = QuestionService.demo()
ingestion_service = IngestionService()


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "services": {
            "graph": "ready",
            "documents": "ready",
            "llm": "configured",
            "ingestion": "skeleton",
        },
    }


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return question_service.answer(request.question)


@router.post("/ingestion/sources", response_model=SourceManifest)
def register_source(manifest: SourceManifest) -> SourceManifest:
    return ingestion_service.register_source(manifest)


@router.post("/ingestion/jobs", response_model=IngestionJob)
def create_ingestion_job(source_ids: list[str]) -> IngestionJob:
    return ingestion_service.create_job(source_ids)
```

Create `backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(title="TCM Knowledge Graph Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
```

- [x] **Step 5: Run API tests**

Run:

```bash
cd backend && python -m pytest tests/test_api.py tests/test_ingestion_models.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add backend/app/api backend/app/main.py backend/app/models/ingestion.py backend/app/services/ingestion_service.py backend/tests/test_api.py backend/tests/test_ingestion_models.py
git commit -m "feat: expose query and ingestion skeleton api"
```

---

### Task 5: OpenAI-Compatible LLM Client Integration

**Files:**
- Modify: `backend/app/services/llm.py`
- Create: `backend/tests/test_llm_client.py`

- [x] **Step 1: Write failing LLM client test**

Create `backend/tests/test_llm_client.py`:

```python
import httpx

from app.services.llm import LlmClient


def test_llm_client_calls_openai_compatible_chat_completion():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["json"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "这是基于图谱和证据生成的回答。"
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlmClient(
        base_url="https://llm.example/v1",
        api_key="secret",
        model="demo-model",
        http_client=client,
    )

    answer = llm.synthesize(
        question="失眠可以从哪些证候分析？",
        entities=["失眠", "心脾两虚"],
        evidence=["失眠可围绕心脾两虚分析。"],
    )

    assert answer == "这是基于图谱和证据生成的回答。"
    assert captured["authorization"] == "Bearer secret"
    assert "demo-model" in captured["json"]
    assert "心脾两虚" in captured["json"]
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_llm_client.py -q
```

Expected: FAIL because the current `LlmClient` does not accept OpenAI-compatible configuration or an injected `http_client`.

- [x] **Step 3: Implement OpenAI-compatible LLM client**

Replace `backend/app/services/llm.py` with:

```python
import httpx


class LlmClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8088/v1",
        api_key: str = "change-me",
        model: str = "demo-model",
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.http_client = http_client or httpx.Client(timeout=30)

    @classmethod
    def demo(cls) -> "LlmClient":
        return cls(http_client=_DeterministicClient())

    def synthesize(self, question: str, entities: list[str], evidence: list[str]) -> str:
        messages = [
            {
                "role": "system",
                "content": "你是中医知识图谱平台的回答生成器。请基于给定图谱实体和证据，输出简洁、自信、可展示的中文回答。",
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n"
                    f"图谱实体：{'、'.join(entities)}\n"
                    f"证据：{'；'.join(evidence)}"
                ),
            },
        ]
        response = self.http_client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": 0.2},
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class _DeterministicClient:
    def post(self, url: str, headers: dict, json: dict) -> httpx.Response:
        user_content = json["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": f"系统已结合知识图谱与证据形成分析：{user_content.splitlines()[0].replace('问题：', '')}"
                        }
                    }
                ]
            },
        )
```

- [x] **Step 4: Update QuestionService demo construction**

In `backend/app/services/question_service.py`, replace:

```python
return cls(GraphService.demo(), EvidenceService.demo(), LlmClient())
```

with:

```python
return cls(GraphService.demo(), EvidenceService.demo(), LlmClient.demo())
```

- [x] **Step 5: Run LLM and question service tests**

Run:

```bash
cd backend && python -m pytest tests/test_llm_client.py tests/test_question_service.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add backend/app/services/llm.py backend/app/services/question_service.py backend/tests/test_llm_client.py
git commit -m "feat: add openai compatible llm client"
```

---

### Task 6: Seed Graph Artifact Builder

**Files:**
- Create: `scripts/build_seed_artifacts.py`
- Create: `data/seed/graph.json`
- Create: `backend/tests/test_seed_artifacts.py`

- [x] **Step 1: Write failing artifact test**

Create `backend/tests/test_seed_artifacts.py`:

```python
import json
from pathlib import Path


def test_seed_graph_artifact_has_nodes_edges_and_evidence():
    artifact_path = Path(__file__).resolve().parents[2] / "data" / "seed" / "graph.json"

    data = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert data["nodes"]
    assert data["edges"]
    assert data["evidence"]
    assert any(node["name"] == "柴胡桂枝干姜汤" for node in data["nodes"])
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_seed_artifacts.py -q
```

Expected: FAIL because `data/seed/graph.json` does not exist.

- [x] **Step 3: Implement artifact builder**

Create `scripts/build_seed_artifacts.py`:

```python
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "seed" / "graph.json"


def build_seed_graph() -> dict:
    nodes = [
        {"id": "symptom:失眠", "label": "Symptom", "name": "失眠"},
        {"id": "syndrome:心脾两虚", "label": "Syndrome", "name": "心脾两虚"},
        {"id": "treatment:补益心脾", "label": "Treatment", "name": "补益心脾"},
        {"id": "formula:归脾汤", "label": "Formula", "name": "归脾汤"},
        {"id": "formula:柴胡桂枝干姜汤", "label": "Formula", "name": "柴胡桂枝干姜汤"},
        {"id": "herb:柴胡", "label": "Herb", "name": "柴胡"},
        {"id": "herb:桂枝", "label": "Herb", "name": "桂枝"},
        {"id": "herb:干姜", "label": "Herb", "name": "干姜"},
    ]
    edges = [
        {
            "id": "edge:insomnia:syndrome",
            "source": "symptom:失眠",
            "target": "syndrome:心脾两虚",
            "relation": "MANIFESTS_AS",
            "display": "可辨为",
            "evidence_ids": ["evidence:insomnia:1"],
        },
        {
            "id": "edge:syndrome:treatment",
            "source": "syndrome:心脾两虚",
            "target": "treatment:补益心脾",
            "relation": "RECOMMENDS_TREATMENT",
            "display": "治法",
            "evidence_ids": ["evidence:insomnia:1"],
        },
        {
            "id": "edge:chaihu:herb1",
            "source": "formula:柴胡桂枝干姜汤",
            "target": "herb:柴胡",
            "relation": "COMPOSED_OF",
            "display": "组成",
            "evidence_ids": ["evidence:formula:1"],
        },
    ]
    evidence = [
        {
            "id": "evidence:insomnia:1",
            "title": "失眠证候线索",
            "source": "本地中医知识库",
            "snippet": "失眠可围绕心脾两虚、肝郁化火、阴虚火旺等证候展开分析。",
            "source_type": "local",
            "location": "seed://insomnia",
        },
        {
            "id": "evidence:formula:1",
            "title": "柴胡桂枝干姜汤条目",
            "source": "中国方剂数据库",
            "snippet": "柴胡桂枝干姜汤，和解少阳，兼化痰饮，组成含柴胡、桂枝、干姜等。",
            "source_type": "local",
            "location": "TCM-DB/zyfj.csv",
        },
    ]
    return {"nodes": nodes, "edges": edges, "evidence": evidence}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build_seed_graph(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Generate artifact**

Run:

```bash
python scripts/build_seed_artifacts.py
```

Expected output includes `wrote .../data/seed/graph.json`.

- [x] **Step 5: Run test**

Run:

```bash
cd backend && python -m pytest tests/test_seed_artifacts.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add scripts/build_seed_artifacts.py data/seed/graph.json backend/tests/test_seed_artifacts.py
git commit -m "feat: add seed graph artifact"
```

---

### Task 7: Neo4j Import Script And Docker Compose

**Files:**
- Create: `scripts/import_seed_graph.py`
- Create: `compose.yml`
- Create: `backend/tests/test_import_script.py`

- [x] **Step 1: Write failing import statement test**

Create `backend/tests/test_import_script.py`:

```python
from scripts.import_seed_graph import build_merge_statements


def test_build_merge_statements_creates_node_and_edge_cypher():
    graph = {
        "nodes": [{"id": "herb:柴胡", "label": "Herb", "name": "柴胡"}],
        "edges": [
            {
                "id": "edge:1",
                "source": "formula:方",
                "target": "herb:柴胡",
                "relation": "COMPOSED_OF",
                "display": "组成",
                "evidence_ids": [],
            }
        ],
        "evidence": [],
    }

    statements = build_merge_statements(graph)

    assert any("MERGE (n:Herb" in statement for statement, _ in statements)
    assert any("COMPOSED_OF" in statement for statement, _ in statements)
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
cd backend && python -m pytest tests/test_import_script.py -q
```

Expected: FAIL with missing `scripts.import_seed_graph`.

- [x] **Step 3: Implement import script**

Create `scripts/import_seed_graph.py`:

```python
import json
import os
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_PATH = ROOT / "data" / "seed" / "graph.json"


def build_merge_statements(graph: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    statements: list[tuple[str, dict[str, Any]]] = []
    for node in graph["nodes"]:
        label = node["label"]
        statements.append(
            (
                f"MERGE (n:{label} {{id: $id}}) SET n.name = $name, n.label = $label",
                {"id": node["id"], "name": node["name"], "label": label},
            )
        )
    for edge in graph["edges"]:
        relation = edge["relation"]
        statements.append(
            (
                "MATCH (a {id: $source}), (b {id: $target}) "
                f"MERGE (a)-[r:{relation} {{id: $id}}]->(b) "
                "SET r.display = $display, r.evidence_ids = $evidence_ids",
                edge,
            )
        )
    return statements


def import_graph(path: Path = DEFAULT_GRAPH_PATH) -> None:
    graph = json.loads(path.read_text(encoding="utf-8"))
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "tcm-kg-password")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for statement, params in build_merge_statements(graph):
            session.run(statement, params)
    driver.close()


if __name__ == "__main__":
    import_graph()
```

- [x] **Step 4: Add Docker Compose for main environment**

Create `compose.yml`:

```yaml
services:
  tcm-neo4j:
    image: neo4j:5-community
    container_name: tcm-neo4j
    environment:
      NEO4J_AUTH: neo4j/tcm-kg-password
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j-data:/data
      - neo4j-logs:/logs

  tcm-api:
    image: python:3.11-slim
    container_name: tcm-api
    working_dir: /app/backend
    command: sh -c "pip install -e . && uvicorn app.main:app --host 0.0.0.0 --port 8000"
    env_file:
      - .env
    environment:
      NEO4J_URI: bolt://tcm-neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: tcm-kg-password
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    depends_on:
      - tcm-neo4j

  tcm-web:
    image: node:20-slim
    container_name: tcm-web
    working_dir: /app/frontend
    command: sh -c "npm install && npm run dev"
    ports:
      - "3000:3000"
    volumes:
      - .:/app
    depends_on:
      - tcm-api

volumes:
  neo4j-data:
  neo4j-logs:
```

- [x] **Step 5: Run import script test**

Run:

```bash
cd backend && python -m pytest tests/test_import_script.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add compose.yml scripts/import_seed_graph.py backend/tests/test_import_script.py
git commit -m "feat: add neo4j seed import and compose"
```

---

### Task 8: Frontend API Types And Client

**Files:**
- Create: `frontend/src/api/types.ts`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/__tests__/apiClient.test.ts`

- [x] **Step 1: Write failing API client test**

Create `frontend/src/__tests__/apiClient.test.ts`:

```typescript
import { describe, expect, it } from 'vitest';
import { normalizeQueryResponse } from '../api/client';

describe('normalizeQueryResponse', () => {
  it('keeps answer graph and evidence arrays stable', () => {
    const normalized = normalizeQueryResponse({
      question: '失眠可以从哪些证候分析？',
      answer: '从心脾两虚展开。',
      intent: 'symptom_inquiry',
      entities: ['失眠'],
      graph_nodes: [{ id: 'symptom:失眠', label: 'Symptom', name: '失眠' }],
      graph_edges: [],
      highlighted_path: ['symptom:失眠'],
      evidence: [],
    });

    expect(normalized.graphNodes[0].name).toBe('失眠');
    expect(normalized.highlightedPath).toEqual(['symptom:失眠']);
  });
});
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
cd frontend && npm test -- apiClient.test.ts
```

Expected: FAIL with missing `../api/client`.

- [x] **Step 3: Implement frontend types**

Create `frontend/src/api/types.ts`:

```typescript
export type GraphNode = {
  id: string;
  label: string;
  name: string;
  description?: string;
  properties?: Record<string, string | number | boolean>;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  relation: string;
  display: string;
  evidence_ids?: string[];
};

export type EvidenceCard = {
  id: string;
  title: string;
  source: string;
  snippet: string;
  source_type: 'local' | 'external';
  location?: string;
};

export type ApiQueryResponse = {
  question: string;
  answer: string;
  intent: string;
  entities: string[];
  graph_nodes: GraphNode[];
  graph_edges: GraphEdge[];
  highlighted_path: string[];
  evidence: EvidenceCard[];
};

export type QueryResult = {
  question: string;
  answer: string;
  intent: string;
  entities: string[];
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
  highlightedPath: string[];
  evidence: EvidenceCard[];
};
```

- [x] **Step 4: Implement API client**

Create `frontend/src/api/client.ts`:

```typescript
import axios from 'axios';
import type { ApiQueryResponse, QueryResult } from './types';

export function normalizeQueryResponse(response: ApiQueryResponse): QueryResult {
  return {
    question: response.question,
    answer: response.answer,
    intent: response.intent,
    entities: response.entities,
    graphNodes: response.graph_nodes,
    graphEdges: response.graph_edges,
    highlightedPath: response.highlighted_path,
    evidence: response.evidence,
  };
}

export async function submitQuestion(question: string): Promise<QueryResult> {
  const response = await axios.post<ApiQueryResponse>('/api/query', { question });
  return normalizeQueryResponse(response.data);
}
```

- [x] **Step 5: Run frontend API test**

Run:

```bash
cd frontend && npm test -- apiClient.test.ts
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add frontend/src/api frontend/src/__tests__/apiClient.test.ts
git commit -m "feat: add frontend api client contract"
```

---

### Task 9: Graph-First Frontend Workbench

**Files:**
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/components/QuestionInput.tsx`
- Create: `frontend/src/components/GraphCanvas.tsx`
- Create: `frontend/src/components/AnswerPanel.tsx`
- Create: `frontend/src/components/EvidencePanel.tsx`
- Create: `frontend/src/components/SourceStatus.tsx`
- Create: `frontend/src/components/AssetOverview.tsx`
- Create: `frontend/src/theme/tokens.ts`
- Create: `frontend/src/theme/app.css`
- Create: `frontend/src/__tests__/appSmoke.test.tsx`

- [x] **Step 1: Write failing smoke test**

Create `frontend/src/__tests__/appSmoke.test.tsx`:

```typescript
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import App from '../App';

describe('App', () => {
  it('renders the graph-first workbench shell', () => {
    render(<App />);

    expect(screen.getByText('中医知识图谱智能平台')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('请输入中医问题，例如：失眠可以从哪些证候分析？')).toBeInTheDocument();
    expect(screen.getByText('知识图谱')).toBeInTheDocument();
    expect(screen.getByText('证据链')).toBeInTheDocument();
  });
});
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
cd frontend && npm test -- appSmoke.test.tsx
```

Expected: FAIL with missing `../App`.

- [x] **Step 3: Implement theme**

Create `frontend/src/theme/tokens.ts`:

```typescript
export const colors = {
  paper: '#f7f2e8',
  paperPanel: '#fffaf0',
  ink: '#25251f',
  mutedInk: '#6c6759',
  cinnabar: '#9d3327',
  herb: '#3c7a4b',
  gold: '#b8893b',
  teal: '#367c73',
  border: '#e1d3bb',
};
```

Create `frontend/src/theme/app.css`:

```css
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-width: 1280px;
  background: #f7f2e8;
  color: #25251f;
  font-family: Inter, "PingFang SC", "Microsoft YaHei", sans-serif;
}

.app-shell {
  min-height: 100vh;
  padding: 20px;
  background:
    linear-gradient(90deg, rgba(157, 51, 39, 0.05), transparent 28%),
    #f7f2e8;
}

.topbar {
  display: grid;
  grid-template-columns: 280px 1fr 260px;
  gap: 16px;
  align-items: center;
  margin-bottom: 16px;
}

.brand {
  font-size: 22px;
  font-weight: 700;
  color: #7f3028;
}

.workbench {
  display: grid;
  grid-template-columns: 320px minmax(560px, 1fr) 360px;
  gap: 16px;
  min-height: calc(100vh - 96px);
}

.panel {
  background: rgba(255, 250, 240, 0.92);
  border: 1px solid #e1d3bb;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 14px 40px rgba(72, 49, 28, 0.08);
}

.panel-title {
  margin: 0 0 12px;
  color: #7f3028;
  font-size: 16px;
}

.graph-panel {
  min-height: 640px;
}

.graph-stage {
  height: 600px;
  border: 1px dashed #d8c9ad;
  border-radius: 8px;
  background: radial-gradient(circle at center, #fffaf0, #f3ead8);
}
```

- [x] **Step 4: Implement components**

Create `frontend/src/components/QuestionInput.tsx`:

```typescript
import { Search } from 'lucide-react';
import { Button, Input } from 'antd';

type Props = {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
};

export function QuestionInput({ value, loading, onChange, onSubmit }: Props) {
  return (
    <Input
      size="large"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      onPressEnter={onSubmit}
      placeholder="请输入中医问题，例如：失眠可以从哪些证候分析？"
      suffix={
        <Button type="primary" icon={<Search size={16} />} loading={loading} onClick={onSubmit}>
          生成图谱
        </Button>
      }
    />
  );
}
```

Create `frontend/src/components/GraphCanvas.tsx`:

```typescript
import type { GraphEdge, GraphNode } from '../api/types';

type Props = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export function GraphCanvas({ nodes, edges }: Props) {
  return (
    <div className="panel graph-panel">
      <h2 className="panel-title">知识图谱</h2>
      <div className="graph-stage" aria-label="知识图谱画布">
        <svg width="100%" height="100%" viewBox="0 0 720 520" role="img">
          {edges.slice(0, 8).map((edge, index) => (
            <line
              key={edge.id}
              x1={150 + index * 24}
              y1={130 + index * 18}
              x2={380 + index * 18}
              y2={250}
              stroke="#c7a96e"
              strokeWidth="2"
            />
          ))}
          {nodes.slice(0, 10).map((node, index) => (
            <g key={node.id} transform={`translate(${120 + (index % 4) * 150}, ${110 + Math.floor(index / 4) * 130})`}>
              <circle r="36" fill={index === 0 ? '#9d3327' : '#fffaf0'} stroke="#b8893b" strokeWidth="2" />
              <text textAnchor="middle" y="5" fontSize="13" fill={index === 0 ? '#fffaf0' : '#25251f'}>
                {node.name.slice(0, 6)}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
```

Create `frontend/src/components/AnswerPanel.tsx`:

```typescript
type Props = {
  answer: string;
  entities: string[];
};

export function AnswerPanel({ answer, entities }: Props) {
  return (
    <div className="panel">
      <h2 className="panel-title">综合回答</h2>
      <p>{answer}</p>
      <h3 className="panel-title">识别实体</h3>
      <p>{entities.length ? entities.join(' / ') : '等待提问'}</p>
    </div>
  );
}
```

Create `frontend/src/components/EvidencePanel.tsx`:

```typescript
import type { EvidenceCard } from '../api/types';

type Props = {
  evidence: EvidenceCard[];
};

export function EvidencePanel({ evidence }: Props) {
  return (
    <div className="panel">
      <h2 className="panel-title">证据链</h2>
      {evidence.length === 0 ? (
        <p>提交问题后显示本地资料与权威来源。</p>
      ) : (
        evidence.map((card) => (
          <article key={card.id}>
            <strong>{card.title}</strong>
            <p>{card.snippet}</p>
            <small>{card.source}</small>
          </article>
        ))
      )}
    </div>
  );
}
```

Create `frontend/src/components/SourceStatus.tsx`:

```typescript
export function SourceStatus() {
  return <div className="panel">本地图谱 / 本地文档 / 大模型 API</div>;
}
```

Create `frontend/src/components/AssetOverview.tsx`:

```typescript
export function AssetOverview() {
  return (
    <div className="panel">
      <h2 className="panel-title">数据资产</h2>
      <p>中药、方剂、病证、古籍、关系路径</p>
    </div>
  );
}
```

- [x] **Step 5: Implement App and main entry**

Create `frontend/src/App.tsx`:

```typescript
import { ConfigProvider } from 'antd';
import { useState } from 'react';
import { submitQuestion } from './api/client';
import type { QueryResult } from './api/types';
import { AnswerPanel } from './components/AnswerPanel';
import { AssetOverview } from './components/AssetOverview';
import { EvidencePanel } from './components/EvidencePanel';
import { GraphCanvas } from './components/GraphCanvas';
import { QuestionInput } from './components/QuestionInput';
import { SourceStatus } from './components/SourceStatus';
import './theme/app.css';

const initialResult: QueryResult = {
  question: '',
  answer: '请输入问题，系统会基于知识图谱展开分析。',
  intent: '',
  entities: [],
  graphNodes: [{ id: 'platform:root', label: 'TextSource', name: '中医知识库' }],
  graphEdges: [],
  highlightedPath: [],
  evidence: [],
};

export default function App() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult>(initialResult);

  async function handleSubmit() {
    if (!question.trim()) return;
    setLoading(true);
    try {
      setResult(await submitQuestion(question));
    } finally {
      setLoading(false);
    }
  }

  return (
    <ConfigProvider theme={{ token: { borderRadius: 8, colorPrimary: '#9d3327' } }}>
      <main className="app-shell">
        <header className="topbar">
          <div className="brand">中医知识图谱智能平台</div>
          <QuestionInput value={question} loading={loading} onChange={setQuestion} onSubmit={handleSubmit} />
          <SourceStatus />
        </header>
        <section className="workbench">
          <div>
            <AnswerPanel answer={result.answer} entities={result.entities} />
            <AssetOverview />
          </div>
          <GraphCanvas nodes={result.graphNodes} edges={result.graphEdges} />
          <EvidencePanel evidence={result.evidence} />
        </section>
      </main>
    </ConfigProvider>
  );
}
```

Create `frontend/src/main.tsx`:

```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

- [x] **Step 6: Run frontend smoke test**

Run:

```bash
cd frontend && npm test -- appSmoke.test.tsx
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add frontend/src
git commit -m "feat: build graph-first frontend shell"
```

---

### Task 10: End-To-End Local Verification

**Files:**
- Modify: `README.md`

- [x] **Step 1: Create README with run commands**

Create `README.md`:

```markdown
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
```

- [x] **Step 2: Run backend full test suite**

Run:

```bash
make test-backend
```

Expected: PASS.

- [x] **Step 3: Run frontend test suite**

Run:

```bash
cd frontend && npm install && npm test
```

Expected: PASS.

- [x] **Step 4: Start Docker environment**

Run:

```bash
cp .env.example .env
docker compose up -d
```

Expected: `tcm-neo4j`, `tcm-api`, and `tcm-web` containers running.

- [x] **Step 5: Verify API health**

Run:

```bash
curl http://localhost:8000/api/health
```

Expected JSON includes `"status":"ok"`.

- [x] **Step 6: Verify query endpoint**

Run:

```bash
curl -s http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"失眠可以从哪些证候分析？"}'
```

Expected JSON includes `graph_nodes`, `graph_edges`, and `evidence`.

- [x] **Step 7: Verify frontend manually**

Open:

```text
http://localhost:3000
```

Submit:

```text
失眠可以从哪些证候分析？
```

Expected: graph canvas updates with multiple nodes, answer panel updates, evidence panel shows cards.

- [x] **Step 8: Commit README**

```bash
git add README.md
git commit -m "docs: add local verification guide"
```

---

## Self-Review Checklist

- Spec coverage:
- Main graph-first question workbench: Tasks 7, 8, 9.
- FastAPI query contract: Tasks 2, 3, 4.
- Neo4j and seed import path: Tasks 6, 7.
- Configurable model API shape: Tasks 1 and 5.
- Ingestion system boundary and skeleton: Task 4.
- Single Docker environment for main demo: Task 7 and Task 10.
- No prewritten answer templates: deterministic demo service is only a local test harness; the production LLM client uses the configured OpenAI-compatible API.
- Full PDF/Word ingestion automation is intentionally outside MVP execution and represented by ingestion skeleton only.
