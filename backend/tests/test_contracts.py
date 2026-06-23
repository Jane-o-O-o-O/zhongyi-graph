import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.models.graph import EvidenceCard, GraphEdge, GraphNode
from app.models.query import QueryRequest, QueryResponse


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_settings_env_file_points_to_repo_root():
    env_file = Settings.model_config["env_file"]
    assert Path(env_file).resolve() == REPO_ROOT / ".env"


def test_vite_html_entrypoint_exists():
    html = (REPO_ROOT / "frontend/index.html").read_text(encoding="utf-8")
    assert 'src="/src/main.tsx"' in html
    assert (REPO_ROOT / "frontend/src/main.tsx").is_file()


def test_frontend_test_environment_declares_jsdom():
    package = json.loads((REPO_ROOT / "frontend/package.json").read_text(encoding="utf-8"))
    assert "jsdom" in package["devDependencies"]


def test_vite_react_plugin_is_dev_dependency():
    package = json.loads((REPO_ROOT / "frontend/package.json").read_text(encoding="utf-8"))
    assert "@vitejs/plugin-react" not in package["dependencies"]
    assert "@vitejs/plugin-react" in package["devDependencies"]


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


def test_query_request_rejects_blank_question_after_trim():
    with pytest.raises(ValidationError):
        QueryRequest(question="   ")


@pytest.mark.parametrize("raw_question", [None, 123, [], {}])
def test_query_request_rejects_non_string_question(raw_question):
    with pytest.raises(ValidationError):
        QueryRequest(question=raw_question)
