import json
from pathlib import Path

from app.core.config import Settings


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
