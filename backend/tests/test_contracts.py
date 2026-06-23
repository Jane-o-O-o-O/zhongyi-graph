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
