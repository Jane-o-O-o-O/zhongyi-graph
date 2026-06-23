from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "TCM Knowledge Graph Platform"
    environment: str = "development"

    llm_base_url: str = "http://localhost:8088/v1"
    llm_api_key: str = "change-me"
    llm_model: str = "demo-model"
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    rerank_model: str = "Qwen/Qwen3-Reranker-8B"
    ocr_model: str = "deepseek-ai/DeepSeek-OCR"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "tcm-kg-password"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "tcm_knowledge"

    postgres_dsn: str = "postgresql+psycopg://tcm:tcm@localhost:5432/tcm_kg"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "tcm-documents"
    minio_secure: bool = False

    ragflow_base_url: str = "http://localhost:8088"
    ragflow_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
