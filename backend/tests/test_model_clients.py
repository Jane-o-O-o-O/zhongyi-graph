import json

import httpx

from app.services.model_clients import EmbeddingClient, RerankClient, StructuredExtractionClient


def test_embedding_client_calls_openai_compatible_embeddings_endpoint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["url"] = str(request.url)
        captured["json"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ]
            },
        )

    client = EmbeddingClient(
        base_url="https://api.siliconflow.cn/v1",
        api_key="secret",
        model="Qwen/Qwen3-Embedding-8B",
        dimensions=1024,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    vectors = client.embed(["失眠", "心脾两虚"])
    payload = json.loads(captured["json"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert captured["url"] == "https://api.siliconflow.cn/v1/embeddings"
    assert captured["authorization"] == "Bearer secret"
    assert payload["model"] == "Qwen/Qwen3-Embedding-8B"
    assert payload["input"] == ["失眠", "心脾两虚"]
    assert payload["dimensions"] == 1024


def test_rerank_client_calls_siliconflow_rerank_endpoint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["url"] = str(request.url)
        captured["json"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.72},
                ]
            },
        )

    client = RerankClient(
        base_url="https://api.siliconflow.cn/v1",
        api_key="secret",
        model="Qwen/Qwen3-Reranker-8B",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    ranked = client.rerank("失眠怎么辨证？", ["失眠 -> 心脾两虚", "失眠 -> 肝郁化火"])
    payload = json.loads(captured["json"])

    assert ranked == [(1, 0.91), (0, 0.72)]
    assert captured["url"] == "https://api.siliconflow.cn/v1/rerank"
    assert captured["authorization"] == "Bearer secret"
    assert payload["model"] == "Qwen/Qwen3-Reranker-8B"
    assert payload["query"] == "失眠怎么辨证？"
    assert payload["documents"] == ["失眠 -> 心脾两虚", "失眠 -> 肝郁化火"]


def test_structured_extraction_client_extracts_relevant_window_around_query_hint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode("utf-8"))
        captured["user"] = payload["messages"][1]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"entities":[],"relations":[]}'}}]},
        )

    client = StructuredExtractionClient(
        base_url="https://api.siliconflow.cn/v1",
        api_key="secret",
        model="nclusionAI/Ling-flash-2.0",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    client.extract_chunk("甲" * 2500 + "头痛偏左，法当平肝潜阳。" + "乙" * 2500, hints=["头痛"])

    user_prompt = captured["user"]
    assert "头痛偏左" in user_prompt
    assert len(user_prompt) < 1800
