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


def test_structured_extraction_client_query_prompt_requests_expanded_entities():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode("utf-8"))
        captured["system"] = payload["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"entities":["头痛","肝"],'
                                '"expanded_entities":["肝阳上亢","肝火上炎","头风"],'
                                '"relations":["相关"]}'
                            )
                        }
                    }
                ]
            },
        )

    client = StructuredExtractionClient(
        base_url="https://api.siliconflow.cn/v1",
        api_key="secret",
        model="nclusionAI/Ling-flash-2.0",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    extracted = client.extract_query("头痛和肝有什么关联？")

    assert extracted["expanded_entities"] == ["肝阳上亢", "肝火上炎", "头风"]
    assert "expanded_entities" in captured["system"]
    assert "衍生" in captured["system"]


def test_structured_extraction_client_extracts_ragflow_query_rewrite_with_type_pool():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode("utf-8"))
        captured["system"] = payload["messages"][0]["content"]
        captured["user"] = payload["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer_type_keywords": ["Formula"],
                                    "entities_from_query": ["失眠", "归脾汤"],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    client = StructuredExtractionClient(
        base_url="https://api.siliconflow.cn/v1",
        api_key="secret",
        model="nclusionAI/Ling-flash-2.0",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    extracted = client.extract_query(
        "睡不着用什么方？",
        type_pool={"Formula": ["归脾汤"], "Syndrome": ["心脾两虚"]},
    )

    assert extracted["answer_type_keywords"] == ["Formula"]
    assert extracted["entities_from_query"] == ["失眠", "归脾汤"]
    assert "answer_type_keywords" in captured["system"]
    assert "entities_from_query" in captured["system"]
    assert '"Formula": ["归脾汤"]' in captured["user"]


def test_structured_extraction_client_extracts_units_in_one_batch_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode("utf-8"))
        captured["json"] = json.dumps(payload)
        captured["user"] = payload["messages"][1]["content"]
        captured["system"] = payload["messages"][0]["content"]
        captured["model"] = payload["model"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "unit_id": "unit:1",
                                            "entities": [
                                                {
                                                    "name": "头痛",
                                                    "label": "Symptom",
                                                    "confidence": 0.9,
                                                }
                                            ],
                                            "relations": [],
                                        },
                                        {
                                            "unit_id": "unit:2",
                                            "entities": [
                                                {
                                                    "name": "肝阳上亢",
                                                    "label": "Syndrome",
                                                    "confidence": 0.88,
                                                }
                                            ],
                                            "relations": [],
                                        },
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    client = StructuredExtractionClient(
        base_url="https://api.siliconflow.cn/v1",
        api_key="secret",
        model="Qwen/Qwen3-8B",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.extract_chunks_batch(
        [
            {"unit_id": "unit:1", "text": "头痛多属于阳。"},
            {"unit_id": "unit:2", "text": "肝阳上亢可见头痛眩晕。"},
        ]
    )

    assert captured["model"] == "Qwen/Qwen3-8B"
    assert json.loads(captured["json"])["enable_thinking"] is False
    assert json.loads(captured["json"])["max_tokens"] == 4096
    assert json.loads(captured["json"])["response_format"] == {"type": "json_object"}
    assert captured["system"].startswith("/no_think")
    assert captured["user"].startswith("/no_think")
    assert captured["user"].count("unit_id") >= 2
    assert result["items"][0]["unit_id"] == "unit:1"
    assert result["items"][1]["entities"][0]["name"] == "肝阳上亢"


def test_structured_extraction_client_normalizes_single_item_batch_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "unit_id": "unit:1",
                                    "entities": [
                                        {
                                            "name": "头痛",
                                            "label": "Symptom",
                                            "confidence": 0.9,
                                        }
                                    ],
                                    "relations": [],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    client = StructuredExtractionClient(
        base_url="https://api.siliconflow.cn/v1",
        api_key="secret",
        model="Qwen/Qwen3-8B",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.extract_chunks_batch(
        [
            {"unit_id": "unit:1", "text": "头痛多属于阳。"},
            {"unit_id": "unit:2", "text": "肝阳上亢可见头痛眩晕。"},
        ]
    )

    assert result == {
        "items": [
            {
                "unit_id": "unit:1",
                "entities": [{"name": "头痛", "label": "Symptom", "confidence": 0.9}],
                "relations": [],
            }
        ]
    }


def test_demo_structured_extraction_client_extracts_each_batch_unit():
    result = StructuredExtractionClient.demo().extract_chunks_batch(
        [
            {"unit_id": "unit:1", "text": "失眠可辨为心脾两虚。"},
            {"unit_id": "unit:2", "text": "归脾汤由党参等药组成。"},
        ]
    )

    assert [item["unit_id"] for item in result["items"]] == ["unit:1", "unit:2"]
    first_item = result["items"][0]
    second_item = result["items"][1]
    assert {entity["name"] for entity in first_item["entities"]} == {"失眠", "心脾两虚"}
    assert first_item["relations"][0]["relation"] == "MANIFESTS_AS"
    assert {entity["name"] for entity in second_item["entities"]} == {"归脾汤", "党参"}
    assert second_item["relations"][0]["relation"] == "COMPOSED_OF"
