import json

import httpx
import pytest

from app.services.llm import LlmClient


def test_llm_client_calls_openai_compatible_chat_completion():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["method"] = request.method
        captured["url"] = str(request.url)
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

    payload = json.loads(captured["json"])

    assert answer == "这是基于图谱和证据生成的回答。"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert payload["model"] == "demo-model"
    assert payload["temperature"] == 0.2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert "心脾两虚" in payload["messages"][1]["content"]


def test_llm_client_requires_markdown_answer_structure():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "### 综合结论\n- 已按 Markdown 输出。"
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

    llm.synthesize(
        question="失眠可以从哪些证候分析？",
        entities=["失眠", "心脾两虚"],
        evidence=["失眠可围绕心脾两虚分析。"],
    )

    payload = json.loads(captured["json"])
    system_prompt = payload["messages"][0]["content"]

    assert "Markdown" in system_prompt
    assert "### 综合结论" in system_prompt
    assert "### 图谱路径" in system_prompt
    assert "### 证候要点" in system_prompt
    assert "### 展开建议" in system_prompt
    assert "不要输出代码块" in system_prompt


def test_llm_client_raises_for_error_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "upstream failed"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlmClient(
        base_url="https://llm.example/v1",
        api_key="secret",
        model="demo-model",
        http_client=client,
    )

    with pytest.raises(httpx.HTTPStatusError):
        llm.synthesize(
            question="失眠可以从哪些证候分析？",
            entities=["失眠"],
            evidence=["失眠可围绕心脾两虚分析。"],
        )
