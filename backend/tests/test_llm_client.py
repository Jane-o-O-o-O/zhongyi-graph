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
