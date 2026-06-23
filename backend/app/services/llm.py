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
            request=httpx.Request("POST", url),
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
