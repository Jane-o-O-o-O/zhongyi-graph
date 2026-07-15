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

    def synthesize(
        self,
        question: str,
        entities: list[str],
        evidence: list[str],
        graph_paths: list[str] | None = None,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是中医知识图谱平台的综合研判生成器。"
                    "必须只基于给定图谱实体和证据生成回答，语气自信、适合领导演示。"
                    "图谱路径优先于普通证据；如果普通证据与图谱路径侧重点不同，回答必须跟随图谱路径。"
                    "输出必须是 Markdown，且严格使用以下结构：\n"
                    "### 图谱路径\n"
                    "- 用 `症状 -> 证候 -> 治法 -> 方药` 形式描述图谱主路径。\n"
                    "### 证候要点\n"
                    "- 列出 2-4 条关键证候、治法或方药依据，关键词用 **加粗**。\n"
                    "### 综合结论\n"
                    "- 综合结论必须放在第三部分，形成完整连贯的研判文字，且不少于 200 个汉字。"
                    "结论必须归纳前述图谱路径和证候依据，不得使用无依据的扩展内容。\n"
                    "不要输出展开建议，不要输出代码块，不要输出 HTML，不要输出免责声明，"
                    "不要输出上述三个标题以外的一级结构。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n"
                    f"图谱实体：{'、'.join(entities)}\n"
                    f"图谱路径：{'；'.join(graph_paths or [])}\n"
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
        question = user_content.splitlines()[0].replace("问题：", "")
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "### 图谱路径\n"
                                "- 症状 -> 证候 -> 治法 -> 方药，优先呈现当前问题对应的高置信路径。\n\n"
                                "### 证候要点\n"
                                "- **图谱实体**：围绕已识别实体展开关系检索。\n"
                                "- **证据来源**：以本地知识库证据卡片作为回答依据。\n\n"
                                "### 综合结论\n"
                                f"本次研判围绕“{question}”展开，系统首先依据本地知识图谱识别问题中的核心实体，"
                                "再沿实体之间已经建立的关系检索相关证候、治法、方剂与中药信息，并将图谱路径与可追溯证据进行对应。"
                                "当前结果强调，页面展示的判断并非脱离资料的自由推测，而是对已检索节点、关系方向和证据内容的综合整理。"
                                "在阅读结果时，应重点关注图谱中由核心问题出发形成的连续关系，以及各条关系所连接的证候依据和治疗方向。"
                                "若不同资料存在侧重点差异，系统以当前图谱中连接明确、证据能够相互支持的路径作为主要研判依据，"
                                "并通过实体名称、关系类型和证据来源共同限定结论范围。由此形成的综合判断能够较完整地说明当前问题在知识图谱中的关联位置、"
                                "主要辨证方向及其对应依据，同时保留对原始资料的可追溯性，便于对展示结果进行核查和进一步分析。"
                            )
                        }
                    }
                ]
            },
        )
