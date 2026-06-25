from collections.abc import Sequence
import hashlib
import json
import math

import httpx


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int = 1024,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.http_client = http_client or httpx.Client(timeout=30)

    @classmethod
    def demo(cls, dimensions: int = 64) -> "EmbeddingClient":
        return cls(
            base_url="http://localhost:8088/v1",
            api_key="demo",
            model="demo-embedding",
            dimensions=dimensions,
            http_client=_DeterministicEmbeddingHttpClient(dimensions),
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.http_client.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": list(texts),
                "dimensions": self.dimensions,
            },
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]


class RerankClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.http_client = http_client or httpx.Client(timeout=30)

    @classmethod
    def demo(cls) -> "RerankClient":
        return cls(
            base_url="http://localhost:8088/v1",
            api_key="demo",
            model="demo-rerank",
            http_client=_DeterministicRerankHttpClient(),
        )

    def rerank(self, query: str, documents: Sequence[str]) -> list[tuple[int, float]]:
        if not documents:
            return []
        response = self.http_client.post(
            f"{self.base_url}/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": list(documents),
            },
        )
        response.raise_for_status()
        data = response.json()
        return [
            (int(item["index"]), float(item["relevance_score"]))
            for item in data.get("results", [])
        ]


class StructuredExtractionClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.http_client = http_client or httpx.Client(timeout=60)

    @classmethod
    def demo(cls) -> "StructuredExtractionClient":
        return cls(
            base_url="http://localhost:8088/v1",
            api_key="demo",
            model="demo-extraction",
            http_client=_DeterministicExtractionHttpClient(),
        )

    def extract_chunk(self, text: str, hints: Sequence[str] | None = None) -> dict:
        focused_text = _focus_text_window(text=text, hints=hints or [])
        return self._chat_json(
            system=(
                "你是中医知识图谱实体关系抽取器。"
                "从给定文本中抽取明确出现的实体和关系，只输出 JSON。"
                "只要文本中出现症状、证候、治法、方剂、中药，就必须抽取。"
                "关系方向遵循：症状->证候 用 MANIFESTS_AS；证候->治法 用 RECOMMENDS_TREATMENT；"
                "治法->方剂 用 RECOMMENDS_FORMULA；方剂->中药 用 COMPOSED_OF；方剂或治法->症状/证候 用 TREATS。"
                "实体 label 只能使用 Symptom、Syndrome、Treatment、Formula、Herb、Indication、Function。"
                "关系 relation 使用英文大写枚举，例如 MANIFESTS_AS、RECOMMENDS_TREATMENT、"
                "RECOMMENDS_FORMULA、COMPOSED_OF、TREATS、RELATED_TO。"
                "输出格式：{\"entities\":[{\"name\":\"\",\"label\":\"Symptom\",\"confidence\":0.9}],"
                "\"relations\":[{\"source\":\"\",\"target\":\"\",\"relation\":\"RELATED_TO\","
                "\"display\":\"相关\",\"confidence\":0.8}]}"
            ),
            user=f"文本：{focused_text}",
        )

    def extract_query(self, question: str) -> dict:
        return self._chat_json(
            system=(
                "你是中医知识图谱查询理解器。"
                "从用户问题中抽取适合图谱检索的原始实体、衍生相关实体和关系意图，只输出 JSON。"
                "entities 必须保留用户明示实体；expanded_entities 输出同义词、标准中医术语、相关脏腑、"
                "常见相关证候或病名，用于扩大关键词检索和向量检索。"
                "例如用户问“头痛和肝有什么关联”，entities 可为 [\"头痛\",\"肝\"]，"
                "expanded_entities 可为 [\"肝阳上亢\",\"肝火上炎\",\"头风\",\"头痛偏左\"]。"
                "输出格式：{\"entities\":[\"头痛\",\"肝\"],"
                "\"expanded_entities\":[\"肝阳上亢\",\"肝火上炎\"],"
                "\"relations\":[\"相关\",\"证候\"]}。"
                "实体应保留用户原词或标准中医术语，不要编造具体结论。"
            ),
            user=f"问题：{question}",
        )

    def _chat_json(self, *, system: str, user: str) -> dict:
        response = self.http_client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _loads_json_object(content)


class _DeterministicEmbeddingHttpClient:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def post(self, url: str, headers: dict, json: dict) -> httpx.Response:
        vectors = [_hash_embedding(text, self.dimensions) for text in json["input"]]
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"data": [{"embedding": vector} for vector in vectors]},
        )


class _DeterministicRerankHttpClient:
    def post(self, url: str, headers: dict, json: dict) -> httpx.Response:
        query_terms = set(_tokenize_for_score(json["query"]))
        results = []
        for index, document in enumerate(json["documents"]):
            document_terms = set(_tokenize_for_score(document))
            overlap = len(query_terms & document_terms)
            score = overlap + min(len(document) / 500, 0.25)
            results.append({"index": index, "relevance_score": score})
        results.sort(key=lambda item: item["relevance_score"], reverse=True)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"results": results},
        )


class _DeterministicExtractionHttpClient:
    def post(self, url: str, headers: dict, json: dict) -> httpx.Response:
        user = json["messages"][1]["content"]
        if user.startswith("问题："):
            question = user.removeprefix("问题：")
            entities = [
                term
                for term in [
                    "柴胡桂枝干姜汤",
                    "心脾两虚",
                    "补益心脾",
                    "归脾汤",
                    "睡不着",
                    "失眠",
                    "不寐",
                    "党参",
                    "柴胡",
                    "桂枝",
                    "干姜",
                    "便秘",
                    "头痛",
                    "肝",
                    "偏头疼",
                    "眩晕",
                ]
                if term in question
            ]
            if not entities:
                entities = [question.strip("？?。！!，,；;")]
            expanded_entities = []
            if "头痛" in entities and "肝" in entities:
                expanded_entities = ["肝阳上亢", "肝火上炎", "头风", "头痛偏左"]
            payload = {
                "entities": entities,
                "expanded_entities": expanded_entities,
                "relations": [],
            }
        else:
            text = user.removeprefix("文本：")
            entities = []
            for name, label in [
                ("头痛", "Symptom"),
                ("发热头痛", "Symptom"),
                ("不寐", "Symptom"),
                ("失眠", "Symptom"),
                ("心脾两虚", "Syndrome"),
                ("补益心脾", "Treatment"),
                ("归脾汤", "Formula"),
                ("党参", "Herb"),
            ]:
                if name in text:
                    entities.append({"name": name, "label": label, "confidence": 0.8})
            relations = []
            names = {entity["name"] for entity in entities}
            if "头痛" in names and "心脾两虚" in names:
                relations.append(
                    {
                        "source": "头痛",
                        "target": "心脾两虚",
                        "relation": "MANIFESTS_AS",
                        "display": "可辨为",
                        "confidence": 0.78,
                    }
                )
            if "失眠" in names and "心脾两虚" in names:
                relations.append(
                    {
                        "source": "失眠",
                        "target": "心脾两虚",
                        "relation": "MANIFESTS_AS",
                        "display": "可辨为",
                        "confidence": 0.78,
                    }
                )
            if "不寐" in names and "心脾两虚" in names:
                relations.append(
                    {
                        "source": "不寐",
                        "target": "心脾两虚",
                        "relation": "MANIFESTS_AS",
                        "display": "可辨为",
                        "confidence": 0.78,
                    }
                )
            if "心脾两虚" in names and "补益心脾" in names:
                relations.append(
                    {
                        "source": "心脾两虚",
                        "target": "补益心脾",
                        "relation": "RECOMMENDS_TREATMENT",
                        "display": "治法",
                        "confidence": 0.78,
                    }
                )
            if "补益心脾" in names and "归脾汤" in names:
                relations.append(
                    {
                        "source": "补益心脾",
                        "target": "归脾汤",
                        "relation": "RECOMMENDS_FORMULA",
                        "display": "推荐方剂",
                        "confidence": 0.78,
                    }
                )
            if "心脾两虚" in names and "归脾汤" in names:
                relations.append(
                    {
                        "source": "心脾两虚",
                        "target": "归脾汤",
                        "relation": "RECOMMENDS_FORMULA",
                        "display": "推荐方剂",
                        "confidence": 0.78,
                    }
                )
            if "归脾汤" in names and "党参" in names:
                relations.append(
                    {
                        "source": "归脾汤",
                        "target": "党参",
                        "relation": "COMPOSED_OF",
                        "display": "组成",
                        "confidence": 0.78,
                    }
                )
            payload = {"entities": entities, "relations": relations}
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": json_module_dumps(payload)}}]},
        )


def _hash_embedding(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokenize_for_score(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _tokenize_for_score(text: str) -> list[str]:
    compact = "".join(ch for ch in text.lower() if not ch.isspace())
    tokens = [compact[index : index + 2] for index in range(max(len(compact) - 1, 0))]
    tokens.extend(compact[index : index + 1] for index in range(len(compact)))
    return [token for token in tokens if token]


def _loads_json_object(content: str) -> dict:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    data = json.loads(stripped)
    return data if isinstance(data, dict) else {}


def _focus_text_window(text: str, hints: Sequence[str], window: int = 700) -> str:
    compact = text.strip()
    positions = [
        compact.find(hint)
        for hint in hints
        if hint and compact.find(hint) >= 0
    ]
    if not positions:
        return compact[:3000]

    center = min(positions)
    start = max(0, center - window)
    end = min(len(compact), center + window)
    return compact[start:end]


def json_module_dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)
