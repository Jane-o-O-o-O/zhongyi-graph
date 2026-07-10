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
                "/no_think 你是中医知识图谱实体关系抽取器。"
                "从给定文本中抽取明确出现的实体和关系，只输出 JSON。"
                "只要文本中出现症状、证候、治法、方剂、中药，就必须抽取。"
                "关系方向遵循：症状->证候 用 MANIFESTS_AS；证候->治法 用 RECOMMENDS_TREATMENT；"
                "治法->方剂 用 RECOMMENDS_FORMULA；方剂->中药 用 COMPOSED_OF；方剂或治法->症状/证候 用 TREATS。"
                "凡出现症状、证候、病机、病名、治法、方剂、中药、舌象、脉象、诊法、体征、功效、主治，均要抽取。"
                "实体 label 只能使用 Symptom、Syndrome、Treatment、Formula、Herb、Indication、Function。"
                "舌象、脉象、诊法、体征归为 Indication；病机、病因、病名、证候归为 Syndrome；功效归为 Function。"
                "关系 relation 使用英文大写枚举，例如 MANIFESTS_AS、RECOMMENDS_TREATMENT、"
                "RECOMMENDS_FORMULA、COMPOSED_OF、TREATS、RELATED_TO。"
                "输出格式：{\"entities\":[{\"name\":\"\",\"label\":\"Symptom\",\"confidence\":0.9}],"
                "\"relations\":[{\"source\":\"\",\"target\":\"\",\"relation\":\"RELATED_TO\","
                "\"display\":\"相关\",\"confidence\":0.8}]}"
            ),
            user=f"/no_think 文本：{focused_text}",
        )

    def extract_chunks_batch(self, items: Sequence[dict]) -> dict:
        normalized_items = [
            {
                "unit_id": str(item.get("unit_id", "")).strip(),
                "text": str(item.get("text", "")).strip(),
            }
            for item in items
            if str(item.get("unit_id", "")).strip() and str(item.get("text", "")).strip()
        ]
        payload = self._chat_json(
            system=(
                "/no_think 你是中医知识图谱实体关系批量抽取器。"
                "输入是多个独立的 extraction unit，每个都有 unit_id 和 text。"
                "必须分别抽取每个 unit 内明确出现的实体和关系，只输出 JSON。"
                "无论输入有几个 unit，顶层 JSON 都必须只有 items 字段，items 必须是数组。"
                "禁止把 unit_id、entities、relations 直接放在顶层。"
                "不要跨 unit 建关系，不要把一个 unit 的实体放到另一个 unit。"
                "凡出现症状、证候、病机、病名、治法、方剂、中药、舌象、脉象、诊法、体征、功效、主治，均要抽取。"
                "每个非空 unit 尽量抽取 3-12 个最核心实体；若文本是总论、诊法、分类、理论，也要抽取核心概念和分类名。"
                "如果关系不明显，可以 relations 为空，但 entities 不应为空。"
                "实体 label 只能使用 Symptom、Syndrome、Treatment、Formula、Herb、Indication、Function。"
                "舌象、脉象、诊法、体征归为 Indication；病机、病因、病名、证候归为 Syndrome；功效归为 Function。"
                "关系 relation 使用 MANIFESTS_AS、RECOMMENDS_TREATMENT、RECOMMENDS_FORMULA、"
                "COMPOSED_OF、TREATS、RELATED_TO。"
                "关系方向遵循：症状->证候，证候->治法，治法->方剂，方剂->中药，"
                "方剂或治法->症状/证候。"
                "输出格式必须为：{\"items\":[{\"unit_id\":\"unit:...\","
                "\"entities\":[{\"name\":\"\",\"label\":\"Symptom\",\"confidence\":0.9}],"
                "\"relations\":[{\"source\":\"\",\"target\":\"\",\"relation\":\"RELATED_TO\","
                "\"display\":\"相关\",\"confidence\":0.8}]}]}。"
                "如果某个 unit 没有可抽取内容，也要返回该 unit_id，entities 和 relations 为空数组。"
            ),
            user="/no_think 批量文本：\n" + json.dumps(normalized_items, ensure_ascii=False),
        )
        return _normalize_batch_extraction_payload(payload)

    def extract_query(self, question: str) -> dict:
        return self._chat_json(
            system=(
                "/no_think 你是中医知识图谱查询理解器。"
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
            user=f"/no_think 问题：{question}",
        )

    def resolve_entity_pairs(
        self,
        *,
        entity_type: str,
        pairs: Sequence[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        if not pairs:
            return []
        payload = self._chat_json(
            system=(
                "/no_think 你是中医知识图谱实体消歧器。"
                "判断每个候选实体对是否表示同一个实体，只输出 JSON。"
                "必须聚焦实体的关键医学含义，忽略别名、简称、繁简、前后缀等噪声。"
                "如果两个名称只是相关但不是同一个实体，same 必须为 false。"
                "输出格式：{\"pairs\":[{\"source\":\"实体A\",\"target\":\"实体B\",\"same\":true}]}"
            ),
            user=(
                "/no_think 实体类型："
                f"{entity_type}\n候选实体对："
                + json.dumps(
                    [
                        {"source": source, "target": target}
                        for source, target in pairs
                    ],
                    ensure_ascii=False,
                )
            ),
        )
        return _selected_resolution_pairs(payload, pairs)

    def _chat_json(self, *, system: str, user: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 4096,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
        }
        response = self.http_client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
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
        normalized_user = user.removeprefix("/no_think").strip()
        if normalized_user.startswith("问题："):
            question = normalized_user.removeprefix("问题：")
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
            text = normalized_user.removeprefix("文本：")
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


def _normalize_batch_extraction_payload(payload: dict) -> dict:
    items = payload.get("items")
    if isinstance(items, list):
        return {"items": [item for item in items if isinstance(item, dict)]}
    if "unit_id" in payload and ("entities" in payload or "relations" in payload):
        return {"items": [payload]}
    return {"items": []}


def _selected_resolution_pairs(
    payload: dict,
    requested_pairs: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    requested = {tuple(pair): tuple(pair) for pair in requested_pairs}
    requested.update({(target, source): (source, target) for source, target in requested_pairs})
    selected: list[tuple[str, str]] = []
    items = payload.get("pairs", [])
    if not isinstance(items, list):
        return selected
    for item in items:
        if not isinstance(item, dict):
            continue
        same_value = item.get("same", item.get("same_entity", item.get("is_same", False)))
        if not _truthy_resolution_value(same_value):
            continue
        source = str(item.get("source", item.get("entity_a", item.get("left", "")))).strip()
        target = str(item.get("target", item.get("entity_b", item.get("right", "")))).strip()
        pair = requested.get((source, target))
        if pair and pair not in selected:
            selected.append(pair)
    return selected


def _truthy_resolution_value(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "same", "1", "是", "同一"}
    return False


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
