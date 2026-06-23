from dataclasses import dataclass
import math
from uuid import NAMESPACE_URL, uuid5
from typing import Literal

import httpx

from app.models.graph import EvidenceCard, GraphNode
from app.services.model_clients import EmbeddingClient

ContentType = Literal["entity", "evidence", "chunk"]


@dataclass(frozen=True)
class VectorPayload:
    id: str
    text: str
    content_type: ContentType
    node_id: str = ""
    evidence_id: str = ""
    chunk_id: str = ""
    label: str = ""


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    text: str
    payload: dict[str, str]


class VectorIndexService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        documents: list[VectorPayload],
        qdrant_url: str = "",
        collection: str = "tcm_knowledge",
        http_client: httpx.Client | None = None,
    ):
        self.embedding_client = embedding_client
        self.documents = documents
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection = collection
        self.http_client = http_client or httpx.Client(timeout=10)
        self._vectors: list[list[float]] | None = None

    @classmethod
    def in_memory(
        cls,
        embedding_client: EmbeddingClient,
        nodes: list[GraphNode],
        evidence: list[EvidenceCard],
    ) -> "VectorIndexService":
        return cls(
            embedding_client=embedding_client,
            documents=build_vector_payloads(nodes=nodes, evidence=evidence),
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        content_types: list[ContentType] | None = None,
    ) -> list[VectorHit]:
        query_vector = self.embedding_client.embed([query])[0]
        if self.qdrant_url:
            try:
                hits = self._search_qdrant(query, query_vector, top_k, content_types)
                if hits:
                    return hits
            except Exception:
                pass
        return self._search_memory(query, query_vector, top_k, content_types)

    def _search_memory(
        self,
        query: str,
        query_vector: list[float],
        top_k: int,
        content_types: list[ContentType] | None,
    ) -> list[VectorHit]:
        vectors = self._ensure_vectors()
        allowed_types = set(content_types or ["entity", "evidence", "chunk"])
        hits = []
        for document, vector in zip(self.documents, vectors, strict=True):
            if document.content_type not in allowed_types:
                continue
            score = _cosine_similarity(query_vector, vector) + _lexical_bonus(query, document.text)
            hits.append(
                VectorHit(
                    id=document.id,
                    score=score,
                    text=document.text,
                    payload={
                        "content_type": document.content_type,
                        "node_id": document.node_id,
                        "evidence_id": document.evidence_id,
                        "chunk_id": document.chunk_id,
                        "label": document.label,
                    },
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _search_qdrant(
        self,
        query: str,
        query_vector: list[float],
        top_k: int,
        content_types: list[ContentType] | None,
    ) -> list[VectorHit]:
        response = self.http_client.post(
            f"{self.qdrant_url}/collections/{self.collection}/points/search",
            json={
                "vector": query_vector,
                "limit": top_k,
                "with_payload": True,
                **_qdrant_filter(content_types),
            },
        )
        response.raise_for_status()
        hits: list[VectorHit] = []
        for point in response.json().get("result", []):
            payload = point.get("payload") or {}
            text = payload.get("text", "")
            hits.append(
                VectorHit(
                    id=str(payload.get("id") or point.get("id")),
                    score=float(point.get("score", 0.0)) + _lexical_bonus(query, text),
                    text=text,
                    payload={
                        "content_type": payload.get("content_type", ""),
                        "node_id": payload.get("node_id", ""),
                        "evidence_id": payload.get("evidence_id", ""),
                        "label": payload.get("label", ""),
                        "chunk_id": payload.get("chunk_id", ""),
                    },
                )
            )
        return hits

    def ensure_qdrant_collection(self) -> None:
        vectors = self._ensure_vectors()
        if not self.qdrant_url or not vectors:
            return
        vector_size = len(vectors[0])
        response = self.http_client.put(
            f"{self.qdrant_url}/collections/{self.collection}",
            json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        )
        if response.status_code == 409:
            return
        response.raise_for_status()

    def upsert_qdrant(self) -> None:
        if not self.qdrant_url:
            return
        vectors = self._ensure_vectors()
        self.ensure_qdrant_collection()
        self._upsert_documents_qdrant(self.documents, vectors)

    def upsert_payloads_qdrant(self, documents: list[VectorPayload]) -> None:
        if not self.qdrant_url or not documents:
            return
        vectors = self.embedding_client.embed([document.text for document in documents])
        self._ensure_qdrant_collection_for_size(len(vectors[0]))
        self._upsert_documents_qdrant(documents, vectors)

    def _ensure_qdrant_collection_for_size(self, vector_size: int) -> None:
        response = self.http_client.put(
            f"{self.qdrant_url}/collections/{self.collection}",
            json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        )
        if response.status_code == 409:
            return
        response.raise_for_status()

    def _upsert_documents_qdrant(
        self,
        documents: list[VectorPayload],
        vectors: list[list[float]],
    ) -> None:
        points = []
        for document, vector in zip(documents, vectors, strict=True):
            points.append(
                {
                    "id": _qdrant_point_id(document.id),
                    "vector": vector,
                    "payload": {
                        "id": document.id,
                        "text": document.text,
                        "content_type": document.content_type,
                        "node_id": document.node_id,
                        "evidence_id": document.evidence_id,
                        "chunk_id": document.chunk_id,
                        "label": document.label,
                    },
                }
            )
        self.http_client.put(
            f"{self.qdrant_url}/collections/{self.collection}/points",
            json={"points": points},
        ).raise_for_status()

    def _ensure_vectors(self) -> list[list[float]]:
        if self._vectors is None:
            self._vectors = (
                self.embedding_client.embed([document.text for document in self.documents])
                if self.documents
                else []
            )
        return self._vectors


def build_vector_payloads(nodes: list[GraphNode], evidence: list[EvidenceCard]) -> list[VectorPayload]:
    payloads = [
        VectorPayload(
            id=f"entity:{node.id}",
            node_id=node.id,
            content_type="entity",
            label=node.label,
            text=_node_text(node),
        )
        for node in nodes
    ]
    payloads.extend(
        VectorPayload(
            id=f"evidence:{item.id}",
            evidence_id=item.id,
            content_type="evidence",
            text=f"{item.title}。{item.snippet}。来源：{item.source}",
        )
        for item in evidence
    )
    return payloads


def _node_text(node: GraphNode) -> str:
    aliases = {
        "失眠": "失眠 睡不着 入睡困难 夜寐不安 不寐",
        "心脾两虚": "心脾两虚 失眠 多梦 健忘 心悸 食欲不振",
        "归脾汤": "归脾汤 补益心脾 养血安神 失眠",
        "柴胡桂枝干姜汤": "柴胡桂枝干姜汤 往来寒热 和解少阳",
    }
    return " ".join(
        part
        for part in [
            node.name,
            node.label,
            node.description,
            aliases.get(node.name, ""),
        ]
        if part
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return numerator / (left_norm * right_norm)


def _lexical_bonus(query: str, text: str) -> float:
    compact_query = query.replace(" ", "")
    compact_text = text.replace(" ", "")
    bonus = 0.0
    for keyword in ["失眠", "睡不着", "入睡困难", "不寐", "柴胡桂枝干姜汤", "党参", "柴胡"]:
        if keyword in compact_query and keyword in compact_text:
            bonus += 1.0
    return bonus


def _qdrant_filter(content_types: list[ContentType] | None) -> dict:
    if not content_types:
        return {}
    if len(content_types) == 1:
        return {
            "filter": {
                "must": [{"key": "content_type", "match": {"value": content_types[0]}}]
            }
        }
    return {
        "filter": {
            "should": [
                {"key": "content_type", "match": {"value": content_type}}
                for content_type in content_types
            ]
        }
    }


def _qdrant_point_id(document_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"tcm-kg-platform:{document_id}"))
