import json

import httpx

from app.data.sample_graph import SAMPLE_EVIDENCE, SAMPLE_NODES
from app.services.model_clients import EmbeddingClient
from app.services.vector_service import VectorIndexService, VectorPayload


class StaticEmbeddingClient:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _text in texts]


def test_vector_search_queries_qdrant_when_configured():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.read().decode("utf-8"))
        return httpx.Response(
            200,
            request=request,
            json={
                "result": [
                    {
                        "id": 1,
                        "score": 0.91,
                        "payload": {
                            "id": "entity:symptom:失眠",
                            "text": "失眠 Symptom 睡不着 入睡困难 不寐",
                            "content_type": "entity",
                            "node_id": "symptom:失眠",
                            "evidence_id": "",
                            "label": "Symptom",
                        },
                    }
                ]
            },
        )

    vector_index = VectorIndexService(
        embedding_client=StaticEmbeddingClient(),
        documents=[],
        qdrant_url="http://qdrant:6333",
        collection="tcm_knowledge",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    hits = vector_index.search("睡不着怎么辨证", top_k=3, content_types=["entity"])

    assert captured["url"] == "http://qdrant:6333/collections/tcm_knowledge/points/search"
    assert captured["json"]["vector"] == [1.0, 0.0, 0.0]
    assert captured["json"]["limit"] == 3
    assert captured["json"]["filter"] == {
        "must": [{"key": "content_type", "match": {"value": "entity"}}]
    }
    assert hits[0].payload["node_id"] == "symptom:失眠"
    assert hits[0].score > 0.91


def test_vector_search_falls_back_to_memory_when_qdrant_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request, json={"status": "not ready"})

    vector_index = VectorIndexService(
        embedding_client=EmbeddingClient.demo(),
        documents=[],
        qdrant_url="http://qdrant:6333",
        collection="tcm_knowledge",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    vector_index.documents = vector_index.in_memory(
        embedding_client=EmbeddingClient.demo(),
        nodes=SAMPLE_NODES,
        evidence=SAMPLE_EVIDENCE,
    ).documents

    hits = vector_index.search("睡不着怎么辨证", top_k=3, content_types=["entity"])

    assert hits
    assert hits[0].payload["node_id"] == "symptom:失眠"


def test_vector_upsert_continues_when_qdrant_collection_already_exists():
    requested_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if request.method == "PUT" and str(request.url).endswith("/collections/tcm_knowledge"):
            return httpx.Response(409, request=request, json={"status": {"error": "already exists"}})
        return httpx.Response(200, request=request, json={"result": {"operation_id": 1}})

    vector_index = VectorIndexService(
        embedding_client=StaticEmbeddingClient(),
        documents=[
            VectorPayload(
                id="evidence:1",
                text="失眠可辨为心脾两虚。",
                content_type="evidence",
                evidence_id="evidence:1",
            )
        ],
        qdrant_url="http://qdrant:6333",
        collection="tcm_knowledge",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    vector_index.upsert_qdrant()

    assert requested_urls == [
        "http://qdrant:6333/collections/tcm_knowledge",
        "http://qdrant:6333/collections/tcm_knowledge/points",
    ]


def test_vector_upsert_uses_stable_document_ids_for_qdrant_points():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/points"):
            captured["json"] = json.loads(request.read().decode("utf-8"))
        return httpx.Response(200, request=request, json={"result": {"operation_id": 1}})

    vector_index = VectorIndexService(
        embedding_client=StaticEmbeddingClient(),
        documents=[
            VectorPayload(
                id="evidence:evidence:source:uploaded:abc:0001",
                text="失眠可辨为心脾两虚。",
                content_type="evidence",
                evidence_id="evidence:source:uploaded:abc:0001",
            )
        ],
        qdrant_url="http://qdrant:6333",
        collection="tcm_knowledge",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    vector_index.upsert_qdrant()

    point_id = captured["json"]["points"][0]["id"]
    assert isinstance(point_id, str)
    assert point_id != "1"
    assert len(point_id) == 36


def test_vector_upsert_payloads_only_embeds_given_documents():
    embedded_texts = []
    captured = {}

    class CapturingEmbeddingClient:
        def embed(self, texts):
            embedded_texts.extend(texts)
            return [[1.0, 0.0, 0.0] for _text in texts]

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/points"):
            captured["json"] = json.loads(request.read().decode("utf-8"))
        return httpx.Response(200, request=request, json={"result": {"operation_id": 1}})

    vector_index = VectorIndexService(
        embedding_client=CapturingEmbeddingClient(),
        documents=[
            VectorPayload(
                id="chunk:old",
                text="旧文档不应被重新嵌入。",
                content_type="chunk",
                chunk_id="chunk:old",
            )
        ],
        qdrant_url="http://qdrant:6333",
        collection="tcm_knowledge",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    vector_index.upsert_payloads_qdrant(
        [
            VectorPayload(
                id="chunk:new",
                text="新文档需要进入向量库。",
                content_type="chunk",
                chunk_id="chunk:new",
            )
        ]
    )

    assert embedded_texts == ["新文档需要进入向量库。"]
    assert captured["json"]["points"][0]["payload"]["chunk_id"] == "chunk:new"
