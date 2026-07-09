import json

import httpx

from scripts.sync_structured_node_vectors import (
    build_node_payloads,
    existing_entity_node_ids,
    missing_payloads,
    sync_payloads,
)


class StaticEmbeddingClient:
    def __init__(self):
        self.embedded_texts = []

    def embed(self, texts):
        self.embedded_texts.extend(texts)
        return [[1.0, 0.0, 0.0] for _text in texts]


def test_build_node_payloads_loads_entity_payloads_from_graph_file(tmp_path):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "herb:柴胡", "label": "Herb", "name": "柴胡"},
                    {"id": "formula:鳖甲汤", "label": "Formula", "name": "鳖甲汤"},
                ],
                "edges": [],
                "evidence": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payloads = build_node_payloads(graph_path)

    assert [payload.id for payload in payloads] == ["entity:herb:柴胡", "entity:formula:鳖甲汤"]
    assert payloads[0].node_id == "herb:柴胡"
    assert payloads[0].content_type == "entity"
    assert payloads[0].label == "Herb"
    assert payloads[0].text == "柴胡 Herb"


def test_existing_entity_node_ids_reads_qdrant_entity_payloads():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.read().decode("utf-8")))
        if "offset" not in requests[-1]:
            return httpx.Response(
                200,
                request=request,
                json={
                    "result": {
                        "points": [
                            {
                                "id": "1",
                                "payload": {
                                    "content_type": "entity",
                                    "node_id": "herb:柴胡",
                                },
                            },
                            {
                                "id": "2",
                                "payload": {
                                    "content_type": "chunk",
                                    "node_id": "",
                                },
                            },
                        ],
                        "next_page_offset": "next",
                    }
                },
            )
        return httpx.Response(
            200,
            request=request,
            json={
                "result": {
                    "points": [
                        {
                            "id": "3",
                            "payload": {
                                "content_type": "entity",
                                "node_id": "formula:鳖甲汤",
                            },
                        }
                    ],
                    "next_page_offset": None,
                }
            },
        )

    ids = existing_entity_node_ids(
        httpx.Client(transport=httpx.MockTransport(handler)),
        qdrant_url="http://qdrant:6333",
        collection="tcm_knowledge",
    )

    assert ids == {"herb:柴胡", "formula:鳖甲汤"}
    assert requests[1]["offset"] == "next"


def test_missing_payloads_skips_existing_nodes(tmp_path):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "herb:柴胡", "label": "Herb", "name": "柴胡"},
                    {"id": "formula:鳖甲汤", "label": "Formula", "name": "鳖甲汤"},
                ],
                "edges": [],
                "evidence": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payloads = missing_payloads(
        build_node_payloads(graph_path),
        existing_node_ids={"herb:柴胡"},
    )

    assert [payload.node_id for payload in payloads] == ["formula:鳖甲汤"]


def test_sync_payloads_embeds_and_upserts_batches(tmp_path):
    embedding_client = StaticEmbeddingClient()
    upserted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/collections/tcm_knowledge"):
            return httpx.Response(409, request=request, json={"status": {"error": "exists"}})
        if str(request.url).endswith("/points"):
            upserted.append(json.loads(request.read().decode("utf-8")))
        return httpx.Response(200, request=request, json={"result": {"operation_id": 1}})

    sync_payloads(
        [
            build_node_payload(tmp_path, "herb:柴胡", "Herb", "柴胡"),
            build_node_payload(tmp_path, "formula:鳖甲汤", "Formula", "鳖甲汤"),
        ],
        embedding_client=embedding_client,
        qdrant_client=httpx.Client(transport=httpx.MockTransport(handler)),
        qdrant_url="http://qdrant:6333",
        collection="tcm_knowledge",
        dimensions=3,
        batch_size=1,
    )

    assert embedding_client.embedded_texts == ["柴胡 Herb", "鳖甲汤 Formula"]
    assert len(upserted) == 2
    assert upserted[0]["points"][0]["payload"]["node_id"] == "herb:柴胡"


def build_node_payload(tmp_path, node_id, label, name):
    graph = {"nodes": [{"id": node_id, "label": label, "name": name}], "edges": [], "evidence": []}
    path = tmp_path / f"{node_id.replace(':', '_')}.json"
    path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    return build_node_payloads(path)[0]
