import httpx

from app.models.graph import GraphNode
from app.services.model_clients import StructuredExtractionClient
from app.services.ragflow_compat.entity_resolution import LlmEntityResolutionDecider


class ResolutionHttpClient:
    def __init__(self):
        self.requests = []

    def post(self, url: str, headers: dict, json: dict) -> httpx.Response:
        self.requests.append(json)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"pairs": ['
                                '{"source": "白芍", "target": "白芍药", "same": true},'
                                '{"source": "柴胡", "target": "北柴胡", "same_entity": false}'
                                "]}"
                            )
                        }
                    }
                ]
            },
        )


def test_structured_extraction_client_resolves_entity_pairs_from_llm_json():
    http_client = ResolutionHttpClient()
    client = StructuredExtractionClient(
        base_url="http://localhost:8088/v1",
        api_key="demo",
        model="demo-extraction",
        http_client=http_client,
    )

    selected = client.resolve_entity_pairs(
        entity_type="Herb",
        pairs=[("白芍", "白芍药"), ("柴胡", "北柴胡")],
    )

    assert selected == [("白芍", "白芍药")]
    payload = http_client.requests[0]
    assert payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}
    assert "白芍药" in payload["messages"][1]["content"]


def test_llm_resolution_decider_delegates_to_structured_client():
    class FakeResolutionClient:
        def resolve_entity_pairs(self, entity_type, pairs):
            assert entity_type == "Herb"
            assert pairs == [("白芍", "白芍药")]
            return [("白芍", "白芍药")]

    decider = LlmEntityResolutionDecider(FakeResolutionClient())

    assert decider.resolve_pairs(
        "Herb",
        [("白芍", "白芍药")],
        {
            "白芍": GraphNode(id="entity:herb:白芍", label="Herb", name="白芍"),
            "白芍药": GraphNode(id="entity:herb:白芍药", label="Herb", name="白芍药"),
        },
    ) == [("白芍", "白芍药")]
