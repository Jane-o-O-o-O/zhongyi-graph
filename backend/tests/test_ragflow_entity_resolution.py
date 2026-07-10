import httpx
from sqlalchemy import create_engine

from app.models.graph import GraphNode
from app.services.model_clients import StructuredExtractionClient
from app.services.ragflow_compat.entity_resolution import (
    LlmEntityResolutionDecider,
    RagflowGraphEntityResolutionService,
)
from app.services.ragflow_compat.checkpoints import RESOLUTION_CHECKPOINT
from app.services.ragflow_compat.repository import RagflowRetrievalRepository


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


def test_ragflow_entity_resolution_filters_digit_different_candidate_names():
    class FailingDecider:
        def resolve_pairs(self, entity_type, pairs, nodes_by_name):
            raise AssertionError(f"digit-different pairs must not be resolved: {pairs}")

    repository = RagflowRetrievalRepository(create_engine("sqlite:///:memory:"))
    nodes = [
        GraphNode(id="formula:归脾汤1号", label="Formula", name="归脾汤1号"),
        GraphNode(id="formula:归脾汤2号", label="Formula", name="归脾汤2号"),
    ]

    result = RagflowGraphEntityResolutionService(decider=FailingDecider()).resolve(
        nodes=nodes,
        edges=[],
        repository=repository,
    )

    assert result.pairs_resolved == 0
    assert result.pairs_merged == 0
    assert [node.name for node in result.nodes] == ["归脾汤1号", "归脾汤2号"]


def test_ragflow_entity_resolution_resolves_candidates_in_ragflow_sized_batches():
    class RecordingDecider:
        def __init__(self):
            self.calls = []

        def resolve_pairs(self, entity_type, pairs, nodes_by_name):
            del nodes_by_name
            self.calls.append((entity_type, list(pairs)))
            return []

    suffixes = [
        "甲",
        "乙",
        "丙",
        "丁",
        "戊",
        "己",
        "庚",
        "辛",
        "壬",
        "癸",
        "子",
        "丑",
        "寅",
        "卯",
        "辰",
        "巳",
    ]
    nodes = [
        GraphNode(id=f"syndrome:心脾两虚{suffix}", label="Syndrome", name=f"心脾两虚{suffix}")
        for suffix in suffixes
    ]
    decider = RecordingDecider()
    repository = RagflowRetrievalRepository(create_engine("sqlite:///:memory:"))

    RagflowGraphEntityResolutionService(decider=decider).resolve(
        nodes=nodes,
        edges=[],
        repository=repository,
    )

    assert [entity_type for entity_type, _pairs in decider.calls] == ["Syndrome", "Syndrome"]
    assert [len(pairs) for _entity_type, pairs in decider.calls] == [100, 20]
    assert repository.load_graphrag_checkpoints(RESOLUTION_CHECKPOINT) == {}
