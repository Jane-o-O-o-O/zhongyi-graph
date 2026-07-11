from sqlalchemy import create_engine

from app.models.graph import GraphEdge, GraphNode
from app.services.graph_community_summary_service import (
    CommunitySummary,
    GraphCommunitySummaryResult,
)
from app.services.ragflow_compat.community_reports import RagflowGraphCommunityReportService
from app.services.ragflow_compat.repository import RagflowRetrievalRepository


class RecordingCommunityReportClient:
    def __init__(self):
        self.calls = []

    def generate_community_report(self, *, community_id, entities, relations):
        self.calls.append(
            {
                "community_id": community_id,
                "entities": list(entities),
                "relations": list(relations),
            }
        )
        return {
            "title": f"社区报告 {community_id}",
            "summary": "结构化社区报告",
            "findings": [{"summary": "主题", "explanation": "按社区层级生成"}],
            "rating": 1.0,
            "rating_explanation": "test",
        }


def test_ragflow_community_report_service_generates_reports_for_hierarchical_levels():
    repository = RagflowRetrievalRepository(create_engine("sqlite:///:memory:"))
    report_client = RecordingCommunityReportClient()
    nodes = [
        GraphNode(
            id="node:0",
            label="Symptom",
            name="失眠",
            properties={"community_id": 0, "community_levels": ["0:0", "1:0"]},
        ),
        GraphNode(
            id="node:1",
            label="Syndrome",
            name="心脾两虚",
            properties={"community_id": 1, "community_levels": ["0:1", "1:0"]},
        ),
        GraphNode(
            id="node:2",
            label="Formula",
            name="归脾汤",
            properties={"community_id": 1, "community_levels": ["0:1", "1:0"]},
        ),
    ]
    edges = [
        GraphEdge(
            id="edge:0",
            source="node:0",
            target="node:1",
            relation="MANIFESTS_AS",
            display="可辨为",
        ),
        GraphEdge(
            id="edge:1",
            source="node:1",
            target="node:2",
            relation="RECOMMENDS_FORMULA",
            display="推荐方剂",
        ),
    ]
    base_summaries = GraphCommunitySummaryResult(
        {
            0: CommunitySummary(
                community_id=0,
                title="失眠",
                summary="失眠社区",
                size=1,
                weight=1.0,
                entities=["失眠"],
                label_counts=["Symptom:1"],
            ),
            1: CommunitySummary(
                community_id=1,
                title="心脾两虚、归脾汤",
                summary="证方社区",
                size=2,
                weight=2.0,
                entities=["心脾两虚", "归脾汤"],
                label_counts=["Syndrome:1", "Formula:1"],
            ),
        }
    )

    result = RagflowGraphCommunityReportService(report_client=report_client).summarize(
        nodes=nodes,
        edges=edges,
        base_summaries=base_summaries,
        repository=repository,
    )

    assert result.reports_generated == 3
    assert [call["entities"] for call in report_client.calls] == [
        ["失眠"],
        ["心脾两虚", "归脾汤"],
        ["失眠", "心脾两虚", "归脾汤"],
    ]
    high_level_summary = result.summaries.by_community_id[1_000_000]
    assert high_level_summary.level == 1
    assert high_level_summary.source_node_ids == ["node:0", "node:1", "node:2"]
