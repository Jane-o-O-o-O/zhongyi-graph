from app.models.graph import GraphEdge, GraphNode
from app.services.graph_analytics_service import GraphAnalyticsService
from app.services.graph_community_summary_service import GraphCommunitySummaryService


def test_community_summary_writes_titles_summaries_and_representative_entities():
    nodes = [
        GraphNode(id="formula:归脾汤", label="Formula", name="归脾汤"),
        GraphNode(id="prescription:归脾汤_1", label="Prescription", name="归脾汤_1"),
        GraphNode(id="herb:人参", label="Herb", name="人参"),
        GraphNode(id="dose:1两", label="Dose", name="1两"),
    ]
    edges = [
        GraphEdge(
            id="edge:formula:prescription",
            source="formula:归脾汤",
            target="prescription:归脾汤_1",
            relation="HAS_PRESCRIPTION",
            display="处方",
        ),
        GraphEdge(
            id="edge:prescription:herb",
            source="prescription:归脾汤_1",
            target="herb:人参",
            relation="COMPOSED_OF",
            display="组成",
        ),
        GraphEdge(
            id="edge:prescription:dose",
            source="prescription:归脾汤_1",
            target="dose:1两",
            relation="HAS_DOSE",
            display="剂量",
        ),
    ]
    analytics = GraphAnalyticsService().analyze(nodes, edges)
    analytics.apply_to_nodes(nodes)

    result = GraphCommunitySummaryService().summarize(nodes)
    result.apply_to_nodes(nodes)

    assert result.by_community_id
    for node in nodes:
        assert node.properties["community_title"]
        assert "归脾汤" in node.properties["community_summary"]
        assert node.properties["community_size"] == 4
        assert "归脾汤" in node.properties["community_entities"]
