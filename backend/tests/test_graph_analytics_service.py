from app.models.graph import GraphEdge, GraphNode
from app.services.graph_analytics_service import GraphAnalyticsService


def test_graph_analytics_computes_rank_components_communities_and_visual_weights():
    nodes = [
        GraphNode(id="formula:核心方", label="Formula", name="核心方"),
        GraphNode(id="prescription:核心方_1", label="Prescription", name="核心方_1"),
        GraphNode(id="herb:柴胡", label="Herb", name="柴胡"),
        GraphNode(id="dose:1两", label="Dose", name="1两"),
        GraphNode(id="formula:小群方", label="Formula", name="小群方"),
        GraphNode(id="herb:桂枝", label="Herb", name="桂枝"),
    ]
    edges = [
        GraphEdge(
            id="edge:formula:prescription",
            source="formula:核心方",
            target="prescription:核心方_1",
            relation="HAS_PRESCRIPTION",
            display="处方",
        ),
        GraphEdge(
            id="edge:prescription:herb",
            source="prescription:核心方_1",
            target="herb:柴胡",
            relation="COMPOSED_OF",
            display="组成",
        ),
        GraphEdge(
            id="edge:prescription:dose",
            source="prescription:核心方_1",
            target="dose:1两",
            relation="HAS_DOSE",
            display="剂量",
        ),
        GraphEdge(
            id="edge:small",
            source="formula:小群方",
            target="herb:桂枝",
            relation="COMPOSED_OF",
            display="组成",
        ),
    ]

    result = GraphAnalyticsService().analyze(nodes, edges)

    assert result.by_node_id["prescription:核心方_1"].degree == 3
    assert result.by_node_id["formula:核心方"].pagerank > 0
    assert result.by_node_id["formula:核心方"].component_id != result.by_node_id["formula:小群方"].component_id
    assert result.by_node_id["formula:核心方"].community_id != result.by_node_id["formula:小群方"].community_id
    assert (
        result.by_node_id["formula:核心方"].visual_weight
        > result.by_node_id["dose:1两"].visual_weight
    )


def test_graph_analytics_can_write_metrics_to_node_properties():
    nodes = [
        GraphNode(id="formula:核心方", label="Formula", name="核心方"),
        GraphNode(id="herb:柴胡", label="Herb", name="柴胡"),
    ]
    edges = [
        GraphEdge(
            id="edge:formula:herb",
            source="formula:核心方",
            target="herb:柴胡",
            relation="COMPOSED_OF",
            display="组成",
        ),
    ]

    result = GraphAnalyticsService().analyze(nodes, edges)
    result.apply_to_nodes(nodes)

    assert nodes[0].properties["degree"] == 1
    assert nodes[0].properties["pagerank"] > 0
    assert nodes[0].properties["component_id"] == nodes[1].properties["component_id"]
    assert nodes[0].properties["community_id"] == nodes[1].properties["community_id"]
    assert nodes[0].properties["visual_weight"] > 0
