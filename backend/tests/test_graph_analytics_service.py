from dataclasses import dataclass

from app.models.graph import GraphEdge, GraphNode
import app.services.graph_analytics_service as graph_analytics_module
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


def test_graph_analytics_falls_back_when_pagerank_optional_dependency_is_missing(monkeypatch):
    def raise_missing_dependency(*args, **kwargs):
        del args, kwargs
        raise ImportError("scipy missing")

    monkeypatch.setattr(graph_analytics_module.nx, "pagerank", raise_missing_dependency)
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

    assert result.by_node_id["formula:核心方"].pagerank == 0.5
    assert result.by_node_id["herb:柴胡"].pagerank == 0.5


def test_graph_analytics_prefers_leiden_community_runner_when_available(monkeypatch):
    @dataclass(frozen=True)
    class FakePartition:
        level: int
        node: str
        cluster: int

    calls = []

    def fake_hierarchical_leiden(graph, max_cluster_size, random_seed):
        calls.append(
            {
                "nodes": sorted(graph.nodes),
                "max_cluster_size": max_cluster_size,
                "random_seed": random_seed,
            }
        )
        return [
            FakePartition(0, "node:0", 20),
            FakePartition(0, "node:1", 10),
            FakePartition(0, "node:2", 20),
            FakePartition(0, "node:3", 10),
            FakePartition(0, "node:4", 20),
            FakePartition(0, "node:5", 10),
            FakePartition(0, "node:6", 20),
        ]

    monkeypatch.setattr(
        graph_analytics_module,
        "_hierarchical_leiden",
        fake_hierarchical_leiden,
        raising=False,
    )
    nodes = [
        GraphNode(id=f"node:{index}", label="Herb", name=f"节点{index}")
        for index in range(7)
    ]
    edges = [
        GraphEdge(
            id=f"edge:{index}",
            source=f"node:{index}",
            target=f"node:{index + 1}",
            relation="RELATED_TO",
            display="相关",
        )
        for index in range(6)
    ]

    result = GraphAnalyticsService().analyze(nodes, edges)

    assert calls == [
        {
            "nodes": [f"node:{index}" for index in range(7)],
            "max_cluster_size": 12,
            "random_seed": 0xDEADBEEF,
        }
    ]
    assert result.by_node_id["node:0"].community_id == result.by_node_id["node:2"].community_id
    assert result.by_node_id["node:1"].community_id == result.by_node_id["node:3"].community_id
    assert result.by_node_id["node:0"].community_id != result.by_node_id["node:1"].community_id


def test_graph_analytics_preserves_hierarchical_leiden_levels(monkeypatch):
    @dataclass(frozen=True)
    class FakePartition:
        level: int
        node: str
        cluster: int

    def fake_hierarchical_leiden(graph, max_cluster_size, random_seed):
        del graph, max_cluster_size, random_seed
        return [
            FakePartition(0, "node:0", 10),
            FakePartition(0, "node:1", 20),
            FakePartition(0, "node:2", 10),
            FakePartition(1, "node:0", 100),
            FakePartition(1, "node:1", 100),
            FakePartition(1, "node:2", 200),
        ]

    monkeypatch.setattr(
        graph_analytics_module,
        "_hierarchical_leiden",
        fake_hierarchical_leiden,
        raising=False,
    )
    nodes = [
        GraphNode(id=f"node:{index}", label="Herb", name=f"节点{index}")
        for index in range(3)
    ]
    edges = [
        GraphEdge(
            id="edge:0",
            source="node:0",
            target="node:1",
            relation="RELATED_TO",
            display="相关",
        ),
        GraphEdge(
            id="edge:1",
            source="node:1",
            target="node:2",
            relation="RELATED_TO",
            display="相关",
        ),
    ]

    result = GraphAnalyticsService().analyze(nodes, edges)
    result.apply_to_nodes(nodes)

    assert result.by_node_id["node:0"].community_id == result.by_node_id["node:2"].community_id
    assert result.by_node_id["node:0"].community_levels == {"0": 0, "1": 0}
    assert result.by_node_id["node:1"].community_levels == {"0": 1, "1": 0}
    assert nodes[0].properties["community_levels"] == ["0:0", "1:0"]
