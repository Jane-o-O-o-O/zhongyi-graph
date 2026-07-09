from app.models.graph import GraphEdge, GraphNode
from app.services.graph_service import GraphService


def test_neighborhood_expands_allowed_relations_without_category_fanout():
    service = GraphService(
        nodes=[
            GraphNode(id="category:方剂", label="Category", name="方剂"),
            GraphNode(id="formula:鳖甲汤", label="Formula", name="鳖甲汤"),
            GraphNode(id="formula:其他方", label="Formula", name="其他方"),
            GraphNode(id="prescription:鳖甲汤_1", label="Prescription", name="鳖甲汤_1"),
            GraphNode(id="herb:柴胡", label="Herb", name="柴胡"),
        ],
        edges=[
            GraphEdge(
                id="edge:category:biejia",
                source="category:方剂",
                target="formula:鳖甲汤",
                relation="INCLUDES",
                display="包含",
            ),
            GraphEdge(
                id="edge:category:other",
                source="category:方剂",
                target="formula:其他方",
                relation="INCLUDES",
                display="包含",
            ),
            GraphEdge(
                id="edge:formula:prescription",
                source="formula:鳖甲汤",
                target="prescription:鳖甲汤_1",
                relation="HAS_PRESCRIPTION",
                display="处方",
            ),
            GraphEdge(
                id="edge:prescription:herb",
                source="prescription:鳖甲汤_1",
                target="herb:柴胡",
                relation="COMPOSED_OF",
                display="组成",
            ),
        ],
    )

    nodes, edges = service.neighborhood(
        ["formula:鳖甲汤"],
        allowed_relations={"HAS_PRESCRIPTION", "COMPOSED_OF"},
        max_depth=2,
    )

    assert {node.id for node in nodes} == {
        "formula:鳖甲汤",
        "prescription:鳖甲汤_1",
        "herb:柴胡",
    }
    assert {edge.id for edge in edges} == {"edge:formula:prescription", "edge:prescription:herb"}


def test_neighborhood_includes_terminal_nodes_without_expanding_through_them():
    service = GraphService(
        nodes=[
            GraphNode(id="formula:鳖甲汤", label="Formula", name="鳖甲汤"),
            GraphNode(id="source:圣济总录", label="Source", name="圣济总录"),
            GraphNode(id="formula:补脾汤", label="Formula", name="补脾汤"),
        ],
        edges=[
            GraphEdge(
                id="edge:biejia:source",
                source="formula:鳖甲汤",
                target="source:圣济总录",
                relation="FROM_SOURCE",
                display="来源",
            ),
            GraphEdge(
                id="edge:bupi:source",
                source="formula:补脾汤",
                target="source:圣济总录",
                relation="FROM_SOURCE",
                display="来源",
            ),
        ],
    )

    nodes, edges = service.neighborhood(
        ["formula:鳖甲汤"],
        allowed_relations={"FROM_SOURCE"},
        terminal_labels={"Source"},
        max_depth=2,
    )

    assert {node.id for node in nodes} == {"formula:鳖甲汤", "source:圣济总录"}
    assert {edge.id for edge in edges} == {"edge:biejia:source"}


def test_neighborhood_limits_nodes_by_discovery_order_not_repository_order():
    service = GraphService(
        nodes=[
            GraphNode(id="dose:30克", label="Dose", name="30克"),
            GraphNode(id="herb:柴胡", label="Herb", name="柴胡"),
            GraphNode(id="formula:鳖甲汤", label="Formula", name="鳖甲汤"),
            GraphNode(id="prescription:鳖甲汤_1", label="Prescription", name="鳖甲汤_1"),
        ],
        edges=[
            GraphEdge(
                id="edge:formula:prescription",
                source="formula:鳖甲汤",
                target="prescription:鳖甲汤_1",
                relation="HAS_PRESCRIPTION",
                display="处方",
            ),
            GraphEdge(
                id="edge:prescription:herb",
                source="prescription:鳖甲汤_1",
                target="herb:柴胡",
                relation="COMPOSED_OF",
                display="组成",
            ),
            GraphEdge(
                id="edge:prescription:dose",
                source="prescription:鳖甲汤_1",
                target="dose:30克",
                relation="HAS_DOSE",
                display="剂量",
            ),
        ],
    )

    nodes, edges = service.neighborhood(
        ["formula:鳖甲汤"],
        allowed_relations={"HAS_PRESCRIPTION", "COMPOSED_OF", "HAS_DOSE"},
        terminal_labels={"Dose"},
        max_depth=2,
        max_nodes=3,
    )

    assert [node.id for node in nodes] == [
        "formula:鳖甲汤",
        "prescription:鳖甲汤_1",
        "herb:柴胡",
    ]
    assert {edge.id for edge in edges} == {"edge:formula:prescription", "edge:prescription:herb"}


def test_matching_nodes_returns_direct_term_matches_without_expansion():
    service = GraphService(
        nodes=[
            GraphNode(id="formula:鳖甲汤", label="Formula", name="鳖甲汤"),
            GraphNode(id="prescription:鳖甲汤_1", label="Prescription", name="鳖甲汤_1"),
        ],
        edges=[
            GraphEdge(
                id="edge:formula:prescription",
                source="formula:鳖甲汤",
                target="prescription:鳖甲汤_1",
                relation="HAS_PRESCRIPTION",
                display="处方",
            )
        ],
    )

    assert [node.id for node in service.matching_nodes(["鳖甲汤"])] == [
        "formula:鳖甲汤",
        "prescription:鳖甲汤_1",
    ]


def test_overview_limits_nodes_and_keeps_edges_inside_visible_graph():
    service = GraphService(
        nodes=[
            GraphNode(id="formula:大方", label="Formula", name="大方"),
            GraphNode(id="herb:柴胡", label="Herb", name="柴胡"),
            GraphNode(id="herb:桂枝", label="Herb", name="桂枝"),
            GraphNode(id="source:典籍", label="Source", name="典籍"),
        ],
        edges=[
            GraphEdge(
                id="edge:formula:chaihu",
                source="formula:大方",
                target="herb:柴胡",
                relation="COMPOSED_OF",
                display="组成",
            ),
            GraphEdge(
                id="edge:formula:guizhi",
                source="formula:大方",
                target="herb:桂枝",
                relation="COMPOSED_OF",
                display="组成",
            ),
            GraphEdge(
                id="edge:guizhi:source",
                source="herb:桂枝",
                target="source:典籍",
                relation="FROM_SOURCE",
                display="来源",
            ),
        ],
    )

    nodes, edges = service.overview(max_nodes=3, max_edges=10)

    visible_ids = {node.id for node in nodes}
    assert len(nodes) == 3
    assert "formula:大方" in visible_ids
    assert edges
    assert all(edge.source in visible_ids and edge.target in visible_ids for edge in edges)
