from app.models.graph import GraphEdge, GraphNode
from app.services.graph_entity_resolution_service import GraphEntityResolutionService


def test_entity_resolution_maps_alias_edges_to_canonical_entities():
    nodes = [
        GraphNode(id="herb:白芍", label="Herb", name="白芍"),
        GraphNode(id="alias:白芍药", label="Alias", name="白芍药"),
    ]
    edges = [
        GraphEdge(
            id="edge:alias",
            source="herb:白芍",
            target="alias:白芍药",
            relation="HAS_ALIAS",
            display="别名",
        )
    ]

    result = GraphEntityResolutionService().resolve(nodes, edges)
    result.apply_to_nodes(nodes)

    herb = nodes[0]
    alias = nodes[1]
    assert herb.properties["canonical_id"] == "herb:白芍"
    assert herb.properties["aliases"] == ["白芍药"]
    assert alias.properties["canonical_id"] == "herb:白芍"
    assert alias.properties["canonical_name"] == "白芍"
    assert alias.properties["is_alias"] is True


def test_entity_resolution_maps_unambiguous_same_name_aliases():
    nodes = [
        GraphNode(id="herb:柴胡", label="Herb", name="柴胡"),
        GraphNode(id="alias:柴胡", label="Alias", name="柴胡"),
    ]

    result = GraphEntityResolutionService().resolve(nodes, [])

    assert result.by_node_id["alias:柴胡"].canonical_id == "herb:柴胡"
    assert result.by_node_id["herb:柴胡"].aliases == ["柴胡"]


def test_entity_resolution_keeps_ambiguous_same_name_entities_separate():
    nodes = [
        GraphNode(id="herb:鳖甲", label="Herb", name="鳖甲"),
        GraphNode(id="formula:鳖甲", label="Formula", name="鳖甲"),
        GraphNode(id="alias:鳖甲", label="Alias", name="鳖甲"),
    ]

    result = GraphEntityResolutionService().resolve(nodes, [])

    assert result.by_node_id["herb:鳖甲"].canonical_id == "herb:鳖甲"
    assert result.by_node_id["formula:鳖甲"].canonical_id == "formula:鳖甲"
    assert result.by_node_id["alias:鳖甲"].canonical_id == "alias:鳖甲"
