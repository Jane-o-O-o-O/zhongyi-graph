from scripts.import_seed_graph import build_merge_statements


def test_build_merge_statements_creates_node_and_edge_cypher():
    graph = {
        "nodes": [{"id": "herb:柴胡", "label": "Herb", "name": "柴胡"}],
        "edges": [
            {
                "id": "edge:1",
                "source": "formula:方",
                "target": "herb:柴胡",
                "relation": "COMPOSED_OF",
                "display": "组成",
                "evidence_ids": [],
            }
        ],
        "evidence": [],
    }

    statements = build_merge_statements(graph)

    assert any("MERGE (n:Herb" in statement for statement, _ in statements)
    assert any("COMPOSED_OF" in statement for statement, _ in statements)
