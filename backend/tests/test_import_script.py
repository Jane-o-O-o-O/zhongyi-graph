import pytest

from scripts import import_seed_graph
from scripts.import_seed_graph import build_merge_statements


def test_build_merge_statements_creates_node_and_edge_cypher():
    graph = {
        "nodes": [
            {
                "id": "herb:柴胡",
                "label": "Herb",
                "name": "柴胡",
                "source_chunks": ["evidence:structured:1"],
            }
        ],
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

    assert statements == [
        (
            "MERGE (n:Herb {id: $id}) "
            "SET n.name = $name, n.label = $label "
            "SET n += $properties",
            {
                "id": "herb:柴胡",
                "name": "柴胡",
                "label": "Herb",
                "properties": {"source_chunks": ["evidence:structured:1"]},
            },
        ),
        (
            "MATCH (a {id: $source}), (b {id: $target}) "
            "MERGE (a)-[r:COMPOSED_OF {id: $id}]->(b) "
            "SET r.display = $display, r.evidence_ids = $evidence_ids",
            {
                "id": "edge:1",
                "source": "formula:方",
                "target": "herb:柴胡",
                "relation": "COMPOSED_OF",
                "display": "组成",
                "evidence_ids": [],
            },
        ),
    ]


def test_build_merge_statements_rejects_invalid_label():
    graph = {
        "nodes": [{"id": "herb:柴胡", "label": "Herb) DETACH DELETE n //", "name": "柴胡"}],
        "edges": [],
        "evidence": [],
    }

    with pytest.raises(ValueError, match="label.*Herb\\) DETACH DELETE n //"):
        build_merge_statements(graph)


def test_build_merge_statements_rejects_invalid_relation():
    graph = {
        "nodes": [],
        "edges": [
            {
                "id": "edge:1",
                "source": "formula:方",
                "target": "herb:柴胡",
                "relation": "COMPOSED_OF]->() DELETE r //",
                "display": "组成",
                "evidence_ids": [],
            }
        ],
        "evidence": [],
    }

    with pytest.raises(ValueError, match="relation.*COMPOSED_OF"):
        build_merge_statements(graph)


def test_import_seed_graph_main_accepts_graph_path(monkeypatch, tmp_path):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text('{"nodes":[],"edges":[],"evidence":[]}', encoding="utf-8")
    imported = []

    monkeypatch.setattr("sys.argv", ["import_seed_graph.py", "--graph", str(graph_path)])
    monkeypatch.setattr(import_seed_graph, "import_graph", lambda path: imported.append(path))

    import_seed_graph.main()

    assert imported == [graph_path]
