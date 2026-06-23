import json
from pathlib import Path


def test_seed_graph_artifact_has_nodes_edges_and_evidence():
    artifact_path = Path(__file__).resolve().parents[2] / "data" / "seed" / "graph.json"

    data = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert data["nodes"]
    assert data["edges"]
    assert data["evidence"]
    assert any(node["name"] == "柴胡桂枝干姜汤" for node in data["nodes"])
