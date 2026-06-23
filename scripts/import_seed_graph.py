import json
import os
import re
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_PATH = ROOT / "data" / "seed" / "graph.json"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(kind: str, value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid Neo4j {kind} identifier: {value}")
    return value


def build_merge_statements(graph: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    statements: list[tuple[str, dict[str, Any]]] = []
    for node in graph["nodes"]:
        label = validate_identifier("label", node["label"])
        statements.append(
            (
                f"MERGE (n:{label} {{id: $id}}) SET n.name = $name, n.label = $label",
                {"id": node["id"], "name": node["name"], "label": label},
            )
        )
    for edge in graph["edges"]:
        relation = validate_identifier("relation", edge["relation"])
        statements.append(
            (
                "MATCH (a {id: $source}), (b {id: $target}) "
                f"MERGE (a)-[r:{relation} {{id: $id}}]->(b) "
                "SET r.display = $display, r.evidence_ids = $evidence_ids",
                edge,
            )
        )
    return statements


def import_graph(path: Path = DEFAULT_GRAPH_PATH) -> None:
    graph = json.loads(path.read_text(encoding="utf-8"))
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "tcm-kg-password")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            for statement, params in build_merge_statements(graph):
                session.run(statement, params)
    finally:
        driver.close()


if __name__ == "__main__":
    import_graph()
