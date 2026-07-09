from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from app.models.graph import GraphEdge, GraphNode
from app.services.graph_service import GraphService


class Neo4jGraphLoader:
    def __init__(self, driver):
        self.driver = driver

    def load_graph_service(self) -> GraphService:
        with self.driver.session() as session:
            nodes = [_node_from_record(record.data()) for record in session.run(_NODE_QUERY)]
            edges = [_edge_from_record(record.data()) for record in session.run(_EDGE_QUERY)]
        return GraphService(nodes=nodes, edges=edges)


def graph_service_from_neo4j(uri: str, user: str, password: str) -> GraphService:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        return Neo4jGraphLoader(driver).load_graph_service()
    finally:
        driver.close()


def _node_from_record(record: dict[str, Any]) -> GraphNode:
    properties = record.get("properties") or {}
    return GraphNode(
        id=str(record["id"]),
        label=record["label"],
        name=str(record["name"]),
        description=str(record.get("description") or ""),
        properties={key: value for key, value in properties.items() if _is_graph_property(value)},
    )


def _edge_from_record(record: dict[str, Any]) -> GraphEdge:
    return GraphEdge(
        id=str(record["id"]),
        source=str(record["source"]),
        target=str(record["target"]),
        relation=str(record["relation"]),
        display=str(record.get("display") or record["relation"]),
        evidence_ids=[str(value) for value in (record.get("evidence_ids") or [])],
    )


def _is_graph_property(value: Any) -> bool:
    if isinstance(value, str | int | float | bool):
        return True
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


_NODE_QUERY = """
MATCH (n)
WHERE n.id IS NOT NULL AND NOT n:Evidence
WITH n, head(labels(n)) AS label
RETURN n.id AS id,
       label AS label,
       coalesce(n.name, n.id) AS name,
       coalesce(n.description, "") AS description,
       properties(n) AS properties
ORDER BY label, name, id
"""

_EDGE_QUERY = """
MATCH (a)-[r]->(b)
WHERE a.id IS NOT NULL AND b.id IS NOT NULL
  AND NOT a:Evidence AND NOT b:Evidence
RETURN coalesce(r.id, elementId(r)) AS id,
       a.id AS source,
       b.id AS target,
       type(r) AS relation,
       coalesce(r.display, type(r)) AS display,
       coalesce(r.evidence_ids, []) AS evidence_ids
ORDER BY source, relation, target, id
"""
