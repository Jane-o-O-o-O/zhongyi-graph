from __future__ import annotations

import re
from typing import Any

from neo4j import GraphDatabase

from app.services.knowledge_publisher import PublishedKnowledgeArtifact


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Neo4jPublisher:
    def __init__(self, driver):
        self.driver = driver

    @classmethod
    def from_config(cls, uri: str, user: str, password: str) -> "Neo4jPublisher":
        return cls(GraphDatabase.driver(uri, auth=(user, password)))

    def publish(self, artifact: PublishedKnowledgeArtifact) -> None:
        with self.driver.session() as session:
            for statement, params in build_merge_statements(artifact):
                session.run(statement, params)

    def close(self) -> None:
        self.driver.close()


def build_merge_statements(artifact: PublishedKnowledgeArtifact) -> list[tuple[str, dict[str, Any]]]:
    statements: list[tuple[str, dict[str, Any]]] = []
    for node in artifact.nodes:
        label = _validate_identifier("label", node.label)
        statements.append(
            (
                f"MERGE (n:{label} {{id: $id}}) "
                "SET n.name = $name, n.label = $label, n.description = $description "
                "SET n += $properties",
                {
                    "id": node.id,
                    "name": node.name,
                    "label": label,
                    "description": node.description,
                    "properties": node.properties,
                },
            )
        )
    for edge in artifact.edges:
        relation = _validate_identifier("relation", edge.relation)
        statements.append(
            (
                "MATCH (a {id: $source}), (b {id: $target}) "
                f"MERGE (a)-[r:{relation} {{id: $id}}]->(b) "
                "SET r.display = $display, r.evidence_ids = $evidence_ids",
                edge.model_dump(),
            )
        )
    for card in artifact.evidence:
        statements.append(
            (
                "MERGE (e:Evidence {id: $id}) "
                "SET e.title = $title, e.source = $source, e.snippet = $snippet, "
                "e.source_type = $source_type, e.location = $location",
                card.model_dump(),
            )
        )
    return statements


def _validate_identifier(kind: str, value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid Neo4j {kind} identifier: {value}")
    return value
