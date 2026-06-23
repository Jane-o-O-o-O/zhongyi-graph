from dataclasses import dataclass

from app.models.graph import EvidenceCard, GraphEdge, GraphNode
from app.models.ingestion import KnowledgeBundle
from app.services.vector_service import VectorPayload


@dataclass(frozen=True)
class PublishedKnowledgeArtifact:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    evidence: list[EvidenceCard]
    vector_payloads: list[VectorPayload]


class KnowledgePublisher:
    def build_artifact(self, bundles: list[KnowledgeBundle]) -> PublishedKnowledgeArtifact:
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        evidence: dict[str, EvidenceCard] = {}

        chunk_lookup = {
            chunk.chunk_id: (bundle.source, chunk)
            for bundle in bundles
            for chunk in bundle.chunks
        }

        for bundle in bundles:
            for entity in bundle.entities:
                node_id = _graph_node_id(entity.entity_id)
                nodes[node_id] = GraphNode(
                    id=node_id,
                    label=entity.label,
                    name=entity.name,
                    properties={"source_chunks": ",".join(entity.source_chunk_ids)},
                )

            for relation in bundle.relations:
                evidence_ids = [
                    _evidence_id(chunk_id) for chunk_id in relation.evidence_chunk_ids
                ]
                edge_id = _edge_id(relation.relation_id)
                edges[edge_id] = GraphEdge(
                    id=edge_id,
                    source=_graph_node_id(relation.source_entity_id),
                    target=_graph_node_id(relation.target_entity_id),
                    relation=relation.relation,
                    display=relation.display,
                    evidence_ids=evidence_ids,
                )
                for chunk_id in relation.evidence_chunk_ids:
                    source, chunk = chunk_lookup[chunk_id]
                    evidence[_evidence_id(chunk_id)] = EvidenceCard(
                        id=_evidence_id(chunk_id),
                        title=f"{source.filename} #{chunk.chunk_index}",
                        source=source.filename,
                        snippet=chunk.content,
                        source_type="local",
                        location=f"{source.object_key or source.filename}:{chunk.chunk_index}",
                    )

        vector_payloads = [
            VectorPayload(
                id=f"entity:{node.id}",
                text=f"{node.name} {node.label} {node.description}",
                content_type="entity",
                node_id=node.id,
                label=node.label,
            )
            for node in nodes.values()
        ]
        vector_payloads.extend(
            VectorPayload(
                id=f"evidence:{card.id}",
                text=f"{card.title}。{card.snippet}。来源：{card.source}",
                content_type="evidence",
                evidence_id=card.id,
            )
            for card in evidence.values()
        )
        vector_payloads.extend(
            VectorPayload(
                id=f"chunk:{chunk.chunk_id}",
                text=chunk.content,
                content_type="chunk",
                chunk_id=chunk.chunk_id,
            )
            for bundle in bundles
            for chunk in bundle.chunks
        )

        return PublishedKnowledgeArtifact(
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            evidence=list(evidence.values()),
            vector_payloads=vector_payloads,
        )


def _graph_node_id(entity_id: str) -> str:
    return entity_id.removeprefix("entity:")


def _edge_id(relation_id: str) -> str:
    return relation_id.replace("relation:", "edge:", 1)


def _evidence_id(chunk_id: str) -> str:
    return chunk_id.replace("chunk:", "evidence:", 1)
