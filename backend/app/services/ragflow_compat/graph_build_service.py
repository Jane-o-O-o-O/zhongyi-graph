from __future__ import annotations

import json
from dataclasses import dataclass

from app.models.graph import GraphEdge, GraphNode
from app.models.ingestion import EntityCandidate, RelationCandidate
from app.services.graph_extractor import GraphExtractor
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.phase_markers import PHASE_COMMUNITY, PHASE_RESOLUTION
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalGraphArtifact


@dataclass(frozen=True)
class RagflowGraphBuildSummary:
    sources_total: int
    sources_skipped: int
    sources_built: int
    sources_failed: int
    subgraphs_merged: int = 0
    global_nodes: int = 0
    global_edges: int = 0
    graph_changed: bool = False
    resolution_marker_cleared: bool = False
    community_marker_cleared: bool = False


class RagflowGraphBuildService:
    def __init__(
        self,
        *,
        ingestion_repository: IngestionRepository,
        retrieval_repository: RagflowRetrievalRepository,
        graph_extractor: GraphExtractor,
        chunk_batch_size: int = 1000,
    ):
        self.ingestion_repository = ingestion_repository
        self.retrieval_repository = retrieval_repository
        self.graph_extractor = graph_extractor
        self.chunk_batch_size = chunk_batch_size

    def build(self, source_ids: list[str] | None = None) -> RagflowGraphBuildSummary:
        selected_source_ids = self._source_ids(source_ids)
        skipped = 0
        built = 0
        failed = 0
        for source_id in selected_source_ids:
            if self.retrieval_repository.get_subgraph_artifact(source_id):
                skipped += 1
                continue
            chunks = self.ingestion_repository.list_chunks(source_id)
            if not chunks:
                failed += 1
                continue
            entities, relations = self.graph_extractor.extract(chunks)
            nodes, edges = _graph_from_candidates(source_id, entities, relations)
            if not nodes:
                failed += 1
                continue
            self.retrieval_repository.save_graph_artifact(_subgraph_artifact(source_id, nodes, edges))
            built += 1
        merged_nodes, merged_edges, merged_sources = _merge_subgraph_artifacts(
            self.retrieval_repository.list_graph_artifacts(available_only=True)
        )
        if merged_nodes or merged_edges:
            self.retrieval_repository.save_graph_artifact(
                _global_graph_artifact(merged_nodes, merged_edges, merged_sources)
            )
        resolution_marker_cleared = False
        community_marker_cleared = False
        if built > 0:
            self.retrieval_repository.clear_graphrag_phase_markers(
                [PHASE_RESOLUTION, PHASE_COMMUNITY]
            )
            resolution_marker_cleared = True
            community_marker_cleared = True
        return RagflowGraphBuildSummary(
            sources_total=len(selected_source_ids),
            sources_skipped=skipped,
            sources_built=built,
            sources_failed=failed,
            subgraphs_merged=len(merged_sources),
            global_nodes=len(merged_nodes),
            global_edges=len(merged_edges),
            graph_changed=built > 0,
            resolution_marker_cleared=resolution_marker_cleared,
            community_marker_cleared=community_marker_cleared,
        )

    def _source_ids(self, source_ids: list[str] | None) -> list[str]:
        if source_ids is not None:
            return list(dict.fromkeys(source_ids))
        return [source.source_id for source in self.ingestion_repository.list_sources()]


def _graph_from_candidates(
    source_id: str,
    entities: list[EntityCandidate],
    relations: list[RelationCandidate],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes_by_id = {
        entity.entity_id: GraphNode(
            id=entity.entity_id,
            label=entity.label,
            name=entity.name,
            description=f"{entity.name} {entity.label}",
            properties={
                "source_id": source_id,
                "source_chunk_ids": entity.source_chunk_ids,
                "confidence": entity.confidence,
            },
        )
        for entity in entities
    }
    edges: list[GraphEdge] = []
    for relation in relations:
        if relation.source_entity_id not in nodes_by_id or relation.target_entity_id not in nodes_by_id:
            continue
        edges.append(
            GraphEdge(
                id=relation.relation_id,
                source=relation.source_entity_id,
                target=relation.target_entity_id,
                relation=relation.relation,
                display=relation.display,
                evidence_ids=relation.evidence_chunk_ids,
            )
        )
    return list(nodes_by_id.values()), edges


def _subgraph_artifact(
    source_id: str,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> RetrievalGraphArtifact:
    return RetrievalGraphArtifact(
        artifact_id=f"subgraph:{source_id}",
        artifact_type="subgraph",
        content_with_weight=json.dumps(_graph_payload(nodes, edges), ensure_ascii=False),
        source_id=[source_id],
        node_count=len(nodes),
        edge_count=len(edges),
        metadata={"scope": "source", "built_from": "graph_extractor"},
    )


def _graph_payload(nodes: list[GraphNode], edges: list[GraphEdge]) -> dict:
    return {
        "nodes": [node.model_dump() for node in nodes],
        "edges": [edge.model_dump() for edge in edges],
    }


def _merge_subgraph_artifacts(artifacts) -> tuple[list[GraphNode], list[GraphEdge], list[str]]:
    nodes_by_id: dict[str, GraphNode] = {}
    edges_by_id: dict[str, GraphEdge] = {}
    merged_sources: set[str] = set()
    for artifact in sorted(artifacts, key=lambda item: item.artifact_id):
        if artifact.artifact_type != "subgraph" or artifact.available_int != 1:
            continue
        payload = json.loads(artifact.content_with_weight)
        merged_sources.update(str(source_id) for source_id in artifact.source_id)
        for node_data in payload.get("nodes", []):
            node = GraphNode.model_validate(node_data)
            existing = nodes_by_id.get(node.id)
            if existing:
                node = _merge_node(existing, node)
            nodes_by_id[node.id] = node
        for edge_data in payload.get("edges", []):
            edge = GraphEdge.model_validate(edge_data)
            existing = edges_by_id.get(edge.id)
            if existing:
                edge = _merge_edge(existing, edge)
            edges_by_id[edge.id] = edge
    return (
        [nodes_by_id[node_id] for node_id in sorted(nodes_by_id)],
        [edges_by_id[edge_id] for edge_id in sorted(edges_by_id)],
        sorted(merged_sources),
    )


def _merge_node(left: GraphNode, right: GraphNode) -> GraphNode:
    properties = dict(left.properties)
    for key, value in right.properties.items():
        if key == "source_chunk_ids":
            properties[key] = sorted(
                set(str(item) for item in properties.get(key, []))
                | set(str(item) for item in value)
            )
        elif key not in properties or not properties[key]:
            properties[key] = value
    return left.model_copy(update={"properties": properties})


def _merge_edge(left: GraphEdge, right: GraphEdge) -> GraphEdge:
    evidence_ids = sorted(set(left.evidence_ids) | set(right.evidence_ids))
    return left.model_copy(update={"evidence_ids": evidence_ids})


def _global_graph_artifact(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    source_ids: list[str],
) -> RetrievalGraphArtifact:
    return RetrievalGraphArtifact(
        artifact_id="graph:global",
        artifact_type="graph",
        content_with_weight=json.dumps(_graph_payload(nodes, edges), ensure_ascii=False),
        source_id=source_ids,
        node_count=len(nodes),
        edge_count=len(edges),
        metadata={"scope": "global", "built_from": "subgraphs"},
    )
