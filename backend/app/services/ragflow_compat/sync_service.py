from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from app.models.graph import GraphNode
from app.models.ingestion import DocumentChunk, EntityCandidate, RelationCandidate, SourceManifest
from app.services.graph_analytics_service import GraphAnalyticsService
from app.services.graph_service import GraphService
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.phase_markers import PHASE_COMMUNITY, PHASE_RESOLUTION
from app.services.ragflow_compat.query import build_content_with_weight, tokenize_query
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import (
    RetrievalCommunityReport,
    RetrievalChunk,
    RetrievalChunkTerm,
    RetrievalDocument,
    RetrievalGraphArtifact,
    RetrievalKgEntity,
    RetrievalKgRelation,
    RetrievalTypeSamples,
)


@dataclass(frozen=True)
class CandidateEvidenceLookup:
    entity_chunks: dict[tuple[str, str], list[str]]
    relation_chunks: dict[tuple[str, str, str], list[str]]


class RagflowRetrievalSyncService:
    def __init__(
        self,
        *,
        ingestion_repository: IngestionRepository,
        retrieval_repository: RagflowRetrievalRepository,
        min_chunk_tokens: int = 8,
        max_chunk_tokens: int = 900,
        chunk_batch_size: int = 1000,
        build_chunk_terms: bool = False,
        graph_service: GraphService | None = None,
        write_graph_artifacts: bool = True,
        mark_resolution_phase: bool = True,
        mark_community_phase: bool = True,
    ):
        self.ingestion_repository = ingestion_repository
        self.retrieval_repository = retrieval_repository
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_batch_size = chunk_batch_size
        self.build_chunk_terms = build_chunk_terms
        self.graph_service = graph_service
        self.write_graph_artifacts = write_graph_artifacts
        self.mark_resolution_phase = mark_resolution_phase
        self.mark_community_phase = mark_community_phase

    def rebuild_from_ingestion(self) -> dict[str, int]:
        sources = self.ingestion_repository.list_sources()
        entities_with_source = self.ingestion_repository.list_entities()
        relations_with_source = self.ingestion_repository.list_relations()
        pages = self.ingestion_repository.list_pages()
        page_numbers = {page.page_id: page.page_number for page in pages}
        chunk_counts = self.ingestion_repository.chunk_counts_by_source(
            min_tokens=self.min_chunk_tokens,
            max_tokens=self.max_chunk_tokens,
        )

        retrieval_documents = [
            self._document_from_source(source, chunk_counts.get(source.source_id, (0, 0)))
            for source in sources
        ]
        entity_lookup = {
            (source_id, entity.entity_id): entity for source_id, entity in entities_with_source
        }
        graph_branch = bool(self.graph_service and self.graph_service.nodes)
        if graph_branch:
            evidence_lookup = _candidate_evidence_lookup(
                entities_with_source,
                relations_with_source,
            )
            retrieval_entities = self._entities_from_graph(
                self.graph_service,
                evidence_lookup,
            )
            retrieval_relations = self._relations_from_graph(
                self.graph_service,
                evidence_lookup,
            )
            community_reports = self._community_reports_from_graph(self.graph_service)
            graph_artifacts = self._graph_artifacts_from_graph(self.graph_service)
            if not self.write_graph_artifacts:
                graph_artifacts = []
        else:
            retrieval_entities = [
                self._entity_from_candidate(source_id, entity)
                for source_id, entity in entities_with_source
            ]
            retrieval_relations = [
                self._relation_from_candidate(source_id, relation, entity_lookup)
                for source_id, relation in relations_with_source
            ]
            community_reports = []
            graph_artifacts = []
        retrieval_type_samples = self._type_samples(retrieval_entities)

        self.retrieval_repository.clear_rebuild_tables(
            include_graph_artifacts=self.write_graph_artifacts
        )
        self.retrieval_repository.append_documents(retrieval_documents)
        chunk_count = 0
        for chunk_batch in self.ingestion_repository.iter_chunk_batches(self.chunk_batch_size):
            retrieval_chunks = [
                self._chunk_from_source_chunk(chunk, page_numbers)
                for chunk in chunk_batch
            ]
            self.retrieval_repository.append_chunks(retrieval_chunks)
            if self.build_chunk_terms:
                self.retrieval_repository.append_chunk_terms(
                    [
                        term
                        for retrieval_chunk in retrieval_chunks
                        for term in self._terms_from_chunk(retrieval_chunk)
                    ]
                )
            chunk_count += len(retrieval_chunks)
        self.retrieval_repository.append_kg_entities(retrieval_entities)
        self.retrieval_repository.append_kg_relations(retrieval_relations)
        self.retrieval_repository.append_community_reports(community_reports)
        self.retrieval_repository.append_graph_artifacts(graph_artifacts)
        self.retrieval_repository.append_type_samples(retrieval_type_samples)
        if graph_branch and self.mark_resolution_phase:
            self.retrieval_repository.set_graphrag_phase_marker(PHASE_RESOLUTION)
        if graph_branch and self.mark_community_phase:
            self.retrieval_repository.set_graphrag_phase_marker(PHASE_COMMUNITY)
        return {
            "documents": len(retrieval_documents),
            "chunks": chunk_count,
            "kg_entities": len(retrieval_entities),
            "kg_relations": len(retrieval_relations),
            "community_reports": len(community_reports),
            "graph_artifacts": len(graph_artifacts),
        }

    def _document_from_source(
        self,
        source: SourceManifest,
        counts: tuple[int, int],
    ) -> RetrievalDocument:
        chunk_count, eligible_chunk_count = counts
        return RetrievalDocument(
            doc_id=source.source_id,
            source_id=source.source_id,
            filename=source.filename,
            mime_type=source.mime_type,
            checksum=source.checksum,
            status=source.status,
            object_key=source.object_key,
            source_version=source.version,
            chunk_count=chunk_count,
            eligible_chunk_count=eligible_chunk_count,
            metadata={"source_status": source.status},
        )

    def _chunk_from_source_chunk(
        self,
        chunk: DocumentChunk,
        page_numbers: dict[str, int],
    ) -> RetrievalChunk:
        metadata = chunk.metadata or {}
        section_path = _string_list(metadata.get("section_path", []))
        title = chunk.section_title or (section_path[-1] if section_path else "")
        important_keywords = tokenize_query(" ".join([title, chunk.content]))[:12]
        content_with_weight = build_content_with_weight(
            title=title,
            section_path=section_path,
            important_keywords=important_keywords,
            content=chunk.content,
        )
        return RetrievalChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.source_id,
            source_id=chunk.source_id,
            parent_unit_id=str(metadata.get("parent_unit_id") or chunk.chunk_id),
            chunk_order_int=chunk.chunk_index,
            page_num_int=page_numbers.get(chunk.page_id, 1),
            title=title,
            section_path=section_path,
            content=chunk.content,
            content_with_weight=content_with_weight,
            content_ltks=" ".join(tokenize_query(chunk.content)),
            title_tks=" ".join(tokenize_query(title)),
            important_kwd=important_keywords,
            content_type=chunk.content_type,
            token_count=chunk.token_count or _estimate_tokens(chunk.content),
            available_int=1 if self._is_eligible_chunk(chunk) else 0,
            metadata=dict(metadata),
        )

    def _entity_from_candidate(
        self,
        source_id: str,
        entity: EntityCandidate,
    ) -> RetrievalKgEntity:
        description = f"{entity.name} {entity.label}"
        return RetrievalKgEntity(
            entity_id=f"{source_id}:{entity.entity_id}",
            entity_name=entity.name,
            entity_type=entity.label,
            source_node_id=entity.entity_id,
            content_with_weight=json.dumps(
                {"description": description, "source": source_id},
                ensure_ascii=False,
            ),
            description=description,
            rank_flt=max(entity.confidence, 0.01),
            n_hop_with_weight=[],
            aliases=_unique([entity.normalized_name, entity.name]),
            evidence_chunk_ids=entity.source_chunk_ids,
            metadata={"source_id": source_id, "confidence": entity.confidence},
        )

    def _relation_from_candidate(
        self,
        source_id: str,
        relation: RelationCandidate,
        entity_lookup: dict[tuple[str, str], EntityCandidate],
    ) -> RetrievalKgRelation:
        source_entity = entity_lookup.get((source_id, relation.source_entity_id))
        target_entity = entity_lookup.get((source_id, relation.target_entity_id))
        from_name = source_entity.name if source_entity else relation.source_entity_id
        to_name = target_entity.name if target_entity else relation.target_entity_id
        display = relation.display or relation.relation
        content = f"{from_name} {display} {to_name}"
        return RetrievalKgRelation(
            relation_id=f"{source_id}:{relation.relation_id}",
            from_entity_kwd=from_name,
            to_entity_kwd=to_name,
            relation_type=relation.relation,
            display=display,
            content_with_weight=content,
            weight_int=max(1, round(relation.confidence)),
            evidence_chunk_ids=relation.evidence_chunk_ids,
            source_edge_id=relation.relation_id,
            metadata={"source_id": source_id, "confidence": relation.confidence},
        )

    def _entities_from_graph(
        self,
        graph_service: GraphService,
        evidence_lookup: "CandidateEvidenceLookup",
    ) -> list[RetrievalKgEntity]:
        n_hop_paths = _n_hop_paths_from_graph(graph_service)
        return [
            RetrievalKgEntity(
                entity_id=node.id,
                entity_name=node.name,
                entity_type=node.label,
                source_node_id=node.id,
                content_with_weight=json.dumps(_graph_entity_content(node), ensure_ascii=False),
                description=node.description or f"{node.name} {node.label}",
                rank_flt=_float_property(node.properties, "pagerank", "rank", fallback=1.0),
                n_hop_with_weight=n_hop_paths.get(node.id, [])[:40],
                aliases=_string_list(node.properties.get("aliases", [])),
                evidence_chunk_ids=_unique(
                    _source_chunks_from_graph_node(node)
                    + evidence_lookup.entity_chunks.get((node.name, node.label), [])
                    + evidence_lookup.entity_chunks.get((node.name, ""), [])
                ),
                metadata={"source": "neo4j", **node.properties},
            )
            for node in graph_service.nodes
        ]

    def _relations_from_graph(
        self,
        graph_service: GraphService,
        evidence_lookup: "CandidateEvidenceLookup",
    ) -> list[RetrievalKgRelation]:
        nodes_by_id = {node.id: node for node in graph_service.nodes}
        relations: list[RetrievalKgRelation] = []
        for edge in graph_service.edges:
            source = nodes_by_id.get(edge.source)
            target = nodes_by_id.get(edge.target)
            from_name = source.name if source else edge.source
            to_name = target.name if target else edge.target
            candidate_evidence = evidence_lookup.relation_chunks.get(
                (from_name, to_name, edge.relation),
                [],
            )
            if not candidate_evidence:
                candidate_evidence = evidence_lookup.relation_chunks.get(
                    (to_name, from_name, edge.relation),
                    [],
                )
            relations.append(
                RetrievalKgRelation(
                    relation_id=edge.id,
                    from_entity_kwd=from_name,
                    to_entity_kwd=to_name,
                    relation_type=edge.relation,
                    display=edge.display,
                    content_with_weight=f"{from_name} {edge.display} {to_name}",
                    weight_int=max(
                        1,
                        round(GraphAnalyticsService.retrieval_edge_weight(edge) * 10),
                    ),
                    evidence_chunk_ids=_unique(
                        _chunk_ids_from_evidence_ids(edge.evidence_ids)
                        + candidate_evidence
                    ),
                    source_edge_id=edge.id,
                    metadata={"source": "neo4j"},
                )
            )
        return relations

    def _community_reports_from_graph(
        self,
        graph_service: GraphService,
    ) -> list[RetrievalCommunityReport]:
        reports: list[RetrievalCommunityReport] = []
        for community_id, summary in sorted(graph_service.community_summaries.by_community_id.items()):
            content = {
                "report": summary.summary,
                "evidences": "；".join(summary.entities),
                "title": summary.title,
                "findings": summary.findings or [],
                "rating": summary.rating,
                "rating_explanation": summary.rating_explanation,
            }
            reports.append(
                RetrievalCommunityReport(
                    report_id=f"community:{community_id}",
                    title=summary.title,
                    content_with_weight=json.dumps(content, ensure_ascii=False),
                    summary=summary.summary,
                    evidences=content["evidences"],
                    entities_kwd=summary.entities,
                    weight_flt=summary.weight,
                    source_id=_community_source_ids(graph_service, community_id),
                    metadata={
                        "community_id": community_id,
                        "community_size": summary.size,
                        "label_counts": summary.label_counts,
                        "findings": summary.findings or [],
                        "rating": summary.rating,
                        "rating_explanation": summary.rating_explanation,
                    },
                )
            )
        return reports

    def _graph_artifacts_from_graph(
        self,
        graph_service: GraphService,
    ) -> list[RetrievalGraphArtifact]:
        artifacts = [
            RetrievalGraphArtifact(
                artifact_id="graph:global",
                artifact_type="graph",
                content_with_weight=json.dumps(
                    _graph_payload(graph_service.nodes, graph_service.edges),
                    ensure_ascii=False,
                ),
                source_id=_graph_source_ids(graph_service.nodes, graph_service.edges),
                node_count=len(graph_service.nodes),
                edge_count=len(graph_service.edges),
                metadata={"scope": "global"},
            )
        ]
        for source_id, payload in sorted(_subgraph_groups(graph_service).items()):
            artifacts.append(
                RetrievalGraphArtifact(
                    artifact_id=f"subgraph:{source_id}",
                    artifact_type="subgraph",
                    content_with_weight=json.dumps(
                        _graph_payload(payload["nodes"], payload["edges"]),
                        ensure_ascii=False,
                    ),
                    source_id=[source_id],
                    node_count=len(payload["nodes"]),
                    edge_count=len(payload["edges"]),
                    metadata={"scope": "source"},
                )
            )
        return artifacts

    def _terms_from_chunk(self, chunk: RetrievalChunk) -> list[RetrievalChunkTerm]:
        terms = [
            *(RetrievalChunkTerm(chunk.chunk_id, term, "content", 1.0) for term in chunk.content_ltks.split()),
            *(RetrievalChunkTerm(chunk.chunk_id, term, "title", 2.0) for term in chunk.title_tks.split()),
            *(RetrievalChunkTerm(chunk.chunk_id, term, "important", 3.0) for term in chunk.important_kwd),
        ]
        unique: dict[tuple[str, str], RetrievalChunkTerm] = {}
        for term in terms:
            unique[(term.term, term.term_type)] = term
        return list(unique.values())

    def _type_samples(self, entities: list[RetrievalKgEntity]) -> list[RetrievalTypeSamples]:
        samples: dict[str, list[str]] = defaultdict(list)
        for entity in entities:
            samples[entity.entity_type].append(entity.entity_name)
        updated_at = datetime.now(UTC).isoformat()
        return [
            RetrievalTypeSamples(
                entity_type=entity_type,
                sample_entities=_unique(names)[:20],
                sample_count=len(_unique(names)),
                updated_at=updated_at,
            )
            for entity_type, names in sorted(samples.items())
        ]

    def _is_eligible_chunk(self, chunk: DocumentChunk) -> bool:
        token_count = chunk.token_count or _estimate_tokens(chunk.content)
        return self.min_chunk_tokens <= token_count <= self.max_chunk_tokens and bool(chunk.content.strip())


def _estimate_tokens(text: str) -> int:
    chinese_chars = len([char for char in text if "\u4e00" <= char <= "\u9fff"])
    ascii_terms = len([term for term in text.split() if term])
    return chinese_chars + ascii_terms


def _candidate_evidence_lookup(
    entities_with_source: list[tuple[str, EntityCandidate]],
    relations_with_source: list[tuple[str, RelationCandidate]],
) -> CandidateEvidenceLookup:
    entities_by_id = {
        (source_id, entity.entity_id): entity
        for source_id, entity in entities_with_source
    }
    entity_chunks: dict[tuple[str, str], list[str]] = defaultdict(list)
    relation_chunks: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for _source_id, entity in entities_with_source:
        entity_chunks[(entity.name, entity.label)].extend(entity.source_chunk_ids)
        entity_chunks[(entity.name, "")].extend(entity.source_chunk_ids)
    for source_id, relation in relations_with_source:
        source_entity = entities_by_id.get((source_id, relation.source_entity_id))
        target_entity = entities_by_id.get((source_id, relation.target_entity_id))
        if not source_entity or not target_entity:
            continue
        relation_chunks[
            (source_entity.name, target_entity.name, relation.relation)
        ].extend(relation.evidence_chunk_ids)
    return CandidateEvidenceLookup(
        entity_chunks={
            key: _unique(chunk_ids)
            for key, chunk_ids in entity_chunks.items()
        },
        relation_chunks={
            key: _unique(chunk_ids)
            for key, chunk_ids in relation_chunks.items()
        },
    )


def _source_chunks_from_graph_node(node: GraphNode) -> list[str]:
    value = node.properties.get("source_chunks")
    if isinstance(value, list):
        return _string_list(value)
    if isinstance(value, str):
        return _string_list(value.split(","))
    return []


def _chunk_ids_from_evidence_ids(evidence_ids: list[str]) -> list[str]:
    chunk_ids = []
    for evidence_id in evidence_ids:
        if evidence_id.startswith("chunk:"):
            chunk_ids.append(evidence_id)
        elif evidence_id.startswith("evidence:"):
            chunk_ids.append(evidence_id.replace("evidence:", "chunk:", 1))
    return chunk_ids


def _community_source_ids(graph_service: GraphService, community_id: int) -> list[str]:
    source_ids: list[str] = []
    node_ids = {
        node.id
        for node in graph_service.nodes
        if int(node.properties.get("community_id", -1)) == community_id
    }
    for node in graph_service.nodes:
        if node.id in node_ids:
            source_ids.extend(_source_chunks_from_graph_node(node))
    for edge in graph_service.edges:
        if edge.source in node_ids and edge.target in node_ids:
            source_ids.extend(edge.evidence_ids)
    return _unique(source_ids)


def _graph_source_ids(nodes: list[GraphNode], edges) -> list[str]:
    source_ids: list[str] = []
    for node in nodes:
        source_ids.extend(_source_groups_from_node(node))
    for edge in edges:
        source_ids.extend(_source_groups_from_evidence(edge.evidence_ids))
    return _unique(source_ids)


def _subgraph_groups(graph_service: GraphService) -> dict[str, dict]:
    nodes_by_id = {node.id: node for node in graph_service.nodes}
    edges_by_id = {edge.id: edge for edge in graph_service.edges}
    grouped_node_ids: dict[str, set[str]] = defaultdict(set)
    grouped_edge_ids: dict[str, set[str]] = defaultdict(set)

    for node in graph_service.nodes:
        for source_id in _source_groups_from_node(node):
            grouped_node_ids[source_id].add(node.id)

    for edge in graph_service.edges:
        source_ids = _source_groups_from_evidence(edge.evidence_ids)
        if not source_ids and edge.source in nodes_by_id and edge.target in nodes_by_id:
            source_ids = sorted(
                set(_source_groups_from_node(nodes_by_id[edge.source]))
                | set(_source_groups_from_node(nodes_by_id[edge.target]))
            )
        for source_id in source_ids:
            grouped_edge_ids[source_id].add(edge.id)
            grouped_node_ids[source_id].add(edge.source)
            grouped_node_ids[source_id].add(edge.target)

    groups: dict[str, dict] = {}
    for source_id, node_ids in grouped_node_ids.items():
        nodes = [
            nodes_by_id[node_id]
            for node_id in sorted(node_ids)
            if node_id in nodes_by_id
        ]
        edge_ids = grouped_edge_ids.get(source_id, set())
        edges = [
            edges_by_id[edge_id]
            for edge_id in sorted(edge_ids)
            if edge_id in edges_by_id
            and edges_by_id[edge_id].source in node_ids
            and edges_by_id[edge_id].target in node_ids
        ]
        if nodes or edges:
            groups[source_id] = {"nodes": nodes, "edges": edges}
    return groups


def _graph_payload(nodes: list[GraphNode], edges) -> dict:
    return {
        "nodes": [node.model_dump() for node in nodes],
        "edges": [edge.model_dump() for edge in edges],
    }


def _source_groups_from_node(node: GraphNode) -> list[str]:
    return _source_groups_from_evidence(_source_chunks_from_graph_node(node))


def _source_groups_from_evidence(evidence_ids: list[str]) -> list[str]:
    return _unique([_source_group(str(value)) for value in evidence_ids if str(value).strip()])


def _source_group(value: str) -> str:
    parts = value.split(":")
    if len(parts) >= 3 and parts[0] in {"chunk", "evidence"}:
        if parts[1] == "source" and len(parts) >= 4:
            return ":".join(parts[1:-1])
        return parts[1]
    return value


def _n_hop_paths_from_graph(
    graph_service: GraphService,
    *,
    max_hops: int = 2,
    max_paths_per_node: int = 40,
) -> dict[str, list[dict]]:
    nodes_by_id = {node.id: node for node in graph_service.nodes}
    names_by_id = {node.id: node.name for node in graph_service.nodes}
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in graph_service.edges:
        if edge.source not in nodes_by_id or edge.target not in nodes_by_id:
            continue
        weight = GraphAnalyticsService.retrieval_edge_weight(edge)
        adjacency[edge.source].append((edge.target, weight))
        adjacency[edge.target].append((edge.source, weight))
    for node_id, neighbors in adjacency.items():
        neighbors.sort(key=lambda item: (-item[1], names_by_id.get(item[0], item[0]), item[0]))

    result: dict[str, list[dict]] = {}
    for node_id in nodes_by_id:
        paths: list[dict] = []

        def walk(current_id: str, path: list[str], weights: list[float]) -> None:
            if len(weights) >= max_hops:
                return
            for next_id, weight in adjacency.get(current_id, []):
                if next_id in path:
                    continue
                next_path = [*path, next_id]
                next_weights = [*weights, weight]
                paths.append(
                    {
                        "path": [names_by_id.get(path_id, path_id) for path_id in next_path],
                        "weights": [round(value, 6) for value in next_weights],
                    }
                )
                walk(next_id, next_path, next_weights)

        walk(node_id, [node_id], [])
        paths.sort(
            key=lambda item: (
                -sum(float(value) for value in item["weights"]),
                len(item["path"]),
                item["path"],
            )
        )
        result[node_id] = paths[:max_paths_per_node]
    return result


def _graph_entity_content(node: GraphNode) -> dict:
    description_parts = [node.name, node.label, node.description]
    aliases = _string_list(node.properties.get("aliases", []))
    if aliases:
        description_parts.append("别名：" + "、".join(aliases))
    community_summary = str(node.properties.get("community_summary") or "")
    if community_summary:
        description_parts.append(community_summary)
    return {
        "description": " ".join(part for part in description_parts if part),
        "source": "neo4j",
        "canonical_id": str(node.properties.get("canonical_id") or node.id),
        "canonical_name": str(node.properties.get("canonical_name") or node.name),
        "canonical_label": str(node.properties.get("canonical_label") or node.label),
        "aliases": aliases,
        "community_id": node.properties.get("community_id", 0),
        "community_title": str(node.properties.get("community_title") or ""),
        "community_summary": community_summary,
    }


def _float_property(properties: dict, *keys: str, fallback: float) -> float:
    for key in keys:
        value = properties.get(key)
        if isinstance(value, int | float):
            return float(value)
    return fallback


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
