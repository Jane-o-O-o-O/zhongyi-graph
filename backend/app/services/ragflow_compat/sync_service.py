from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from app.models.graph import GraphNode
from app.models.ingestion import DocumentChunk, EntityCandidate, RelationCandidate, SourceManifest
from app.services.graph_service import GraphService
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.query import build_content_with_weight, tokenize_query
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import (
    RetrievalChunk,
    RetrievalChunkTerm,
    RetrievalDocument,
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
    ):
        self.ingestion_repository = ingestion_repository
        self.retrieval_repository = retrieval_repository
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_batch_size = chunk_batch_size
        self.build_chunk_terms = build_chunk_terms
        self.graph_service = graph_service

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
        if self.graph_service and self.graph_service.nodes:
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
        else:
            retrieval_entities = [
                self._entity_from_candidate(source_id, entity)
                for source_id, entity in entities_with_source
            ]
            retrieval_relations = [
                self._relation_from_candidate(source_id, relation, entity_lookup)
                for source_id, relation in relations_with_source
            ]
        retrieval_type_samples = self._type_samples(retrieval_entities)

        self.retrieval_repository.clear_rebuild_tables()
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
        self.retrieval_repository.append_type_samples(retrieval_type_samples)
        return {
            "documents": len(retrieval_documents),
            "chunks": chunk_count,
            "kg_entities": len(retrieval_entities),
            "kg_relations": len(retrieval_relations),
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
        adjacency: dict[str, list[dict]] = defaultdict(list)
        names_by_id = {node.id: node.name for node in graph_service.nodes}
        for edge in graph_service.edges:
            source = names_by_id.get(edge.source, edge.source)
            target = names_by_id.get(edge.target, edge.target)
            adjacency[edge.source].append(
                {"path": [source, target], "weights": [1.0]}
            )
            adjacency[edge.target].append(
                {"path": [target, source], "weights": [1.0]}
            )
        return [
            RetrievalKgEntity(
                entity_id=node.id,
                entity_name=node.name,
                entity_type=node.label,
                source_node_id=node.id,
                content_with_weight=json.dumps(
                    {
                        "description": " ".join(
                            part for part in [node.name, node.label, node.description] if part
                        ),
                        "source": "neo4j",
                    },
                    ensure_ascii=False,
                ),
                description=node.description or f"{node.name} {node.label}",
                rank_flt=float(node.properties.get("rank", 1.0))
                if isinstance(node.properties.get("rank", 1.0), int | float)
                else 1.0,
                n_hop_with_weight=adjacency.get(node.id, [])[:20],
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
                    weight_int=1,
                    evidence_chunk_ids=_unique(
                        _chunk_ids_from_evidence_ids(edge.evidence_ids)
                        + candidate_evidence
                    ),
                    source_edge_id=edge.id,
                    metadata={"source": "neo4j"},
                )
            )
        return relations

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
