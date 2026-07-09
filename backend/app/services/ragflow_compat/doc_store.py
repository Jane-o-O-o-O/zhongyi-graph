from __future__ import annotations

from dataclasses import dataclass
import math

import httpx

from app.services.model_clients import EmbeddingClient
from app.services.ragflow_compat.query import (
    candidate_search_keywords,
    token_similarity,
    tokenize_query,
)
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalChunk, RetrievalKgEntity, RetrievalKgRelation


@dataclass(frozen=True)
class ChunkSearchHit:
    chunk: RetrievalChunk
    score: float
    keyword_score: float
    vector_score: float

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass(frozen=True)
class EntitySearchHit:
    entity: RetrievalKgEntity
    score: float


@dataclass(frozen=True)
class RelationSearchHit:
    relation: RetrievalKgRelation
    score: float


class RagflowVectorSearchClient:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        qdrant_url: str,
        collection: str,
        http_client: httpx.Client | None = None,
    ):
        self.embedding_client = embedding_client
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection = collection
        self.http_client = http_client or httpx.Client(timeout=10)

    def search_chunks(self, query: str, *, top_k: int) -> dict[str, float]:
        return self._search_ids(query, top_k=top_k, content_type="chunk", payload_key="chunk_id")

    def search_entities(self, query: str, *, top_k: int) -> dict[str, float]:
        return self._search_ids(
            query,
            top_k=top_k,
            content_type="kg_entity",
            payload_key="entity_id",
        )

    def search_relations(self, query: str, *, top_k: int) -> dict[str, float]:
        return self._search_ids(
            query,
            top_k=top_k,
            content_type="kg_relation",
            payload_key="relation_id",
        )

    def _search_ids(
        self,
        query: str,
        *,
        top_k: int,
        content_type: str,
        payload_key: str,
    ) -> dict[str, float]:
        if not self.qdrant_url or top_k <= 0:
            return {}
        try:
            query_vector = self.embedding_client.embed([query])[0]
            response = self.http_client.post(
                f"{self.qdrant_url}/collections/{self.collection}/points/search",
                json={
                    "vector": query_vector,
                    "limit": top_k,
                    "with_payload": True,
                    "filter": {
                        "must": [{"key": "content_type", "match": {"value": content_type}}]
                    },
                },
            )
            response.raise_for_status()
        except Exception:
            return {}
        scores: dict[str, float] = {}
        for point in response.json().get("result", []):
            payload = point.get("payload") or {}
            item_id = str(payload.get(payload_key) or "")
            if not item_id:
                continue
            scores[item_id] = float(point.get("score", 0.0))
        return scores


class RagflowDocStore:
    def __init__(
        self,
        repository: RagflowRetrievalRepository,
        embedding_client: EmbeddingClient | None = None,
        *,
        vector_weight: float = 0.7,
        token_weight: float = 0.3,
        vector_search_client: RagflowVectorSearchClient | None = None,
        min_vector_chunks_for_search: int = 0,
        min_vector_kg_entities_for_search: int = 0,
        min_vector_kg_relations_for_search: int = 0,
        min_vector_chunk_coverage_for_search: float = 0.0,
        min_vector_kg_entity_coverage_for_search: float = 0.0,
        min_vector_kg_relation_coverage_for_search: float = 0.0,
    ):
        self.repository = repository
        self.embedding_client = embedding_client
        self.vector_weight = vector_weight
        self.token_weight = token_weight
        self.vector_search_client = vector_search_client
        self.min_vector_chunks_for_search = min_vector_chunks_for_search
        self.min_vector_kg_entities_for_search = min_vector_kg_entities_for_search
        self.min_vector_kg_relations_for_search = min_vector_kg_relations_for_search
        self.min_vector_chunk_coverage_for_search = min_vector_chunk_coverage_for_search
        self.min_vector_kg_entity_coverage_for_search = (
            min_vector_kg_entity_coverage_for_search
        )
        self.min_vector_kg_relation_coverage_for_search = (
            min_vector_kg_relation_coverage_for_search
        )

    def search_chunks(
        self,
        query: str,
        keywords: list[str],
        *,
        top_k: int = 10,
        candidates: int = 80,
    ) -> list[ChunkSearchHit]:
        search_keywords = keywords or tokenize_query(query)
        lexical_chunks = self.repository.search_chunk_candidates(
            candidate_search_keywords(search_keywords),
            limit=candidates,
        )
        vector_scores_by_chunk_id = self._qdrant_chunk_scores(query, top_k=candidates)
        vector_chunks = (
            self.repository.get_chunks_by_ids(list(vector_scores_by_chunk_id))
            if vector_scores_by_chunk_id
            else []
        )
        chunks = _merge_chunks(lexical_chunks, vector_chunks)
        if not chunks:
            return []
        keyword_query = " ".join(search_keywords)
        all_keyword_scores = token_similarity(
            keyword_query,
            [chunk.content_ltks or chunk.content_with_weight for chunk in chunks],
        )
        lexical_hits = [
            (chunk, all_keyword_scores[index])
            for index, chunk in enumerate(chunks)
        ]
        lexical_hits.sort(
            key=lambda item: (
                item[1] + _literal_bonus(keywords, item[0].content_with_weight),
                -item[0].chunk_order_int,
            ),
            reverse=True,
        )
        lexical_hits = [
            (chunk, score)
            for chunk, score in lexical_hits
            if (
                score > 1e-6
                or _literal_bonus(keywords, chunk.content_with_weight) > 0
                or chunk.chunk_id in vector_scores_by_chunk_id
            )
        ]
        candidate_hits = lexical_hits
        candidate_chunks = [chunk for chunk, _score in candidate_hits]
        keyword_scores = [score for _chunk, score in candidate_hits]
        vector_scores = self._chunk_vector_scores(
            query,
            candidate_chunks,
            vector_scores_by_chunk_id,
        )
        hits = [
            ChunkSearchHit(
                chunk=chunk,
                score=keyword_scores[index] * self.token_weight
                + vector_scores[index] * self.vector_weight
                + _literal_bonus(keywords, chunk.content_with_weight),
                keyword_score=keyword_scores[index],
                vector_score=vector_scores[index],
            )
            for index, chunk in enumerate(candidate_chunks)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def search_entities(
        self,
        query: str,
        keywords: list[str],
        *,
        top_k: int = 56,
        sim_threshold: float = 0.0,
    ) -> list[EntitySearchHit]:
        entities = self.repository.list_kg_entities(available_only=True)
        vector_scores_by_entity_id = self._qdrant_entity_scores(query, top_k=top_k * 4)
        vector_entities = (
            self.repository.get_kg_entities_by_ids(list(vector_scores_by_entity_id))
            if vector_scores_by_entity_id
            else []
        )
        entities = _merge_entities(entities, vector_entities)
        if not entities:
            return []
        entities = _prefilter_entities(
            entities,
            keywords,
            top_k * 4,
            force_ids=set(vector_scores_by_entity_id),
        )
        scores = self._hybrid_scores(
            query,
            keywords,
            [entity.content_with_weight for entity in entities],
            [entity.vector_status for entity in entities],
            [
                vector_scores_by_entity_id.get(entity.entity_id)
                for entity in entities
            ],
        )
        hits = [
            EntitySearchHit(entity=entity, score=score)
            for entity, score in zip(entities, scores, strict=True)
            if score >= sim_threshold
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def search_relations(
        self,
        query: str,
        *,
        top_k: int = 56,
        sim_threshold: float = 0.0,
    ) -> list[RelationSearchHit]:
        relations = self.repository.list_kg_relations(available_only=True)
        vector_scores_by_relation_id = self._qdrant_relation_scores(query, top_k=top_k * 4)
        vector_relations = (
            self.repository.get_kg_relations_by_ids(list(vector_scores_by_relation_id))
            if vector_scores_by_relation_id
            else []
        )
        relations = _merge_relations(relations, vector_relations)
        if not relations:
            return []
        query_keywords = tokenize_query(query)
        relations = _prefilter_relations(
            relations,
            query_keywords,
            top_k * 4,
            force_ids=set(vector_scores_by_relation_id),
        )
        scores = self._hybrid_scores(
            query,
            query_keywords,
            [relation.content_with_weight for relation in relations],
            [relation.vector_status for relation in relations],
            [
                vector_scores_by_relation_id.get(relation.relation_id)
                for relation in relations
            ],
        )
        hits = [
            RelationSearchHit(relation=relation, score=score)
            for relation, score in zip(relations, scores, strict=True)
            if score >= sim_threshold
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _hybrid_scores(
        self,
        query: str,
        keywords: list[str],
        documents: list[str],
        vector_statuses: list[str],
        preset_vector_scores: list[float | None] | None = None,
    ) -> list[float]:
        keyword_scores = token_similarity(" ".join(keywords), [tokenize_query(doc) for doc in documents])
        vector_scores = self._status_vector_scores(
            query,
            documents,
            vector_statuses,
            preset_vector_scores=preset_vector_scores,
        )
        return [
            keyword_scores[index] * self.token_weight + vector_scores[index] * self.vector_weight
            for index in range(len(documents))
        ]

    def _vector_scores(self, query: str, documents: list[str]) -> list[float]:
        if not self.embedding_client or not documents:
            return [0.0 for _ in documents]
        try:
            vectors = self.embedding_client.embed([query, *documents])
        except Exception:
            return [0.0 for _ in documents]
        query_vector = vectors[0]
        return [_cosine_similarity(query_vector, vector) for vector in vectors[1:]]

    def _status_vector_scores(
        self,
        query: str,
        documents: list[str],
        vector_statuses: list[str],
        preset_vector_scores: list[float | None] | None = None,
    ) -> list[float]:
        preset_vector_scores = preset_vector_scores or [None for _document in documents]
        vector_scores = [
            float(score) if score is not None else 0.0
            for score in preset_vector_scores
        ]
        embedded_positions = [
            index
            for index, status in enumerate(vector_statuses)
            if status == "embedded" and preset_vector_scores[index] is None
        ]
        if not embedded_positions:
            return vector_scores
        embedded_scores = self._vector_scores(
            query,
            [documents[index] for index in embedded_positions],
        )
        for position, score in zip(embedded_positions, embedded_scores, strict=True):
            vector_scores[position] = score
        return vector_scores

    def _chunk_vector_scores(
        self,
        query: str,
        chunks: list[RetrievalChunk],
        vector_scores_by_chunk_id: dict[str, float] | None = None,
    ) -> list[float]:
        vector_scores_by_chunk_id = vector_scores_by_chunk_id or {}
        vector_scores = [
            vector_scores_by_chunk_id.get(chunk.chunk_id, 0.0)
            for chunk in chunks
        ]
        missing_embedded_positions = [
            index
            for index, chunk in enumerate(chunks)
            if chunk.vector_status == "embedded" and chunk.chunk_id not in vector_scores_by_chunk_id
        ]
        if not missing_embedded_positions:
            return vector_scores
        fallback_scores = self._status_vector_scores(
            query,
            [chunks[index].content_with_weight for index in missing_embedded_positions],
            [chunks[index].vector_status for index in missing_embedded_positions],
        )
        for position, score in zip(missing_embedded_positions, fallback_scores, strict=True):
            vector_scores[position] = score
        return vector_scores

    def _qdrant_chunk_scores(self, query: str, *, top_k: int) -> dict[str, float]:
        if not self.vector_search_client:
            return {}
        if not self._has_enough_chunk_vectors_for_search():
            return {}
        return self.vector_search_client.search_chunks(query, top_k=top_k)

    def _qdrant_entity_scores(self, query: str, *, top_k: int) -> dict[str, float]:
        if not self.vector_search_client:
            return {}
        if not self._has_enough_kg_entity_vectors_for_search():
            return {}
        return self.vector_search_client.search_entities(query, top_k=top_k)

    def _qdrant_relation_scores(self, query: str, *, top_k: int) -> dict[str, float]:
        if not self.vector_search_client:
            return {}
        if not self._has_enough_kg_relation_vectors_for_search():
            return {}
        return self.vector_search_client.search_relations(query, top_k=top_k)

    def _has_enough_chunk_vectors_for_search(self) -> bool:
        if (
            self.min_vector_chunks_for_search <= 0
            and self.min_vector_chunk_coverage_for_search <= 0
        ):
            return True
        try:
            audit = self.repository.audit()
        except Exception:
            return False
        return _meets_vector_search_gate(
            embedded=audit.chunks_with_vectors,
            total=audit.chunks,
            min_indexed=self.min_vector_chunks_for_search,
            min_coverage=self.min_vector_chunk_coverage_for_search,
        )

    def _has_enough_kg_entity_vectors_for_search(self) -> bool:
        if (
            self.min_vector_kg_entities_for_search <= 0
            and self.min_vector_kg_entity_coverage_for_search <= 0
        ):
            return True
        try:
            audit = self.repository.audit()
        except Exception:
            return False
        return _meets_vector_search_gate(
            embedded=audit.kg_entities_with_vectors,
            total=audit.kg_entities,
            min_indexed=self.min_vector_kg_entities_for_search,
            min_coverage=self.min_vector_kg_entity_coverage_for_search,
        )

    def _has_enough_kg_relation_vectors_for_search(self) -> bool:
        if (
            self.min_vector_kg_relations_for_search <= 0
            and self.min_vector_kg_relation_coverage_for_search <= 0
        ):
            return True
        try:
            audit = self.repository.audit()
        except Exception:
            return False
        return _meets_vector_search_gate(
            embedded=audit.kg_relations_with_vectors,
            total=audit.kg_relations,
            min_indexed=self.min_vector_kg_relations_for_search,
            min_coverage=self.min_vector_kg_relation_coverage_for_search,
        )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return numerator / (left_norm * right_norm)


def _literal_bonus(keywords: list[str], text: str) -> float:
    if not keywords:
        return 0.0
    compact = text.replace(" ", "")
    return sum(0.1 for keyword in keywords if keyword and keyword.replace(" ", "") in compact)


def _meets_vector_search_gate(
    *,
    embedded: int,
    total: int,
    min_indexed: int,
    min_coverage: float,
) -> bool:
    if min_indexed > 0 and embedded < min_indexed:
        return False
    if min_coverage > 0:
        if total <= 0:
            return False
        if embedded / total < min_coverage:
            return False
    return True


def _prefilter_entities(
    entities: list[RetrievalKgEntity],
    keywords: list[str],
    limit: int,
    *,
    force_ids: set[str] | None = None,
) -> list[RetrievalKgEntity]:
    force_ids = force_ids or set()
    if not keywords:
        return entities[:limit]
    scored = [
        (entity, _literal_bonus(keywords, f"{entity.entity_name} {entity.content_with_weight}"))
        for entity in entities
    ]
    scored = [
        (entity, score)
        for entity, score in scored
        if score > 0 or entity.entity_id in force_ids
    ]
    scored.sort(key=lambda item: (item[1], item[0].rank_flt), reverse=True)
    return [entity for entity, _score in scored[:limit]]


def _prefilter_relations(
    relations: list[RetrievalKgRelation],
    keywords: list[str],
    limit: int,
    *,
    force_ids: set[str] | None = None,
) -> list[RetrievalKgRelation]:
    force_ids = force_ids or set()
    if not keywords:
        return relations[:limit]
    scored = [
        (relation, _literal_bonus(keywords, relation.content_with_weight))
        for relation in relations
    ]
    scored = [
        (relation, score)
        for relation, score in scored
        if score > 0 or relation.relation_id in force_ids
    ]
    scored.sort(key=lambda item: (item[1], item[0].weight_int), reverse=True)
    return [relation for relation, _score in scored[:limit]]


def _merge_chunks(
    lexical_chunks: list[RetrievalChunk],
    vector_chunks: list[RetrievalChunk],
) -> list[RetrievalChunk]:
    chunks: dict[str, RetrievalChunk] = {}
    for chunk in [*lexical_chunks, *vector_chunks]:
        chunks.setdefault(chunk.chunk_id, chunk)
    return list(chunks.values())


def _merge_entities(
    lexical_entities: list[RetrievalKgEntity],
    vector_entities: list[RetrievalKgEntity],
) -> list[RetrievalKgEntity]:
    entities: dict[str, RetrievalKgEntity] = {}
    for entity in [*lexical_entities, *vector_entities]:
        entities.setdefault(entity.entity_id, entity)
    return list(entities.values())


def _merge_relations(
    lexical_relations: list[RetrievalKgRelation],
    vector_relations: list[RetrievalKgRelation],
) -> list[RetrievalKgRelation]:
    relations: dict[str, RetrievalKgRelation] = {}
    for relation in [*lexical_relations, *vector_relations]:
        relations.setdefault(relation.relation_id, relation)
    return list(relations.values())
