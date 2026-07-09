from __future__ import annotations

from dataclasses import dataclass

from app.services.model_clients import RerankClient
from app.services.ragflow_compat.doc_store import ChunkSearchHit, RagflowDocStore
from app.services.ragflow_compat.query import tokenize_query


@dataclass(frozen=True)
class RankedChunkHit:
    chunk: object
    score: float
    keyword_score: float
    vector_score: float
    rerank_score: float


@dataclass(frozen=True)
class FulltextRetrievalResult:
    keywords: list[str]
    hits: list[RankedChunkHit]


class RagflowFulltextRetriever:
    def __init__(
        self,
        *,
        doc_store: RagflowDocStore,
        rerank_client: RerankClient | None = None,
        rerank_weight: float = 0.5,
    ):
        self.doc_store = doc_store
        self.rerank_client = rerank_client
        self.rerank_weight = rerank_weight

    def retrieve(self, question: str, *, top_k: int = 8) -> FulltextRetrievalResult:
        keywords = tokenize_query(question)
        candidates = self.doc_store.search_chunks(
            question,
            keywords,
            top_k=max(top_k * 4, top_k),
        )
        rerank_scores = self._rerank(question, candidates)
        ranked_hits = [
            RankedChunkHit(
                chunk=hit.chunk,
                score=hit.score * (1 - self.rerank_weight)
                + rerank_scores[index] * self.rerank_weight,
                keyword_score=hit.keyword_score,
                vector_score=hit.vector_score,
                rerank_score=rerank_scores[index],
            )
            for index, hit in enumerate(candidates)
        ]
        ranked_hits.sort(key=lambda hit: hit.score, reverse=True)
        return FulltextRetrievalResult(keywords=keywords, hits=ranked_hits[:top_k])

    def _rerank(self, question: str, hits: list[ChunkSearchHit]) -> list[float]:
        if not hits:
            return []
        if not self.rerank_client:
            return [hit.score for hit in hits]
        try:
            ranking = self.rerank_client.rerank(
                question,
                [hit.chunk.content_with_weight for hit in hits],
            )
        except Exception:
            return [hit.score for hit in hits]
        scores = [0.0 for _ in hits]
        for index, score in ranking:
            if 0 <= index < len(scores):
                scores[index] = float(score)
        return scores
