from app.models.ingestion import DocumentChunk
from app.services.ingestion_repository import IngestionRepository
from app.services.vector_service import VectorIndexService


class ChunkRetriever:
    def __init__(self, repository: IngestionRepository, vector_index: VectorIndexService):
        self.repository = repository
        self.vector_index = vector_index

    def retrieve(self, question: str, top_k: int = 5) -> list[DocumentChunk]:
        try:
            hits = self.vector_index.search(question, top_k=top_k, content_types=["chunk", "evidence"])
        except Exception:
            return []

        chunks: list[DocumentChunk] = []
        seen: set[str] = set()
        for hit in hits:
            chunk_id = hit.payload.get("chunk_id", "")
            if not chunk_id:
                chunk_id = _chunk_id_from_evidence_id(hit.payload.get("evidence_id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            chunk = self.repository.get_chunk(chunk_id)
            if chunk:
                chunks.append(chunk)
                seen.add(chunk_id)
        return chunks


def _chunk_id_from_evidence_id(evidence_id: str) -> str:
    if evidence_id.startswith("evidence:"):
        return evidence_id.replace("evidence:", "chunk:", 1)
    return ""
