from app.models.ingestion import DocumentChunk, SourceManifest
from app.services.chunk_retriever import ChunkRetriever
from app.services.ingestion_repository import IngestionRepository


class FakeVectorIndex:
    def search(self, query, top_k=10, content_types=None):
        assert content_types == ["chunk", "evidence"]

        class Hit:
            payload = {
                "chunk_id": "chunk:source:uploaded:abc:0001",
                "evidence_id": "",
            }

        return [Hit()]


def test_chunk_retriever_hydrates_qdrant_chunk_hit_from_repository():
    repository = IngestionRepository.in_memory()
    source = SourceManifest(
        source_id="source:uploaded:abc",
        filename="资料.txt",
        mime_type="text/plain",
        checksum="abc123",
        status="parsed",
    )
    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:abc:0001",
        source_id=source.source_id,
        page_id="page:source:uploaded:abc:1",
        chunk_index=1,
        content="失眠可辨为心脾两虚。",
    )
    repository.upsert_source(source)
    repository.replace_pages_and_chunks(source.source_id, [], [chunk])

    chunks = ChunkRetriever(repository=repository, vector_index=FakeVectorIndex()).retrieve("睡不着", top_k=3)

    assert chunks == [chunk]
