from app.models.ingestion import DocumentChunk
from scripts.sync_chunk_vectors import build_chunk_payload, iter_batches, qdrant_point_id


def test_build_chunk_payload_preserves_chunk_identity_and_text():
    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:abc:0001",
        source_id="source:uploaded:abc",
        page_id="page:source:uploaded:abc:1",
        chunk_index=1,
        content="归脾汤可用于心脾两虚所致不寐。",
    )

    payload = build_chunk_payload(chunk)

    assert payload.id == "chunk:chunk:source:uploaded:abc:0001"
    assert payload.content_type == "chunk"
    assert payload.chunk_id == chunk.chunk_id
    assert payload.text == "归脾汤可用于心脾两虚所致不寐。"


def test_qdrant_point_id_is_stable_uuid():
    first = qdrant_point_id("chunk:chunk:source:uploaded:abc:0001")
    second = qdrant_point_id("chunk:chunk:source:uploaded:abc:0001")

    assert first == second
    assert len(first) == 36


def test_iter_batches_splits_without_dropping_tail():
    assert list(iter_batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
