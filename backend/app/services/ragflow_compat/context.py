from __future__ import annotations

from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalChunk


def expand_parent_context(
    repository: RagflowRetrievalRepository,
    chunk_ids: list[str],
    *,
    window: int = 1,
) -> list[RetrievalChunk]:
    if not chunk_ids:
        return []
    all_chunks = repository.list_chunks(available_only=True)
    selected = {chunk_id for chunk_id in chunk_ids}
    seed_chunks = [chunk for chunk in all_chunks if chunk.chunk_id in selected]
    if not seed_chunks:
        return []

    expanded: dict[str, RetrievalChunk] = {}
    for seed in seed_chunks:
        lower = seed.chunk_order_int - window
        upper = seed.chunk_order_int + window
        for chunk in all_chunks:
            if chunk.source_id != seed.source_id:
                continue
            if chunk.parent_unit_id != seed.parent_unit_id:
                continue
            if lower <= chunk.chunk_order_int <= upper:
                expanded[chunk.chunk_id] = chunk
    return sorted(
        expanded.values(),
        key=lambda chunk: (chunk.source_id, chunk.parent_unit_id, chunk.chunk_order_int),
    )
