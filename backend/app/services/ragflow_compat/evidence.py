from __future__ import annotations

from app.models.graph import EvidenceCard, GraphEdge


def assemble_evidence_cards(chunk_hits, graph_edges: list[GraphEdge]) -> list[EvidenceCard]:
    cards: dict[str, EvidenceCard] = {}
    for hit in chunk_hits:
        chunk = hit.chunk
        evidence_id = _evidence_id(chunk.chunk_id)
        cards[evidence_id] = EvidenceCard(
            id=evidence_id,
            title=chunk.title or f"{chunk.source_id} #{chunk.chunk_order_int}",
            source=chunk.source_id,
            snippet=chunk.content,
            source_type="local",
            location=f"{chunk.source_id}:{chunk.page_num_int}:{chunk.chunk_order_int}",
        )

    for edge in graph_edges:
        for raw_id in edge.evidence_ids:
            evidence_id = _evidence_id(raw_id)
            if evidence_id in cards:
                continue
            cards[evidence_id] = EvidenceCard(
                id=evidence_id,
                title=edge.display,
                source=edge.id,
                snippet=f"{edge.source} {edge.display} {edge.target}",
                source_type="local",
                location=edge.id,
            )
    return list(cards.values())


def _evidence_id(raw_id: str) -> str:
    if raw_id.startswith("evidence:"):
        return raw_id
    if raw_id.startswith("chunk:"):
        return raw_id.replace("chunk:", "evidence:", 1)
    return f"evidence:{raw_id}"
