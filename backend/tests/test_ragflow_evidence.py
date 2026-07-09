from app.models.graph import GraphEdge
from app.services.ragflow_compat.evidence import assemble_evidence_cards
from app.services.ragflow_compat.fulltext import RankedChunkHit
from app.services.ragflow_compat.schemas import RetrievalChunk


def test_assemble_evidence_cards_deduplicates_chunks_and_kg_relation_evidence():
    chunk = RetrievalChunk(
        chunk_id="chunk:source:abc:0001",
        doc_id="source:abc",
        source_id="source:abc",
        parent_unit_id="unit:1",
        chunk_order_int=1,
        page_num_int=1,
        title="不寐",
        section_path=[],
        content="失眠可辨为心脾两虚。",
        content_with_weight="不寐 失眠可辨为心脾两虚。",
        token_count=12,
    )
    hit = RankedChunkHit(
        chunk=chunk,
        score=1.0,
        keyword_score=1.0,
        vector_score=0.0,
        rerank_score=1.0,
    )
    edge = GraphEdge(
        id="edge:1",
        source="symptom:失眠",
        target="syndrome:心脾两虚",
        relation="MANIFESTS_AS",
        display="可辨为",
        evidence_ids=["chunk:source:abc:0001"],
    )

    cards = assemble_evidence_cards([hit], [edge])

    assert len(cards) == 1
    assert cards[0].id == "evidence:source:abc:0001"
    assert cards[0].title == "不寐"
    assert cards[0].snippet == "失眠可辨为心脾两虚。"
