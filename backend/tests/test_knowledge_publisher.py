from app.models.ingestion import (
    DocumentChunk,
    EntityCandidate,
    KnowledgeBundle,
    RelationCandidate,
    SourceManifest,
)
from app.services.knowledge_publisher import KnowledgePublisher


def test_knowledge_publisher_builds_graph_evidence_and_vector_payloads():
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
    entities = [
        EntityCandidate(
            entity_id="entity:symptom:失眠",
            name="失眠",
            label="Symptom",
            normalized_name="失眠",
            source_chunk_ids=[chunk.chunk_id],
        ),
        EntityCandidate(
            entity_id="entity:syndrome:心脾两虚",
            name="心脾两虚",
            label="Syndrome",
            normalized_name="心脾两虚",
            source_chunk_ids=[chunk.chunk_id],
        ),
    ]
    relations = [
        RelationCandidate(
            relation_id="relation:失眠:心脾两虚",
            source_entity_id="entity:symptom:失眠",
            target_entity_id="entity:syndrome:心脾两虚",
            relation="MANIFESTS_AS",
            display="可辨为",
            evidence_chunk_ids=[chunk.chunk_id],
        )
    ]
    bundle = KnowledgeBundle(
        source=source,
        chunks=[chunk],
        entities=entities,
        relations=relations,
    )

    artifact = KnowledgePublisher().build_artifact([bundle])

    assert [node.name for node in artifact.nodes] == ["失眠", "心脾两虚"]
    assert artifact.edges[0].source == "symptom:失眠"
    assert artifact.edges[0].target == "syndrome:心脾两虚"
    assert artifact.evidence[0].id == "evidence:source:uploaded:abc:0001"
    assert {payload.content_type for payload in artifact.vector_payloads} == {
        "chunk",
        "entity",
        "evidence",
    }
    chunk_payload = next(payload for payload in artifact.vector_payloads if payload.content_type == "chunk")
    assert chunk_payload.chunk_id == chunk.chunk_id
    assert chunk_payload.text == "失眠可辨为心脾两虚。"
