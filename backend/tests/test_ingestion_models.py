from app.models.ingestion import (
    DocumentChunk,
    DocumentPage,
    EntityCandidate,
    KnowledgeBundle,
    PublishBatch,
    RelationCandidate,
    SourceManifest,
)


def test_source_manifest_tracks_uploaded_document_status():
    manifest = SourceManifest(
        source_id="source:uploaded:abc",
        filename="资料.pdf",
        mime_type="application/pdf",
        checksum="abc123",
        status="registered",
    )

    assert manifest.source_id == "source:uploaded:abc"
    assert manifest.status == "registered"


def test_knowledge_bundle_tracks_chunks_candidates_and_publish_batch():
    source = SourceManifest(
        source_id="source:uploaded:abc",
        filename="资料.txt",
        mime_type="text/plain",
        checksum="abc123",
        status="parsed",
        object_key="sources/abc123/资料.txt",
    )
    page = DocumentPage(
        page_id="page:source:uploaded:abc:1",
        source_id=source.source_id,
        page_number=1,
        text="失眠可辨为心脾两虚，治以补益心脾。",
    )
    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:abc:0001",
        source_id=source.source_id,
        page_id=page.page_id,
        chunk_index=1,
        content="失眠可辨为心脾两虚，治以补益心脾。",
        content_type="text",
        token_count=18,
        char_start=0,
        char_end=18,
    )
    entity = EntityCandidate(
        entity_id="entity:symptom:失眠",
        name="失眠",
        label="Symptom",
        normalized_name="失眠",
        source_chunk_ids=[chunk.chunk_id],
        confidence=0.91,
    )
    relation = RelationCandidate(
        relation_id="relation:失眠:心脾两虚",
        source_entity_id="entity:symptom:失眠",
        target_entity_id="entity:syndrome:心脾两虚",
        relation="MANIFESTS_AS",
        display="可辨为",
        evidence_chunk_ids=[chunk.chunk_id],
        confidence=0.88,
    )
    batch = PublishBatch(
        batch_id="publish:1",
        source_ids=[source.source_id],
        status="published",
        node_count=2,
        edge_count=1,
        chunk_count=1,
    )

    bundle = KnowledgeBundle(
        source=source,
        pages=[page],
        chunks=[chunk],
        entities=[entity],
        relations=[relation],
        publish_batch=batch,
    )

    assert bundle.chunks[0].source_id == source.source_id
    assert bundle.relations[0].evidence_chunk_ids == [chunk.chunk_id]
    assert bundle.publish_batch.node_count == 2
