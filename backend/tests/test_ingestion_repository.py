from app.models.ingestion import (
    DocumentChunk,
    DocumentPage,
    EntityCandidate,
    RelationCandidate,
    SourceManifest,
)
from app.services.ingestion_repository import IngestionRepository


def test_repository_persists_source_pages_chunks_and_candidates():
    repository = IngestionRepository.in_memory()
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
        token_count=18,
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

    repository.upsert_source(source)
    repository.replace_pages_and_chunks(source.source_id, [page], [chunk])
    repository.save_candidates(source.source_id, [entity], [relation])
    bundle = repository.get_bundle(source.source_id)

    assert bundle.source.source_id == source.source_id
    assert bundle.pages == [page]
    assert bundle.chunks == [chunk]
    assert bundle.entities == [entity]
    assert bundle.relations == [relation]


def test_repository_allows_same_candidate_ids_from_different_sources():
    repository = IngestionRepository.in_memory()
    entity = EntityCandidate(
        entity_id="entity:symptom:失眠",
        name="失眠",
        label="Symptom",
        normalized_name="失眠",
        source_chunk_ids=["chunk:source:uploaded:first:0001"],
        confidence=0.91,
    )
    relation = RelationCandidate(
        relation_id="relation:失眠:心脾两虚",
        source_entity_id="entity:symptom:失眠",
        target_entity_id="entity:syndrome:心脾两虚",
        relation="MANIFESTS_AS",
        display="可辨为",
        evidence_chunk_ids=["chunk:source:uploaded:first:0001"],
        confidence=0.88,
    )
    first = SourceManifest(
        source_id="source:uploaded:first",
        filename="first.txt",
        mime_type="text/plain",
        checksum="first",
        status="parsed",
        object_key="sources/first/first.txt",
    )
    second = SourceManifest(
        source_id="source:uploaded:second",
        filename="second.txt",
        mime_type="text/plain",
        checksum="second",
        status="parsed",
        object_key="sources/second/second.txt",
    )

    repository.upsert_source(first)
    repository.upsert_source(second)
    repository.save_candidates(first.source_id, [entity], [relation])
    repository.save_candidates(second.source_id, [entity], [relation])

    assert repository.get_bundle(first.source_id).entities == [entity]
    assert repository.get_bundle(second.source_id).entities == [entity]


def test_repository_merges_candidates_without_replacing_existing_source_graph():
    repository = IngestionRepository.in_memory()
    source = SourceManifest(
        source_id="source:uploaded:merge",
        filename="merge.txt",
        mime_type="text/plain",
        checksum="merge",
        status="parsed",
        object_key="sources/merge/merge.txt",
    )
    insomnia = EntityCandidate(
        entity_id="entity:symptom:失眠",
        name="失眠",
        label="Symptom",
        normalized_name="失眠",
        source_chunk_ids=["chunk:1"],
        confidence=0.7,
    )
    headache = EntityCandidate(
        entity_id="entity:symptom:头痛",
        name="头痛",
        label="Symptom",
        normalized_name="头痛",
        source_chunk_ids=["chunk:2"],
        confidence=0.9,
    )
    headache_more_evidence = headache.model_copy(
        update={"source_chunk_ids": ["chunk:2", "chunk:3"], "confidence": 0.8}
    )
    relation = RelationCandidate(
        relation_id="relation:headache:syndrome",
        source_entity_id="entity:symptom:头痛",
        target_entity_id="entity:syndrome:心脾两虚",
        relation="MANIFESTS_AS",
        display="可辨为",
        evidence_chunk_ids=["chunk:2"],
        confidence=0.75,
    )
    relation_more_evidence = relation.model_copy(
        update={"evidence_chunk_ids": ["chunk:2", "chunk:3"], "confidence": 0.8}
    )

    repository.upsert_source(source)
    repository.save_candidates(source.source_id, [insomnia, headache], [relation])
    repository.merge_candidates(source.source_id, [headache_more_evidence], [relation_more_evidence])
    bundle = repository.get_bundle(source.source_id)

    entities_by_id = {entity.entity_id: entity for entity in bundle.entities}
    relations_by_id = {item.relation_id: item for item in bundle.relations}
    assert set(entities_by_id) == {"entity:symptom:失眠", "entity:symptom:头痛"}
    assert entities_by_id["entity:symptom:头痛"].source_chunk_ids == ["chunk:2", "chunk:3"]
    assert entities_by_id["entity:symptom:头痛"].confidence == 0.9
    assert relations_by_id["relation:headache:syndrome"].evidence_chunk_ids == ["chunk:2", "chunk:3"]
    assert relations_by_id["relation:headache:syndrome"].confidence == 0.8


def test_repository_records_publish_batch_counts():
    repository = IngestionRepository.in_memory()

    batch = repository.record_publish_batch(
        source_ids=["source:uploaded:abc"],
        node_count=3,
        edge_count=2,
        chunk_count=4,
    )

    assert batch.status == "published"
    assert batch.node_count == 3
    assert batch.edge_count == 2
    assert batch.chunk_count == 4


def test_repository_lists_published_source_ids_from_source_status():
    repository = IngestionRepository.in_memory()
    repository.upsert_source(
        SourceManifest(
            source_id="source:uploaded:published",
            filename="published.txt",
            mime_type="text/plain",
            checksum="published",
            status="published",
            object_key="sources/published/published.txt",
        )
    )
    repository.upsert_source(
        SourceManifest(
            source_id="source:uploaded:parsed",
            filename="parsed.txt",
            mime_type="text/plain",
            checksum="parsed",
            status="parsed",
            object_key="sources/parsed/parsed.txt",
        )
    )

    assert repository.list_published_source_ids() == ["source:uploaded:published"]


def test_repository_generates_unique_publish_batch_ids():
    repository = IngestionRepository.in_memory()

    first = repository.record_publish_batch(["source:1"], node_count=1, edge_count=1, chunk_count=1)
    second = repository.record_publish_batch(["source:1"], node_count=1, edge_count=1, chunk_count=1)

    assert first.batch_id != second.batch_id
