from app.models.ingestion import DocumentChunk, EntityCandidate, RelationCandidate, SourceManifest
from app.services.ingestion_repository import IngestionRepository
from app.services.ingestion_service import IngestionService


def test_ingestion_service_restores_published_artifact_from_repository():
    repository = IngestionRepository.in_memory()
    source = SourceManifest(
        source_id="source:uploaded:published",
        filename="published.txt",
        mime_type="text/plain",
        checksum="published",
        status="published",
        object_key="sources/published/published.txt",
    )
    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:published:0001",
        source_id=source.source_id,
        page_id="page:source:uploaded:published:1",
        chunk_index=1,
        content="不寐可辨为心脾两虚。",
    )
    repository.upsert_source(source)
    repository.replace_pages_and_chunks(source.source_id, [], [chunk])
    repository.save_candidates(
        source.source_id,
        [
            EntityCandidate(
                entity_id="entity:symptom:不寐",
                name="不寐",
                label="Symptom",
                normalized_name="不寐",
                source_chunk_ids=[chunk.chunk_id],
            ),
            EntityCandidate(
                entity_id="entity:syndrome:心脾两虚",
                name="心脾两虚",
                label="Syndrome",
                normalized_name="心脾两虚",
                source_chunk_ids=[chunk.chunk_id],
            ),
        ],
        [
            RelationCandidate(
                relation_id="relation:不寐:心脾两虚",
                source_entity_id="entity:symptom:不寐",
                target_entity_id="entity:syndrome:心脾两虚",
                relation="MANIFESTS_AS",
                display="可辨为",
                evidence_chunk_ids=[chunk.chunk_id],
            )
        ],
    )

    artifact = IngestionService(repository=repository).restore_published_artifact()

    assert {node.name for node in artifact.nodes} == {"不寐", "心脾两虚"}
    assert artifact.edges[0].source == "symptom:不寐"
    assert artifact.evidence[0].snippet == "不寐可辨为心脾两虚。"
