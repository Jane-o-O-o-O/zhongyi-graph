from app.models.ingestion import (
    DocumentChunk,
    EntityCandidate,
    ExtractionUnit,
    RelationCandidate,
    SourceManifest,
)
from app.services.ingestion_repository import IngestionRepository
from app.services.ingestion_service import IngestionService
from app.services.object_storage import LocalObjectStorage


class RecordingExtractor:
    def __init__(self):
        self.received_chunks = []

    def extract(self, chunks):
        self.received_chunks = chunks
        return [], []


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


def test_ingestion_service_extracts_from_parent_units_and_stores_child_chunks(tmp_path):
    repository = IngestionRepository.in_memory()
    extractor = RecordingExtractor()
    service = IngestionService(
        repository=repository,
        storage=LocalObjectStorage(tmp_path),
        extractor=extractor,
    )
    source = service.upload_source(
        filename="普济方.txt",
        mime_type="text/plain",
        content="""
<篇名>普济方

<目录>卷一\\方脉总论

<篇名>五常大论

属性：头者诸阳之会。是以头痛多属于阳也。独厥阴肝脉，上入颃颡，连目系，上出额。
""".strip().encode("utf-8"),
    )
    job = service.create_job([source.source_id])

    result = service.run_job(job.job_id)
    bundle = repository.get_bundle(source.source_id)

    assert result["unit_count"] == 1
    assert result["chunk_count"] == 1
    assert bundle.extraction_units[0].title == "五常大论"
    assert bundle.chunks[0].parent_unit_id == bundle.extraction_units[0].unit_id
    assert [chunk.chunk_id for chunk in extractor.received_chunks] == [
        bundle.extraction_units[0].unit_id
    ]
    assert extractor.received_chunks[0].content == bundle.extraction_units[0].content
