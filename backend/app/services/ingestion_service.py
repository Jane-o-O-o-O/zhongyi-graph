from pathlib import Path
import hashlib
import tempfile

from app.models.ingestion import DocumentChunk, IngestionJob, SourceManifest
from app.services.document_parser import DocumentParser
from app.services.graph_extractor import GraphExtractor
from app.services.ingestion_repository import IngestionRepository
from app.services.knowledge_publisher import KnowledgePublisher, PublishedKnowledgeArtifact
from app.services.object_storage import LocalObjectStorage, object_key_for


class IngestionService:
    def __init__(
        self,
        repository: IngestionRepository | None = None,
        storage=None,
        parser: DocumentParser | None = None,
        extractor: GraphExtractor | None = None,
        publisher: KnowledgePublisher | None = None,
    ):
        self.sources: dict[str, SourceManifest] = {}
        self.jobs: dict[str, IngestionJob] = {}
        self.repository = repository or IngestionRepository.in_memory()
        self.storage = storage or LocalObjectStorage(Path(tempfile.gettempdir()) / "tcm-kg-objects")
        self.parser = parser or DocumentParser()
        self.extractor = extractor or GraphExtractor()
        self.publisher = publisher or KnowledgePublisher()

    def register_source(self, manifest: SourceManifest) -> SourceManifest:
        self.sources[manifest.source_id] = manifest
        self.repository.upsert_source(manifest)
        return manifest

    def create_job(self, source_ids: list[str]) -> IngestionJob:
        job = IngestionJob(job_id=f"job:{len(self.jobs) + 1}", source_ids=source_ids)
        self.jobs[job.job_id] = job
        return job

    def upload_source(self, filename: str, mime_type: str, content: bytes) -> SourceManifest:
        checksum = hashlib.sha256(content).hexdigest()
        source_id = f"source:uploaded:{checksum[:12]}"
        object_key = object_key_for(checksum=checksum, filename=filename)
        self.storage.put_bytes(object_key, content)
        manifest = SourceManifest(
            source_id=source_id,
            filename=filename,
            mime_type=mime_type,
            checksum=checksum,
            status="registered",
            object_key=object_key,
        )
        return self.register_source(manifest)

    def run_job(self, job_id: str) -> dict:
        job = self.jobs[job_id]
        total_chunks = 0
        total_entities = 0
        total_relations = 0
        job.status = "running"
        for source_id in job.source_ids:
            source = self.sources[source_id]
            content = self.storage.get_bytes(source.object_key)
            pages, chunks, status = self.parser.parse(source, content)
            source.status = status
            self.register_source(source)
            self.repository.replace_pages_and_chunks(source_id, pages, chunks)
            entities, relations = self.extractor.extract(chunks)
            self.repository.save_candidates(source_id, entities, relations)
            total_chunks += len(chunks)
            total_entities += len(entities)
            total_relations += len(relations)
        job.status = "completed"
        return {
            "job_id": job.job_id,
            "status": job.status,
            "chunk_count": total_chunks,
            "entity_count": total_entities,
            "relation_count": total_relations,
        }

    def publish_sources(self, source_ids: list[str]) -> tuple[PublishedKnowledgeArtifact, dict]:
        bundles = [self.repository.get_bundle(source_id) for source_id in source_ids]
        artifact = self.publisher.build_artifact(bundles)
        batch = self.repository.record_publish_batch(
            source_ids=source_ids,
            node_count=len(artifact.nodes),
            edge_count=len(artifact.edges),
            chunk_count=sum(len(bundle.chunks) for bundle in bundles),
        )
        for bundle in bundles:
            source = bundle.source
            source.status = "published"
            self.register_source(source)
        return artifact, batch.model_dump()

    def restore_published_artifact(self) -> PublishedKnowledgeArtifact:
        source_ids = self.repository.list_published_source_ids()
        bundles = [self.repository.get_bundle(source_id) for source_id in source_ids]
        return self.publisher.build_artifact(bundles)

    def list_chunks(self, source_id: str) -> list[DocumentChunk]:
        return self.repository.list_chunks(source_id)
