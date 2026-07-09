from app.models.ingestion import DocumentChunk, DocumentPage, ExtractionUnit, SourceManifest
from app.services.ingestion_repository import IngestionRepository
from scripts.rebuild_structural_chunks import (
    RebuildProgress,
    StructuralChunkRebuilder,
    list_rebuildable_sources,
)


class MemoryStorage:
    def __init__(self, objects):
        self.objects = objects

    def get_bytes(self, key):
        return self.objects[key]


class FakeParser:
    def parse(self, source, content):
        page = DocumentPage(
            page_id=f"page:{source.source_id}:1",
            source_id=source.source_id,
            page_number=1,
            text=content.decode("utf-8"),
        )
        unit = ExtractionUnit(
            unit_id=f"unit:{source.source_id}:0001",
            source_id=source.source_id,
            page_id=page.page_id,
            unit_index=1,
            title="五常大论",
            content="头痛多属于阳也。",
        )
        chunk = DocumentChunk(
            chunk_id=f"chunk:{source.source_id}:0001",
            source_id=source.source_id,
            page_id=page.page_id,
            chunk_index=1,
            content="头痛多属于阳也。",
            parent_unit_id=unit.unit_id,
            unit_index=unit.unit_index,
        )
        return [page], [unit], [chunk], "parsed"


def test_list_rebuildable_sources_skips_sources_without_object_key():
    repository = IngestionRepository.in_memory()
    with_key = SourceManifest(
        source_id="source:with-key",
        filename="with.txt",
        mime_type="text/plain",
        checksum="with",
        status="parsed",
        object_key="sources/with/with.txt",
    )
    without_key = SourceManifest(
        source_id="source:without-key",
        filename="without.txt",
        mime_type="text/plain",
        checksum="without",
        status="parsed",
        object_key="",
    )
    repository.upsert_source(with_key)
    repository.upsert_source(without_key)

    assert [source.source_id for source in list_rebuildable_sources(repository)] == [with_key.source_id]


def test_structural_chunk_rebuilder_saves_units_and_chunks(tmp_path):
    repository = IngestionRepository.in_memory()
    source = SourceManifest(
        source_id="source:uploaded:abc",
        filename="资料.txt",
        mime_type="text/plain",
        checksum="abc",
        status="parsed",
        object_key="sources/abc/资料.txt",
    )
    repository.upsert_source(source)
    progress = RebuildProgress(tmp_path / "progress.jsonl")
    rebuilder = StructuralChunkRebuilder(
        repository=repository,
        storage=MemoryStorage({"sources/abc/资料.txt": b"raw"}),
        parser=FakeParser(),
        progress=progress,
        dry_run=False,
    )

    rebuilder.run([source])
    bundle = repository.get_bundle(source.source_id)

    assert rebuilder.summary["rebuilt"] == 1
    assert rebuilder.summary["total_units"] == 1
    assert rebuilder.summary["total_chunks"] == 1
    assert bundle.extraction_units[0].content == "头痛多属于阳也。"
    assert bundle.chunks[0].parent_unit_id == bundle.extraction_units[0].unit_id


def test_structural_chunk_rebuilder_preserves_published_status(tmp_path):
    repository = IngestionRepository.in_memory()
    source = SourceManifest(
        source_id="source:uploaded:published",
        filename="资料.txt",
        mime_type="text/plain",
        checksum="published",
        status="published",
        object_key="sources/published/资料.txt",
    )
    repository.upsert_source(source)
    rebuilder = StructuralChunkRebuilder(
        repository=repository,
        storage=MemoryStorage({"sources/published/资料.txt": b"raw"}),
        parser=FakeParser(),
        progress=RebuildProgress(tmp_path / "progress.jsonl"),
        dry_run=False,
    )

    rebuilder.run([source])

    assert repository.get_bundle(source.source_id).source.status == "published"
