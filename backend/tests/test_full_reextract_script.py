from scripts.reextract_existing_graphrag import (
    adaptive_extraction_batches,
    ExtractionProgress,
    build_extractor_workers,
    chunk_batches,
    extract_batch_with_llm,
    should_replace_candidates,
    limited_chunks,
    units_as_extraction_chunks,
)
from app.models.ingestion import DocumentChunk, ExtractionUnit


def test_extraction_progress_tracks_completed_sources_and_retries_failed(tmp_path):
    progress_path = tmp_path / "reextract.jsonl"
    progress_path.write_text(
        '{"source_id":"source:done","status":"extracted"}\n'
        '{"source_id":"source:published","status":"published"}\n'
        '{"source_id":"source:failed","status":"failed"}\n',
        encoding="utf-8",
    )

    progress = ExtractionProgress(progress_path)

    assert progress.completed_source_ids == {"source:done", "source:published"}
    assert "source:failed" not in progress.completed_source_ids


def test_chunk_batches_splits_tail():
    assert list(chunk_batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_adaptive_extraction_batches_keeps_short_units_between_five_and_ten():
    chunks = [
        DocumentChunk(
            chunk_id=f"unit:{index}",
            source_id="source:1",
            page_id="page:1",
            chunk_index=index,
            content=f"短父单元{index}。头痛。",
        )
        for index in range(23)
    ]

    batches = list(
        adaptive_extraction_batches(
            chunks,
            min_items=5,
            max_items=10,
            max_chars=10_000,
        )
    )

    assert [len(batch) for batch in batches] == [10, 8, 5]
    assert [chunk.chunk_id for batch in batches for chunk in batch] == [
        chunk.chunk_id for chunk in chunks
    ]


def test_adaptive_extraction_batches_uses_smaller_batches_for_long_units():
    chunks = [
        DocumentChunk(
            chunk_id=f"unit:{index}",
            source_id="source:1",
            page_id="page:1",
            chunk_index=index,
            content="头痛。" + ("证候。" * 800),
        )
        for index in range(6)
    ]

    batches = list(
        adaptive_extraction_batches(
            chunks,
            min_items=5,
            max_items=10,
            max_chars=8_000,
        )
    )

    assert [len(batch) for batch in batches] == [3, 3]


def test_limited_chunks_keeps_shortest_chunks_when_limit_is_set():
    chunks = [
        DocumentChunk(
            chunk_id="chunk:1",
            source_id="source:1",
            page_id="page:1",
            chunk_index=1,
            content="x" * 50,
        ),
        DocumentChunk(
            chunk_id="chunk:2",
            source_id="source:1",
            page_id="page:1",
            chunk_index=2,
            content="x" * 10,
        ),
        DocumentChunk(
            chunk_id="chunk:3",
            source_id="source:1",
            page_id="page:1",
            chunk_index=3,
            content="x" * 30,
        ),
    ]

    selected = limited_chunks(chunks, 2)

    assert [chunk.chunk_id for chunk in selected] == ["chunk:2", "chunk:3"]


def test_limited_chunks_prioritizes_medical_keyword_chunks_before_shortness():
    chunks = [
        DocumentChunk(
            chunk_id="chunk:short",
            source_id="source:1",
            page_id="page:1",
            chunk_index=1,
            content="目录 卷一",
        ),
        DocumentChunk(
            chunk_id="chunk:medical",
            source_id="source:1",
            page_id="page:1",
            chunk_index=2,
            content="头痛可辨肝阳上亢，治以平肝潜阳。",
        ),
    ]

    selected = limited_chunks(chunks, 1)

    assert [chunk.chunk_id for chunk in selected] == ["chunk:medical"]


def test_should_replace_candidates_keeps_existing_graph_when_extraction_is_empty():
    assert not should_replace_candidates([], [], replace_empty=False)
    assert should_replace_candidates(["entity"], [], replace_empty=False)
    assert should_replace_candidates([], ["relation"], replace_empty=False)
    assert should_replace_candidates([], [], replace_empty=True)


def test_units_as_extraction_chunks_preserves_unit_ids_for_llm_evidence():
    unit = ExtractionUnit(
        unit_id="unit:source:uploaded:abc:0001",
        source_id="source:uploaded:abc",
        page_id="page:source:uploaded:abc:1",
        unit_index=1,
        title="五常大论",
        content="头痛多属于阳也。",
        section_path=["普济方", "卷一\\方脉总论", "五常大论"],
    )

    chunks = units_as_extraction_chunks([unit])

    assert chunks[0].chunk_id == unit.unit_id
    assert chunks[0].content == unit.content
    assert chunks[0].metadata["section_path"] == unit.section_path
    assert chunks[0].metadata["extraction_unit"]


def test_build_extractor_workers_expands_keys_by_concurrency():
    workers = build_extractor_workers(
        api_keys=["key-a", "key-b"],
        per_key_concurrency=2,
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen3-8B",
        timeout=30,
    )

    assert len(workers) == 4
    assert [worker.api_key for worker in workers] == ["key-a", "key-a", "key-b", "key-b"]
    assert all(worker.model == "Qwen/Qwen3-8B" for worker in workers)


class BatchExtractor:
    def extract_chunks_batch(self, items):
        return {
            "items": [
                {
                    "unit_id": item["unit_id"],
                    "entities": [
                        {"name": "头痛", "label": "Symptom", "confidence": 0.9}
                    ],
                    "relations": [],
                }
                for item in items
            ]
        }


class MixedSchemaBatchExtractor:
    def extract_chunks_batch(self, items):
        return {
            "items": [
                {
                    "unit_id": items[0]["unit_id"],
                    "entities": [
                        {"text": "归脾汤", "type": "Medicine", "confidence": 0.9},
                        {"value": "党参", "type": "药材", "confidence": 0.9},
                    ],
                    "relations": [
                        {
                            "subject": "归脾汤",
                            "object": "党参",
                            "relation": "COMPOSED_OF",
                            "display": "组成",
                            "confidence": 0.9,
                        }
                    ],
                }
            ]
        }


class FlakyBatchExtractor:
    def __init__(self):
        self.calls = 0

    def extract_chunks_batch(self, items):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("temporary timeout")
        return {
            "items": [
                {
                    "unit_id": item["unit_id"],
                    "entities": [
                        {"name": "头痛", "label": "Symptom", "confidence": 0.9}
                    ],
                    "relations": [],
                }
                for item in items
            ]
        }


def test_extract_batch_with_llm_maps_batch_items_back_to_unit_evidence():
    chunks = [
        DocumentChunk(
            chunk_id="unit:1",
            source_id="source:1",
            page_id="page:1",
            chunk_index=1,
            content="头痛多属于阳。",
        ),
        DocumentChunk(
            chunk_id="unit:2",
            source_id="source:1",
            page_id="page:1",
            chunk_index=2,
            content="头痛连齿。",
        ),
    ]

    entities, relations = extract_batch_with_llm(BatchExtractor(), chunks)

    assert len(entities) == 1
    assert entities[0].source_chunk_ids == ["unit:1", "unit:2"]
    assert relations == []


def test_extract_batch_with_llm_normalizes_fast_model_label_aliases():
    chunks = [
        DocumentChunk(
            chunk_id="unit:1",
            source_id="source:1",
            page_id="page:1",
            chunk_index=1,
            content="归脾汤由党参等组成。",
        )
    ]

    entities, relations = extract_batch_with_llm(MixedSchemaBatchExtractor(), chunks)

    labels = {entity.name: entity.label for entity in entities}
    assert labels == {"归脾汤": "Formula", "党参": "Herb"}
    assert relations[0].source_entity_id == "entity:formula:归脾汤"
    assert relations[0].target_entity_id == "entity:herb:党参"


def test_extract_batch_with_llm_retries_transient_failures():
    chunks = [
        DocumentChunk(
            chunk_id="unit:1",
            source_id="source:1",
            page_id="page:1",
            chunk_index=1,
            content="头痛多属于阳。",
        )
    ]
    extractor = FlakyBatchExtractor()

    entities, _relations = extract_batch_with_llm(
        extractor,
        chunks,
        retries=2,
        sleep_seconds=0,
    )

    assert extractor.calls == 2
    assert entities[0].name == "头痛"
