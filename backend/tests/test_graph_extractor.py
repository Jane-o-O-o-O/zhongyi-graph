from app.models.ingestion import DocumentChunk
from app.services.graph_extractor import GraphExtractor
from app.services.model_clients import StructuredExtractionClient


def test_graph_extractor_requires_llm_extractor_instead_of_fixed_dictionary():
    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:abc:0001",
        source_id="source:uploaded:abc",
        page_id="page:source:uploaded:abc:1",
        chunk_index=1,
        content="失眠可辨为心脾两虚，治以补益心脾，常用归脾汤，药味包含党参。",
    )

    entities, relations = GraphExtractor().extract([chunk])

    assert entities == []
    assert relations == []


def test_graph_extractor_extracts_tcm_candidates_from_llm_output():
    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:abc:0001",
        source_id="source:uploaded:abc",
        page_id="page:source:uploaded:abc:1",
        chunk_index=1,
        content="失眠可辨为心脾两虚，治以补益心脾，常用归脾汤，药味包含党参。",
    )

    entities, relations = GraphExtractor(
        llm_extractor=StructuredExtractionClient.demo()
    ).extract([chunk])

    entity_names = {entity.name for entity in entities}
    assert {"失眠", "心脾两虚", "补益心脾", "归脾汤", "党参"} <= entity_names
    assert any(relation.relation == "MANIFESTS_AS" for relation in relations)
    assert any(relation.relation == "RECOMMENDS_FORMULA" for relation in relations)
    assert all(relation.evidence_chunk_ids == [chunk.chunk_id] for relation in relations)


def test_graph_extractor_ner_method_extracts_entities_without_llm():
    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:ner:0001",
        source_id="source:uploaded:ner",
        page_id="page:source:uploaded:ner:1",
        chunk_index=1,
        content="白芍具有养血敛阴功效，常用于失眠。",
    )

    entities, relations = GraphExtractor(method="ner").extract([chunk])

    assert {
        (entity.name, entity.label)
        for entity in entities
    } >= {
        ("白芍", "Herb"),
        ("养血敛阴", "Function"),
        ("失眠", "Symptom"),
    }
    assert any(
        relation.source_entity_id == "entity:herb:白芍"
        and relation.target_entity_id == "entity:function:养血敛阴"
        and relation.relation == "RELATED_TO"
        for relation in relations
    )


def test_graph_extractor_links_insomnia_synonym_to_clinical_path():
    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:abc:0001",
        source_id="source:uploaded:abc",
        page_id="page:source:uploaded:abc:1",
        chunk_index=1,
        content="不寐可见心脾两虚，治以补益心脾，方选归脾汤。",
    )

    _entities, relations = GraphExtractor(
        llm_extractor=StructuredExtractionClient.demo()
    ).extract([chunk])

    assert any(
        relation.source_entity_id == "entity:symptom:不寐"
        and relation.target_entity_id == "entity:syndrome:心脾两虚"
        and relation.relation == "MANIFESTS_AS"
        for relation in relations
    )


def test_graph_extractor_uses_llm_structured_output_for_new_entities():
    class FakeLlmExtractor:
        def extract_chunk(self, text, hints=None):
            assert "头痛" in text
            return {
                "entities": [
                    {"name": "头痛", "label": "Symptom", "confidence": 0.91},
                    {"name": "心脾两虚", "label": "Syndrome", "confidence": 0.88},
                    {"name": "归脾汤", "label": "Formula", "confidence": 0.86},
                ],
                "relations": [
                    {
                        "source": "头痛",
                        "target": "心脾两虚",
                        "relation": "MANIFESTS_AS",
                        "display": "可辨为",
                        "confidence": 0.82,
                    },
                    {
                        "source": "心脾两虚",
                        "target": "归脾汤",
                        "relation": "RECOMMENDS_FORMULA",
                        "display": "推荐方剂",
                        "confidence": 0.8,
                    },
                ],
            }

    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:headache:0001",
        source_id="source:uploaded:headache",
        page_id="page:source:uploaded:headache:1",
        chunk_index=1,
        content="头痛日久，伴心悸健忘、少寐，可从心脾两虚辨治，方选归脾汤加减。",
    )

    entities, relations = GraphExtractor(llm_extractor=FakeLlmExtractor()).extract([chunk])

    assert {entity.name for entity in entities} == {"头痛", "心脾两虚", "归脾汤"}
    assert any(entity.entity_id == "entity:symptom:头痛" for entity in entities)
    assert any(
        relation.source_entity_id == "entity:symptom:头痛"
        and relation.target_entity_id == "entity:syndrome:心脾两虚"
        and relation.relation == "MANIFESTS_AS"
        for relation in relations
    )


def test_graph_extractor_general_method_uses_batch_extraction_units():
    class FakeBatchExtractor:
        def __init__(self):
            self.batch_calls = []
            self.chunk_calls = 0

        def extract_chunk(self, text, hints=None):
            self.chunk_calls += 1
            raise AssertionError("general method should use batch extraction")

        def extract_chunks_batch(self, items):
            self.batch_calls.append(items)
            return {
                "items": [
                    {
                        "unit_id": items[0]["unit_id"],
                        "entities": [
                            {"name": "失眠", "label": "Symptom", "confidence": 0.91},
                            {"name": "心脾两虚", "label": "Syndrome", "confidence": 0.88},
                        ],
                        "relations": [
                            {
                                "source": "失眠",
                                "target": "心脾两虚",
                                "relation": "MANIFESTS_AS",
                                "display": "可辨为",
                                "confidence": 0.82,
                            }
                        ],
                    },
                    {
                        "unit_id": items[1]["unit_id"],
                        "entities": [
                            {"name": "归脾汤", "label": "Formula", "confidence": 0.86},
                            {"name": "党参", "label": "Herb", "confidence": 0.84},
                        ],
                        "relations": [
                            {
                                "source": "归脾汤",
                                "target": "党参",
                                "relation": "COMPOSED_OF",
                                "display": "组成",
                                "confidence": 0.81,
                            }
                        ],
                    },
                ]
            }

    extractor = FakeBatchExtractor()
    chunks = [
        DocumentChunk(
            chunk_id="chunk:general:1",
            source_id="source:general",
            page_id="page:general:1",
            chunk_index=1,
            content="失眠可辨为心脾两虚。",
        ),
        DocumentChunk(
            chunk_id="chunk:general:2",
            source_id="source:general",
            page_id="page:general:1",
            chunk_index=2,
            content="归脾汤由党参等药组成。",
        ),
    ]

    entities, relations = GraphExtractor(
        llm_extractor=extractor,
        method="general",
    ).extract(chunks)

    assert extractor.chunk_calls == 0
    assert extractor.batch_calls == [
        [
            {"unit_id": "chunk:general:1", "text": "失眠可辨为心脾两虚。"},
            {"unit_id": "chunk:general:2", "text": "归脾汤由党参等药组成。"},
        ]
    ]
    assert {entity.name for entity in entities} == {"失眠", "心脾两虚", "归脾汤", "党参"}
    assert any(
        relation.relation == "MANIFESTS_AS"
        and relation.evidence_chunk_ids == ["chunk:general:1"]
        for relation in relations
    )
    assert any(
        relation.relation == "COMPOSED_OF"
        and relation.evidence_chunk_ids == ["chunk:general:2"]
        for relation in relations
    )


def test_graph_extractor_normalizes_llm_symptom_aliases_after_extraction():
    class FakeLlmExtractor:
        def extract_chunk(self, text, hints=None):
            return {
                "entities": [
                    {"name": "发热头痛", "label": "Symptom", "confidence": 0.9},
                    {"name": "补中益气汤", "label": "Formula", "confidence": 0.85},
                ],
                "relations": [
                    {
                        "source": "补中益气汤",
                        "target": "发热头痛",
                        "relation": "TREATS",
                        "display": "治疗",
                        "confidence": 0.82,
                    }
                ],
            }

    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:headache:0002",
        source_id="source:uploaded:headache",
        page_id="page:source:uploaded:headache:1",
        chunk_index=2,
        content="补中益气汤治疗发热头痛。",
    )

    entities, relations = GraphExtractor(llm_extractor=FakeLlmExtractor()).extract([chunk])

    assert any(entity.entity_id == "entity:symptom:头痛" and entity.name == "头痛" for entity in entities)
    assert any(relation.target_entity_id == "entity:symptom:头痛" for relation in relations)


def test_graph_extractor_does_not_fallback_to_dictionary_when_llm_fails():
    class BrokenLlmExtractor:
        def extract_chunk(self, text, hints=None):
            raise RuntimeError("llm unavailable")

    chunk = DocumentChunk(
        chunk_id="chunk:source:uploaded:abc:0003",
        source_id="source:uploaded:abc",
        page_id="page:source:uploaded:abc:1",
        chunk_index=3,
        content="失眠可辨为心脾两虚，常用归脾汤。",
    )

    entities, relations = GraphExtractor(llm_extractor=BrokenLlmExtractor()).extract([chunk])

    assert entities == []
    assert relations == []
