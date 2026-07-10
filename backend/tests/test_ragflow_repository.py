from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import (
    RetrievalGraphRagBuildRun,
    RetrievalCommunityReport,
    RetrievalChunk,
    RetrievalDocument,
    RetrievalGraphArtifact,
    RetrievalKgEntity,
    RetrievalKgRelation,
    RetrievalTypeSamples,
)


def _repository() -> RagflowRetrievalRepository:
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return RagflowRetrievalRepository(engine)


def test_repository_creates_ragflow_compatible_tables():
    repository = _repository()

    tables = set(inspect(repository.engine).get_table_names())
    assert {
        "retrieval_documents",
        "retrieval_chunks",
        "retrieval_chunk_terms",
        "retrieval_kg_entities",
        "retrieval_kg_relations",
        "retrieval_kg_community_reports",
        "retrieval_kg_graph_artifacts",
        "retrieval_graphrag_checkpoints",
        "retrieval_graphrag_phase_markers",
        "retrieval_kg_type_samples",
        "retrieval_sync_state",
        "retrieval_query_logs",
    }.issubset(tables)


def test_repository_round_trips_retrieval_rows():
    repository = _repository()
    document = RetrievalDocument(
        doc_id="source:uploaded:abc",
        source_id="source:uploaded:abc",
        filename="abc.txt",
        mime_type="text/plain",
        checksum="abc",
        status="parsed",
        object_key="sources/abc/abc.txt",
        source_version=1,
        chunk_count=1,
        eligible_chunk_count=1,
        metadata={"kind": "fixture"},
    )
    chunk = RetrievalChunk(
        chunk_id="chunk:source:uploaded:abc:0001",
        doc_id=document.doc_id,
        source_id=document.source_id,
        parent_unit_id="unit:source:uploaded:abc:0001",
        chunk_order_int=1,
        page_num_int=1,
        title="失眠",
        section_path=["资料", "失眠"],
        content="失眠可辨为心脾两虚。",
        content_with_weight="失眠 资料 失眠 失眠可辨为心脾两虚。",
        content_ltks="失眠 心脾两虚",
        title_tks="失眠",
        important_kwd=["失眠", "心脾两虚"],
        token_count=12,
        metadata={"short_chunk": True},
    )
    entity = RetrievalKgEntity(
        entity_id="entity:syndrome:心脾两虚",
        entity_name="心脾两虚",
        entity_type="Syndrome",
        source_node_id="syndrome:心脾两虚",
        content_with_weight='{"description":"心脾两虚 Syndrome"}',
        description="心脾两虚 Syndrome",
        rank_flt=2.0,
        n_hop_with_weight=[{"path": ["失眠", "心脾两虚"], "weights": [1.0]}],
        aliases=["心脾不足"],
        evidence_chunk_ids=[chunk.chunk_id],
    )
    relation = RetrievalKgRelation(
        relation_id="relation:失眠:心脾两虚",
        from_entity_kwd="失眠",
        to_entity_kwd="心脾两虚",
        relation_type="MANIFESTS_AS",
        display="可辨为",
        content_with_weight="失眠 可辨为 心脾两虚",
        weight_int=3,
        evidence_chunk_ids=[chunk.chunk_id],
        source_edge_id="edge:1",
    )
    samples = RetrievalTypeSamples(
        entity_type="Syndrome",
        sample_entities=["心脾两虚", "肝郁化火"],
        sample_count=2,
        updated_at="2026-06-28T00:00:00Z",
    )
    community_report = RetrievalCommunityReport(
        report_id="community:1",
        title="失眠、心脾两虚",
        content_with_weight="失眠、心脾两虚相关社区报告",
        summary="失眠与心脾两虚相关",
        evidences="图谱社区摘要",
        entities_kwd=["失眠", "心脾两虚"],
        weight_flt=0.9,
        source_id=[document.doc_id],
    )
    graph_artifact = RetrievalGraphArtifact(
        artifact_id="graph:global",
        artifact_type="graph",
        content_with_weight='{"nodes":[],"edges":[]}',
        source_id=[document.doc_id],
        node_count=2,
        edge_count=1,
        metadata={"kind": "global"},
    )

    repository.replace_documents([document])
    repository.replace_chunks([chunk])
    repository.replace_kg_entities([entity])
    repository.replace_kg_relations([relation])
    repository.replace_community_reports([community_report])
    repository.replace_graph_artifacts([graph_artifact])
    repository.replace_type_samples([samples])

    assert repository.list_documents() == [document]
    assert repository.list_chunks() == [chunk]
    assert repository.list_kg_entities() == [entity]
    assert repository.list_kg_relations() == [relation]
    assert repository.list_community_reports() == [community_report]
    assert repository.list_graph_artifacts() == [graph_artifact]
    assert repository.list_type_samples() == [samples]


def test_repository_saves_and_loads_graph_artifact_checkpoints_by_source():
    repository = _repository()
    global_artifact = RetrievalGraphArtifact(
        artifact_id="graph:global",
        artifact_type="graph",
        content_with_weight='{"nodes":["全局"],"edges":[]}',
        source_id=["doc:a"],
        node_count=1,
        edge_count=0,
    )
    subgraph_a = RetrievalGraphArtifact(
        artifact_id="subgraph:doc:a",
        artifact_type="subgraph",
        content_with_weight='{"nodes":["白芍"],"edges":[]}',
        source_id=["doc:a"],
        node_count=1,
        edge_count=0,
    )
    subgraph_b = RetrievalGraphArtifact(
        artifact_id="subgraph:doc:b",
        artifact_type="subgraph",
        content_with_weight='{"nodes":["柴胡"],"edges":[]}',
        source_id=["doc:b"],
        node_count=1,
        edge_count=0,
    )

    repository.save_graph_artifact(global_artifact)
    repository.save_graph_artifact(subgraph_a)
    repository.save_graph_artifact(subgraph_b)

    assert repository.get_graph_artifact("graph:global") == global_artifact
    assert repository.get_subgraph_artifact("doc:a") == subgraph_a
    assert repository.get_subgraph_artifact("doc:missing") is None

    replacement = RetrievalGraphArtifact(
        artifact_id="subgraph:doc:a",
        artifact_type="subgraph",
        content_with_weight='{"nodes":["白芍","白芍药"],"edges":[]}',
        source_id=["doc:a"],
        node_count=2,
        edge_count=0,
    )
    repository.save_graph_artifact(replacement)

    assert repository.get_subgraph_artifact("doc:a") == replacement
    assert repository.get_graph_artifact("graph:global") == global_artifact


def test_repository_manages_graphrag_checkpoints_and_phase_markers():
    repository = _repository()

    assert repository.save_graphrag_checkpoint(
        "resolution",
        "checkpoint:a",
        {"pairs": [["白芍", "白芍药"]]},
    )
    assert repository.save_graphrag_checkpoint(
        "resolution",
        "checkpoint:b",
        {"pairs": [["柴胡", "北柴胡"]]},
    )

    assert repository.load_graphrag_checkpoints("resolution") == {
        "checkpoint:a": {"pairs": [["白芍", "白芍药"]]},
        "checkpoint:b": {"pairs": [["柴胡", "北柴胡"]]},
    }
    assert repository.cleanup_graphrag_checkpoints("resolution") == 2
    assert repository.load_graphrag_checkpoints("resolution") == {}

    assert repository.has_graphrag_phase_marker("resolution_done") is False
    assert repository.set_graphrag_phase_marker("resolution_done")
    assert repository.has_graphrag_phase_marker("resolution_done") is True
    repository.clear_graphrag_phase_markers(["resolution_done"])
    assert repository.has_graphrag_phase_marker("resolution_done") is False


def test_repository_persists_graphrag_build_runs_in_sync_state():
    repository = _repository()
    running = RetrievalGraphRagBuildRun(
        run_id="graphrag:build:test",
        status="running",
        started_at="2026-07-10T00:00:00Z",
        finished_at="",
        total=2,
        processed=0,
        failed=0,
        metadata={
            "source_ids": ["doc:a", "doc:b"],
            "with_resolution": True,
            "with_community": False,
        },
    )

    repository.save_graphrag_build_run(running)

    assert repository.get_graphrag_build_run("graphrag:build:test") == running

    completed = RetrievalGraphRagBuildRun(
        run_id="graphrag:build:test",
        status="completed",
        started_at="2026-07-10T00:00:00Z",
        finished_at="2026-07-10T00:00:10Z",
        total=2,
        processed=2,
        failed=0,
        metadata={
            "source_ids": ["doc:a", "doc:b"],
            "with_resolution": True,
            "with_community": False,
            "summary": {"global_nodes": 18, "global_edges": 24},
        },
    )
    repository.save_graphrag_build_run(completed)

    assert repository.get_graphrag_build_run("graphrag:build:test") == completed
    assert repository.get_graphrag_build_run("missing") is None


def test_repository_claims_and_releases_graphrag_build_lock():
    repository = _repository()

    assert repository.claim_graphrag_build_lock(
        "graphrag:build:first",
        started_at="2026-07-10T00:00:00Z",
        metadata={"source_ids": ["doc:a"]},
    )
    assert not repository.claim_graphrag_build_lock(
        "graphrag:build:second",
        started_at="2026-07-10T00:00:01Z",
        metadata={"source_ids": ["doc:b"]},
    )

    repository.release_graphrag_build_lock("graphrag:build:first")

    assert repository.claim_graphrag_build_lock(
        "graphrag:build:second",
        started_at="2026-07-10T00:00:02Z",
        metadata={"source_ids": ["doc:b"]},
    )


def test_repository_audit_counts_vectors_and_chunk_lengths():
    repository = _repository()
    repository.replace_documents(
        [
            RetrievalDocument(
                doc_id="doc",
                source_id="doc",
                filename="doc.txt",
                chunk_count=2,
                eligible_chunk_count=2,
            )
        ]
    )
    repository.replace_chunks(
        [
            RetrievalChunk(
                chunk_id="chunk:short",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=1,
                page_num_int=1,
                title="",
                section_path=[],
                content="短切片",
                content_with_weight="短切片",
                token_count=20,
                vector_status="embedded",
                vector_point_id="point:1",
            ),
            RetrievalChunk(
                chunk_id="chunk:long",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:2",
                chunk_order_int=2,
                page_num_int=1,
                title="",
                section_path=[],
                content="长切片",
                content_with_weight="长切片",
                token_count=620,
                vector_status="failed",
            ),
        ]
    )
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id="entity:1",
                entity_name="失眠",
                entity_type="Symptom",
                source_node_id="symptom:失眠",
                content_with_weight="失眠 Symptom",
                vector_status="embedded",
                vector_point_id="point:entity",
            )
        ]
    )
    repository.replace_kg_relations(
        [
            RetrievalKgRelation(
                relation_id="relation:1",
                from_entity_kwd="失眠",
                to_entity_kwd="心脾两虚",
                relation_type="MANIFESTS_AS",
                display="可辨为",
                content_with_weight="失眠 可辨为 心脾两虚",
            )
        ]
    )

    audit = repository.audit()

    assert audit.documents == 1
    assert audit.chunks == 2
    assert audit.chunks_with_vectors == 1
    assert audit.chunks_failed_vectors == 1
    assert audit.kg_entities == 1
    assert audit.kg_entities_with_vectors == 1
    assert audit.kg_entities_failed_vectors == 0
    assert audit.kg_relations == 1
    assert audit.kg_relations_with_vectors == 0
    assert audit.kg_relations_failed_vectors == 0
    assert audit.short_chunks == 1
    assert audit.long_chunks == 1


def test_repository_searches_available_chunk_candidates_by_keywords_with_limit():
    repository = _repository()
    repository.replace_documents(
        [RetrievalDocument(doc_id="doc", source_id="doc", filename="doc.txt")]
    )
    repository.replace_chunks(
        [
            RetrievalChunk(
                chunk_id="chunk:1",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=1,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为心脾两虚。",
                content_with_weight="不寐 失眠 心脾两虚",
                content_ltks="失眠 心脾两虚",
                title_tks="不寐",
                token_count=18,
            ),
            RetrievalChunk(
                chunk_id="chunk:2",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:2",
                chunk_order_int=2,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为肝郁化火。",
                content_with_weight="不寐 失眠 肝郁化火",
                content_ltks="失眠 肝郁化火",
                token_count=18,
            ),
            RetrievalChunk(
                chunk_id="chunk:3",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:3",
                chunk_order_int=3,
                page_num_int=1,
                title="少阳病",
                section_path=[],
                content="柴胡桂枝干姜汤主往来寒热。",
                content_with_weight="少阳 柴胡桂枝干姜汤 往来寒热",
                content_ltks="柴胡桂枝干姜汤 往来寒热",
                token_count=18,
            ),
            RetrievalChunk(
                chunk_id="chunk:4",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:4",
                chunk_order_int=4,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠但不可用。",
                content_with_weight="不寐 失眠 不可用",
                content_ltks="失眠",
                token_count=18,
                available_int=0,
            ),
        ]
    )

    candidates = repository.search_chunk_candidates(["失眠"], limit=1)

    assert [chunk.chunk_id for chunk in candidates] == ["chunk:1"]


def test_repository_gets_chunks_by_ids_preserving_requested_order():
    repository = _repository()
    repository.replace_documents(
        [RetrievalDocument(doc_id="doc", source_id="doc", filename="doc.txt")]
    )
    repository.replace_chunks(
        [
            RetrievalChunk(
                chunk_id=f"chunk:{index}",
                doc_id="doc",
                source_id="doc",
                parent_unit_id=f"unit:{index}",
                chunk_order_int=index,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content=f"失眠可辨为心脾两虚 {index}",
                content_with_weight=f"不寐 失眠 心脾两虚 {index}",
                content_ltks="失眠 心脾两虚",
                token_count=18,
            )
            for index in range(1, 4)
        ]
    )

    chunks = repository.get_chunks_by_ids(["chunk:3", "chunk:missing", "chunk:1"])

    assert [chunk.chunk_id for chunk in chunks] == ["chunk:3", "chunk:1"]


def test_repository_counts_embedded_chunks():
    repository = _repository()
    repository.replace_documents(
        [RetrievalDocument(doc_id="doc", source_id="doc", filename="doc.txt")]
    )
    repository.replace_chunks(
        [
            RetrievalChunk(
                chunk_id="chunk:embedded",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=1,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为心脾两虚。",
                content_with_weight="不寐 失眠 心脾两虚",
                content_ltks="失眠 心脾两虚",
                token_count=18,
                vector_status="embedded",
            ),
            RetrievalChunk(
                chunk_id="chunk:missing",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:2",
                chunk_order_int=2,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为肝郁化火。",
                content_with_weight="不寐 失眠 肝郁化火",
                content_ltks="失眠 肝郁化火",
                token_count=18,
                vector_status="missing",
            ),
        ]
    )

    assert repository.count_embedded_chunks() == 1


def test_repository_resets_failed_vector_statuses_by_content_type():
    repository = _repository()
    repository.replace_documents(
        [RetrievalDocument(doc_id="doc", source_id="doc", filename="doc.txt")]
    )
    repository.replace_chunks(
        [
            RetrievalChunk(
                chunk_id="chunk:failed",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=1,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为心脾两虚。",
                content_with_weight="不寐 失眠 心脾两虚",
                token_count=18,
                vector_status="failed",
            ),
            RetrievalChunk(
                chunk_id="chunk:embedded",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:2",
                chunk_order_int=2,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为肝郁化火。",
                content_with_weight="不寐 失眠 肝郁化火",
                token_count=18,
                vector_status="embedded",
            ),
        ]
    )
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id="entity:failed",
                entity_name="失眠",
                entity_type="Symptom",
                source_node_id="symptom:失眠",
                content_with_weight="失眠 Symptom",
                vector_status="failed",
            )
        ]
    )

    summary = repository.reset_failed_vector_statuses(content_types=["chunk"])

    assert summary == {
        "reset_failed_chunks": 1,
        "reset_failed_kg_entities": 0,
        "reset_failed_kg_relations": 0,
    }
    assert repository.get_chunk("chunk:failed").vector_status == "missing"
    assert repository.get_chunk("chunk:embedded").vector_status == "embedded"
    assert repository.list_kg_entities()[0].vector_status == "failed"


def test_repository_claims_missing_vectors_and_excludes_queued_records():
    repository = _repository()
    repository.replace_documents(
        [RetrievalDocument(doc_id="doc", source_id="doc", filename="doc.txt")]
    )
    repository.replace_chunks(
        [
            RetrievalChunk(
                chunk_id=f"chunk:{index}",
                doc_id="doc",
                source_id="doc",
                parent_unit_id=f"unit:{index}",
                chunk_order_int=index,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content=f"失眠可辨为心脾两虚 {index}",
                content_with_weight=f"不寐 失眠 心脾两虚 {index}",
                token_count=18,
            )
            for index in range(1, 4)
        ]
    )

    first_claim = repository.claim_missing_vector_records(
        content_types=["chunk"],
        limit=2,
    )
    second_claim = repository.claim_missing_vector_records(
        content_types=["chunk"],
        limit=2,
    )

    assert [record.chunk_id for record in first_claim] == ["chunk:1", "chunk:2"]
    assert [record.chunk_id for record in second_claim] == ["chunk:3"]
    assert repository.get_chunk("chunk:1").vector_status == "queued"
    assert repository.get_chunk("chunk:2").vector_status == "queued"
    assert repository.get_chunk("chunk:3").vector_status == "queued"
    assert repository.list_missing_vector_records(content_types=["chunk"], limit=10) == []


def test_repository_gets_kg_records_by_ids_and_counts_embedded_vectors():
    repository = _repository()
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id="entity:失眠",
                entity_name="失眠",
                entity_type="Symptom",
                source_node_id="symptom:失眠",
                content_with_weight="失眠 Symptom",
                vector_status="embedded",
            ),
            RetrievalKgEntity(
                entity_id="entity:心脾两虚",
                entity_name="心脾两虚",
                entity_type="Syndrome",
                source_node_id="syndrome:心脾两虚",
                content_with_weight="心脾两虚 Syndrome",
            ),
        ]
    )
    repository.replace_kg_relations(
        [
            RetrievalKgRelation(
                relation_id="relation:1",
                from_entity_kwd="失眠",
                to_entity_kwd="心脾两虚",
                relation_type="MANIFESTS_AS",
                display="可辨为",
                content_with_weight="失眠 可辨为 心脾两虚",
                vector_status="embedded",
            ),
            RetrievalKgRelation(
                relation_id="relation:2",
                from_entity_kwd="心脾两虚",
                to_entity_kwd="归脾汤",
                relation_type="RECOMMENDS_FORMULA",
                display="推荐方剂",
                content_with_weight="心脾两虚 推荐方剂 归脾汤",
            ),
        ]
    )

    entities = repository.get_kg_entities_by_ids(
        ["entity:心脾两虚", "entity:missing", "entity:失眠"]
    )
    relations = repository.get_kg_relations_by_ids(
        ["relation:2", "relation:missing", "relation:1"]
    )

    assert [entity.entity_id for entity in entities] == ["entity:心脾两虚", "entity:失眠"]
    assert [relation.relation_id for relation in relations] == ["relation:2", "relation:1"]
    assert repository.count_embedded_kg_entities() == 1
    assert repository.count_embedded_kg_relations() == 1


def test_repository_readiness_reports_vector_coverage_chunk_buckets_and_blockers():
    repository = _repository()
    repository.replace_documents(
        [RetrievalDocument(doc_id="doc", source_id="doc", filename="doc.txt")]
    )
    repository.replace_chunks(
        [
            RetrievalChunk(
                chunk_id="chunk:short",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=1,
                page_num_int=1,
                title="",
                section_path=[],
                content="短切片",
                content_with_weight="短切片",
                token_count=20,
                vector_status="embedded",
            ),
            RetrievalChunk(
                chunk_id="chunk:good",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:2",
                chunk_order_int=2,
                page_num_int=1,
                title="",
                section_path=[],
                content="有效切片",
                content_with_weight="有效切片",
                token_count=180,
            ),
            RetrievalChunk(
                chunk_id="chunk:long",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:3",
                chunk_order_int=3,
                page_num_int=1,
                title="",
                section_path=[],
                content="长切片",
                content_with_weight="长切片",
                token_count=620,
                vector_status="failed",
            ),
        ]
    )
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id="entity:失眠",
                entity_name="失眠",
                entity_type="Symptom",
                source_node_id="symptom:失眠",
                content_with_weight="失眠 Symptom",
                vector_status="embedded",
            ),
            RetrievalKgEntity(
                entity_id="entity:心脾两虚",
                entity_name="心脾两虚",
                entity_type="Syndrome",
                source_node_id="syndrome:心脾两虚",
                content_with_weight="心脾两虚 Syndrome",
                vector_status="failed",
            ),
        ]
    )
    repository.replace_kg_relations(
        [
            RetrievalKgRelation(
                relation_id="relation:1",
                from_entity_kwd="失眠",
                to_entity_kwd="心脾两虚",
                relation_type="MANIFESTS_AS",
                display="可辨为",
                content_with_weight="失眠 可辨为 心脾两虚",
            )
        ]
    )

    readiness = repository.readiness(
        min_chunk_vector_coverage=0.8,
        min_kg_entity_vector_coverage=0.8,
        min_kg_relation_vector_coverage=0.8,
        min_kg_entity_evidence_coverage=0.8,
        min_kg_relation_evidence_coverage=0.8,
        max_short_chunk_ratio=0.2,
        max_long_chunk_ratio=0.2,
    )

    assert readiness["ready"] is False
    assert readiness["chunk_token_buckets"] == {
        "lt_30": 1,
        "30_127": 0,
        "128_256": 1,
        "257_512": 0,
        "gt_512": 1,
    }
    assert readiness["vector_coverage"]["chunks"] == {
        "embedded": 1,
        "failed": 1,
        "total": 3,
        "ratio": 0.333333,
    }
    assert readiness["vector_coverage"]["kg_entities"]["ratio"] == 0.5
    assert readiness["vector_coverage"]["kg_entities"]["failed"] == 1
    assert readiness["vector_coverage"]["kg_relations"]["ratio"] == 0.0
    assert readiness["evidence_coverage"]["kg_entities"] == {
        "with_evidence": 0,
        "total": 2,
        "ratio": 0.0,
    }
    assert readiness["evidence_coverage"]["kg_relations"] == {
        "with_evidence": 0,
        "total": 1,
        "ratio": 0.0,
    }
    assert "chunk_vector_coverage_below_threshold" in readiness["blockers"]
    assert "kg_relation_vector_coverage_below_threshold" in readiness["blockers"]
    assert "kg_entity_evidence_coverage_below_threshold" in readiness["blockers"]
    assert "kg_relation_evidence_coverage_below_threshold" in readiness["blockers"]
    assert "short_chunk_ratio_above_threshold" in readiness["warnings"]
    assert "long_chunk_ratio_above_threshold" in readiness["warnings"]
    assert "failed_vectors_present" in readiness["warnings"]
    assert readiness["vector_sync_plan"]["chunks"] == {
        "target_ratio": 0.8,
        "target_embedded": 3,
        "current_embedded": 1,
        "remaining": 2,
    }
    assert readiness["vector_sync_plan"]["kg_entities"]["remaining"] == 1
    assert readiness["vector_sync_plan"]["kg_relations"]["remaining"] == 1
    assert readiness["vector_sync_plan"]["balanced_limit"] == 4
    assert (
        readiness["vector_sync_plan"]["recommended_command"]
        == "python ../scripts/sync_ragflow_retrieval_vectors.py --balanced --limit 4 --content-types chunk,kg_entity,kg_relation"
    )


def test_repository_readiness_recommends_only_content_types_below_vector_target():
    repository = _repository()
    repository.replace_documents(
        [RetrievalDocument(doc_id="doc", source_id="doc", filename="doc.txt")]
    )
    repository.replace_chunks(
        [
            RetrievalChunk(
                chunk_id=f"chunk:{index}",
                doc_id="doc",
                source_id="doc",
                parent_unit_id=f"unit:{index}",
                chunk_order_int=index,
                page_num_int=1,
                title="",
                section_path=[],
                content=f"切片 {index}",
                content_with_weight=f"切片 {index}",
                token_count=180,
                vector_status="embedded" if index == 1 else "missing",
            )
            for index in range(1, 6)
        ]
    )
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id=f"entity:{index}",
                entity_name=f"证候{index}",
                entity_type="Syndrome",
                source_node_id=f"syndrome:{index}",
                content_with_weight=f"证候{index} Syndrome",
                vector_status="embedded",
                evidence_chunk_ids=["chunk:1"],
            )
            for index in range(1, 6)
        ]
    )
    repository.replace_kg_relations(
        [
            RetrievalKgRelation(
                relation_id=f"relation:{index}",
                from_entity_kwd="失眠",
                to_entity_kwd=f"证候{index}",
                relation_type="MANIFESTS_AS",
                display="可辨为",
                content_with_weight=f"失眠 可辨为 证候{index}",
                vector_status="embedded",
                evidence_chunk_ids=["chunk:1"],
            )
            for index in range(1, 6)
        ]
    )

    readiness = repository.readiness(
        min_chunk_vector_coverage=0.8,
        min_kg_entity_vector_coverage=0.8,
        min_kg_relation_vector_coverage=0.8,
        min_kg_entity_evidence_coverage=0.8,
        min_kg_relation_evidence_coverage=0.8,
    )

    assert readiness["blockers"] == ["chunk_vector_coverage_below_threshold"]
    assert readiness["vector_sync_plan"]["balanced_limit"] == 3
    assert readiness["vector_sync_plan"]["content_types"] == ["chunk"]
    assert (
        readiness["vector_sync_plan"]["recommended_command"]
        == "python ../scripts/sync_ragflow_retrieval_vectors.py --limit 3 --content-types chunk"
    )
