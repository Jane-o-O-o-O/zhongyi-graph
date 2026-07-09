from scripts.audit_ragflow_retrieval_index import format_audit
from scripts.rebuild_ragflow_retrieval_index import build_summary_message
from app.services.ragflow_compat.repository import MissingVectorRecord
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import (
    RetrievalChunk,
    RetrievalDocument,
    RetrievalKgEntity,
    RetrievalKgRelation,
)
from scripts.sync_ragflow_retrieval_vectors import (
    build_dry_run_summary,
    dry_run_missing_records,
    format_sync_summary,
    reset_failed_records,
)
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


def test_ragflow_script_formatters_are_stable_for_cli_output():
    assert "documents=2" in build_summary_message({"documents": 2, "chunks": 5})
    assert "chunks_with_vectors=3" in format_audit({"chunks_with_vectors": 3})
    assert "embedded=4" in format_sync_summary({"embedded": 4, "failed": 0})


def test_vector_sync_formatter_includes_dry_run_type_breakdown():
    output = format_sync_summary(
        {
            "missing_preview": 9,
            "missing_chunks_preview": 3,
            "missing_kg_entities_preview": 3,
            "missing_kg_relations_preview": 3,
        }
    )

    assert "missing_preview=9" in output
    assert "missing_chunks_preview=3" in output
    assert "missing_kg_entities_preview=3" in output
    assert "missing_kg_relations_preview=3" in output


def test_vector_sync_dry_run_summary_counts_missing_records_by_type():
    summary = build_dry_run_summary(
        [
            MissingVectorRecord(id="chunk:1", text="a", content_type="chunk"),
            MissingVectorRecord(id="entity:1", text="b", content_type="kg_entity"),
            MissingVectorRecord(id="relation:1", text="c", content_type="kg_relation"),
            MissingVectorRecord(id="relation:2", text="d", content_type="kg_relation"),
        ]
    )

    assert summary == {
        "missing_preview": 4,
        "missing_chunks_preview": 1,
        "missing_kg_entities_preview": 1,
        "missing_kg_relations_preview": 2,
    }


def test_vector_sync_balanced_dry_run_uses_balanced_content_type_preview():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
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
                token_count=12,
            )
            for index in range(1, 4)
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
            )
            for index in range(1, 4)
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
            )
            for index in range(1, 4)
        ]
    )

    records = dry_run_missing_records(
        repository,
        content_types=["chunk", "kg_entity", "kg_relation"],
        limit=6,
        balanced=True,
    )

    assert build_dry_run_summary(records) == {
        "missing_preview": 6,
        "missing_chunks_preview": 2,
        "missing_kg_entities_preview": 2,
        "missing_kg_relations_preview": 2,
    }


def test_vector_sync_parser_accepts_reset_collection_flag(monkeypatch):
    from scripts import sync_ragflow_retrieval_vectors

    monkeypatch.setattr(
        "sys.argv",
        ["sync_ragflow_retrieval_vectors.py", "--reset-collection"],
    )

    args = sync_ragflow_retrieval_vectors.parse_args()

    assert args.reset_collection is True


def test_vector_sync_parser_accepts_retry_failed_flag(monkeypatch):
    from scripts import sync_ragflow_retrieval_vectors

    monkeypatch.setattr(
        "sys.argv",
        ["sync_ragflow_retrieval_vectors.py", "--retry-failed"],
    )

    args = sync_ragflow_retrieval_vectors.parse_args()

    assert args.retry_failed is True


def test_vector_sync_parser_accepts_workers_flag(monkeypatch):
    from scripts import sync_ragflow_retrieval_vectors

    monkeypatch.setattr(
        "sys.argv",
        ["sync_ragflow_retrieval_vectors.py", "--workers", "4"],
    )

    args = sync_ragflow_retrieval_vectors.parse_args()

    assert args.workers == 4


def test_vector_sync_parser_accepts_retry_tuning_flags(monkeypatch):
    from scripts import sync_ragflow_retrieval_vectors

    monkeypatch.setattr(
        "sys.argv",
        [
            "sync_ragflow_retrieval_vectors.py",
            "--max-attempts",
            "5",
            "--retry-backoff-seconds",
            "2.5",
        ],
    )

    args = sync_ragflow_retrieval_vectors.parse_args()

    assert args.max_attempts == 5
    assert args.retry_backoff_seconds == 2.5


def test_vector_sync_retry_failed_resets_selected_content_types():
    class FakeRepository:
        def __init__(self):
            self.content_types = None

        def reset_failed_vector_statuses(self, *, content_types):
            self.content_types = content_types
            return {
                "reset_failed_chunks": 2,
                "reset_failed_kg_entities": 0,
                "reset_failed_kg_relations": 1,
            }

    repository = FakeRepository()

    summary = reset_failed_records(
        repository,
        content_types=["chunk", "kg_relation"],
    )

    assert repository.content_types == ["chunk", "kg_relation"]
    assert summary == {
        "reset_failed_chunks": 2,
        "reset_failed_kg_entities": 0,
        "reset_failed_kg_relations": 1,
    }


def test_audit_formatter_includes_readiness_summary_when_available():
    output = format_audit(
        {
            "documents": 2,
            "chunks": 5,
            "readiness": {
                "ready": False,
                "blockers": ["chunk_vector_coverage_below_threshold"],
                "warnings": ["long_chunk_ratio_above_threshold"],
                "vector_coverage": {
                    "chunks": {"embedded": 1, "total": 5, "ratio": 0.2}
                },
                "evidence_coverage": {
                    "kg_entities": {"with_evidence": 2, "total": 5, "ratio": 0.4}
                },
                "chunk_token_buckets": {"lt_30": 1, "128_256": 3},
                "vector_sync_plan": {
                    "balanced_limit": 4,
                    "recommended_command": "python ../scripts/sync_ragflow_retrieval_vectors.py --balanced --limit 4 --content-types chunk,kg_entity,kg_relation",
                },
            },
        }
    )

    assert "ready=False" in output
    assert "blockers=chunk_vector_coverage_below_threshold" in output
    assert "warnings=long_chunk_ratio_above_threshold" in output
    assert "chunks_vector_ratio=0.2" in output
    assert "kg_entities_evidence_ratio=0.4" in output
    assert "kg_entities_evidence=2/5" in output
    assert "bucket_lt_30=1" in output
    assert "sync_limit=4" in output
    assert "sync_command='python ../scripts/sync_ragflow_retrieval_vectors.py --balanced --limit 4 --content-types chunk,kg_entity,kg_relation'" in output
