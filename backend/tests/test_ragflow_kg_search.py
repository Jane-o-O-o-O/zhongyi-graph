from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.services.model_clients import EmbeddingClient
from app.services.ragflow_compat.doc_store import EntitySearchHit, RagflowDocStore
from app.services.ragflow_compat.kg_search import RagflowKgSearch
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import (
    RetrievalCommunityReport,
    RetrievalKgEntity,
    RetrievalKgRelation,
    RetrievalTypeSamples,
)


def test_kg_search_uses_query_entities_for_entity_retrieval_query():
    class EmptyRepository:
        def list_kg_entities(self, *, available_only=False):
            del available_only
            return []

    class RecordingDocStore:
        def __init__(self):
            self.repository = EmptyRepository()
            self.entity_calls = []

        def search_entities(self, query, keywords, **kwargs):
            self.entity_calls.append((query, keywords, kwargs))
            return [
                EntitySearchHit(
                    entity=RetrievalKgEntity(
                        entity_id="entity:归脾汤",
                        entity_name="归脾汤",
                        entity_type="Formula",
                        source_node_id="formula:归脾汤",
                        content_with_weight='{"description":"归脾汤 Formula"}',
                        rank_flt=1.0,
                    ),
                    score=1.0,
                )
            ]

        def search_relations(self, query, **kwargs):
            del query, kwargs
            return []

        def search_community_reports(self, entities, *, top_k=1):
            del entities, top_k
            return []

    doc_store = RecordingDocStore()

    RagflowKgSearch(doc_store).retrieve(
        "睡不着用什么方？",
        answer_type_keywords=[],
        entities_from_query=["失眠", "归脾汤"],
    )

    assert doc_store.entity_calls == [
        (
            "失眠, 归脾汤",
            ["失眠", "归脾汤"],
            {"top_k": 56, "sim_threshold": 0.0},
        )
    ]


def test_kg_search_combines_entity_type_relation_and_nhop_scores():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id="entity:失眠",
                entity_name="失眠",
                entity_type="Symptom",
                source_node_id="symptom:失眠",
                content_with_weight='{"description":"失眠 Symptom"}',
                description="失眠 Symptom",
                rank_flt=2.0,
                n_hop_with_weight=[
                    {"path": ["失眠", "心脾两虚", "归脾汤"], "weights": [3, 4]}
                ],
                evidence_chunk_ids=["chunk:1"],
            ),
            RetrievalKgEntity(
                entity_id="entity:心脾两虚",
                entity_name="心脾两虚",
                entity_type="Syndrome",
                source_node_id="syndrome:心脾两虚",
                content_with_weight='{"description":"心脾两虚 Syndrome"}',
                description="心脾两虚 Syndrome",
                rank_flt=3.0,
                evidence_chunk_ids=["chunk:1"],
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
                weight_int=2,
                evidence_chunk_ids=["chunk:1"],
            )
        ]
    )
    repository.replace_type_samples(
        [RetrievalTypeSamples("Syndrome", ["心脾两虚"], 1, "2026-06-28T00:00:00Z")]
    )
    repository.replace_community_reports(
        [
            RetrievalCommunityReport(
                report_id="community:失眠",
                title="失眠、心脾两虚",
                content_with_weight="社区报告：失眠常与心脾两虚相关。",
                summary="失眠与心脾两虚相关",
                evidences="失眠 -> 心脾两虚",
                entities_kwd=["失眠", "心脾两虚"],
                weight_flt=0.8,
                source_id=["doc"],
            )
        ]
    )

    result = RagflowKgSearch(RagflowDocStore(repository, EmbeddingClient.demo())).retrieve(
        "失眠从哪些证候分析？",
        answer_type_keywords=["Syndrome"],
        entities_from_query=["失眠"],
    )

    assert result.entities[0].entity in {"失眠", "心脾两虚"}
    assert any(relation.to_entity == "心脾两虚" for relation in result.relations)
    assert result.community_reports[0].title == "失眠、心脾两虚"
    assert result.graph_edges[0].source == "symptom:失眠"
    assert result.graph_edges[0].target == "syndrome:心脾两虚"


def test_kg_search_honors_community_report_topn():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id="entity:失眠",
                entity_name="失眠",
                entity_type="Symptom",
                source_node_id="symptom:失眠",
                content_with_weight='{"description":"失眠 Symptom"}',
                description="失眠 Symptom",
                rank_flt=2.0,
                evidence_chunk_ids=["chunk:1"],
            )
        ]
    )
    repository.replace_community_reports(
        [
            RetrievalCommunityReport(
                report_id="community:失眠:1",
                title="失眠核心社区",
                content_with_weight="社区报告：失眠核心证候。",
                summary="失眠核心证候",
                evidences="失眠",
                entities_kwd=["失眠"],
                weight_flt=0.9,
                source_id=["doc"],
            ),
            RetrievalCommunityReport(
                report_id="community:失眠:2",
                title="失眠治疗社区",
                content_with_weight="社区报告：失眠治疗关系。",
                summary="失眠治疗关系",
                evidences="失眠",
                entities_kwd=["失眠"],
                weight_flt=0.8,
                source_id=["doc"],
            ),
        ]
    )

    result = RagflowKgSearch(RagflowDocStore(repository, EmbeddingClient.demo())).retrieve(
        "失眠怎么处理？",
        answer_type_keywords=[],
        entities_from_query=["失眠"],
        comm_topn=2,
    )

    assert [report.title for report in result.community_reports] == [
        "失眠核心社区",
        "失眠治疗社区",
    ]


def test_kg_search_does_not_promote_type_only_entities_without_query_overlap():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_kg_entities(
        [
            RetrievalKgEntity(
                entity_id="entity:失眠",
                entity_name="失眠",
                entity_type="Symptom",
                source_node_id="symptom:失眠",
                content_with_weight='{"description":"失眠 Symptom"}',
                rank_flt=2.0,
            ),
            RetrievalKgEntity(
                entity_id="entity:无关证候",
                entity_name="无关证候",
                entity_type="Syndrome",
                source_node_id="syndrome:无关证候",
                content_with_weight='{"description":"无关证候 Syndrome"}',
                rank_flt=100.0,
            ),
        ]
    )
    repository.replace_type_samples(
        [RetrievalTypeSamples("Syndrome", ["无关证候"], 1, "2026-06-28T00:00:00Z")]
    )

    result = RagflowKgSearch(RagflowDocStore(repository, EmbeddingClient.demo())).retrieve(
        "失眠从哪些证候分析？",
        answer_type_keywords=["Syndrome"],
        entities_from_query=["失眠"],
    )

    assert "失眠" in {entity.entity for entity in result.entities}
    assert "无关证候" not in {entity.entity for entity in result.entities}
