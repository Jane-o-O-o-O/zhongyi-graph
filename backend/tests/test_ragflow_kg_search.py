from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.services.model_clients import EmbeddingClient
from app.services.ragflow_compat.doc_store import (
    EntitySearchHit,
    RagflowDocStore,
    RelationSearchHit,
)
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


def test_kg_search_includes_relation_endpoint_nodes_in_query_graph():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    headache = RetrievalKgEntity(
        entity_id="entity:头痛",
        entity_name="头痛",
        entity_type="Indication",
        source_node_id="indication:头痛",
        content_with_weight='{"description":"头痛 Indication"}',
        description="头痛 Indication",
        rank_flt=2.0,
    )
    herb = RetrievalKgEntity(
        entity_id="entity:白芍",
        entity_name="白芍",
        entity_type="Herb",
        source_node_id="herb:白芍",
        content_with_weight='{"description":"白芍 Herb"}',
        description="白芍 Herb",
        rank_flt=1.0,
    )
    relation = RetrievalKgRelation(
        relation_id="relation:白芍:头痛",
        from_entity_kwd="白芍",
        to_entity_kwd="头痛",
        relation_type="TREATS",
        display="主治",
        content_with_weight="白芍 主治 头痛",
        weight_int=2,
        evidence_chunk_ids=["chunk:1"],
    )
    repository.replace_kg_entities([headache, herb])
    repository.replace_kg_relations([relation])

    class EndpointDocStore:
        def __init__(self, repository):
            self.repository = repository

        def search_entities(self, query, keywords, **kwargs):
            del query, keywords, kwargs
            return [EntitySearchHit(entity=headache, score=1.0)]

        def search_relations(self, query, **kwargs):
            del query, kwargs
            return [RelationSearchHit(relation=relation, score=1.0)]

        def search_community_reports(self, entities, *, top_k=1):
            del entities, top_k
            return []

    result = RagflowKgSearch(EndpointDocStore(repository)).retrieve(
        "头痛怎么办",
        answer_type_keywords=[],
        entities_from_query=["头痛"],
    )

    graph_node_ids = {node.id for node in result.graph_nodes}
    assert {edge.source for edge in result.graph_edges}.issubset(graph_node_ids)
    assert {edge.target for edge in result.graph_edges}.issubset(graph_node_ids)
    assert "herb:白芍" in graph_node_ids
    assert "indication:头痛" in graph_node_ids


def test_kg_search_backfills_nhop_relation_details_from_repository():
    repository = _repository_with_relation()

    class NhopOnlyDocStore:
        def __init__(self, repository):
            self.repository = repository

        def search_entities(self, query, keywords, **kwargs):
            del query, keywords, kwargs
            return [
                EntitySearchHit(
                    entity=RetrievalKgEntity(
                        entity_id="entity:心脾两虚",
                        entity_name="心脾两虚",
                        entity_type="Syndrome",
                        source_node_id="syndrome:心脾两虚",
                        content_with_weight='{"description":"心脾两虚 Syndrome"}',
                        description="心脾两虚 Syndrome",
                        rank_flt=2.0,
                        n_hop_with_weight=[
                            {"path": ["心脾两虚", "归脾汤"], "weights": [4]}
                        ],
                        evidence_chunk_ids=["chunk:1"],
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

    result = RagflowKgSearch(NhopOnlyDocStore(repository)).retrieve(
        "心脾两虚",
        answer_type_keywords=[],
        entities_from_query=["心脾两虚"],
    )

    assert result.graph_edges[0].id == "edge:formula"
    assert result.graph_edges[0].relation == "RECOMMENDS_FORMULA"
    assert result.graph_edges[0].display == "推荐方剂"
    assert result.graph_edges[0].evidence_ids == ["chunk:1"]


def _repository_with_relation():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_kg_relations(
        [
            RetrievalKgRelation(
                relation_id="relation:心脾两虚:归脾汤",
                from_entity_kwd="心脾两虚",
                to_entity_kwd="归脾汤",
                relation_type="RECOMMENDS_FORMULA",
                display="推荐方剂",
                content_with_weight="心脾两虚 推荐方剂 归脾汤",
                weight_int=4,
                evidence_chunk_ids=["chunk:1"],
                source_edge_id="edge:formula",
            )
        ]
    )
    return repository


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
