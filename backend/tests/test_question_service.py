from app.services.question_service import QuestionService
from app.models.ingestion import DocumentChunk


def test_question_service_uses_real_llm_client_when_api_key_is_configured():
    service = QuestionService.from_settings(
        llm_base_url="https://llm.example/v1",
        llm_api_key="secret-key",
        llm_model="demo-model",
        embedding_model="Qwen/Qwen3-Embedding-8B",
        rerank_model="Qwen/Qwen3-Reranker-8B",
        qdrant_url="http://qdrant:6333",
        qdrant_collection="tcm_knowledge",
    )

    assert service.llm_client.base_url == "https://llm.example/v1"
    assert service.llm_client.api_key == "secret-key"
    assert service.llm_client.model == "demo-model"
    assert service.vector_index.collection == "tcm_knowledge"


def test_question_service_keeps_deterministic_client_without_api_key():
    service = QuestionService.from_settings(
        llm_base_url="https://llm.example/v1",
        llm_api_key="replace-with-your-key",
        llm_model="demo-model",
        embedding_model="Qwen/Qwen3-Embedding-8B",
        rerank_model="Qwen/Qwen3-Reranker-8B",
        qdrant_url="",
        qdrant_collection="tcm_knowledge",
    )

    response = service.answer("失眠可以从哪些证候分析？")

    assert response.answer.startswith("### 综合结论")
    assert "失眠" in response.answer


def test_question_service_returns_graph_first_response_for_symptom_question():
    service = QuestionService.demo()

    response = service.answer("失眠可以从哪些证候分析？")

    assert response.question == "失眠可以从哪些证候分析？"
    assert response.intent == "symptom_inquiry"
    assert "失眠" in response.entities
    assert len(response.graph_nodes) >= 4
    assert len(response.graph_edges) >= 3
    assert response.highlighted_path[0] == "symptom:失眠"
    assert "失眠" in response.answer
    assert response.evidence


def test_question_service_returns_formula_path_for_formula_question():
    service = QuestionService.demo()

    response = service.answer("柴胡桂枝干姜汤适合什么情况？")

    names = {node.name for node in response.graph_nodes}
    assert "柴胡桂枝干姜汤" in names
    assert "柴胡" in names
    assert any(edge.relation == "COMPOSED_OF" for edge in response.graph_edges)


def test_question_service_uses_vector_recall_for_semantic_symptom_query():
    service = QuestionService.demo()

    response = service.answer("睡不着怎么辨证？")

    names = {node.name for node in response.graph_nodes}
    assert "失眠" in names
    assert "心脾两虚" in names
    assert response.evidence


def test_question_service_expands_path_for_syndrome_question():
    service = QuestionService.demo()

    response = service.answer("心脾两虚怎么处理？")

    names = {node.name for node in response.graph_nodes}
    assert {"心脾两虚", "补益心脾", "归脾汤"} <= names
    assert response.evidence


def test_question_service_extracts_core_graph_terms_from_question():
    service = QuestionService.demo()

    assert service._extract_terms("心脾两虚怎么处理？") == ["心脾两虚"]
    assert service._extract_terms("归脾汤里面党参有什么作用？") == ["归脾汤", "党参"]


def test_question_service_includes_vector_recalled_chunks_as_evidence():
    class FakeChunkRetriever:
        def retrieve(self, question, top_k=5):
            return [
                DocumentChunk(
                    chunk_id="chunk:source:uploaded:abc:0001",
                    source_id="source:uploaded:abc",
                    page_id="page:source:uploaded:abc:1",
                    chunk_index=1,
                    content="归脾汤可用于心脾两虚所致不寐。",
                )
            ]

    service = QuestionService.demo()
    service.chunk_retriever = FakeChunkRetriever()

    response = service.answer("归脾汤适合什么情况？")

    assert any("归脾汤可用于心脾两虚所致不寐" in card.snippet for card in response.evidence)


def test_question_service_uses_llm_query_entities_before_keyword_fallback():
    class FakeQueryExtractor:
        def extract_query(self, question):
            assert question == "偏头疼和眩晕有什么关系？"
            return {"entities": ["偏头疼", "眩晕"], "relations": ["相关"]}

    service = QuestionService.demo()
    service.query_extractor = FakeQueryExtractor()

    assert service._extract_terms("偏头疼和眩晕有什么关系？") == ["偏头疼", "眩晕"]


def test_question_service_merges_llm_query_entities_and_expanded_entities():
    class FakeQueryExtractor:
        def extract_query(self, question):
            assert question == "头痛和肝有什么关联？"
            return {
                "entities": ["头痛", "肝"],
                "expanded_entities": ["肝阳上亢", "肝火上炎", "头痛", "头风"],
                "relations": ["相关"],
            }

    service = QuestionService.demo()
    service.query_extractor = FakeQueryExtractor()

    assert service._extract_terms("头痛和肝有什么关联？") == [
        "头痛",
        "肝",
        "肝阳上亢",
        "肝火上炎",
        "头风",
    ]


def test_question_service_sends_expanded_entities_into_retrieval_terms():
    captured = {}

    class FakeQueryExtractor:
        def extract_query(self, question):
            return {
                "entities": ["头痛", "肝"],
                "expanded_entities": ["肝阳上亢", "肝火上炎"],
                "relations": ["相关"],
            }

    class FakeRetriever:
        def retrieve(self, question, terms, top_k=8):
            captured["terms"] = terms
            return type(
                "Retrieval",
                (),
                {"nodes": [], "edges": [], "evidence_ids": [], "seed_node_ids": []},
            )()

    service = QuestionService.demo()
    service.query_extractor = FakeQueryExtractor()
    service.hybrid_retriever = FakeRetriever()

    service.answer("头痛和肝有什么关联？")

    assert captured["terms"] == ["头痛", "肝", "肝阳上亢", "肝火上炎"]


def test_question_service_uses_question_text_when_query_extractor_fails():
    class BrokenQueryExtractor:
        def extract_query(self, question):
            raise RuntimeError("llm unavailable")

    service = QuestionService.demo()
    service.query_extractor = BrokenQueryExtractor()

    assert service._extract_terms("头痛和归脾汤有什么关系？") == ["头痛和归脾汤有什么关系"]
