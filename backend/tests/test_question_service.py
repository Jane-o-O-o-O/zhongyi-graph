from app.services.question_service import QuestionService


def test_question_service_uses_real_llm_client_when_api_key_is_configured():
    service = QuestionService.from_settings(
        llm_base_url="https://llm.example/v1",
        llm_api_key="secret-key",
        llm_model="demo-model",
    )

    assert service.llm_client.base_url == "https://llm.example/v1"
    assert service.llm_client.api_key == "secret-key"
    assert service.llm_client.model == "demo-model"


def test_question_service_keeps_deterministic_client_without_api_key():
    service = QuestionService.from_settings(
        llm_base_url="https://llm.example/v1",
        llm_api_key="replace-with-your-key",
        llm_model="demo-model",
    )

    response = service.answer("失眠可以从哪些证候分析？")

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
