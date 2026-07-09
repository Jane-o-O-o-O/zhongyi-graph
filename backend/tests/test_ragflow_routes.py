from app.api import routes


def test_retrieval_status_endpoint_reports_engine_and_audit_counts():
    response = routes.retrieval_status()

    assert response["retrieval_engine"] in {"legacy", "ragflow_compat"}
    assert "ragflow_compat" in response
    assert "chunks" in response["ragflow_compat"]


def test_question_service_delegates_to_ragflow_service_when_configured():
    captured = {}

    class FakeRagflowService:
        def answer(self, question):
            captured["question"] = question
            return "ragflow-response"

    service = routes.QuestionService.demo()
    service.retrieval_engine = "ragflow_compat"
    service.ragflow_retrieval_service = FakeRagflowService()

    assert service.answer("失眠怎么辨证？") == "ragflow-response"
    assert captured["question"] == "失眠怎么辨证？"
