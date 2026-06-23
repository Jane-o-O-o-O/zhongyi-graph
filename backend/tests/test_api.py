from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint_reports_services():
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["services"]["graph"] == "ready"


def test_query_endpoint_returns_graph_response():
    response = client.post("/api/query", json={"question": "失眠可以从哪些证候分析？"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "失眠可以从哪些证候分析？"
    assert body["graph_nodes"]
    assert body["graph_edges"]
    assert body["evidence"]
