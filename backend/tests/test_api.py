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


def test_register_ingestion_source_echoes_manifest():
    response = client.post(
        "/api/ingestion/sources",
        json={
            "source_id": "source:uploaded:test",
            "filename": "资料.pdf",
            "mime_type": "application/pdf",
            "checksum": "abc123",
            "status": "registered",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source_id"] == "source:uploaded:test"
    assert body["status"] == "registered"
    assert body["version"] == 1


def test_create_ingestion_job_accepts_bare_source_id_array():
    response = client.post("/api/ingestion/jobs", json=["source:uploaded:test"])

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["source_ids"] == ["source:uploaded:test"]
    assert body["job_id"].startswith("job:")
