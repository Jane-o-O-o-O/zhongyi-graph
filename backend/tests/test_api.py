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


def test_upload_run_publish_ingestion_flow_updates_query_sources():
    upload = client.post(
        "/api/ingestion/upload",
        files={"file": ("资料.txt", "失眠可辨为心脾两虚，治以补益心脾，常用归脾汤，药味包含党参。", "text/plain")},
    )

    assert upload.status_code == 200
    source_id = upload.json()["source_id"]
    job_id = client.post("/api/ingestion/jobs", json=[source_id]).json()["job_id"]

    run = client.post(f"/api/ingestion/jobs/{job_id}/run")
    assert run.status_code == 200
    assert run.json()["chunk_count"] >= 1
    assert run.json()["entity_count"] >= 5

    publish = client.post("/api/ingestion/publish", json=[source_id])
    assert publish.status_code == 200
    assert publish.json()["node_count"] >= 5
    assert publish.json()["edge_count"] >= 3

    chunks = client.get(f"/api/ingestion/sources/{source_id}/chunks")
    assert chunks.status_code == 200
    assert "失眠" in chunks.json()[0]["content"]

    query = client.post("/api/query", json={"question": "失眠怎么辨证？"})
    assert query.status_code == 200
    body = query.json()
    assert any(node["name"] == "心脾两虚" for node in body["graph_nodes"])
    assert any("失眠可辨为心脾两虚" in card["snippet"] for card in body["evidence"])


def test_publish_uses_repository_source_after_service_restart():
    upload = client.post(
        "/api/ingestion/upload",
        files={"file": ("重启资料.txt", "失眠可辨为心脾两虚，治以补益心脾。", "text/plain")},
    )
    source_id = upload.json()["source_id"]
    job_id = client.post("/api/ingestion/jobs", json=[source_id]).json()["job_id"]
    client.post(f"/api/ingestion/jobs/{job_id}/run")

    from app.api import routes

    routes.ingestion_service.sources.clear()
    publish = client.post("/api/ingestion/publish", json=[source_id])

    assert publish.status_code == 200
    assert publish.json()["status"] == "published"


def test_publish_persists_artifact_to_neo4j_publisher():
    from app.api import routes

    published = []

    class FakeNeo4jPublisher:
        def publish(self, artifact):
            published.append(artifact)

    previous_publisher = routes.neo4j_publisher
    routes.neo4j_publisher = FakeNeo4jPublisher()
    try:
        upload = client.post(
            "/api/ingestion/upload",
            files={"file": ("图谱持久化.txt", "失眠可辨为心脾两虚，治以补益心脾。", "text/plain")},
        )
        source_id = upload.json()["source_id"]
        job_id = client.post("/api/ingestion/jobs", json=[source_id]).json()["job_id"]
        client.post(f"/api/ingestion/jobs/{job_id}/run")

        publish = client.post("/api/ingestion/publish", json=[source_id])
    finally:
        routes.neo4j_publisher = previous_publisher

    assert publish.status_code == 200
    assert publish.json()["graph_persisted"] is True
    assert published
    assert any(node.name == "失眠" for node in published[0].nodes)
