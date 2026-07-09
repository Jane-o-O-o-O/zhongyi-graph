from app.services.neo4j_graph_loader import Neo4jGraphLoader, graph_service_from_neo4j


def test_neo4j_graph_loader_builds_graph_service_from_records():
    class FakeRecord(dict):
        def data(self):
            return dict(self)

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def run(self, statement):
            if "MATCH (n)" in statement:
                return [
                    FakeRecord(
                        id="formula:鳖甲汤",
                        label="Formula",
                        name="鳖甲汤",
                        description="",
                        properties={"source_chunks": ["evidence:structured:1"]},
                    ),
                    FakeRecord(
                        id="prescription:鳖甲汤_1",
                        label="Prescription",
                        name="鳖甲汤_1",
                        description="",
                        properties={},
                    ),
                ]
            return [
                FakeRecord(
                    id="edge:1",
                    source="formula:鳖甲汤",
                    target="prescription:鳖甲汤_1",
                    relation="HAS_PRESCRIPTION",
                    display="处方",
                    evidence_ids=[],
                )
            ]

    class FakeDriver:
        def session(self):
            return FakeSession()

        def close(self):
            pass

    service = Neo4jGraphLoader(FakeDriver()).load_graph_service()

    assert [node.id for node in service.nodes] == ["formula:鳖甲汤", "prescription:鳖甲汤_1"]
    assert service.nodes[0].properties["source_chunks"] == ["evidence:structured:1"]
    assert service.edges[0].source == "formula:鳖甲汤"
    assert service.edges[0].relation == "HAS_PRESCRIPTION"


def test_graph_service_from_neo4j_closes_driver_after_loading(monkeypatch):
    closed = []

    class FakeLoader:
        def __init__(self, driver):
            self.driver = driver

        def load_graph_service(self):
            return "graph-service"

    class FakeDriver:
        def close(self):
            closed.append(True)

    def fake_driver(uri, auth):
        assert uri == "bolt://neo4j:7687"
        assert auth == ("neo4j", "secret")
        return FakeDriver()

    monkeypatch.setattr("app.services.neo4j_graph_loader.GraphDatabase.driver", fake_driver)
    monkeypatch.setattr("app.services.neo4j_graph_loader.Neo4jGraphLoader", FakeLoader)

    service = graph_service_from_neo4j("bolt://neo4j:7687", "neo4j", "secret")

    assert service == "graph-service"
    assert closed == [True]
