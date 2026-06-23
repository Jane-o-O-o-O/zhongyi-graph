from app.models.graph import EvidenceCard, GraphEdge, GraphNode
from app.services.knowledge_publisher import PublishedKnowledgeArtifact
from app.services.neo4j_publisher import Neo4jPublisher, build_merge_statements


def test_build_merge_statements_creates_nodes_edges_and_evidence():
    artifact = PublishedKnowledgeArtifact(
        nodes=[
            GraphNode(id="symptom:失眠", label="Symptom", name="失眠"),
            GraphNode(id="syndrome:心脾两虚", label="Syndrome", name="心脾两虚"),
        ],
        edges=[
            GraphEdge(
                id="edge:失眠:心脾两虚",
                source="symptom:失眠",
                target="syndrome:心脾两虚",
                relation="MANIFESTS_AS",
                display="可辨为",
                evidence_ids=["evidence:1"],
            )
        ],
        evidence=[
            EvidenceCard(
                id="evidence:1",
                title="资料.txt #1",
                source="资料.txt",
                snippet="失眠可辨为心脾两虚。",
                source_type="local",
                location="sources/demo/资料.txt:1",
            )
        ],
        vector_payloads=[],
    )

    statements = build_merge_statements(artifact)

    assert statements[0] == (
        "MERGE (n:Symptom {id: $id}) "
        "SET n.name = $name, n.label = $label, n.description = $description "
        "SET n += $properties",
        {
            "id": "symptom:失眠",
            "name": "失眠",
            "label": "Symptom",
            "description": "",
            "properties": {},
        },
    )
    assert statements[2][0].startswith("MATCH (a {id: $source}), (b {id: $target}) MERGE (a)-[r:MANIFESTS_AS")
    assert statements[3][0].startswith("MERGE (e:Evidence {id: $id})")
    assert statements[3][1]["snippet"] == "失眠可辨为心脾两虚。"


def test_neo4j_publisher_runs_all_merge_statements():
    executed = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def run(self, statement, params):
            executed.append((statement, params))

    class FakeDriver:
        def session(self):
            return FakeSession()

        def close(self):
            executed.append(("closed", {}))

    artifact = PublishedKnowledgeArtifact(
        nodes=[GraphNode(id="symptom:失眠", label="Symptom", name="失眠")],
        edges=[],
        evidence=[],
        vector_payloads=[],
    )

    Neo4jPublisher(FakeDriver()).publish(artifact)

    assert executed[0][0].startswith("MERGE (n:Symptom")
