from app.data.sample_graph import SAMPLE_EDGES, SAMPLE_EVIDENCE, SAMPLE_NODES
from app.services.graph_service import GraphService
from app.services.hybrid_retriever import HybridRetriever
from app.services.model_clients import EmbeddingClient, RerankClient
from app.services.vector_service import VectorIndexService
from app.models.graph import GraphEdge, GraphNode


def test_graph_service_expands_related_subgraph_from_node_ids():
    graph_service = GraphService.demo()

    nodes, edges = graph_service.related_to_node_ids(["symptom:失眠"])

    names = {node.name for node in nodes}
    assert "失眠" in names
    assert "心脾两虚" in names
    assert any(edge.source == "symptom:失眠" for edge in edges)


def test_graph_service_expands_related_subgraph_until_stable_with_unordered_edges():
    graph_service = GraphService(
        nodes=[
            GraphNode(id="symptom:不寐", label="Symptom", name="不寐"),
            GraphNode(id="syndrome:心脾两虚", label="Syndrome", name="心脾两虚"),
            GraphNode(id="treatment:补益心脾", label="Treatment", name="补益心脾"),
            GraphNode(id="formula:归脾汤", label="Formula", name="归脾汤"),
        ],
        edges=[
            GraphEdge(
                id="edge:treatment:formula",
                source="treatment:补益心脾",
                target="formula:归脾汤",
                relation="RECOMMENDS_FORMULA",
                display="推荐方剂",
            ),
            GraphEdge(
                id="edge:syndrome:treatment",
                source="syndrome:心脾两虚",
                target="treatment:补益心脾",
                relation="RECOMMENDS_TREATMENT",
                display="治法",
            ),
            GraphEdge(
                id="edge:symptom:syndrome",
                source="symptom:不寐",
                target="syndrome:心脾两虚",
                relation="MANIFESTS_AS",
                display="可辨为",
            ),
        ],
    )

    nodes, edges = graph_service.related_to_node_ids(["symptom:不寐"])

    assert {node.name for node in nodes} == {"不寐", "心脾两虚", "补益心脾", "归脾汤"}
    assert len(edges) == 3


def test_vector_retriever_recalls_semantic_entity_without_exact_keyword():
    vector_index = VectorIndexService.in_memory(
        embedding_client=EmbeddingClient.demo(),
        nodes=SAMPLE_NODES,
        evidence=SAMPLE_EVIDENCE,
    )

    hits = vector_index.search("睡不着怎么辨证", top_k=5, content_types=["entity"])

    assert hits
    assert hits[0].payload["node_id"] == "symptom:失眠"


def test_hybrid_retriever_merges_keyword_vector_and_rerank_candidates():
    graph_service = GraphService.demo()
    vector_index = VectorIndexService.in_memory(
        embedding_client=EmbeddingClient.demo(),
        nodes=SAMPLE_NODES,
        evidence=SAMPLE_EVIDENCE,
    )
    retriever = HybridRetriever(
        graph_service=graph_service,
        vector_index=vector_index,
        rerank_client=RerankClient.demo(),
    )

    result = retriever.retrieve(question="睡不着怎么辨证", terms=["睡不着"], top_k=5)

    names = {node.name for node in result.nodes}
    assert "失眠" in names
    assert result.edges
    assert result.evidence_ids


def test_hybrid_retriever_keeps_symptom_query_on_connected_clinical_path():
    graph_service = GraphService.demo()
    vector_index = VectorIndexService.in_memory(
        embedding_client=EmbeddingClient.demo(),
        nodes=SAMPLE_NODES,
        evidence=SAMPLE_EVIDENCE,
    )
    retriever = HybridRetriever(
        graph_service=graph_service,
        vector_index=vector_index,
        rerank_client=RerankClient.demo(),
    )

    result = retriever.retrieve(question="睡不着怎么辨证", terms=["睡不着"], top_k=5)

    names = {node.name for node in result.nodes}
    assert {"失眠", "心脾两虚", "补益心脾", "归脾汤", "党参"} <= names
    assert "柴胡桂枝干姜汤" not in names
    assert "往来寒热" not in names
    assert result.seed_node_ids[:4] == [
        "symptom:失眠",
        "syndrome:心脾两虚",
        "treatment:补益心脾",
        "formula:归脾汤",
    ]


def test_hybrid_retriever_keeps_formula_query_on_formula_branch():
    graph_service = GraphService.demo()
    vector_index = VectorIndexService.in_memory(
        embedding_client=EmbeddingClient.demo(),
        nodes=SAMPLE_NODES,
        evidence=SAMPLE_EVIDENCE,
    )
    retriever = HybridRetriever(
        graph_service=graph_service,
        vector_index=vector_index,
        rerank_client=RerankClient.demo(),
    )

    result = retriever.retrieve(question="柴胡桂枝干姜汤适合什么情况？", terms=["柴胡桂枝干姜汤"], top_k=5)

    names = {node.name for node in result.nodes}
    assert {"柴胡桂枝干姜汤", "柴胡", "桂枝", "干姜", "往来寒热"} <= names
    assert "失眠" not in names
    assert "归脾汤" not in names
    assert result.seed_node_ids[0] == "formula:柴胡桂枝干姜汤"
    assert "formula:归脾汤" not in result.seed_node_ids
    assert set(result.seed_node_ids) <= {node.id for node in result.nodes}


def test_hybrid_retriever_falls_back_to_keyword_when_vector_service_fails():
    class FailingVectorIndex:
        def search(self, question, top_k=10, content_types=None):
            raise RuntimeError("embedding service unavailable")

    retriever = HybridRetriever(
        graph_service=GraphService.demo(),
        vector_index=FailingVectorIndex(),
        rerank_client=RerankClient.demo(),
    )

    result = retriever.retrieve(question="失眠可以从哪些证候分析？", terms=["失眠"], top_k=5)

    names = {node.name for node in result.nodes}
    assert "失眠" in names
    assert result.edges
