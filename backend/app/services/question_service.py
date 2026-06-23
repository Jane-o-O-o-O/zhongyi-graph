from app.data.sample_graph import SAMPLE_EVIDENCE, SAMPLE_NODES
from app.models.query import QueryResponse
from app.models.graph import EvidenceCard
from app.services.evidence_service import EvidenceService
from app.services.graph_service import GraphService
from app.services.hybrid_retriever import HybridRetriever
from app.services.llm import LlmClient
from app.services.model_clients import EmbeddingClient, RerankClient, StructuredExtractionClient
from app.services.vector_service import VectorIndexService, build_vector_payloads
from app.services.knowledge_publisher import PublishedKnowledgeArtifact
from app.services.chunk_retriever import ChunkRetriever


class QuestionService:
    def __init__(
        self,
        graph_service: GraphService,
        evidence_service: EvidenceService,
        llm_client: LlmClient,
        vector_index: VectorIndexService,
        hybrid_retriever: HybridRetriever,
        chunk_retriever: ChunkRetriever | None = None,
        query_extractor=None,
    ):
        self.graph_service = graph_service
        self.evidence_service = evidence_service
        self.llm_client = llm_client
        self.vector_index = vector_index
        self.hybrid_retriever = hybrid_retriever
        self.chunk_retriever = chunk_retriever
        self.query_extractor = query_extractor

    @classmethod
    def demo(cls) -> "QuestionService":
        graph_service = GraphService.demo()
        evidence_service = EvidenceService.demo()
        vector_index = VectorIndexService.in_memory(
            embedding_client=EmbeddingClient.demo(),
            nodes=SAMPLE_NODES,
            evidence=SAMPLE_EVIDENCE,
        )
        hybrid_retriever = HybridRetriever(
            graph_service=graph_service,
            vector_index=vector_index,
            rerank_client=RerankClient.demo(),
        )
        return cls(
            graph_service=graph_service,
            evidence_service=evidence_service,
            llm_client=LlmClient.demo(),
            vector_index=vector_index,
            hybrid_retriever=hybrid_retriever,
            query_extractor=StructuredExtractionClient.demo(),
        )

    @classmethod
    def from_settings(
        cls,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        embedding_model: str,
        rerank_model: str,
        qdrant_url: str,
        qdrant_collection: str,
    ) -> "QuestionService":
        if llm_api_key and llm_api_key != "replace-with-your-key":
            llm_client = LlmClient(
                base_url=llm_base_url,
                api_key=llm_api_key,
                model=llm_model,
            )
            embedding_client = EmbeddingClient(
                base_url=llm_base_url,
                api_key=llm_api_key,
                model=embedding_model,
            )
            rerank_client = RerankClient(
                base_url=llm_base_url,
                api_key=llm_api_key,
                model=rerank_model,
            )
        else:
            llm_client = LlmClient.demo()
            embedding_client = EmbeddingClient.demo()
            rerank_client = RerankClient.demo()

        graph_service = GraphService.demo()
        evidence_service = EvidenceService.demo()
        vector_index = VectorIndexService(
            embedding_client=embedding_client,
            documents=build_vector_payloads(nodes=SAMPLE_NODES, evidence=SAMPLE_EVIDENCE),
            qdrant_url=qdrant_url,
            collection=qdrant_collection,
        )
        hybrid_retriever = HybridRetriever(
            graph_service=graph_service,
            vector_index=vector_index,
            rerank_client=rerank_client,
        )
        return cls(graph_service, evidence_service, llm_client, vector_index, hybrid_retriever)

    def publish_artifact(self, artifact: PublishedKnowledgeArtifact) -> None:
        existing_node_ids = {node.id for node in self.graph_service.nodes}
        existing_edge_ids = {edge.id for edge in self.graph_service.edges}
        self.graph_service.nodes.extend(
            node for node in artifact.nodes if node.id not in existing_node_ids
        )
        self.graph_service.edges.extend(
            edge for edge in artifact.edges if edge.id not in existing_edge_ids
        )
        self.evidence_service.upsert_many(artifact.evidence)

        existing_payload_ids = {document.id for document in self.vector_index.documents}
        self.vector_index.documents.extend(
            payload for payload in artifact.vector_payloads if payload.id not in existing_payload_ids
        )
        self.vector_index._vectors = None

    def answer(self, question: str) -> QueryResponse:
        terms = self._extract_terms(question)
        retrieval = self.hybrid_retriever.retrieve(question=question, terms=terms, top_k=8)
        nodes = retrieval.nodes
        edges = retrieval.edges
        evidence_ids = retrieval.evidence_ids
        evidence = self.evidence_service.by_edge_ids(evidence_ids)
        evidence.extend(self._retrieve_chunk_evidence(question, existing_ids={card.id for card in evidence}))
        entities = [node.name for node in nodes if any(term in node.name for term in terms)]
        if not entities:
            entities = [node.name for node in nodes[:5]]
        answer = self.llm_client.synthesize(
            question=question,
            entities=entities,
            evidence=[card.snippet for card in evidence],
            graph_paths=_format_graph_paths(nodes, edges),
        )
        return QueryResponse(
            question=question,
            answer=answer,
            intent=self._infer_intent(question),
            entities=entities,
            graph_nodes=nodes,
            graph_edges=edges,
            highlighted_path=retrieval.seed_node_ids[:5] or [node.id for node in nodes[:5]],
            evidence=evidence,
        )

    def _retrieve_chunk_evidence(self, question: str, existing_ids: set[str]) -> list[EvidenceCard]:
        if not self.chunk_retriever:
            return []
        chunks = self.chunk_retriever.retrieve(question, top_k=5)
        cards: list[EvidenceCard] = []
        for chunk in chunks:
            evidence_id = chunk.chunk_id.replace("chunk:", "evidence:", 1)
            if evidence_id in existing_ids:
                continue
            cards.append(
                EvidenceCard(
                    id=evidence_id,
                    title=f"{chunk.source_id} #{chunk.chunk_index}",
                    source=chunk.source_id,
                    snippet=chunk.content,
                    source_type="local",
                    location=f"{chunk.source_id}:{chunk.chunk_index}",
                )
            )
            existing_ids.add(evidence_id)
        return cards

    def _extract_terms(self, question: str) -> list[str]:
        if self.query_extractor:
            try:
                extracted = self.query_extractor.extract_query(question)
                entities = [
                    str(entity).strip()
                    for entity in extracted.get("entities", [])
                    if str(entity).strip()
                ]
                if entities:
                    return entities
            except Exception:
                pass
        compact = "".join(question.split()).strip("？?。！!，,；;")
        return [compact] if compact else []

    def _infer_intent(self, question: str) -> str:
        if any(term in question for term in ["汤", "方", "方剂"]):
            return "formula_inquiry"
        if any(term in question for term in ["药", "中药", "功效", "归经"]):
            return "herb_inquiry"
        return "symptom_inquiry"


def _format_graph_paths(nodes, edges) -> list[str]:
    node_names = {node.id: node.name for node in nodes}
    paths = []
    for edge in edges[:12]:
        source = node_names.get(edge.source, edge.source)
        target = node_names.get(edge.target, edge.target)
        paths.append(f"{source} -> {edge.display} -> {target}")
    return paths
