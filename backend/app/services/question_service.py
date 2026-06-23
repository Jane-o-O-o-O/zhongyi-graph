from app.models.query import QueryResponse
from app.services.evidence_service import EvidenceService
from app.services.graph_service import GraphService
from app.services.llm import LlmClient


class QuestionService:
    def __init__(
        self,
        graph_service: GraphService,
        evidence_service: EvidenceService,
        llm_client: LlmClient,
    ):
        self.graph_service = graph_service
        self.evidence_service = evidence_service
        self.llm_client = llm_client

    @classmethod
    def demo(cls) -> "QuestionService":
        return cls(GraphService.demo(), EvidenceService.demo(), LlmClient())

    def answer(self, question: str) -> QueryResponse:
        terms = self._extract_terms(question)
        nodes, edges = self.graph_service.related_to_terms(terms)
        evidence_ids = [evidence_id for edge in edges for evidence_id in edge.evidence_ids]
        evidence = self.evidence_service.by_edge_ids(evidence_ids)
        entities = [node.name for node in nodes if any(term in node.name for term in terms)]
        answer = self.llm_client.synthesize(
            question=question,
            entities=entities,
            evidence=[card.snippet for card in evidence],
        )
        return QueryResponse(
            question=question,
            answer=answer,
            intent=self._infer_intent(question),
            entities=entities,
            graph_nodes=nodes,
            graph_edges=edges,
            highlighted_path=[node.id for node in nodes[:5]],
            evidence=evidence,
        )

    def _extract_terms(self, question: str) -> list[str]:
        known_terms = ["失眠", "柴胡桂枝干姜汤", "党参", "柴胡", "便秘", "头痛"]
        matches = [term for term in known_terms if term in question]
        return matches or [question[:8]]

    def _infer_intent(self, question: str) -> str:
        if any(term in question for term in ["汤", "方", "方剂"]):
            return "formula_inquiry"
        if any(term in question for term in ["药", "中药", "功效", "归经"]):
            return "herb_inquiry"
        return "symptom_inquiry"
