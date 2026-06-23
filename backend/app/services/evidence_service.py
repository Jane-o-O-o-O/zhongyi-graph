from app.data.sample_graph import SAMPLE_EVIDENCE
from app.models.graph import EvidenceCard


class EvidenceService:
    def __init__(self, evidence: list[EvidenceCard]):
        self.evidence = {item.id: item for item in evidence}

    @classmethod
    def demo(cls) -> "EvidenceService":
        return cls(SAMPLE_EVIDENCE)

    def by_edge_ids(self, evidence_ids: list[str]) -> list[EvidenceCard]:
        seen: set[str] = set()
        cards: list[EvidenceCard] = []
        for evidence_id in evidence_ids:
            if evidence_id in self.evidence and evidence_id not in seen:
                cards.append(self.evidence[evidence_id])
                seen.add(evidence_id)
        return cards

    def upsert_many(self, evidence: list[EvidenceCard]) -> None:
        for item in evidence:
            self.evidence[item.id] = item
