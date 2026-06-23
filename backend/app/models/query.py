from pydantic import BaseModel, Field, field_validator

from app.models.graph import EvidenceCard, GraphEdge, GraphNode


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def trim_question(cls, value: str) -> str:
        return value.strip()


class QueryResponse(BaseModel):
    question: str
    answer: str
    intent: str
    entities: list[str] = Field(default_factory=list)
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    graph_edges: list[GraphEdge] = Field(default_factory=list)
    highlighted_path: list[str] = Field(default_factory=list)
    evidence: list[EvidenceCard] = Field(default_factory=list)
