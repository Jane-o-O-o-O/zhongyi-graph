from pydantic import BaseModel, Field, field_validator

from app.models.graph import EvidenceCard, GraphEdge, GraphNode


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question", mode="before")
    @classmethod
    def trim_question(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("question must be a string")
        question = value.strip()
        if not question:
            raise ValueError("question must not be blank")
        return question


class QueryResponse(BaseModel):
    question: str
    answer: str
    intent: str
    entities: list[str] = Field(default_factory=list)
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    graph_edges: list[GraphEdge] = Field(default_factory=list)
    highlighted_path: list[str] = Field(default_factory=list)
    evidence: list[EvidenceCard] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)


class GraphOverviewResponse(BaseModel):
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    graph_edges: list[GraphEdge] = Field(default_factory=list)
    highlighted_path: list[str] = Field(default_factory=list)


class GraphBuildRequest(BaseModel):
    source_ids: list[str] | None = None
    with_resolution: bool = True
    with_community: bool = True

    @field_validator("source_ids", mode="before")
    @classmethod
    def normalize_source_ids(cls, value):
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("source_ids must be a list")
        source_ids: list[str] = []
        seen: set[str] = set()
        for item in value:
            source_id = str(item).strip()
            if not source_id or source_id in seen:
                continue
            source_ids.append(source_id)
            seen.add(source_id)
        return source_ids or None


class GraphBuildResponse(BaseModel):
    status: str
    sources_total: int
    sources_skipped: int
    sources_built: int
    sources_failed: int
    subgraphs_merged: int
    global_nodes: int
    global_edges: int
    graph_changed: bool
    resolution_marker_cleared: bool
    community_marker_cleared: bool
    resolution_marker_set: bool
    community_marker_set: bool
    graph_refreshed: bool
