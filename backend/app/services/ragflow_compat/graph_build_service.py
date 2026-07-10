from __future__ import annotations

from dataclasses import dataclass

from app.services.graph_extractor import GraphExtractor
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.repository import RagflowRetrievalRepository


@dataclass(frozen=True)
class RagflowGraphBuildSummary:
    sources_total: int
    sources_skipped: int
    sources_built: int
    sources_failed: int
    subgraphs_merged: int = 0
    global_nodes: int = 0
    global_edges: int = 0
    graph_changed: bool = False
    resolution_marker_cleared: bool = False
    community_marker_cleared: bool = False


class RagflowGraphBuildService:
    def __init__(
        self,
        *,
        ingestion_repository: IngestionRepository,
        retrieval_repository: RagflowRetrievalRepository,
        graph_extractor: GraphExtractor,
        chunk_batch_size: int = 1000,
    ):
        self.ingestion_repository = ingestion_repository
        self.retrieval_repository = retrieval_repository
        self.graph_extractor = graph_extractor
        self.chunk_batch_size = chunk_batch_size

    def build(self, source_ids: list[str] | None = None) -> RagflowGraphBuildSummary:
        selected_source_ids = self._source_ids(source_ids)
        skipped = 0
        failed = 0
        for source_id in selected_source_ids:
            if self.retrieval_repository.get_subgraph_artifact(source_id):
                skipped += 1
                continue
            failed += 1
        return RagflowGraphBuildSummary(
            sources_total=len(selected_source_ids),
            sources_skipped=skipped,
            sources_built=0,
            sources_failed=failed,
        )

    def _source_ids(self, source_ids: list[str] | None) -> list[str]:
        if source_ids is not None:
            return list(dict.fromkeys(source_ids))
        return [source.source_id for source in self.ingestion_repository.list_sources()]
