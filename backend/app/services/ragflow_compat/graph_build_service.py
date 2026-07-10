from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import time
from uuid import uuid4

from app.models.graph import GraphEdge, GraphNode
from app.models.ingestion import EntityCandidate, RelationCandidate
from app.services.graph_extractor import GraphExtractor
from app.services.graph_service import GraphService
from app.services.ingestion_repository import IngestionRepository
from app.services.ragflow_compat.community_reports import RagflowGraphCommunityReportService
from app.services.ragflow_compat.entity_resolution import RagflowGraphEntityResolutionService
from app.services.ragflow_compat.phase_markers import PHASE_COMMUNITY, PHASE_RESOLUTION
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import (
    RetrievalGraphArtifact,
    RetrievalGraphRagBuildRun,
)
from app.services.ragflow_compat.sync_service import RagflowRetrievalSyncService


@dataclass(frozen=True)
class RagflowGraphBuildSummary:
    run_id: str
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
    resolution_marker_set: bool = False
    community_marker_set: bool = False
    resolution_pairs_replayed: int = 0
    resolution_pairs_resolved: int = 0
    resolution_pairs_merged: int = 0
    community_reports_replayed: int = 0
    community_reports_generated: int = 0
    source_events: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class RagflowGraphBuildSubmission:
    run: RetrievalGraphRagBuildRun
    source_ids: list[str]
    with_resolution: bool
    with_community: bool


class RagflowGraphBuildAlreadyRunningError(RuntimeError):
    pass


class RagflowGraphBuildCanceledError(RuntimeError):
    pass


class RagflowGraphBuildService:
    def __init__(
        self,
        *,
        ingestion_repository: IngestionRepository,
        retrieval_repository: RagflowRetrievalRepository,
        graph_extractor: GraphExtractor,
        entity_resolution_service: RagflowGraphEntityResolutionService | None = None,
        community_report_service: RagflowGraphCommunityReportService | None = None,
        chunk_batch_size: int = 1000,
        batch_chunk_token_size: int = 4096,
        retry_attempts: int = 2,
        retry_backoff_seconds: float = 2.0,
        retry_backoff_max_seconds: float = 60.0,
        source_timeout_seconds: float = 600.0,
        method: str = "light",
        sleep_fn=time.sleep,
        monotonic_fn=time.monotonic,
    ):
        self.ingestion_repository = ingestion_repository
        self.retrieval_repository = retrieval_repository
        self.graph_extractor = graph_extractor
        self.entity_resolution_service = (
            entity_resolution_service or RagflowGraphEntityResolutionService()
        )
        self.community_report_service = (
            community_report_service or RagflowGraphCommunityReportService()
        )
        self.chunk_batch_size = chunk_batch_size
        self.batch_chunk_token_size = max(1, int(batch_chunk_token_size or 4096))
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.retry_backoff_max_seconds = max(
            self.retry_backoff_seconds,
            retry_backoff_max_seconds,
        )
        self.source_timeout_seconds = max(0.0, source_timeout_seconds)
        self.method = _normalize_method(method)
        self.sleep_fn = sleep_fn
        self.monotonic_fn = monotonic_fn

    def build(
        self,
        source_ids: list[str] | None = None,
        *,
        with_resolution: bool = True,
        with_community: bool = True,
    ) -> RagflowGraphBuildSummary:
        submission = self.submit(
            source_ids,
            with_resolution=with_resolution,
            with_community=with_community,
            execution_mode="sync",
        )
        return self.run_submitted(submission)

    def submit(
        self,
        source_ids: list[str] | None = None,
        *,
        with_resolution: bool = True,
        with_community: bool = True,
        execution_mode: str = "sync",
    ) -> RagflowGraphBuildSubmission:
        run_id = f"graphrag:build:{uuid4().hex}"
        started_at = _utc_now()
        selected_source_ids = self._source_ids(source_ids)
        metadata = self._run_metadata(
            source_ids=selected_source_ids,
            with_resolution=with_resolution,
            with_community=with_community,
            execution_mode=execution_mode,
        )
        if not self.retrieval_repository.claim_graphrag_build_lock(
            run_id,
            started_at=started_at,
            metadata=metadata,
        ):
            raise RagflowGraphBuildAlreadyRunningError("GraphRAG build is already running")
        run = RetrievalGraphRagBuildRun(
            run_id=run_id,
            status="running",
            started_at=started_at,
            finished_at="",
            total=len(selected_source_ids),
            processed=0,
            failed=0,
            metadata=metadata,
        )
        self.retrieval_repository.save_graphrag_build_run(run)
        return RagflowGraphBuildSubmission(
            run=run,
            source_ids=selected_source_ids,
            with_resolution=with_resolution,
            with_community=with_community,
        )

    def run_submitted(
        self,
        submission: RagflowGraphBuildSubmission,
    ) -> RagflowGraphBuildSummary:
        run_id = submission.run.run_id
        started_at = submission.run.started_at
        selected_source_ids = submission.source_ids
        with_resolution = submission.with_resolution
        with_community = submission.with_community
        try:
            try:
                summary = self._build_selected_sources(
                    run_id=run_id,
                    started_at=started_at,
                    selected_source_ids=selected_source_ids,
                    with_resolution=with_resolution,
                    with_community=with_community,
                )
            except RagflowGraphBuildCanceledError:
                current_run = self.retrieval_repository.get_graphrag_build_run(run_id)
                metadata = dict(current_run.metadata if current_run else {})
                metadata["cancel_requested"] = True
                self._save_build_run(
                    run_id=run_id,
                    status="canceled",
                    started_at=started_at,
                    finished_at=_utc_now(),
                    total=len(selected_source_ids),
                    processed=current_run.processed if current_run else 0,
                    failed=current_run.failed if current_run else 0,
                    metadata=metadata,
                )
                raise
            except Exception as exc:
                self._save_build_run(
                    run_id=run_id,
                    status="failed",
                    started_at=started_at,
                    finished_at=_utc_now(),
                    total=len(selected_source_ids),
                    processed=0,
                    failed=1,
                    metadata={
                        **self._run_metadata(
                            source_ids=selected_source_ids,
                            with_resolution=with_resolution,
                            with_community=with_community,
                            execution_mode=submission.run.metadata.get(
                                "execution_mode", "sync"
                            ),
                        ),
                        "error": repr(exc),
                    },
                )
                raise
            self._save_build_run(
                run_id=run_id,
                status="completed",
                started_at=started_at,
                finished_at=_utc_now(),
                total=summary.sources_total,
                processed=summary.sources_total,
                failed=summary.sources_failed,
                metadata={
                    **self._run_metadata(
                        source_ids=selected_source_ids,
                        with_resolution=with_resolution,
                        with_community=with_community,
                        execution_mode=submission.run.metadata.get(
                            "execution_mode", "sync"
                        ),
                    ),
                    "summary": asdict(summary),
                    "source_events": summary.source_events,
                },
            )
            return summary
        finally:
            self.retrieval_repository.release_graphrag_build_lock(run_id)

    def _run_metadata(
        self,
        *,
        source_ids: list[str],
        with_resolution: bool,
        with_community: bool,
        execution_mode: str,
    ) -> dict:
        return {
            "source_ids": source_ids,
            "with_resolution": with_resolution,
            "with_community": with_community,
            "batch_chunk_token_size": self.batch_chunk_token_size,
            "retry_attempts": self.retry_attempts,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "retry_backoff_max_seconds": self.retry_backoff_max_seconds,
            "source_timeout_seconds": self.source_timeout_seconds,
            "method": self.method,
            "execution_mode": execution_mode,
        }

    def _build_selected_sources(
        self,
        *,
        run_id: str,
        started_at: str,
        selected_source_ids: list[str],
        with_resolution: bool,
        with_community: bool,
    ) -> RagflowGraphBuildSummary:
        skipped = 0
        built = 0
        failed = 0
        source_events: list[dict] = []
        for source_id in selected_source_ids:
            self._raise_if_cancel_requested(run_id)
            if self.retrieval_repository.get_subgraph_artifact(source_id):
                skipped += 1
                self._record_source_progress(
                    run_id=run_id,
                    started_at=started_at,
                    selected_source_ids=selected_source_ids,
                    with_resolution=with_resolution,
                    with_community=with_community,
                    source_id=source_id,
                    status="skipped",
                    processed=skipped + built + failed,
                    failed=failed,
                    source_events=source_events,
                )
                continue
            chunks = self.ingestion_repository.list_chunks(source_id)
            if not chunks:
                failed += 1
                self._record_source_progress(
                    run_id=run_id,
                    started_at=started_at,
                    selected_source_ids=selected_source_ids,
                    with_resolution=with_resolution,
                    with_community=with_community,
                    source_id=source_id,
                    status="failed",
                    processed=skipped + built + failed,
                    failed=failed,
                    source_events=source_events,
                )
                continue
            try:
                entities, relations = self._extract_source_graph(chunks)
                nodes, edges = _graph_from_candidates(source_id, entities, relations)
                if not nodes:
                    failed += 1
                    self._record_source_progress(
                        run_id=run_id,
                        started_at=started_at,
                        selected_source_ids=selected_source_ids,
                        with_resolution=with_resolution,
                        with_community=with_community,
                        source_id=source_id,
                        status="failed",
                        processed=skipped + built + failed,
                        failed=failed,
                        source_events=source_events,
                    )
                    continue
                self.retrieval_repository.save_graph_artifact(
                    _subgraph_artifact(source_id, nodes, edges)
                )
                built += 1
            except Exception:
                failed += 1
                status = "failed"
            else:
                status = "built"
            self._record_source_progress(
                run_id=run_id,
                started_at=started_at,
                selected_source_ids=selected_source_ids,
                with_resolution=with_resolution,
                with_community=with_community,
                source_id=source_id,
                status=status,
                processed=skipped + built + failed,
                failed=failed,
                source_events=source_events,
            )
        self._raise_if_cancel_requested(run_id)
        merged_nodes, merged_edges, merged_sources = _merge_subgraph_artifacts(
            self.retrieval_repository.list_graph_artifacts(available_only=True)
        )
        resolution_pairs_replayed = 0
        resolution_pairs_resolved = 0
        resolution_pairs_merged = 0
        if with_resolution and (merged_nodes or merged_edges):
            resolution_result = self.entity_resolution_service.resolve(
                nodes=merged_nodes,
                edges=merged_edges,
                repository=self.retrieval_repository,
            )
            merged_nodes = resolution_result.nodes
            merged_edges = resolution_result.edges
            resolution_pairs_replayed = resolution_result.pairs_replayed
            resolution_pairs_resolved = resolution_result.pairs_resolved
            resolution_pairs_merged = resolution_result.pairs_merged
        if merged_nodes or merged_edges:
            self.retrieval_repository.save_graph_artifact(
                _global_graph_artifact(merged_nodes, merged_edges, merged_sources)
            )
        resolution_marker_cleared = False
        community_marker_cleared = False
        if built > 0:
            self.retrieval_repository.clear_graphrag_phase_markers(
                [PHASE_RESOLUTION, PHASE_COMMUNITY]
            )
            resolution_marker_cleared = True
            community_marker_cleared = True
        resolution_marker_set = False
        community_marker_set = False
        community_reports_replayed = 0
        community_reports_generated = 0
        if merged_nodes or merged_edges:
            graph_service = GraphService(merged_nodes, merged_edges)
            if with_community:
                community_result = self.community_report_service.summarize(
                    nodes=graph_service.nodes,
                    edges=graph_service.edges,
                    base_summaries=graph_service.community_summaries,
                    repository=self.retrieval_repository,
                )
                graph_service.community_summaries = community_result.summaries
                graph_service.community_summaries.apply_to_nodes(graph_service.nodes)
                community_reports_replayed = community_result.reports_replayed
                community_reports_generated = community_result.reports_generated
            RagflowRetrievalSyncService(
                ingestion_repository=self.ingestion_repository,
                retrieval_repository=self.retrieval_repository,
                graph_service=graph_service,
                write_graph_artifacts=False,
                mark_resolution_phase=False,
                mark_community_phase=False,
            ).rebuild_from_ingestion()
            if with_resolution:
                resolution_marker_set = self.retrieval_repository.set_graphrag_phase_marker(
                    PHASE_RESOLUTION
                )
            if with_community:
                community_marker_set = self.retrieval_repository.set_graphrag_phase_marker(
                    PHASE_COMMUNITY
                )
        return RagflowGraphBuildSummary(
            run_id=run_id,
            sources_total=len(selected_source_ids),
            sources_skipped=skipped,
            sources_built=built,
            sources_failed=failed,
            subgraphs_merged=len(merged_sources),
            global_nodes=len(merged_nodes),
            global_edges=len(merged_edges),
            graph_changed=built > 0,
            resolution_marker_cleared=resolution_marker_cleared,
            community_marker_cleared=community_marker_cleared,
            resolution_marker_set=resolution_marker_set,
            community_marker_set=community_marker_set,
            resolution_pairs_replayed=resolution_pairs_replayed,
            resolution_pairs_resolved=resolution_pairs_resolved,
            resolution_pairs_merged=resolution_pairs_merged,
            community_reports_replayed=community_reports_replayed,
            community_reports_generated=community_reports_generated,
            source_events=source_events,
        )

    def _extract_source_graph(
        self,
        chunks,
    ) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                started_at = self.monotonic_fn()
                result = self.graph_extractor.extract(chunks)
                elapsed = self.monotonic_fn() - started_at
                if self.source_timeout_seconds and elapsed > self.source_timeout_seconds:
                    raise TimeoutError(
                        f"GraphRAG source extraction timed out after {elapsed:.3f}s"
                    )
                return result
            except Exception as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    raise
                self._sleep_before_retry(attempt)
        raise last_error or RuntimeError("GraphRAG source extraction failed")

    def _sleep_before_retry(self, failed_attempt: int) -> None:
        if self.retry_backoff_seconds <= 0:
            return
        wait_seconds = min(
            self.retry_backoff_max_seconds,
            self.retry_backoff_seconds * (2 ** (failed_attempt - 1)),
        )
        if wait_seconds > 0:
            self.sleep_fn(wait_seconds)

    def _source_ids(self, source_ids: list[str] | None) -> list[str]:
        if source_ids is not None:
            return list(dict.fromkeys(source_ids))
        return [source.source_id for source in self.ingestion_repository.list_sources()]

    def _save_build_run(
        self,
        *,
        run_id: str,
        status: str,
        started_at: str,
        finished_at: str,
        total: int,
        processed: int,
        failed: int,
        metadata: dict,
    ) -> None:
        self.retrieval_repository.save_graphrag_build_run(
            RetrievalGraphRagBuildRun(
                run_id=run_id,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                total=total,
                processed=processed,
                failed=failed,
                metadata=metadata,
            )
        )

    def _record_source_progress(
        self,
        *,
        run_id: str,
        started_at: str,
        selected_source_ids: list[str],
        with_resolution: bool,
        with_community: bool,
        source_id: str,
        status: str,
        processed: int,
        failed: int,
        source_events: list[dict],
    ) -> None:
        source_events.append(
            _source_event(
                source_id,
                status=status,
                processed=processed,
                failed=failed,
            )
        )
        self._save_progress_run(
            run_id=run_id,
            started_at=started_at,
            selected_source_ids=selected_source_ids,
            with_resolution=with_resolution,
            with_community=with_community,
            current_source_id=source_id,
            processed=processed,
            failed=failed,
            source_events=source_events,
        )

    def _raise_if_cancel_requested(self, run_id: str) -> None:
        if self.retrieval_repository.is_graphrag_build_cancel_requested(run_id):
            raise RagflowGraphBuildCanceledError("GraphRAG build was canceled")

    def _save_progress_run(
        self,
        *,
        run_id: str,
        started_at: str,
        selected_source_ids: list[str],
        with_resolution: bool,
        with_community: bool,
        current_source_id: str,
        processed: int,
        failed: int,
        source_events: list[dict],
    ) -> None:
        current_run = self.retrieval_repository.get_graphrag_build_run(run_id)
        current_metadata = current_run.metadata if current_run else {}
        metadata = {
            "source_ids": selected_source_ids,
            "with_resolution": with_resolution,
            "with_community": with_community,
            "batch_chunk_token_size": self.batch_chunk_token_size,
            "retry_attempts": self.retry_attempts,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "retry_backoff_max_seconds": self.retry_backoff_max_seconds,
            "source_timeout_seconds": self.source_timeout_seconds,
            "method": self.method,
            "execution_mode": current_metadata.get("execution_mode", "sync"),
            "current_source_id": current_source_id,
            "source_events": list(source_events),
        }
        if current_metadata.get("cancel_requested"):
            metadata["cancel_requested"] = True
        self._save_build_run(
            run_id=run_id,
            status="running",
            started_at=started_at,
            finished_at="",
            total=len(selected_source_ids),
            processed=processed,
            failed=failed,
            metadata=metadata,
        )


def _graph_from_candidates(
    source_id: str,
    entities: list[EntityCandidate],
    relations: list[RelationCandidate],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes_by_id = {
        entity.entity_id: GraphNode(
            id=entity.entity_id,
            label=entity.label,
            name=entity.name,
            description=f"{entity.name} {entity.label}",
            properties={
                "source_id": source_id,
                "source_chunk_ids": entity.source_chunk_ids,
                "confidence": entity.confidence,
            },
        )
        for entity in entities
    }
    edges: list[GraphEdge] = []
    for relation in relations:
        if relation.source_entity_id not in nodes_by_id or relation.target_entity_id not in nodes_by_id:
            continue
        edges.append(
            GraphEdge(
                id=relation.relation_id,
                source=relation.source_entity_id,
                target=relation.target_entity_id,
                relation=relation.relation,
                display=relation.display,
                evidence_ids=relation.evidence_chunk_ids,
            )
        )
    return list(nodes_by_id.values()), edges


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _source_event(
    source_id: str,
    *,
    status: str,
    processed: int,
    failed: int,
) -> dict:
    return {
        "source_id": source_id,
        "status": status,
        "processed": processed,
        "failed": failed,
    }


def _normalize_method(method: str) -> str:
    normalized = str(method or "light").strip().lower()
    return normalized if normalized in {"light", "general", "ner"} else "light"


def _subgraph_artifact(
    source_id: str,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> RetrievalGraphArtifact:
    return RetrievalGraphArtifact(
        artifact_id=f"subgraph:{source_id}",
        artifact_type="subgraph",
        content_with_weight=json.dumps(_graph_payload(nodes, edges), ensure_ascii=False),
        source_id=[source_id],
        node_count=len(nodes),
        edge_count=len(edges),
        metadata={"scope": "source", "built_from": "graph_extractor"},
    )


def _graph_payload(nodes: list[GraphNode], edges: list[GraphEdge]) -> dict:
    return {
        "nodes": [node.model_dump() for node in nodes],
        "edges": [edge.model_dump() for edge in edges],
    }


def _merge_subgraph_artifacts(artifacts) -> tuple[list[GraphNode], list[GraphEdge], list[str]]:
    nodes_by_id: dict[str, GraphNode] = {}
    edges_by_id: dict[str, GraphEdge] = {}
    merged_sources: set[str] = set()
    for artifact in sorted(artifacts, key=lambda item: item.artifact_id):
        if artifact.artifact_type != "subgraph" or artifact.available_int != 1:
            continue
        payload = json.loads(artifact.content_with_weight)
        merged_sources.update(str(source_id) for source_id in artifact.source_id)
        for node_data in payload.get("nodes", []):
            node = GraphNode.model_validate(node_data)
            existing = nodes_by_id.get(node.id)
            if existing:
                node = _merge_node(existing, node)
            nodes_by_id[node.id] = node
        for edge_data in payload.get("edges", []):
            edge = GraphEdge.model_validate(edge_data)
            existing = edges_by_id.get(edge.id)
            if existing:
                edge = _merge_edge(existing, edge)
            edges_by_id[edge.id] = edge
    return (
        [nodes_by_id[node_id] for node_id in sorted(nodes_by_id)],
        [edges_by_id[edge_id] for edge_id in sorted(edges_by_id)],
        sorted(merged_sources),
    )


def _merge_node(left: GraphNode, right: GraphNode) -> GraphNode:
    properties = dict(left.properties)
    for key, value in right.properties.items():
        if key == "source_chunk_ids":
            properties[key] = sorted(
                set(str(item) for item in properties.get(key, []))
                | set(str(item) for item in value)
            )
        elif key not in properties or not properties[key]:
            properties[key] = value
    return left.model_copy(update={"properties": properties})


def _merge_edge(left: GraphEdge, right: GraphEdge) -> GraphEdge:
    evidence_ids = sorted(set(left.evidence_ids) | set(right.evidence_ids))
    return left.model_copy(update={"evidence_ids": evidence_ids})


def _global_graph_artifact(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    source_ids: list[str],
) -> RetrievalGraphArtifact:
    return RetrievalGraphArtifact(
        artifact_id="graph:global",
        artifact_type="graph",
        content_with_weight=json.dumps(_graph_payload(nodes, edges), ensure_ascii=False),
        source_id=source_ids,
        node_count=len(nodes),
        edge_count=len(edges),
        metadata={"scope": "global", "built_from": "subgraphs"},
    )
