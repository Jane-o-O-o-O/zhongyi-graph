from __future__ import annotations

from dataclasses import dataclass

from app.models.graph import GraphEdge, GraphNode
from app.services.graph_community_summary_service import (
    CommunitySummary,
    GraphCommunitySummaryResult,
)
from app.services.ragflow_compat.checkpoints import (
    COMMUNITY_CHECKPOINT,
    community_checkpoint_key,
)
from app.services.ragflow_compat.repository import RagflowRetrievalRepository


@dataclass(frozen=True)
class RagflowGraphCommunityReportResult:
    summaries: GraphCommunitySummaryResult
    reports_replayed: int = 0
    reports_generated: int = 0


class HeuristicCommunityReportClient:
    def generate_community_report(self, *, community_id, entities, relations):
        title = "、".join(entities[:3]) or f"社区 {community_id}"
        return {
            "title": title,
            "summary": (
                f"{title} 相关社区，包含 {len(entities)} 个实体；"
                f"主要关系：{'；'.join(relations[:5]) or '暂无关系'}。"
            ),
            "findings": [
                {
                    "summary": "社区主题",
                    "explanation": "该社区由图谱结构自动聚合生成。",
                }
            ],
            "rating": 1.0,
            "rating_explanation": "heuristic",
        }


class RagflowGraphCommunityReportService:
    def __init__(self, report_client=None):
        self.report_client = report_client or HeuristicCommunityReportClient()

    def summarize(
        self,
        *,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        base_summaries: GraphCommunitySummaryResult,
        repository: RagflowRetrievalRepository,
    ) -> RagflowGraphCommunityReportResult:
        checkpoints = repository.load_graphrag_checkpoints(COMMUNITY_CHECKPOINT)
        reports_replayed = 0
        reports_generated = 0
        summaries: dict[int, CommunitySummary] = {}
        nodes_by_community = _nodes_by_community(nodes)
        for community_id, base_summary in sorted(base_summaries.by_community_id.items()):
            community_nodes = nodes_by_community.get(community_id, [])
            entity_names = _ordered_entity_names(base_summary, community_nodes)
            checkpoint_key = community_checkpoint_key("0", str(community_id), entity_names)
            report = checkpoints.get(checkpoint_key)
            if isinstance(report, dict):
                reports_replayed += 1
            else:
                report = self.report_client.generate_community_report(
                    community_id=community_id,
                    entities=entity_names,
                    relations=_community_relations(edges, {node.id for node in community_nodes}),
                )
                repository.save_graphrag_checkpoint(COMMUNITY_CHECKPOINT, checkpoint_key, report)
                reports_generated += 1
            summaries[community_id] = _summary_from_report(
                community_id,
                base_summary,
                report,
                entity_names,
            )
        if reports_replayed or reports_generated:
            repository.cleanup_graphrag_checkpoints(COMMUNITY_CHECKPOINT)
        return RagflowGraphCommunityReportResult(
            summaries=GraphCommunitySummaryResult(summaries),
            reports_replayed=reports_replayed,
            reports_generated=reports_generated,
        )


def _nodes_by_community(nodes: list[GraphNode]) -> dict[int, list[GraphNode]]:
    groups: dict[int, list[GraphNode]] = {}
    for node in nodes:
        community_id = int(node.properties.get("community_id", 0))
        groups.setdefault(community_id, []).append(node)
    return {
        community_id: sorted(group_nodes, key=lambda node: (node.name, node.id))
        for community_id, group_nodes in groups.items()
    }


def _ordered_entity_names(
    base_summary: CommunitySummary,
    community_nodes: list[GraphNode],
) -> list[str]:
    available_names = {node.name for node in community_nodes}
    names = [name for name in base_summary.entities if name in available_names]
    for node in community_nodes:
        if node.name not in names:
            names.append(node.name)
    return names


def _community_relations(edges: list[GraphEdge], node_ids: set[str]) -> list[str]:
    return [
        f"{edge.source} {edge.display} {edge.target}"
        for edge in edges
        if edge.source in node_ids and edge.target in node_ids
    ]


def _summary_from_report(
    community_id: int,
    base_summary: CommunitySummary,
    report,
    entity_names: list[str],
) -> CommunitySummary:
    if not isinstance(report, dict):
        report = {}
    title = str(report.get("title") or base_summary.title)
    summary = str(report.get("summary") or base_summary.summary)
    return CommunitySummary(
        community_id=community_id,
        title=title,
        summary=summary,
        size=base_summary.size,
        weight=base_summary.weight,
        entities=entity_names or base_summary.entities,
        label_counts=base_summary.label_counts,
    )
