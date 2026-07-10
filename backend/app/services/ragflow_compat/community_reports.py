from __future__ import annotations

from collections import Counter
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
            report, replayed = _load_or_generate_report(
                checkpoint_level=0,
                checkpoint_community_id=community_id,
                client_community_id=community_id,
                entity_names=entity_names,
                relation_texts=_community_relations(edges, {node.id for node in community_nodes}),
                checkpoints=checkpoints,
                repository=repository,
                report_client=self.report_client,
            )
            if replayed:
                reports_replayed += 1
            else:
                reports_generated += 1
            summaries[community_id] = _summary_from_report(
                community_id,
                base_summary,
                report,
                entity_names,
                level=0,
                source_node_ids=[node.id for node in community_nodes],
            )
        for (level, community_id), community_nodes in _nodes_by_level_community(nodes).items():
            if level == 0 or len(community_nodes) < 2:
                continue
            summary_id = _summary_id(level, community_id)
            if summary_id in summaries:
                continue
            base_summary = _summary_from_nodes(summary_id, level, community_nodes)
            entity_names = list(base_summary.entities)
            report, replayed = _load_or_generate_report(
                checkpoint_level=level,
                checkpoint_community_id=community_id,
                client_community_id=summary_id,
                entity_names=entity_names,
                relation_texts=_community_relations(edges, {node.id for node in community_nodes}),
                checkpoints=checkpoints,
                repository=repository,
                report_client=self.report_client,
            )
            if replayed:
                reports_replayed += 1
            else:
                reports_generated += 1
            summaries[summary_id] = _summary_from_report(
                summary_id,
                base_summary,
                report,
                entity_names,
                level=level,
                source_node_ids=[node.id for node in community_nodes],
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


def _nodes_by_level_community(nodes: list[GraphNode]) -> dict[tuple[int, int], list[GraphNode]]:
    groups: dict[tuple[int, int], list[GraphNode]] = {}
    for node in nodes:
        for level, community_id in _community_levels(node).items():
            groups.setdefault((level, community_id), []).append(node)
    return {
        key: sorted(group_nodes, key=lambda node: node.id)
        for key, group_nodes in sorted(groups.items())
    }


def _community_levels(node: GraphNode) -> dict[int, int]:
    value = node.properties.get("community_levels", [])
    if isinstance(value, dict):
        return {
            int(level): int(community_id)
            for level, community_id in value.items()
            if _is_int_like(level) and _is_int_like(community_id)
        }
    if not isinstance(value, list):
        return {}
    levels: dict[int, int] = {}
    for item in value:
        if not isinstance(item, str) or ":" not in item:
            continue
        level, community_id = item.split(":", 1)
        if _is_int_like(level) and _is_int_like(community_id):
            levels[int(level)] = int(community_id)
    return levels


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


def _load_or_generate_report(
    *,
    checkpoint_level: int,
    checkpoint_community_id: int,
    client_community_id: int,
    entity_names: list[str],
    relation_texts: list[str],
    checkpoints: dict,
    repository: RagflowRetrievalRepository,
    report_client,
) -> tuple[dict, bool]:
    checkpoint_key = community_checkpoint_key(
        str(checkpoint_level),
        str(checkpoint_community_id),
        entity_names,
    )
    report = checkpoints.get(checkpoint_key)
    if isinstance(report, dict):
        return report, True
    report = report_client.generate_community_report(
        community_id=client_community_id,
        entities=entity_names,
        relations=relation_texts,
    )
    repository.save_graphrag_checkpoint(COMMUNITY_CHECKPOINT, checkpoint_key, report)
    return report, False


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
    *,
    level: int = 0,
    source_node_ids: list[str] | None = None,
) -> CommunitySummary:
    if not isinstance(report, dict):
        report = {}
    title = str(report.get("title") or base_summary.title)
    summary = str(report.get("summary") or base_summary.summary)
    findings = report.get("findings", [])
    if not isinstance(findings, list):
        findings = []
    try:
        rating = float(report.get("rating", 0.0) or 0.0)
    except (TypeError, ValueError):
        rating = 0.0
    return CommunitySummary(
        community_id=community_id,
        title=title,
        summary=summary,
        size=base_summary.size,
        weight=base_summary.weight,
        entities=entity_names or base_summary.entities,
        label_counts=base_summary.label_counts,
        findings=[finding for finding in findings if isinstance(finding, (dict, str))],
        rating=rating,
        rating_explanation=str(report.get("rating_explanation") or ""),
        level=level,
        source_node_ids=source_node_ids or base_summary.source_node_ids,
    )


def _summary_from_nodes(
    summary_id: int,
    level: int,
    community_nodes: list[GraphNode],
) -> CommunitySummary:
    label_counts = Counter(node.label for node in community_nodes)
    entity_names = [node.name for node in community_nodes]
    title_names = entity_names[:3] or [f"社区 {summary_id}"]
    title = "、".join(title_names)
    label_text = "、".join(
        f"{label} {count}"
        for label, count in label_counts.most_common(5)
    )
    return CommunitySummary(
        community_id=summary_id,
        title=title,
        summary=(
            f"{title} 第 {level} 层社区，包含 {len(community_nodes)} 个实体；"
            f"主要类型：{label_text}。"
        ),
        size=len(community_nodes),
        weight=round(
            sum(float(node.properties.get("visual_weight", 0.0)) for node in community_nodes),
            6,
        ),
        entities=entity_names,
        label_counts=[
            f"{label}:{count}"
            for label, count in label_counts.most_common()
        ],
        level=level,
        source_node_ids=[node.id for node in community_nodes],
    )


def _summary_id(level: int, community_id: int) -> int:
    return level * 1_000_000 + community_id


def _is_int_like(value) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True
