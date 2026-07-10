from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.models.graph import GraphNode
from app.services.graph_analytics_service import LOW_VALUE_LABELS


@dataclass(frozen=True)
class CommunitySummary:
    community_id: int
    title: str
    summary: str
    size: int
    weight: float
    entities: list[str]
    label_counts: list[str]
    findings: list[dict[str, Any] | str] | None = None
    rating: float = 0.0
    rating_explanation: str = ""


@dataclass(frozen=True)
class GraphCommunitySummaryResult:
    by_community_id: dict[int, CommunitySummary]

    def apply_to_nodes(self, nodes: list[GraphNode]) -> None:
        for node in nodes:
            community_id = int(node.properties.get("community_id", 0))
            summary = self.by_community_id.get(community_id)
            if not summary:
                continue
            communities = _unique_names([*node.properties.get("communities", []), summary.title])
            node.properties.update(
                {
                    "communities": communities,
                    "community_title": summary.title,
                    "community_summary": summary.summary,
                    "community_size": summary.size,
                    "community_weight": summary.weight,
                    "community_entities": summary.entities,
                    "community_label_counts": summary.label_counts,
                    "community_rating": summary.rating,
                    "community_rating_explanation": summary.rating_explanation,
                }
            )


class GraphCommunitySummaryService:
    def summarize(self, nodes: list[GraphNode]) -> GraphCommunitySummaryResult:
        groups: dict[int, list[GraphNode]] = {}
        for node in nodes:
            community_id = int(node.properties.get("community_id", 0))
            groups.setdefault(community_id, []).append(node)

        summaries: dict[int, CommunitySummary] = {}
        for community_id, group_nodes in groups.items():
            ranked_nodes = sorted(group_nodes, key=_summary_node_sort_key)
            representative_nodes = [
                node for node in ranked_nodes if node.label not in LOW_VALUE_LABELS
            ] or ranked_nodes
            representative_names = _unique_names(node.name for node in representative_nodes)[:6]
            title_names = representative_names[:3] or [f"社区 {community_id}"]
            label_counts = Counter(node.label for node in group_nodes)
            label_text = "、".join(
                f"{label} {count}"
                for label, count in label_counts.most_common(5)
            )
            title = "、".join(title_names)
            weight = round(
                sum(float(node.properties.get("visual_weight", 0.0)) for node in group_nodes),
                6,
            )
            summaries[community_id] = CommunitySummary(
                community_id=community_id,
                title=title,
                summary=(
                    f"{title} 相关社区，包含 {len(group_nodes)} 个实体；"
                    f"代表实体：{'、'.join(representative_names[:5])}；"
                    f"主要类型：{label_text}。"
                ),
                size=len(group_nodes),
                weight=weight,
                entities=representative_names,
                label_counts=[
                    f"{label}:{count}"
                    for label, count in label_counts.most_common()
                ],
            )
        return GraphCommunitySummaryResult(by_community_id=summaries)


def _summary_node_sort_key(node: GraphNode):
    return (
        -float(node.properties.get("visual_weight", 0.0)),
        -int(node.properties.get("degree", 0)),
        node.label,
        node.name,
        node.id,
    )


def _unique_names(names) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result
