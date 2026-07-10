from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.models.graph import GraphEdge, GraphNode


@dataclass(frozen=True)
class NodeResolution:
    node_id: str
    canonical_id: str
    canonical_name: str
    canonical_label: str
    aliases: list[str] = field(default_factory=list)
    is_alias: bool = False


@dataclass(frozen=True)
class GraphEntityResolutionResult:
    by_node_id: dict[str, NodeResolution]

    def apply_to_nodes(self, nodes: list[GraphNode]) -> None:
        for node in nodes:
            resolution = self.by_node_id.get(node.id)
            if not resolution:
                continue
            node.properties.update(
                {
                    "canonical_id": resolution.canonical_id,
                    "canonical_name": resolution.canonical_name,
                    "canonical_label": resolution.canonical_label,
                    "aliases": resolution.aliases,
                    "is_alias": resolution.is_alias,
                }
            )


class GraphEntityResolutionService:
    def resolve(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> GraphEntityResolutionResult:
        nodes_by_id = {node.id: node for node in nodes}
        canonical_by_node_id = {node.id: node.id for node in nodes}
        aliases_by_canonical_id: dict[str, set[str]] = defaultdict(set)

        for edge in edges:
            if edge.relation != "HAS_ALIAS":
                continue
            source = nodes_by_id.get(edge.source)
            target = nodes_by_id.get(edge.target)
            if not source or not target:
                continue
            canonical, alias = _canonical_alias_pair(source, target)
            if not canonical or not alias:
                continue
            canonical_by_node_id[alias.id] = canonical.id
            aliases_by_canonical_id[canonical.id].add(alias.name)

        non_alias_by_name: dict[str, list[GraphNode]] = defaultdict(list)
        for node in nodes:
            if node.label != "Alias":
                non_alias_by_name[node.name].append(node)

        for node in nodes:
            if node.label != "Alias" or canonical_by_node_id[node.id] != node.id:
                continue
            candidates = non_alias_by_name.get(node.name, [])
            if len(candidates) != 1:
                continue
            canonical = candidates[0]
            canonical_by_node_id[node.id] = canonical.id
            aliases_by_canonical_id[canonical.id].add(node.name)

        by_node_id: dict[str, NodeResolution] = {}
        for node in nodes:
            canonical_id = canonical_by_node_id[node.id]
            canonical = nodes_by_id[canonical_id]
            aliases = sorted(aliases_by_canonical_id.get(canonical_id, set()))
            by_node_id[node.id] = NodeResolution(
                node_id=node.id,
                canonical_id=canonical_id,
                canonical_name=canonical.name,
                canonical_label=canonical.label,
                aliases=aliases,
                is_alias=node.id != canonical_id or node.label == "Alias",
            )
        return GraphEntityResolutionResult(by_node_id=by_node_id)


def _canonical_alias_pair(
    source: GraphNode,
    target: GraphNode,
) -> tuple[GraphNode | None, GraphNode | None]:
    if source.label == "Alias" and target.label != "Alias":
        return target, source
    if target.label == "Alias" and source.label != "Alias":
        return source, target
    return None, None
