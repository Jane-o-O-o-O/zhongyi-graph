from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from app.models.graph import GraphEdge, GraphNode
from app.services.ragflow_compat.checkpoints import (
    RESOLUTION_CHECKPOINT,
    resolution_checkpoint_key,
)
from app.services.ragflow_compat.repository import RagflowRetrievalRepository

RESOLUTION_BATCH_SIZE = 100


@dataclass(frozen=True)
class RagflowGraphEntityResolutionResult:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    pairs_replayed: int = 0
    pairs_resolved: int = 0
    pairs_merged: int = 0


class HeuristicEntityResolutionDecider:
    def resolve_pairs(
        self,
        entity_type: str,
        pairs: list[tuple[str, str]],
        nodes_by_name: dict[str, GraphNode],
    ) -> list[tuple[str, str]]:
        del entity_type, nodes_by_name
        return [
            pair
            for pair in pairs
            if _is_likely_same_entity(pair[0], pair[1])
        ]


class LlmEntityResolutionDecider:
    def __init__(self, resolution_client):
        self.resolution_client = resolution_client

    def resolve_pairs(
        self,
        entity_type: str,
        pairs: list[tuple[str, str]],
        nodes_by_name: dict[str, GraphNode],
    ) -> list[tuple[str, str]]:
        del nodes_by_name
        selected_pairs = self.resolution_client.resolve_entity_pairs(
            entity_type=entity_type,
            pairs=pairs,
        )
        allowed = {pair: pair for pair in pairs}
        allowed.update({(target, source): (source, target) for source, target in pairs})
        result: list[tuple[str, str]] = []
        for pair in selected_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            selected = allowed.get((str(pair[0]), str(pair[1])))
            if selected and selected not in result:
                result.append(selected)
        return result


class RagflowGraphEntityResolutionService:
    def __init__(self, decider=None):
        self.decider = decider or HeuristicEntityResolutionDecider()

    def resolve(
        self,
        *,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        repository: RagflowRetrievalRepository,
    ) -> RagflowGraphEntityResolutionResult:
        selected_pairs: list[tuple[str, str]] = []
        pairs_replayed = 0
        pairs_resolved = 0
        checkpoints = repository.load_graphrag_checkpoints(RESOLUTION_CHECKPOINT)
        for entity_type, group_nodes in _nodes_by_label(nodes).items():
            pairs = _candidate_pairs(group_nodes)
            if not pairs:
                continue
            nodes_by_name = {node.name: node for node in group_nodes}
            for batch in _pair_batches(pairs, RESOLUTION_BATCH_SIZE):
                checkpoint_key = resolution_checkpoint_key(entity_type, batch)
                checkpoint = checkpoints.get(checkpoint_key)
                if isinstance(checkpoint, list):
                    replayed_pairs = _valid_pairs_from_payload(checkpoint, nodes_by_name)
                    selected_pairs.extend(replayed_pairs)
                    pairs_replayed += len(replayed_pairs)
                    continue
                resolved_pairs = self.decider.resolve_pairs(entity_type, batch, nodes_by_name)
                resolved_pairs = _valid_pairs_from_payload(resolved_pairs, nodes_by_name)
                repository.save_graphrag_checkpoint(
                    RESOLUTION_CHECKPOINT,
                    checkpoint_key,
                    [list(pair) for pair in resolved_pairs],
                )
                selected_pairs.extend(resolved_pairs)
                pairs_resolved += len(resolved_pairs)
        resolved_nodes, resolved_edges, pairs_merged = _merge_pairs(nodes, edges, selected_pairs)
        if selected_pairs:
            repository.cleanup_graphrag_checkpoints(RESOLUTION_CHECKPOINT)
        return RagflowGraphEntityResolutionResult(
            nodes=resolved_nodes,
            edges=resolved_edges,
            pairs_replayed=pairs_replayed,
            pairs_resolved=pairs_resolved,
            pairs_merged=pairs_merged,
        )


def _nodes_by_label(nodes: list[GraphNode]) -> dict[str, list[GraphNode]]:
    groups: dict[str, list[GraphNode]] = defaultdict(list)
    for node in nodes:
        groups[node.label].append(node)
    return {
        label: sorted(group_nodes, key=lambda node: (node.name, node.id))
        for label, group_nodes in groups.items()
        if len(group_nodes) > 1
    }


def _candidate_pairs(nodes: list[GraphNode]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for left, right in combinations(nodes, 2):
        if _is_candidate_name_pair(left.name, right.name):
            pairs.append((left.name, right.name))
    return pairs


def _pair_batches(pairs: list[tuple[str, str]], batch_size: int) -> list[list[tuple[str, str]]]:
    return [
        pairs[index:index + batch_size]
        for index in range(0, len(pairs), batch_size)
    ]


def _is_candidate_name_pair(left: str, right: str) -> bool:
    if not left or not right or left == right:
        return False
    if _has_digit_in_2gram_diff(left, right):
        return False
    if _is_ascii_text(left) and _is_ascii_text(right):
        return _levenshtein_distance(left, right) <= min(len(left), len(right)) // 2
    left_chars = set(left)
    right_chars = set(right)
    max_length = max(len(left_chars), len(right_chars))
    if max_length < 4:
        return len(left_chars & right_chars) > 1
    return len(left_chars & right_chars) / max_length >= 0.8


def _has_digit_in_2gram_diff(left: str, right: str) -> bool:
    def to_2gram_set(value: str) -> set[str]:
        return {value[index:index + 2] for index in range(len(value) - 1)}

    diff = to_2gram_set(left) ^ to_2gram_set(right)
    return any(any(char.isdigit() for char in value) for value in diff)


def _is_ascii_text(value: str) -> bool:
    return value.isascii() and any(char.isalpha() for char in value)


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _is_likely_same_entity(left: str, right: str) -> bool:
    return left in right or right in left


def _valid_pairs_from_payload(
    payload,
    nodes_by_name: dict[str, GraphNode],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for pair in payload:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        left = str(pair[0])
        right = str(pair[1])
        if left in nodes_by_name and right in nodes_by_name and left != right:
            pairs.append((left, right))
    return pairs


def _merge_pairs(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    pairs: list[tuple[str, str]],
) -> tuple[list[GraphNode], list[GraphEdge], int]:
    if not pairs:
        return nodes, edges, 0
    nodes_by_name = {node.name: node for node in nodes}
    components = _connected_name_components(pairs)
    alias_to_canonical_id: dict[str, str] = {}
    aliases_by_canonical_id: dict[str, set[str]] = defaultdict(set)
    for component in components:
        component_nodes = [nodes_by_name[name] for name in component if name in nodes_by_name]
        if len(component_nodes) < 2:
            continue
        canonical = sorted(component_nodes, key=lambda node: (len(node.name), node.name, node.id))[0]
        for node in component_nodes:
            if node.id == canonical.id:
                continue
            alias_to_canonical_id[node.id] = canonical.id
            aliases_by_canonical_id[canonical.id].add(node.name)

    resolved_nodes: list[GraphNode] = []
    for node in nodes:
        if node.id in alias_to_canonical_id:
            continue
        aliases = sorted(
            set(str(alias) for alias in node.properties.get("aliases", []))
            | aliases_by_canonical_id.get(node.id, set())
        )
        if aliases:
            properties = dict(node.properties)
            properties["aliases"] = aliases
            node = node.model_copy(update={"properties": properties})
        resolved_nodes.append(node)
    resolved_edges = _retarget_edges(edges, alias_to_canonical_id)
    return resolved_nodes, resolved_edges, len(alias_to_canonical_id)


def _connected_name_components(pairs: list[tuple[str, str]]) -> list[set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for left, right in pairs:
        neighbors[left].add(right)
        neighbors[right].add(left)
    visited: set[str] = set()
    components: list[set[str]] = []
    for name in sorted(neighbors):
        if name in visited:
            continue
        stack = [name]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(sorted(neighbors[current] - visited))
        components.append(component)
    return components


def _retarget_edges(
    edges: list[GraphEdge],
    alias_to_canonical_id: dict[str, str],
) -> list[GraphEdge]:
    edges_by_signature: dict[tuple[str, str, str, str], GraphEdge] = {}
    for edge in edges:
        source = alias_to_canonical_id.get(edge.source, edge.source)
        target = alias_to_canonical_id.get(edge.target, edge.target)
        if source == target:
            continue
        signature = (source, target, edge.relation, edge.display)
        existing = edges_by_signature.get(signature)
        if existing:
            edge = existing.model_copy(
                update={
                    "evidence_ids": sorted(set(existing.evidence_ids) | set(edge.evidence_ids))
                }
            )
        else:
            edge = edge.model_copy(
                update={
                    "id": f"relation:{source}:{edge.relation}:{target}",
                    "source": source,
                    "target": target,
                }
            )
        edges_by_signature[signature] = edge
    return [edges_by_signature[key] for key in sorted(edges_by_signature)]
