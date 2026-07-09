from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoredEntity:
    entity: str
    score: float
    description: str = ""


@dataclass(frozen=True)
class ScoredRelation:
    from_entity: str
    to_entity: str
    score: float
    description: str = ""


def score_nhop_paths(entities_from_query: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    """Port of RAGFlow KG n-hop path scoring.

    Mirrors rag/graphrag/search.py and internal/service/kg/scoring.go:
    every path edge receives entity similarity divided by 2 + hop_index, while
    pagerank keeps the maximum relation/path weight seen for that edge.
    """

    paths: dict[tuple[str, str], dict[str, float]] = {}
    for entity in entities_from_query.values():
        sim = float(entity.get("sim", entity.get("similarity", 0.0)) or 0.0)
        n_hop_ents = entity.get("n_hop_ents", [])
        if not isinstance(n_hop_ents, list):
            continue
        for neighbor in n_hop_ents:
            path = neighbor.get("path", []) if isinstance(neighbor, dict) else []
            weights = neighbor.get("weights", []) if isinstance(neighbor, dict) else []
            if not isinstance(path, list):
                continue
            for index in range(len(path) - 1):
                edge = (str(path[index]), str(path[index + 1]))
                edge_score = paths.setdefault(edge, {"sim": 0.0, "pagerank": 0.0})
                edge_score["sim"] += sim / (2.0 + index)
                if index < len(weights):
                    edge_score["pagerank"] = max(edge_score["pagerank"], float(weights[index] or 0.0))
    return paths


def double_hit_boost(entities_from_query: dict[str, dict[str, Any]], entities_from_types: set[str]) -> None:
    for entity_name, entity in entities_from_query.items():
        if entity_name in entities_from_types:
            entity["sim"] = float(entity.get("sim", 0.0) or 0.0) * 2


def fuse_relation_scores(
    relations_from_text: dict[tuple[str, str], dict[str, Any]],
    entities_from_types: set[str],
    nhop_paths: dict[tuple[str, str], dict[str, float]],
) -> None:
    for edge, relation in list(relations_from_text.items()):
        score_boost = 0.0
        pair = tuple(sorted(edge))
        path_score = nhop_paths.pop(pair, None) or nhop_paths.pop(edge, None)
        if path_score:
            score_boost += float(path_score.get("sim", 0.0) or 0.0)
        if edge[0] in entities_from_types:
            score_boost += 1
        if edge[1] in entities_from_types:
            score_boost += 1
        relation["sim"] = float(relation.get("sim", 0.0) or 0.0) * (score_boost + 1)

    for edge, path_score in list(nhop_paths.items()):
        score_boost = 0.0
        if edge[0] in entities_from_types:
            score_boost += 1
        if edge[1] in entities_from_types:
            score_boost += 1
        relations_from_text[edge] = {
            "sim": float(path_score.get("sim", 0.0) or 0.0) * (score_boost + 1),
            "pagerank": float(path_score.get("pagerank", 0.0) or 0.0),
            "description": "",
        }


def sort_entities(
    entities_from_query: dict[str, dict[str, Any]],
    *,
    top_n: int = 6,
) -> list[ScoredEntity]:
    scored = [
        ScoredEntity(
            entity=name,
            score=float(entity.get("sim", 0.0) or 0.0)
            * float(entity.get("pagerank", entity.get("rank_flt", 0.0)) or 0.0),
            description=str(entity.get("description", "")),
        )
        for name, entity in entities_from_query.items()
    ]
    return sorted(scored, key=lambda item: item.score, reverse=True)[:top_n]


def sort_relations(
    relations_from_text: dict[tuple[str, str], dict[str, Any]],
    *,
    top_n: int = 6,
) -> list[ScoredRelation]:
    scored = [
        ScoredRelation(
            from_entity=edge[0],
            to_entity=edge[1],
            score=float(relation.get("sim", 0.0) or 0.0)
            * float(relation.get("pagerank", relation.get("weight_int", 0.0)) or 0.0),
            description=str(relation.get("description", "")),
        )
        for edge, relation in relations_from_text.items()
    ]
    return sorted(scored, key=lambda item: item.score, reverse=True)[:top_n]
