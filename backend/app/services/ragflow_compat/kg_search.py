from __future__ import annotations

from dataclasses import dataclass
import json

from app.models.graph import GraphEdge, GraphNode
from app.services.ragflow_compat.schemas import RetrievalCommunityReport
from app.services.ragflow_compat.doc_store import RagflowDocStore
from app.services.ragflow_compat.scoring import (
    ScoredEntity,
    ScoredRelation,
    double_hit_boost,
    fuse_relation_scores,
    score_nhop_paths,
    sort_entities,
    sort_relations,
)


@dataclass(frozen=True)
class KgSearchResult:
    entities: list[ScoredEntity]
    relations: list[ScoredRelation]
    community_reports: list[RetrievalCommunityReport]
    graph_nodes: list[GraphNode]
    graph_edges: list[GraphEdge]


class RagflowKgSearch:
    def __init__(self, doc_store: RagflowDocStore):
        self.doc_store = doc_store

    def retrieve(
        self,
        question: str,
        *,
        answer_type_keywords: list[str],
        entities_from_query: list[str],
        ent_topn: int = 6,
        rel_topn: int = 6,
        comm_topn: int = 1,
        ent_sim_threshold: float = 0.0,
        rel_sim_threshold: float = 0.0,
    ) -> KgSearchResult:
        entity_query = ", ".join(entities_from_query) if entities_from_query else question
        entity_hits = self.doc_store.search_entities(
            entity_query,
            entities_from_query,
            top_k=56,
            sim_threshold=ent_sim_threshold,
        )
        relation_hits = self.doc_store.search_relations(
            question,
            top_k=56,
            sim_threshold=rel_sim_threshold,
        )
        entities_from_types = self._entities_from_types(answer_type_keywords)
        ents = {
            hit.entity.entity_name: {
                "sim": hit.score,
                "pagerank": hit.entity.rank_flt,
                "n_hop_ents": hit.entity.n_hop_with_weight,
                "description": hit.entity.content_with_weight,
                "entity": hit.entity,
            }
            for hit in entity_hits
        }
        rels = {
            tuple(sorted((hit.relation.from_entity_kwd, hit.relation.to_entity_kwd))): {
                "sim": hit.score,
                "pagerank": hit.relation.weight_int,
                "description": hit.relation.content_with_weight,
                "relation": hit.relation,
            }
            for hit in relation_hits
        }

        type_names = set(entities_from_types)
        nhop_paths = score_nhop_paths(ents)
        double_hit_boost(ents, type_names)
        fuse_relation_scores(rels, type_names, nhop_paths)
        self._backfill_relation_details(rels)
        scored_entities = sort_entities(ents, top_n=ent_topn)
        scored_relations = sort_relations(rels, top_n=rel_topn)
        community_reports = [
            hit.report
            for hit in self.doc_store.search_community_reports(
                [entity.entity for entity in scored_entities],
                top_k=max(0, int(comm_topn)),
            )
        ]
        graph_nodes = self._graph_nodes(scored_entities, ents)
        graph_edges = self._graph_edges(scored_relations, rels)
        return KgSearchResult(
            entities=scored_entities,
            relations=scored_relations,
            community_reports=community_reports,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )

    def _entities_from_types(self, answer_type_keywords: list[str]) -> dict[str, object]:
        entities = self.doc_store.repository.list_kg_entities(available_only=True)
        type_set = set(answer_type_keywords)
        if not type_set:
            return {}
        return {
            entity.entity_name: entity
            for entity in entities
            if entity.entity_type in type_set or entity.entity_name in type_set
        }

    def _backfill_relation_details(
        self,
        relation_map: dict[tuple[str, str], dict],
    ) -> None:
        find_relation = getattr(
            self.doc_store.repository,
            "find_kg_relation_by_entities",
            None,
        )
        if not callable(find_relation):
            return
        for edge, relation_data in relation_map.items():
            if relation_data.get("relation"):
                continue
            relation = find_relation(edge[0], edge[1])
            if not relation:
                continue
            relation_data["description"] = relation.content_with_weight
            relation_data["relation"] = relation

    def _graph_nodes(
        self,
        scored_entities: list[ScoredEntity],
        entity_map: dict[str, dict],
    ) -> list[GraphNode]:
        nodes: dict[str, GraphNode] = {}
        for item in scored_entities:
            entity = entity_map.get(item.entity, {}).get("entity")
            if not entity:
                continue
            node_id = entity.source_node_id or _node_id(entity.entity_type, entity.entity_name)
            nodes[node_id] = GraphNode(
                id=node_id,
                label=_graph_label(entity.entity_type),
                name=entity.entity_name,
                description=_extract_description(entity.content_with_weight),
                properties={"score": round(item.score, 6)},
            )
        return list(nodes.values())

    def _graph_edges(
        self,
        scored_relations: list[ScoredRelation],
        relation_map: dict[tuple[str, str], dict],
    ) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        for item in scored_relations:
            relation_data = relation_map.get(
                (item.from_entity, item.to_entity),
                relation_map.get(tuple(sorted((item.from_entity, item.to_entity))), {}),
            )
            relation = relation_data.get("relation")
            if relation:
                relation_type = relation.relation_type
                display = relation.display
                edge_id = relation.source_edge_id or relation.relation_id
                evidence_ids = relation.evidence_chunk_ids
                source_name = relation.from_entity_kwd
                target_name = relation.to_entity_kwd
            else:
                relation_type = "RELATED_TO"
                display = "相关"
                edge_id = f"edge:{item.from_entity}:{item.to_entity}"
                evidence_ids = []
                source_name = item.from_entity
                target_name = item.to_entity
            edges.append(
                GraphEdge(
                    id=edge_id,
                    source=_node_id("", source_name),
                    target=_node_id("", target_name),
                    relation=relation_type,
                    display=display,
                    evidence_ids=evidence_ids,
                )
            )
        return sorted(edges, key=lambda edge: (0 if edge.evidence_ids else 1, edge.id))


def _node_id(entity_type: str, entity_name: str) -> str:
    prefix = {
        "Symptom": "symptom",
        "Syndrome": "syndrome",
        "Treatment": "treatment",
        "Formula": "formula",
        "Prescription": "prescription",
        "Herb": "herb",
    }.get(entity_type, "")
    if not prefix:
        if "失眠" in entity_name:
            prefix = "symptom"
        elif "虚" in entity_name or "证" in entity_name:
            prefix = "syndrome"
        else:
            prefix = "entity"
        return f"{prefix}:{entity_name}"
    return f"{prefix}:{entity_name}"


def _graph_label(entity_type: str):
    allowed = {
        "Symptom",
        "Syndrome",
        "Treatment",
        "Formula",
        "Prescription",
        "Herb",
        "Dosage",
        "Function",
        "Indication",
        "Channel",
        "Property",
        "Flavor",
        "TextSource",
        "Evidence",
        "ExternalSource",
    }
    return entity_type if entity_type in allowed else "ExternalSource"


def _extract_description(content_with_weight: str) -> str:
    try:
        data = json.loads(content_with_weight)
    except json.JSONDecodeError:
        return content_with_weight
    return str(data.get("description", content_with_weight))
