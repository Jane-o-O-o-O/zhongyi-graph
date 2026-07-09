from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT.parent / "中医文献资料" / "完整中医资料" / "04_中医知识图谱_中药方剂"
DEFAULT_OUT = ROOT / "data" / "imports" / "tcm_structured_graph.json"

NODE_TYPES = {
    "中药名": ("Herb", "herb"),
    "方名": ("Formula", "formula"),
    "处方": ("Prescription", "prescription"),
    "主治": ("Indication", "indication"),
    "功能主治": ("Indication", "indication"),
    "功能": ("Function", "function"),
    "别名": ("Alias", "alias"),
    "来源": ("Source", "source"),
    "分布": ("DistributionArea", "distribution_area"),
    "四气": ("Property", "property"),
    "五味": ("Flavor", "flavor"),
    "归经": ("Meridian", "meridian"),
    "剂量": ("Dose", "dose"),
    "中药材": ("Category", "category"),
    "方剂": ("Category", "category"),
}

RELATIONS = {
    "include": ("INCLUDES", "包含"),
    "from": ("FROM_SOURCE", "来源"),
    "distribution area": ("DISTRIBUTED_IN", "分布"),
    "four properties": ("HAS_PROPERTY", "四气"),
    "five flavors": ("HAS_FLAVOR", "五味"),
    "channel tropism": ("ENTERS_MERIDIAN", "归经"),
    "functions": ("HAS_FUNCTION", "功能"),
    "attending": ("TREATS", "主治"),
    "another name": ("HAS_ALIAS", "别名"),
    "prescription type": ("HAS_PRESCRIPTION", "处方"),
    "composition": ("COMPOSED_OF", "组成"),
    "dose": ("HAS_DOSE", "剂量"),
}

FILES = (
    ("zhongyao/data_zhongyao/nodes_zhongyao.txt", "nodes"),
    ("fangji/data_fangji/nodes_fangji.txt", "nodes"),
    ("zhongyao/data_zhongyao/relations_zhongyao.json", "relations"),
    ("fangji/data_fangji/relations_fangji.json", "relations"),
)


def build_structured_tcm_graph(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, str]] = {}

    for relative_path, kind in FILES:
        path = source_root / relative_path
        if kind == "nodes":
            _read_nodes(path, nodes)
        else:
            _read_relations(path, nodes, edges, evidence)

    return {
        "nodes": sorted(nodes.values(), key=lambda node: (node["label"], node["name"], node["id"])),
        "edges": sorted(edges.values(), key=lambda edge: edge["id"]),
        "evidence": sorted(evidence.values(), key=lambda item: item["id"]),
    }


def write_structured_tcm_graph(source_root: Path, out_path: Path) -> dict[str, Any]:
    graph = build_structured_tcm_graph(source_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "path": str(out_path),
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "evidence": len(graph["evidence"]),
    }


def _read_nodes(path: Path, nodes: dict[str, dict[str, Any]]) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_typed_value(line)
        if not parsed:
            continue
        _ensure_node(nodes, *parsed)


def _read_relations(
    path: Path,
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, str]],
) -> None:
    relations = json.loads(path.read_text(encoding="utf-8"))
    last_prescription_id: str | None = None
    for index, relation in enumerate(relations, start=1):
        source_type, source_name = _require_typed_value(relation["node_1"])
        target_type, target_name = _require_typed_value(relation["node_2"])
        raw_relation = relation["relation"]

        source_id = _ensure_node(nodes, source_type, source_name)
        target_id = _ensure_node(nodes, target_type, target_name)

        if target_type == "处方":
            last_prescription_id = target_id

        if raw_relation == "dose" and last_prescription_id:
            source_id = last_prescription_id

        relation_type, display = RELATIONS[raw_relation]
        if raw_relation == "functions" and source_type in {"处方", "方名"}:
            relation_type = "TREATS"
            display = "功能主治"

        evidence_id = _evidence_id(path, index)
        evidence[evidence_id] = _evidence_card(
            evidence_id=evidence_id,
            path=path,
            source_type=source_type,
            source_name=source_name,
            relation=display,
            target_type=target_type,
            target_name=target_name,
        )
        _ensure_edge(edges, source_id, relation_type, target_id, display, evidence_id)
        _add_evidence_id(nodes[source_id], "source_chunks", evidence_id)
        _add_evidence_id(nodes[target_id], "source_chunks", evidence_id)


def _ensure_node(nodes: dict[str, dict[str, Any]], node_type: str, name: str) -> str:
    label, prefix = NODE_TYPES[node_type]
    node_id = f"{prefix}:{name}"
    nodes.setdefault(node_id, {"id": node_id, "label": label, "name": name, "source_chunks": []})
    return node_id


def _ensure_edge(
    edges: dict[str, dict[str, Any]],
    source_id: str,
    relation: str,
    target_id: str,
    display: str,
    evidence_id: str,
) -> str:
    edge_id = _edge_id(source_id, relation, target_id)
    edge = edges.setdefault(
        edge_id,
        {
            "id": edge_id,
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "display": display,
            "evidence_ids": [],
        },
    )
    _add_evidence_id(edge, "evidence_ids", evidence_id)
    return edge_id


def _add_evidence_id(item: dict[str, Any], key: str, evidence_id: str) -> None:
    values = item.setdefault(key, [])
    if evidence_id not in values:
        values.append(evidence_id)


def _evidence_id(path: Path, index: int) -> str:
    digest = hashlib.sha1(f"{path.name}\t{index}".encode("utf-8")).hexdigest()
    return f"evidence:structured-tcm:{digest[:16]}"


def _evidence_card(
    *,
    evidence_id: str,
    path: Path,
    source_type: str,
    source_name: str,
    relation: str,
    target_type: str,
    target_name: str,
) -> dict[str, str]:
    return {
        "id": evidence_id,
        "title": f"{source_name} {relation} {target_name}",
        "source": path.name,
        "snippet": f"{source_type}\t{source_name} {relation} {target_type}\t{target_name}",
        "source_type": "local",
        "location": path.name,
    }


def _edge_id(source_id: str, relation: str, target_id: str) -> str:
    digest = hashlib.sha1(f"{source_id}\t{relation}\t{target_id}".encode("utf-8")).hexdigest()
    return f"edge:structured-tcm:{digest[:16]}"


def _parse_typed_value(value: str) -> tuple[str, str] | None:
    if "\t" not in value:
        return None
    node_type, name = value.split("\t", 1)
    node_type = node_type.strip()
    name = name.strip()
    if not node_type or not name:
        return None
    if node_type not in NODE_TYPES:
        raise ValueError(f"Unsupported node type: {node_type}")
    return node_type, name


def _require_typed_value(value: str) -> tuple[str, str]:
    parsed = _parse_typed_value(value)
    if parsed is None:
        raise ValueError(f"Expected typed value with tab separator: {value}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a graph artifact from structured TCM data.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summary = write_structured_tcm_graph(args.source_root, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
