import json

from scripts.build_structured_tcm_graph import build_structured_tcm_graph


def test_build_structured_tcm_graph_maps_nodes_edges_and_deduplicates(tmp_path):
    root = tmp_path
    zhongyao = root / "zhongyao" / "data_zhongyao"
    fangji = root / "fangji" / "data_fangji"
    zhongyao.mkdir(parents=True)
    fangji.mkdir(parents=True)

    (zhongyao / "nodes_zhongyao.txt").write_text(
        "中药名\t柴胡\n"
        "功能\t疏肝解郁\n"
        "主治\t寒热往来\n",
        encoding="utf-8",
    )
    (fangji / "nodes_fangji.txt").write_text(
        "方名\t小柴胡汤\n"
        "处方\t小柴胡汤_1\n"
        "中药名\t柴胡\n"
        "剂量\t半斤\n"
        "功能主治\t寒热往来\n",
        encoding="utf-8",
    )
    (zhongyao / "relations_zhongyao.json").write_text(
        json.dumps(
            [
                {
                    "node_1": "中药名\t柴胡",
                    "relation": "functions",
                    "node_2": "功能\t疏肝解郁",
                },
                {
                    "node_1": "中药名\t柴胡",
                    "relation": "attending",
                    "node_2": "主治\t寒热往来",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (fangji / "relations_fangji.json").write_text(
        json.dumps(
            [
                {
                    "node_1": "方剂\t方剂",
                    "relation": "include",
                    "node_2": "方名\t小柴胡汤",
                },
                {
                    "node_1": "方名\t小柴胡汤",
                    "relation": "prescription type",
                    "node_2": "处方\t小柴胡汤_1",
                },
                {
                    "node_1": "处方\t小柴胡汤_1",
                    "relation": "composition",
                    "node_2": "中药名\t柴胡",
                },
                {
                    "node_1": "中药名\t柴胡",
                    "relation": "dose",
                    "node_2": "剂量\t半斤",
                },
                {
                    "node_1": "处方\t小柴胡汤_1",
                    "relation": "functions",
                    "node_2": "功能主治\t寒热往来",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    graph = build_structured_tcm_graph(root)

    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    assert nodes_by_id["herb:柴胡"]["label"] == "Herb"
    assert nodes_by_id["herb:柴胡"]["name"] == "柴胡"
    assert nodes_by_id["herb:柴胡"]["source_chunks"]
    assert nodes_by_id["formula:小柴胡汤"]["label"] == "Formula"
    assert nodes_by_id["prescription:小柴胡汤_1"]["label"] == "Prescription"
    assert nodes_by_id["dose:半斤"]["label"] == "Dose"

    edges_by_triple = {
        (edge["source"], edge["relation"], edge["target"]): edge
        for edge in graph["edges"]
    }
    edges = set(edges_by_triple)
    assert ("herb:柴胡", "HAS_FUNCTION", "function:疏肝解郁") in edges
    assert ("herb:柴胡", "TREATS", "indication:寒热往来") in edges
    assert ("category:方剂", "INCLUDES", "formula:小柴胡汤") in edges
    assert ("formula:小柴胡汤", "HAS_PRESCRIPTION", "prescription:小柴胡汤_1") in edges
    assert ("prescription:小柴胡汤_1", "COMPOSED_OF", "herb:柴胡") in edges
    assert ("prescription:小柴胡汤_1", "HAS_DOSE", "dose:半斤") in edges
    assert ("prescription:小柴胡汤_1", "TREATS", "indication:寒热往来") in edges
    assert edges_by_triple[
        ("prescription:小柴胡汤_1", "COMPOSED_OF", "herb:柴胡")
    ]["evidence_ids"]

    assert len([node for node in graph["nodes"] if node["id"] == "herb:柴胡"]) == 1
    assert graph["evidence"]
    evidence_by_id = {item["id"]: item for item in graph["evidence"]}
    composed_evidence_id = edges_by_triple[
        ("prescription:小柴胡汤_1", "COMPOSED_OF", "herb:柴胡")
    ]["evidence_ids"][0]
    assert evidence_by_id[composed_evidence_id]["source"] == "relations_fangji.json"
    assert "小柴胡汤_1" in evidence_by_id[composed_evidence_id]["snippet"]
