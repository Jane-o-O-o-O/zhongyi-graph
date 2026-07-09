from app.services.ragflow_compat.scoring import (
    double_hit_boost,
    fuse_relation_scores,
    score_nhop_paths,
    sort_entities,
    sort_relations,
)


def test_score_nhop_paths_applies_distance_decay_and_max_pagerank():
    entities = {
        "失眠": {
            "sim": 0.9,
            "pagerank": 2.0,
            "n_hop_ents": [
                {"path": ["失眠", "心脾两虚", "归脾汤"], "weights": [3, 5]},
                {"path": ["失眠", "心脾两虚"], "weights": [4]},
            ],
        }
    }

    paths = score_nhop_paths(entities)

    assert paths[("失眠", "心脾两虚")]["sim"] == 0.9
    assert paths[("失眠", "心脾两虚")]["pagerank"] == 4
    assert paths[("心脾两虚", "归脾汤")]["sim"] == 0.3
    assert paths[("心脾两虚", "归脾汤")]["pagerank"] == 5


def test_double_hit_boost_matches_ragflow_type_boost():
    entities = {
        "失眠": {"sim": 0.4, "pagerank": 2},
        "归脾汤": {"sim": 0.7, "pagerank": 1},
    }

    double_hit_boost(entities, {"失眠", "Syndrome"})

    assert entities["失眠"]["sim"] == 0.8
    assert entities["归脾汤"]["sim"] == 0.7


def test_fuse_relation_scores_boosts_existing_and_adds_nhop_relations():
    relations = {
        ("失眠", "心脾两虚"): {
            "sim": 0.5,
            "pagerank": 3,
            "description": "失眠可辨为心脾两虚",
        }
    }
    nhop_paths = {
        ("失眠", "心脾两虚"): {"sim": 0.25, "pagerank": 4},
        ("心脾两虚", "归脾汤"): {"sim": 0.2, "pagerank": 5},
    }

    fuse_relation_scores(relations, {"心脾两虚"}, nhop_paths)

    assert relations[("失眠", "心脾两虚")]["sim"] == 1.125
    assert ("心脾两虚", "归脾汤") in relations
    assert relations[("心脾两虚", "归脾汤")]["sim"] == 0.4


def test_sort_entities_and_relations_rank_by_similarity_times_pagerank():
    entities = {
        "低相似高权重": {"sim": 0.5, "pagerank": 10, "description": "a"},
        "高相似低权重": {"sim": 0.9, "pagerank": 1, "description": "b"},
    }
    relations = {
        ("失眠", "心脾两虚"): {"sim": 0.5, "pagerank": 4, "description": "a"},
        ("失眠", "归脾汤"): {"sim": 0.9, "pagerank": 1, "description": "b"},
    }

    assert sort_entities(entities, top_n=1)[0].entity == "低相似高权重"
    assert sort_relations(relations, top_n=1)[0].to_entity == "心脾两虚"
