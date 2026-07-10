from app.services.ragflow_compat.checkpoints import (
    COMMUNITY_CHECKPOINT,
    RESOLUTION_CHECKPOINT,
    community_checkpoint_key,
    resolution_checkpoint_key,
    stable_checkpoint_key,
)
from app.services.ragflow_compat.phase_markers import (
    ALL_PHASES,
    PHASE_COMMUNITY,
    PHASE_RESOLUTION,
)


def test_graphrag_checkpoint_constants_match_ragflow_names():
    assert COMMUNITY_CHECKPOINT == "graphrag_checkpoint_community"
    assert RESOLUTION_CHECKPOINT == "graphrag_checkpoint_resolution"


def test_stable_checkpoint_key_is_ordered_json_sha256():
    assert stable_checkpoint_key("resolution", "Herb", [["白芍", "白芍药"]]) == (
        "e0c9d4081b4f919a0a480d309444bae79e5cdc282bdb25f2ffc45a2e3394ad19"
    )


def test_community_checkpoint_key_is_stable_for_node_order():
    assert community_checkpoint_key("0", "12", ["白芍", "柴胡"]) == community_checkpoint_key(
        "0",
        "12",
        ["柴胡", "白芍"],
    )


def test_resolution_checkpoint_key_is_stable_for_pair_order():
    assert resolution_checkpoint_key(
        "Herb",
        [("白芍", "白芍药"), ("柴胡", "北柴胡")],
    ) == resolution_checkpoint_key(
        "Herb",
        [("北柴胡", "柴胡"), ("白芍药", "白芍")],
    )


def test_graphrag_phase_marker_constants_match_ragflow_names():
    assert PHASE_RESOLUTION == "resolution_done"
    assert PHASE_COMMUNITY == "community_done"
    assert ALL_PHASES == (PHASE_RESOLUTION, PHASE_COMMUNITY)
