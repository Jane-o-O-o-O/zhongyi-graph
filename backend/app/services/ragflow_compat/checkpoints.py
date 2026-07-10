from __future__ import annotations

import hashlib
import json
from typing import Any

COMMUNITY_CHECKPOINT = "graphrag_checkpoint_community"
RESOLUTION_CHECKPOINT = "graphrag_checkpoint_resolution"


def stable_checkpoint_key(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def community_checkpoint_key(level: str, community_id: str, nodes: list[str]) -> str:
    return stable_checkpoint_key("community", str(level), str(community_id), sorted(nodes))


def resolution_checkpoint_key(entity_type: str, pairs: list[tuple[str, str]]) -> str:
    normalized_pairs = sorted([sorted([source, target]) for source, target in pairs])
    return stable_checkpoint_key("resolution", entity_type, normalized_pairs)
