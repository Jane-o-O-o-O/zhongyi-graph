from typing import Any

from app.models.ingestion import DocumentChunk, EntityCandidate, RelationCandidate

DEFAULT_GENERAL_BATCH_TOKEN_LIMIT = 4096


class GraphExtractor:
    def __init__(
        self,
        llm_extractor=None,
        method: str = "light",
        batch_token_limit: int = DEFAULT_GENERAL_BATCH_TOKEN_LIMIT,
    ):
        self.llm_extractor = llm_extractor
        self.method = _normalize_method(method)
        self.batch_token_limit = max(1, int(batch_token_limit or DEFAULT_GENERAL_BATCH_TOKEN_LIMIT))

    def extract(
        self,
        chunks: list[DocumentChunk],
        hint_terms: list[str] | None = None,
    ) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        if self.method == "ner":
            return self._extract_with_ner(chunks)
        if self.method == "general" and self.llm_extractor:
            return self._extract_with_general(chunks)
        if self.llm_extractor:
            return self._extract_with_llm(chunks, hint_terms=hint_terms or [])
        return [], []

    def _extract_with_general(
        self,
        chunks: list[DocumentChunk],
    ) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        if not hasattr(self.llm_extractor, "extract_chunks_batch"):
            return self._extract_with_llm(chunks, hint_terms=[])
        batches = _general_extraction_batches(chunks, self.batch_token_limit)
        if not batches:
            return [], []
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        entities: dict[str, EntityCandidate] = {}
        relations: dict[str, RelationCandidate] = {}
        try:
            for batch in batches:
                payload = self.llm_extractor.extract_chunks_batch(batch)
                for item in payload.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    chunk = chunks_by_id.get(str(item.get("unit_id", "")).strip())
                    if not chunk:
                        continue
                    _merge_extracted_payload(entities, relations, chunk, item)
        except Exception:
            return [], []
        return list(entities.values()), list(relations.values())

    def _extract_with_ner(
        self,
        chunks: list[DocumentChunk],
    ) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        entities: dict[str, EntityCandidate] = {}
        relations: dict[str, RelationCandidate] = {}
        for chunk in chunks:
            found = _ner_entities(chunk.content)
            for name, label in found:
                entity_id = _entity_id(label, name)
                existing = entities.get(entity_id)
                chunk_ids = [chunk.chunk_id]
                if existing:
                    chunk_ids = sorted(set(existing.source_chunk_ids + chunk_ids))
                entities[entity_id] = EntityCandidate(
                    entity_id=entity_id,
                    name=name,
                    label=label,
                    normalized_name=name,
                    source_chunk_ids=chunk_ids,
                    confidence=0.65,
                )
            for source_name, source_label, target_name, target_label, relation in _ner_relations(found):
                source_id = _entity_id(source_label, source_name)
                target_id = _entity_id(target_label, target_name)
                relation_id = f"relation:{source_id}:{relation}:{target_id}"
                existing = relations.get(relation_id)
                evidence_chunk_ids = [chunk.chunk_id]
                if existing:
                    evidence_chunk_ids = sorted(set(existing.evidence_chunk_ids + evidence_chunk_ids))
                relations[relation_id] = RelationCandidate(
                    relation_id=relation_id,
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relation=relation,
                    display=_display_for_relation(relation),
                    evidence_chunk_ids=evidence_chunk_ids,
                    confidence=0.6,
                )
        return list(entities.values()), list(relations.values())

    def _extract_with_llm(
        self,
        chunks: list[DocumentChunk],
        hint_terms: list[str],
    ) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        entities: dict[str, EntityCandidate] = {}
        relations: dict[str, RelationCandidate] = {}
        for chunk in chunks:
            try:
                extracted = self.llm_extractor.extract_chunk(chunk.content, hints=hint_terms)
            except Exception:
                continue
            _merge_extracted_payload(entities, relations, chunk, extracted)
        return list(entities.values()), list(relations.values())


def _entity_id(label: str, name: str) -> str:
    label_prefix = {
        "Symptom": "symptom",
        "Syndrome": "syndrome",
        "Treatment": "treatment",
        "Formula": "formula",
        "Herb": "herb",
    }.get(label, label.lower())
    return f"entity:{label_prefix}:{name}"


def _general_extraction_batches(
    chunks: list[DocumentChunk],
    token_limit: int,
) -> list[list[dict[str, str]]]:
    batches: list[list[dict[str, str]]] = []
    current_batch: list[dict[str, str]] = []
    current_tokens = 0
    for chunk in chunks:
        content = chunk.content.strip()
        if not content:
            continue
        chunk_tokens = _chunk_token_count(chunk)
        if current_batch and current_tokens + chunk_tokens > token_limit:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append({"unit_id": chunk.chunk_id, "text": content})
        current_tokens += chunk_tokens
    if current_batch:
        batches.append(current_batch)
    return batches


def _chunk_token_count(chunk: DocumentChunk) -> int:
    if chunk.token_count > 0:
        return chunk.token_count
    return max(1, len(chunk.content.strip()))


def _merge_extracted_payload(
    entities: dict[str, EntityCandidate],
    relations: dict[str, RelationCandidate],
    chunk: DocumentChunk,
    extracted: dict,
) -> None:
    entity_labels: dict[str, str] = {}
    canonical_names: dict[str, str] = {}
    for item in extracted.get("entities", []):
        raw_name = _first_text(item, "name", "text", "value", "entity", "entity_id")
        label = _normalize_label(_first_text(item, "label", "type", "category"))
        name = _canonical_name(raw_name, label)
        if not name or not label:
            continue
        entity_id = _entity_id(label, name)
        canonical_names[raw_name] = name
        entity_labels[name] = label
        existing = entities.get(entity_id)
        chunk_ids = [chunk.chunk_id]
        if existing:
            chunk_ids = sorted(set(existing.source_chunk_ids + chunk_ids))
        entities[entity_id] = EntityCandidate(
            entity_id=entity_id,
            name=name,
            label=label,
            normalized_name=name,
            source_chunk_ids=chunk_ids,
            confidence=float(item.get("confidence") or 0.75),
        )

    for item in extracted.get("relations", []):
        raw_source_name = _first_text(item, "source", "subject", "head", "from")
        raw_target_name = _first_text(item, "target", "object", "tail", "to")
        source_name = canonical_names.get(raw_source_name, raw_source_name)
        target_name = canonical_names.get(raw_target_name, raw_target_name)
        relation = _normalize_relation(str(item.get("relation", "")))
        display = str(item.get("display", "")).strip() or _display_for_relation(relation)
        if not source_name or not target_name or not relation:
            continue
        source_label = entity_labels.get(source_name)
        target_label = entity_labels.get(target_name)
        if not source_label or not target_label:
            continue
        source_id = _entity_id(source_label, source_name)
        target_id = _entity_id(target_label, target_name)
        relation_id = f"relation:{source_id}:{relation}:{target_id}"
        existing = relations.get(relation_id)
        evidence_chunk_ids = [chunk.chunk_id]
        if existing:
            evidence_chunk_ids = sorted(set(existing.evidence_chunk_ids + evidence_chunk_ids))
        relations[relation_id] = RelationCandidate(
            relation_id=relation_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation=relation,
            display=display,
            evidence_chunk_ids=evidence_chunk_ids,
            confidence=float(item.get("confidence") or 0.72),
        )


def _first_text(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_label(label: str) -> str:
    aliases = {
        "symptom": "Symptom",
        "syndrome": "Syndrome",
        "treatment": "Treatment",
        "formula": "Formula",
        "herb": "Herb",
        "indication": "Indication",
        "function": "Function",
        "症状": "Symptom",
        "证候": "Syndrome",
        "病名": "Syndrome",
        "病机": "Syndrome",
        "治法": "Treatment",
        "方剂": "Formula",
        "药方": "Formula",
        "处方": "Formula",
        "medicine": "Formula",
        "prescription": "Formula",
        "中药": "Herb",
        "药物": "Herb",
        "药材": "Herb",
        "herbal": "Herb",
        "病因": "Syndrome",
        "舌象": "Indication",
        "脉象": "Indication",
        "诊法": "Indication",
        "体征": "Indication",
        "主治": "Indication",
        "功效": "Function",
    }
    normalized = aliases.get(label.strip(), aliases.get(label.strip().lower(), label.strip()))
    allowed = {"Symptom", "Syndrome", "Treatment", "Formula", "Herb", "Indication", "Function"}
    return normalized if normalized in allowed else ""


def _canonical_name(name: str, label: str) -> str:
    if label == "Symptom":
        symptom_aliases = {
            "头疼": "头痛",
            "偏头疼": "头痛",
            "偏头痛": "头痛",
            "发热头痛": "头痛",
            "头痛发热": "头痛",
        }
        if name in symptom_aliases:
            return symptom_aliases[name]
        if name.endswith("头痛") and name != "头痛":
            return "头痛"
    return name


def _normalize_relation(relation: str) -> str:
    normalized = relation.strip().upper()
    allowed = {
        "MANIFESTS_AS",
        "RECOMMENDS_TREATMENT",
        "RECOMMENDS_FORMULA",
        "COMPOSED_OF",
        "TREATS",
        "RELATED_TO",
    }
    aliases = {
        "可辨为": "MANIFESTS_AS",
        "证候": "MANIFESTS_AS",
        "治法": "RECOMMENDS_TREATMENT",
        "推荐方剂": "RECOMMENDS_FORMULA",
        "组成": "COMPOSED_OF",
        "主治": "TREATS",
        "相关": "RELATED_TO",
    }
    normalized = aliases.get(relation.strip(), normalized)
    return normalized if normalized in allowed else "RELATED_TO"


def _display_for_relation(relation: str) -> str:
    return {
        "MANIFESTS_AS": "可辨为",
        "RECOMMENDS_TREATMENT": "治法",
        "RECOMMENDS_FORMULA": "推荐方剂",
        "COMPOSED_OF": "组成",
        "TREATS": "主治",
        "RELATED_TO": "相关",
    }.get(relation, "相关")


def _normalize_method(method: str) -> str:
    normalized = str(method or "light").strip().lower()
    return normalized if normalized in {"light", "general", "ner"} else "light"


def _ner_entities(text: str) -> list[tuple[str, str]]:
    terms = {
        "头痛": "Symptom",
        "不寐": "Symptom",
        "失眠": "Symptom",
        "心脾两虚": "Syndrome",
        "肝郁化火": "Syndrome",
        "补益心脾": "Treatment",
        "养血敛阴": "Function",
        "归脾汤": "Formula",
        "白芍": "Herb",
        "白芍药": "Herb",
        "党参": "Herb",
        "柴胡": "Herb",
        "桂枝": "Herb",
        "干姜": "Herb",
    }
    found = [(name, label) for name, label in terms.items() if name in text]
    found.sort(key=lambda item: (text.find(item[0]), -len(item[0]), item[0]))
    return found


def _ner_relations(found: list[tuple[str, str]]) -> list[tuple[str, str, str, str, str]]:
    relations = []
    for source_name, source_label in found:
        for target_name, target_label in found:
            if source_name == target_name:
                continue
            relation = _ner_relation_for_labels(source_label, target_label)
            if relation:
                relations.append(
                    (source_name, source_label, target_name, target_label, relation)
                )
    return relations


def _ner_relation_for_labels(source_label: str, target_label: str) -> str:
    if source_label == "Symptom" and target_label == "Syndrome":
        return "MANIFESTS_AS"
    if source_label == "Syndrome" and target_label == "Treatment":
        return "RECOMMENDS_TREATMENT"
    if source_label == "Treatment" and target_label == "Formula":
        return "RECOMMENDS_FORMULA"
    if source_label == "Formula" and target_label == "Herb":
        return "COMPOSED_OF"
    if source_label == "Herb" and target_label == "Function":
        return "RELATED_TO"
    if source_label in {"Formula", "Treatment"} and target_label in {"Symptom", "Syndrome"}:
        return "TREATS"
    return ""
