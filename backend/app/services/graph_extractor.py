from app.models.ingestion import DocumentChunk, EntityCandidate, RelationCandidate


class GraphExtractor:
    def __init__(self, llm_extractor=None):
        self.llm_extractor = llm_extractor

    def extract(
        self,
        chunks: list[DocumentChunk],
        hint_terms: list[str] | None = None,
    ) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        if self.llm_extractor:
            return self._extract_with_llm(chunks, hint_terms=hint_terms or [])
        return [], []

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

            entity_labels: dict[str, str] = {}
            canonical_names: dict[str, str] = {}
            for item in extracted.get("entities", []):
                raw_name = str(item.get("name", "")).strip()
                label = _normalize_label(str(item.get("label", "")))
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
                raw_source_name = str(item.get("source", "")).strip()
                raw_target_name = str(item.get("target", "")).strip()
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
        "治法": "Treatment",
        "方剂": "Formula",
        "中药": "Herb",
        "药物": "Herb",
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
