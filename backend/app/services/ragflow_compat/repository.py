from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import math
from threading import Lock
from typing import Any, TypeVar

from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.engine import Engine

from app.services.ragflow_compat.schemas import (
    RetrievalAudit,
    RetrievalCommunityReport,
    RetrievalChunk,
    RetrievalDocument,
    RetrievalGraphRagBuildRun,
    RetrievalGraphArtifact,
    RetrievalKgEntity,
    RetrievalKgRelation,
    RetrievalTypeSamples,
)
from app.services.ragflow_compat.tables import (
    retrieval_chunk_terms_table,
    retrieval_chunks_table,
    retrieval_documents_table,
    retrieval_graphrag_checkpoints_table,
    retrieval_graphrag_phase_markers_table,
    retrieval_kg_entities_table,
    retrieval_kg_community_reports_table,
    retrieval_kg_graph_artifacts_table,
    retrieval_kg_relations_table,
    retrieval_kg_type_samples_table,
    retrieval_metadata,
    retrieval_sync_state_table,
)

T = TypeVar("T")
GRAPHRAG_BUILD_LOCK_KEY = "graphrag:build:lock"


@dataclass(frozen=True)
class MissingVectorRecord:
    id: str
    text: str
    content_type: str
    chunk_id: str = ""
    entity_id: str = ""
    relation_id: str = ""
    label: str = ""


class RagflowRetrievalRepository:
    def __init__(self, engine: Engine):
        self.engine = engine
        self._claim_lock = Lock()
        retrieval_metadata.create_all(engine)

    def replace_documents(self, documents: list[RetrievalDocument]) -> None:
        self._replace_all(retrieval_documents_table, [_row(document) for document in documents])

    def replace_chunks(self, chunks: list[RetrievalChunk]) -> None:
        self._replace_all(retrieval_chunks_table, [_row(chunk) for chunk in chunks])

    def replace_chunk_terms(self, terms: list) -> None:
        self._replace_all(retrieval_chunk_terms_table, [_row(term) for term in terms])

    def clear_rebuild_tables(self, *, include_graph_artifacts: bool = True) -> None:
        with self.engine.begin() as connection:
            tables = [
                retrieval_chunk_terms_table,
                retrieval_chunks_table,
                retrieval_documents_table,
                retrieval_kg_entities_table,
                retrieval_kg_relations_table,
                retrieval_kg_community_reports_table,
                retrieval_kg_type_samples_table,
            ]
            if include_graph_artifacts:
                tables.append(retrieval_kg_graph_artifacts_table)
            for table in tables:
                connection.execute(delete(table))

    def append_documents(self, documents: list[RetrievalDocument]) -> None:
        self._append(retrieval_documents_table, [_row(document) for document in documents])

    def append_chunks(self, chunks: list[RetrievalChunk]) -> None:
        self._append(retrieval_chunks_table, [_row(chunk) for chunk in chunks])

    def append_chunk_terms(self, terms: list) -> None:
        self._append(retrieval_chunk_terms_table, [_row(term) for term in terms])

    def append_kg_entities(self, entities: list[RetrievalKgEntity]) -> None:
        self._append(retrieval_kg_entities_table, [_row(entity) for entity in entities])

    def append_kg_relations(self, relations: list[RetrievalKgRelation]) -> None:
        self._append(retrieval_kg_relations_table, [_row(relation) for relation in relations])

    def append_community_reports(self, reports: list[RetrievalCommunityReport]) -> None:
        self._append(retrieval_kg_community_reports_table, [_row(report) for report in reports])

    def append_graph_artifacts(self, artifacts: list[RetrievalGraphArtifact]) -> None:
        self._append(retrieval_kg_graph_artifacts_table, [_row(artifact) for artifact in artifacts])

    def append_type_samples(self, samples: list[RetrievalTypeSamples]) -> None:
        self._append(retrieval_kg_type_samples_table, [_row(sample) for sample in samples])

    def replace_kg_entities(self, entities: list[RetrievalKgEntity]) -> None:
        self._replace_all(retrieval_kg_entities_table, [_row(entity) for entity in entities])

    def replace_kg_relations(self, relations: list[RetrievalKgRelation]) -> None:
        self._replace_all(retrieval_kg_relations_table, [_row(relation) for relation in relations])

    def replace_community_reports(self, reports: list[RetrievalCommunityReport]) -> None:
        self._replace_all(retrieval_kg_community_reports_table, [_row(report) for report in reports])

    def replace_graph_artifacts(self, artifacts: list[RetrievalGraphArtifact]) -> None:
        self._replace_all(retrieval_kg_graph_artifacts_table, [_row(artifact) for artifact in artifacts])

    def save_graph_artifact(self, artifact: RetrievalGraphArtifact) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                delete(retrieval_kg_graph_artifacts_table)
                .where(retrieval_kg_graph_artifacts_table.c.artifact_id == artifact.artifact_id)
            )
            connection.execute(retrieval_kg_graph_artifacts_table.insert(), _row(artifact))

    def replace_type_samples(self, samples: list[RetrievalTypeSamples]) -> None:
        self._replace_all(retrieval_kg_type_samples_table, [_row(sample) for sample in samples])

    def save_graphrag_checkpoint(
        self,
        checkpoint_type: str,
        checkpoint_key: str,
        payload: Any,
    ) -> bool:
        if not checkpoint_type or not checkpoint_key:
            return False
        with self.engine.begin() as connection:
            connection.execute(
                delete(retrieval_graphrag_checkpoints_table)
                .where(retrieval_graphrag_checkpoints_table.c.checkpoint_type == checkpoint_type)
                .where(retrieval_graphrag_checkpoints_table.c.checkpoint_key == checkpoint_key)
            )
            connection.execute(
                retrieval_graphrag_checkpoints_table.insert(),
                {
                    "checkpoint_type": checkpoint_type,
                    "checkpoint_key": checkpoint_key,
                    "payload": payload,
                    "updated_at": _utc_now(),
                    "metadata": {},
                },
            )
        return True

    def load_graphrag_checkpoints(self, checkpoint_type: str) -> dict[str, Any]:
        if not checkpoint_type:
            return {}
        statement = (
            select(
                retrieval_graphrag_checkpoints_table.c.checkpoint_key,
                retrieval_graphrag_checkpoints_table.c.payload,
            )
            .where(retrieval_graphrag_checkpoints_table.c.checkpoint_type == checkpoint_type)
            .order_by(retrieval_graphrag_checkpoints_table.c.checkpoint_key)
        )
        with self.engine.begin() as connection:
            return {
                str(row.checkpoint_key): row.payload
                for row in connection.execute(statement)
            }

    def cleanup_graphrag_checkpoints(self, checkpoint_type: str) -> int:
        if not checkpoint_type:
            return 0
        with self.engine.begin() as connection:
            result = connection.execute(
                delete(retrieval_graphrag_checkpoints_table)
                .where(retrieval_graphrag_checkpoints_table.c.checkpoint_type == checkpoint_type)
            )
        return result.rowcount or 0

    def has_graphrag_phase_marker(self, phase: str) -> bool:
        if not phase:
            return False
        statement = (
            select(retrieval_graphrag_phase_markers_table.c.phase)
            .where(retrieval_graphrag_phase_markers_table.c.phase == phase)
            .limit(1)
        )
        with self.engine.begin() as connection:
            return connection.execute(statement).first() is not None

    def set_graphrag_phase_marker(self, phase: str) -> bool:
        if not phase:
            return False
        with self.engine.begin() as connection:
            connection.execute(
                delete(retrieval_graphrag_phase_markers_table)
                .where(retrieval_graphrag_phase_markers_table.c.phase == phase)
            )
            connection.execute(
                retrieval_graphrag_phase_markers_table.insert(),
                {
                    "phase": phase,
                    "marked_at": _utc_now(),
                    "metadata": {},
                },
            )
        return True

    def clear_graphrag_phase_markers(self, phases: list[str] | tuple[str, ...]) -> None:
        clean_phases = _candidate_keywords(list(phases))
        if not clean_phases:
            return
        with self.engine.begin() as connection:
            connection.execute(
                delete(retrieval_graphrag_phase_markers_table)
                .where(retrieval_graphrag_phase_markers_table.c.phase.in_(clean_phases))
            )

    def save_graphrag_build_run(self, run: RetrievalGraphRagBuildRun) -> None:
        if not run.run_id:
            return
        with self.engine.begin() as connection:
            connection.execute(
                delete(retrieval_sync_state_table)
                .where(retrieval_sync_state_table.c.sync_key == run.run_id)
            )
            connection.execute(
                retrieval_sync_state_table.insert(),
                {
                    "sync_key": run.run_id,
                    "status": run.status,
                    "started_at": run.started_at,
                    "finished_at": run.finished_at,
                    "cursor": run.cursor,
                    "total": run.total,
                    "processed": run.processed,
                    "failed": run.failed,
                    "metadata": run.metadata,
                },
            )

    def get_graphrag_build_run(self, run_id: str) -> RetrievalGraphRagBuildRun | None:
        if not run_id:
            return None
        statement = (
            select(retrieval_sync_state_table)
            .where(retrieval_sync_state_table.c.sync_key == run_id)
            .limit(1)
        )
        with self.engine.begin() as connection:
            row = connection.execute(statement).first()
            if not row:
                return None
            return _graphrag_build_run_from_row(row._mapping)

    def claim_graphrag_build_lock(
        self,
        run_id: str,
        *,
        started_at: str,
        metadata: dict[str, Any],
    ) -> bool:
        if not run_id:
            return False
        if self.engine.dialect.name != "postgresql":
            with self._claim_lock:
                return self._claim_graphrag_build_lock(
                    run_id,
                    started_at=started_at,
                    metadata=metadata,
                )
        return self._claim_graphrag_build_lock(
            run_id,
            started_at=started_at,
            metadata=metadata,
        )

    def _claim_graphrag_build_lock(
        self,
        run_id: str,
        *,
        started_at: str,
        metadata: dict[str, Any],
    ) -> bool:
        with self.engine.begin() as connection:
            current = connection.execute(
                select(
                    retrieval_sync_state_table.c.status,
                    retrieval_sync_state_table.c.cursor,
                )
                .where(retrieval_sync_state_table.c.sync_key == GRAPHRAG_BUILD_LOCK_KEY)
                .limit(1)
            ).first()
            if current and current.status == "running":
                return False
            connection.execute(
                delete(retrieval_sync_state_table)
                .where(retrieval_sync_state_table.c.sync_key == GRAPHRAG_BUILD_LOCK_KEY)
            )
            connection.execute(
                retrieval_sync_state_table.insert(),
                {
                    "sync_key": GRAPHRAG_BUILD_LOCK_KEY,
                    "status": "running",
                    "started_at": started_at,
                    "finished_at": "",
                    "cursor": run_id,
                    "total": 1,
                    "processed": 0,
                    "failed": 0,
                    "metadata": metadata,
                },
            )
        return True

    def release_graphrag_build_lock(self, run_id: str) -> None:
        if not run_id:
            return
        with self.engine.begin() as connection:
            connection.execute(
                retrieval_sync_state_table.update()
                .where(retrieval_sync_state_table.c.sync_key == GRAPHRAG_BUILD_LOCK_KEY)
                .where(retrieval_sync_state_table.c.cursor == run_id)
                .values(status="released", finished_at=_utc_now(), processed=1)
            )

    def request_graphrag_build_cancel(self, run_id: str) -> bool:
        run = self.get_graphrag_build_run(run_id)
        if not run or run.status != "running":
            return False
        metadata = {**run.metadata, "cancel_requested": True}
        self.save_graphrag_build_run(
            RetrievalGraphRagBuildRun(
                run_id=run.run_id,
                status=run.status,
                started_at=run.started_at,
                finished_at=run.finished_at,
                cursor=run.cursor,
                total=run.total,
                processed=run.processed,
                failed=run.failed,
                metadata=metadata,
            )
        )
        return True

    def is_graphrag_build_cancel_requested(self, run_id: str) -> bool:
        run = self.get_graphrag_build_run(run_id)
        if not run:
            return False
        return bool(run.metadata.get("cancel_requested"))

    def update_chunk_vector_status(self, chunk_id: str, *, point_id: str, status: str) -> None:
        if self.engine.dialect.name != "postgresql":
            with self._claim_lock:
                self._update_chunk_vector_status(chunk_id, point_id=point_id, status=status)
            return
        self._update_chunk_vector_status(chunk_id, point_id=point_id, status=status)

    def _update_chunk_vector_status(self, chunk_id: str, *, point_id: str, status: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                retrieval_chunks_table.update()
                .where(retrieval_chunks_table.c.chunk_id == chunk_id)
                .values(vector_point_id=point_id, vector_status=status)
            )

    def update_entity_vector_status(self, entity_id: str, *, point_id: str, status: str) -> None:
        if self.engine.dialect.name != "postgresql":
            with self._claim_lock:
                self._update_entity_vector_status(entity_id, point_id=point_id, status=status)
            return
        self._update_entity_vector_status(entity_id, point_id=point_id, status=status)

    def _update_entity_vector_status(self, entity_id: str, *, point_id: str, status: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                retrieval_kg_entities_table.update()
                .where(retrieval_kg_entities_table.c.entity_id == entity_id)
                .values(vector_point_id=point_id, vector_status=status)
            )

    def update_relation_vector_status(self, relation_id: str, *, point_id: str, status: str) -> None:
        if self.engine.dialect.name != "postgresql":
            with self._claim_lock:
                self._update_relation_vector_status(
                    relation_id,
                    point_id=point_id,
                    status=status,
                )
            return
        self._update_relation_vector_status(relation_id, point_id=point_id, status=status)

    def _update_relation_vector_status(
        self,
        relation_id: str,
        *,
        point_id: str,
        status: str,
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                retrieval_kg_relations_table.update()
                .where(retrieval_kg_relations_table.c.relation_id == relation_id)
                .values(vector_point_id=point_id, vector_status=status)
            )

    def list_missing_vector_records(
        self,
        *,
        content_types: list[str] | None = None,
        limit: int = 1000,
    ) -> list[MissingVectorRecord]:
        allowed = set(content_types or ["chunk", "kg_entity", "kg_relation"])
        records: list[MissingVectorRecord] = []
        remaining = limit
        if "chunk" in allowed and remaining > 0:
            chunk_rows = self._missing_chunk_vectors(remaining)
            records.extend(chunk_rows)
            remaining -= len(chunk_rows)
        if "kg_entity" in allowed and remaining > 0:
            entity_rows = self._missing_entity_vectors(remaining)
            records.extend(entity_rows)
            remaining -= len(entity_rows)
        if "kg_relation" in allowed and remaining > 0:
            records.extend(self._missing_relation_vectors(remaining))
        return records

    def claim_missing_vector_records(
        self,
        *,
        content_types: list[str] | None = None,
        limit: int = 1000,
    ) -> list[MissingVectorRecord]:
        allowed = set(content_types or ["chunk", "kg_entity", "kg_relation"])
        records: list[MissingVectorRecord] = []
        remaining = limit
        if "chunk" in allowed and remaining > 0:
            chunk_rows = self._claim_chunk_vectors(remaining)
            records.extend(chunk_rows)
            remaining -= len(chunk_rows)
        if "kg_entity" in allowed and remaining > 0:
            entity_rows = self._claim_entity_vectors(remaining)
            records.extend(entity_rows)
            remaining -= len(entity_rows)
        if "kg_relation" in allowed and remaining > 0:
            records.extend(self._claim_relation_vectors(remaining))
        return records

    def list_documents(self) -> list[RetrievalDocument]:
        return self._list_all(
            retrieval_documents_table,
            RetrievalDocument,
            retrieval_documents_table.c.doc_id,
        )

    def list_chunks(
        self,
        *,
        source_id: str | None = None,
        available_only: bool = False,
    ) -> list[RetrievalChunk]:
        statement = select(retrieval_chunks_table).order_by(
            retrieval_chunks_table.c.source_id,
            retrieval_chunks_table.c.chunk_order_int,
            retrieval_chunks_table.c.chunk_id,
        )
        if source_id:
            statement = statement.where(retrieval_chunks_table.c.source_id == source_id)
        if available_only:
            statement = statement.where(retrieval_chunks_table.c.available_int == 1)
        with self.engine.begin() as connection:
            return [
                RetrievalChunk(**dict(row._mapping))
                for row in connection.execute(statement)
            ]

    def search_chunk_candidates(
        self,
        keywords: list[str],
        *,
        limit: int,
    ) -> list[RetrievalChunk]:
        clean_keywords = _candidate_keywords(keywords)
        statement = (
            select(retrieval_chunks_table)
            .where(retrieval_chunks_table.c.available_int == 1)
            .order_by(
                retrieval_chunks_table.c.source_id,
                retrieval_chunks_table.c.chunk_order_int,
                retrieval_chunks_table.c.chunk_id,
            )
            .limit(limit)
        )
        if clean_keywords:
            predicates = []
            for keyword in clean_keywords:
                pattern = f"%{_escape_like(keyword)}%"
                predicates.extend(
                    [
                        retrieval_chunks_table.c.content_with_weight.ilike(pattern, escape="\\"),
                        retrieval_chunks_table.c.content_ltks.ilike(pattern, escape="\\"),
                        retrieval_chunks_table.c.title_tks.ilike(pattern, escape="\\"),
                        retrieval_chunks_table.c.title.ilike(pattern, escape="\\"),
                    ]
                )
            statement = statement.where(or_(*predicates))
        with self.engine.begin() as connection:
            return [
                RetrievalChunk(**dict(row._mapping))
                for row in connection.execute(statement)
            ]

    def list_kg_entities(self, *, available_only: bool = False) -> list[RetrievalKgEntity]:
        statement = select(retrieval_kg_entities_table).order_by(
            retrieval_kg_entities_table.c.entity_name,
            retrieval_kg_entities_table.c.entity_id,
        )
        if available_only:
            statement = statement.where(retrieval_kg_entities_table.c.available_int == 1)
        with self.engine.begin() as connection:
            return [
                RetrievalKgEntity(**dict(row._mapping))
                for row in connection.execute(statement)
            ]

    def list_kg_relations(self, *, available_only: bool = False) -> list[RetrievalKgRelation]:
        statement = select(retrieval_kg_relations_table).order_by(
            retrieval_kg_relations_table.c.from_entity_kwd,
            retrieval_kg_relations_table.c.to_entity_kwd,
            retrieval_kg_relations_table.c.relation_id,
        )
        if available_only:
            statement = statement.where(retrieval_kg_relations_table.c.available_int == 1)
        with self.engine.begin() as connection:
            return [
                RetrievalKgRelation(**dict(row._mapping))
                for row in connection.execute(statement)
            ]

    def list_community_reports(
        self,
        *,
        available_only: bool = False,
    ) -> list[RetrievalCommunityReport]:
        statement = select(retrieval_kg_community_reports_table).order_by(
            retrieval_kg_community_reports_table.c.weight_flt.desc(),
            retrieval_kg_community_reports_table.c.title,
            retrieval_kg_community_reports_table.c.report_id,
        )
        if available_only:
            statement = statement.where(retrieval_kg_community_reports_table.c.available_int == 1)
        with self.engine.begin() as connection:
            return [
                RetrievalCommunityReport(**dict(row._mapping))
                for row in connection.execute(statement)
            ]

    def search_community_reports(
        self,
        entities: list[str],
        *,
        limit: int = 1,
    ) -> list[RetrievalCommunityReport]:
        entity_set = set(_candidate_keywords(entities))
        if not entity_set:
            return []
        reports = self.list_community_reports(available_only=True)
        matched = [
            report
            for report in reports
            if entity_set.intersection(str(entity) for entity in report.entities_kwd)
        ]
        matched.sort(
            key=lambda report: (
                len(entity_set.intersection(str(entity) for entity in report.entities_kwd)),
                report.weight_flt,
                report.title,
            ),
            reverse=True,
        )
        return matched[:limit]

    def list_graph_artifacts(
        self,
        *,
        available_only: bool = False,
    ) -> list[RetrievalGraphArtifact]:
        statement = select(retrieval_kg_graph_artifacts_table).order_by(
            retrieval_kg_graph_artifacts_table.c.artifact_type,
            retrieval_kg_graph_artifacts_table.c.artifact_id,
        )
        if available_only:
            statement = statement.where(retrieval_kg_graph_artifacts_table.c.available_int == 1)
        with self.engine.begin() as connection:
            return [
                RetrievalGraphArtifact(**dict(row._mapping))
                for row in connection.execute(statement)
            ]

    def get_graph_artifact(
        self,
        artifact_id: str,
        *,
        available_only: bool = False,
    ) -> RetrievalGraphArtifact | None:
        if not artifact_id:
            return None
        statement = select(retrieval_kg_graph_artifacts_table).where(
            retrieval_kg_graph_artifacts_table.c.artifact_id == artifact_id
        )
        if available_only:
            statement = statement.where(retrieval_kg_graph_artifacts_table.c.available_int == 1)
        with self.engine.begin() as connection:
            row = connection.execute(statement).first()
            return RetrievalGraphArtifact(**dict(row._mapping)) if row else None

    def get_subgraph_artifact(
        self,
        source_id: str,
        *,
        available_only: bool = True,
    ) -> RetrievalGraphArtifact | None:
        if not source_id:
            return None
        statement = (
            select(retrieval_kg_graph_artifacts_table)
            .where(retrieval_kg_graph_artifacts_table.c.artifact_type == "subgraph")
            .order_by(retrieval_kg_graph_artifacts_table.c.artifact_id)
        )
        if available_only:
            statement = statement.where(retrieval_kg_graph_artifacts_table.c.available_int == 1)
        with self.engine.begin() as connection:
            for row in connection.execute(statement):
                artifact = RetrievalGraphArtifact(**dict(row._mapping))
                if source_id in {str(value) for value in artifact.source_id}:
                    return artifact
        return None

    def list_type_samples(self) -> list[RetrievalTypeSamples]:
        return self._list_all(
            retrieval_kg_type_samples_table,
            RetrievalTypeSamples,
            retrieval_kg_type_samples_table.c.entity_type,
        )

    def get_chunk(self, chunk_id: str) -> RetrievalChunk | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(retrieval_chunks_table).where(retrieval_chunks_table.c.chunk_id == chunk_id)
            ).first()
            return RetrievalChunk(**dict(row._mapping)) if row else None

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[RetrievalChunk]:
        ordered_ids = _candidate_keywords(chunk_ids)
        if not ordered_ids:
            return []
        statement = select(retrieval_chunks_table).where(
            retrieval_chunks_table.c.chunk_id.in_(ordered_ids)
        )
        with self.engine.begin() as connection:
            chunks = {
                row.chunk_id: RetrievalChunk(**dict(row._mapping))
                for row in connection.execute(statement)
            }
        return [chunks[chunk_id] for chunk_id in ordered_ids if chunk_id in chunks]

    def count_embedded_chunks(self) -> int:
        with self.engine.begin() as connection:
            return _count_where(
                connection,
                retrieval_chunks_table,
                retrieval_chunks_table.c.vector_status == "embedded",
            )

    def get_kg_entities_by_ids(self, entity_ids: list[str]) -> list[RetrievalKgEntity]:
        ordered_ids = _candidate_keywords(entity_ids)
        if not ordered_ids:
            return []
        statement = select(retrieval_kg_entities_table).where(
            retrieval_kg_entities_table.c.entity_id.in_(ordered_ids)
        )
        with self.engine.begin() as connection:
            entities = {
                row.entity_id: RetrievalKgEntity(**dict(row._mapping))
                for row in connection.execute(statement)
            }
        return [entities[entity_id] for entity_id in ordered_ids if entity_id in entities]

    def get_kg_relations_by_ids(self, relation_ids: list[str]) -> list[RetrievalKgRelation]:
        ordered_ids = _candidate_keywords(relation_ids)
        if not ordered_ids:
            return []
        statement = select(retrieval_kg_relations_table).where(
            retrieval_kg_relations_table.c.relation_id.in_(ordered_ids)
        )
        with self.engine.begin() as connection:
            relations = {
                row.relation_id: RetrievalKgRelation(**dict(row._mapping))
                for row in connection.execute(statement)
            }
        return [
            relations[relation_id]
            for relation_id in ordered_ids
            if relation_id in relations
        ]

    def count_embedded_kg_entities(self) -> int:
        with self.engine.begin() as connection:
            return _count_where(
                connection,
                retrieval_kg_entities_table,
                retrieval_kg_entities_table.c.vector_status == "embedded",
            )

    def count_embedded_kg_relations(self) -> int:
        with self.engine.begin() as connection:
            return _count_where(
                connection,
                retrieval_kg_relations_table,
                retrieval_kg_relations_table.c.vector_status == "embedded",
            )

    def reset_failed_vector_statuses(
        self,
        *,
        content_types: list[str] | None = None,
    ) -> dict[str, int]:
        allowed = set(content_types or ["chunk", "kg_entity", "kg_relation"])
        summary = {
            "reset_failed_chunks": 0,
            "reset_failed_kg_entities": 0,
            "reset_failed_kg_relations": 0,
        }
        with self.engine.begin() as connection:
            if "chunk" in allowed:
                result = connection.execute(
                    retrieval_chunks_table.update()
                    .where(retrieval_chunks_table.c.vector_status == "failed")
                    .values(vector_status="missing", vector_point_id="")
                )
                summary["reset_failed_chunks"] = result.rowcount or 0
            if "kg_entity" in allowed:
                result = connection.execute(
                    retrieval_kg_entities_table.update()
                    .where(retrieval_kg_entities_table.c.vector_status == "failed")
                    .values(vector_status="missing", vector_point_id="")
                )
                summary["reset_failed_kg_entities"] = result.rowcount or 0
            if "kg_relation" in allowed:
                result = connection.execute(
                    retrieval_kg_relations_table.update()
                    .where(retrieval_kg_relations_table.c.vector_status == "failed")
                    .values(vector_status="missing", vector_point_id="")
                )
                summary["reset_failed_kg_relations"] = result.rowcount or 0
        return summary

    def audit(self) -> RetrievalAudit:
        with self.engine.begin() as connection:
            documents = _count(connection, retrieval_documents_table)
            chunks = _count(connection, retrieval_chunks_table)
            chunks_with_vectors = _count_where(
                connection,
                retrieval_chunks_table,
                retrieval_chunks_table.c.vector_status == "embedded",
            )
            chunks_failed_vectors = _count_where(
                connection,
                retrieval_chunks_table,
                retrieval_chunks_table.c.vector_status == "failed",
            )
            kg_entities = _count(connection, retrieval_kg_entities_table)
            kg_entities_with_vectors = _count_where(
                connection,
                retrieval_kg_entities_table,
                retrieval_kg_entities_table.c.vector_status == "embedded",
            )
            kg_entities_failed_vectors = _count_where(
                connection,
                retrieval_kg_entities_table,
                retrieval_kg_entities_table.c.vector_status == "failed",
            )
            kg_entities_with_evidence = _count_where(
                connection,
                retrieval_kg_entities_table,
                func.json_array_length(retrieval_kg_entities_table.c.evidence_chunk_ids) > 0,
            )
            kg_relations = _count(connection, retrieval_kg_relations_table)
            community_reports = _count(connection, retrieval_kg_community_reports_table)
            graph_artifacts = _count(connection, retrieval_kg_graph_artifacts_table)
            kg_relations_with_vectors = _count_where(
                connection,
                retrieval_kg_relations_table,
                retrieval_kg_relations_table.c.vector_status == "embedded",
            )
            kg_relations_failed_vectors = _count_where(
                connection,
                retrieval_kg_relations_table,
                retrieval_kg_relations_table.c.vector_status == "failed",
            )
            kg_relations_with_evidence = _count_where(
                connection,
                retrieval_kg_relations_table,
                func.json_array_length(retrieval_kg_relations_table.c.evidence_chunk_ids) > 0,
            )
            short_chunks = _count_where(
                connection,
                retrieval_chunks_table,
                retrieval_chunks_table.c.token_count < 30,
            )
            long_chunks = _count_where(
                connection,
                retrieval_chunks_table,
                retrieval_chunks_table.c.token_count > 512,
            )
        return RetrievalAudit(
            documents=documents,
            chunks=chunks,
            chunks_with_vectors=chunks_with_vectors,
            chunks_failed_vectors=chunks_failed_vectors,
            kg_entities=kg_entities,
            kg_entities_with_vectors=kg_entities_with_vectors,
            kg_entities_failed_vectors=kg_entities_failed_vectors,
            kg_entities_with_evidence=kg_entities_with_evidence,
            kg_relations=kg_relations,
            community_reports=community_reports,
            graph_artifacts=graph_artifacts,
            kg_relations_with_vectors=kg_relations_with_vectors,
            kg_relations_failed_vectors=kg_relations_failed_vectors,
            kg_relations_with_evidence=kg_relations_with_evidence,
            short_chunks=short_chunks,
            long_chunks=long_chunks,
        )

    def readiness(
        self,
        *,
        min_chunk_vector_coverage: float = 0.8,
        min_kg_entity_vector_coverage: float = 0.8,
        min_kg_relation_vector_coverage: float = 0.8,
        min_kg_entity_evidence_coverage: float = 0.8,
        min_kg_relation_evidence_coverage: float = 0.8,
        max_short_chunk_ratio: float = 0.2,
        max_long_chunk_ratio: float = 0.2,
    ) -> dict[str, Any]:
        audit = self.audit()
        chunk_token_buckets = self._chunk_token_buckets()
        vector_coverage = {
            "chunks": _coverage(
                audit.chunks_with_vectors,
                audit.chunks,
                failed=audit.chunks_failed_vectors,
            ),
            "kg_entities": _coverage(
                audit.kg_entities_with_vectors,
                audit.kg_entities,
                failed=audit.kg_entities_failed_vectors,
            ),
            "kg_relations": _coverage(
                audit.kg_relations_with_vectors,
                audit.kg_relations,
                failed=audit.kg_relations_failed_vectors,
            ),
        }
        evidence_coverage = {
            "kg_entities": _evidence_coverage(
                audit.kg_entities_with_evidence,
                audit.kg_entities,
            ),
            "kg_relations": _evidence_coverage(
                audit.kg_relations_with_evidence,
                audit.kg_relations,
            ),
        }
        vector_sync_plan = _vector_sync_plan(
            chunks=(audit.chunks_with_vectors, audit.chunks, min_chunk_vector_coverage),
            kg_entities=(
                audit.kg_entities_with_vectors,
                audit.kg_entities,
                min_kg_entity_vector_coverage,
            ),
            kg_relations=(
                audit.kg_relations_with_vectors,
                audit.kg_relations,
                min_kg_relation_vector_coverage,
            ),
        )
        short_chunk_ratio = _ratio(audit.short_chunks, audit.chunks)
        long_chunk_ratio = _ratio(audit.long_chunks, audit.chunks)
        blockers = []
        warnings = []
        if vector_coverage["chunks"]["ratio"] < min_chunk_vector_coverage:
            blockers.append("chunk_vector_coverage_below_threshold")
        if vector_coverage["kg_entities"]["ratio"] < min_kg_entity_vector_coverage:
            blockers.append("kg_entity_vector_coverage_below_threshold")
        if vector_coverage["kg_relations"]["ratio"] < min_kg_relation_vector_coverage:
            blockers.append("kg_relation_vector_coverage_below_threshold")
        if evidence_coverage["kg_entities"]["ratio"] < min_kg_entity_evidence_coverage:
            blockers.append("kg_entity_evidence_coverage_below_threshold")
        if evidence_coverage["kg_relations"]["ratio"] < min_kg_relation_evidence_coverage:
            blockers.append("kg_relation_evidence_coverage_below_threshold")
        if short_chunk_ratio > max_short_chunk_ratio:
            warnings.append("short_chunk_ratio_above_threshold")
        if long_chunk_ratio > max_long_chunk_ratio:
            warnings.append("long_chunk_ratio_above_threshold")
        if (
            audit.chunks_failed_vectors
            + audit.kg_entities_failed_vectors
            + audit.kg_relations_failed_vectors
            > 0
        ):
            warnings.append("failed_vectors_present")
        return {
            "ready": not blockers,
            "blockers": blockers,
            "warnings": warnings,
            "documents": audit.documents,
            "chunks": audit.chunks,
            "kg_entities": audit.kg_entities,
            "kg_relations": audit.kg_relations,
            "community_reports": audit.community_reports,
            "graph_artifacts": audit.graph_artifacts,
            "vector_coverage": vector_coverage,
            "evidence_coverage": evidence_coverage,
            "vector_sync_plan": vector_sync_plan,
            "chunk_token_buckets": chunk_token_buckets,
            "chunk_quality": {
                "short_chunks": audit.short_chunks,
                "short_chunk_ratio": short_chunk_ratio,
                "long_chunks": audit.long_chunks,
                "long_chunk_ratio": long_chunk_ratio,
            },
        }

    def _chunk_token_buckets(self) -> dict[str, int]:
        with self.engine.begin() as connection:
            return {
                "lt_30": _count_where(
                    connection,
                    retrieval_chunks_table,
                    retrieval_chunks_table.c.token_count < 30,
                ),
                "30_127": _count_where(
                    connection,
                    retrieval_chunks_table,
                    (retrieval_chunks_table.c.token_count >= 30)
                    & (retrieval_chunks_table.c.token_count < 128),
                ),
                "128_256": _count_where(
                    connection,
                    retrieval_chunks_table,
                    (retrieval_chunks_table.c.token_count >= 128)
                    & (retrieval_chunks_table.c.token_count <= 256),
                ),
                "257_512": _count_where(
                    connection,
                    retrieval_chunks_table,
                    (retrieval_chunks_table.c.token_count > 256)
                    & (retrieval_chunks_table.c.token_count <= 512),
                ),
                "gt_512": _count_where(
                    connection,
                    retrieval_chunks_table,
                    retrieval_chunks_table.c.token_count > 512,
                ),
            }

    def _replace_all(self, table, rows: list[dict[str, Any]]) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(table))
            if rows:
                connection.execute(table.insert(), rows)

    def _append(self, table, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.engine.begin() as connection:
            connection.execute(table.insert(), rows)

    def _list_all(self, table, schema: type[T], *order_by) -> list[T]:
        statement = select(table).order_by(*order_by)
        with self.engine.begin() as connection:
            return [schema(**dict(row._mapping)) for row in connection.execute(statement)]

    def _missing_chunk_vectors(self, limit: int) -> list[MissingVectorRecord]:
        statement = (
            select(
                retrieval_chunks_table.c.chunk_id,
                retrieval_chunks_table.c.content_with_weight,
            )
            .where(retrieval_chunks_table.c.available_int == 1)
            .where(retrieval_chunks_table.c.vector_status == "missing")
            .order_by(retrieval_chunks_table.c.source_id, retrieval_chunks_table.c.chunk_order_int)
            .limit(limit)
        )
        with self.engine.begin() as connection:
            return [
                MissingVectorRecord(
                    id=f"chunk:{row.chunk_id}",
                    text=row.content_with_weight,
                    content_type="chunk",
                    chunk_id=row.chunk_id,
                )
                for row in connection.execute(statement)
            ]

    def _missing_entity_vectors(self, limit: int) -> list[MissingVectorRecord]:
        statement = (
            select(
                retrieval_kg_entities_table.c.entity_id,
                retrieval_kg_entities_table.c.content_with_weight,
                retrieval_kg_entities_table.c.entity_type,
            )
            .where(retrieval_kg_entities_table.c.available_int == 1)
            .where(retrieval_kg_entities_table.c.vector_status == "missing")
            .order_by(retrieval_kg_entities_table.c.entity_name)
            .limit(limit)
        )
        with self.engine.begin() as connection:
            return [
                MissingVectorRecord(
                    id=f"kg_entity:{row.entity_id}",
                    text=row.content_with_weight,
                    content_type="kg_entity",
                    entity_id=row.entity_id,
                    label=row.entity_type,
                )
                for row in connection.execute(statement)
            ]

    def _missing_relation_vectors(self, limit: int) -> list[MissingVectorRecord]:
        statement = (
            select(
                retrieval_kg_relations_table.c.relation_id,
                retrieval_kg_relations_table.c.content_with_weight,
                retrieval_kg_relations_table.c.relation_type,
            )
            .where(retrieval_kg_relations_table.c.available_int == 1)
            .where(retrieval_kg_relations_table.c.vector_status == "missing")
            .order_by(retrieval_kg_relations_table.c.from_entity_kwd)
            .limit(limit)
        )
        with self.engine.begin() as connection:
            return [
                MissingVectorRecord(
                    id=f"kg_relation:{row.relation_id}",
                    text=row.content_with_weight,
                    content_type="kg_relation",
                    relation_id=row.relation_id,
                    label=row.relation_type,
                )
                for row in connection.execute(statement)
            ]

    def _claim_chunk_vectors(self, limit: int) -> list[MissingVectorRecord]:
        if self.engine.dialect.name == "postgresql":
            with self.engine.begin() as connection:
                rows = connection.execute(
                    text(
                        """
                        WITH candidates AS (
                            SELECT chunk_id
                            FROM retrieval_chunks
                            WHERE available_int = 1 AND vector_status = 'missing'
                            ORDER BY source_id, chunk_order_int
                            FOR UPDATE SKIP LOCKED
                            LIMIT :limit
                        )
                        UPDATE retrieval_chunks AS target
                        SET vector_status = 'queued', vector_point_id = ''
                        FROM candidates
                        WHERE target.chunk_id = candidates.chunk_id
                        RETURNING target.chunk_id, target.content_with_weight
                        """
                    ),
                    {"limit": limit},
                ).mappings()
                return [
                    MissingVectorRecord(
                        id=f"chunk:{row['chunk_id']}",
                        text=row["content_with_weight"],
                        content_type="chunk",
                        chunk_id=row["chunk_id"],
                    )
                    for row in rows
                ]
        with self._claim_lock:
            return self._claim_chunk_vectors_unlocked(limit)

    def _claim_chunk_vectors_unlocked(self, limit: int) -> list[MissingVectorRecord]:
        with self.engine.begin() as connection:
            rows = list(
                connection.execute(
                    select(
                        retrieval_chunks_table.c.chunk_id,
                        retrieval_chunks_table.c.content_with_weight,
                    )
                    .where(retrieval_chunks_table.c.available_int == 1)
                    .where(retrieval_chunks_table.c.vector_status == "missing")
                    .order_by(
                        retrieval_chunks_table.c.source_id,
                        retrieval_chunks_table.c.chunk_order_int,
                    )
                    .limit(limit)
                )
            )
            chunk_ids = [row.chunk_id for row in rows]
            if chunk_ids:
                connection.execute(
                    retrieval_chunks_table.update()
                    .where(retrieval_chunks_table.c.chunk_id.in_(chunk_ids))
                    .where(retrieval_chunks_table.c.vector_status == "missing")
                    .values(vector_status="queued", vector_point_id="")
                )
            return [
                MissingVectorRecord(
                    id=f"chunk:{row.chunk_id}",
                    text=row.content_with_weight,
                    content_type="chunk",
                    chunk_id=row.chunk_id,
                )
                for row in rows
            ]

    def _claim_entity_vectors(self, limit: int) -> list[MissingVectorRecord]:
        if self.engine.dialect.name == "postgresql":
            with self.engine.begin() as connection:
                rows = connection.execute(
                    text(
                        """
                        WITH candidates AS (
                            SELECT entity_id
                            FROM retrieval_kg_entities
                            WHERE available_int = 1 AND vector_status = 'missing'
                            ORDER BY entity_name
                            FOR UPDATE SKIP LOCKED
                            LIMIT :limit
                        )
                        UPDATE retrieval_kg_entities AS target
                        SET vector_status = 'queued', vector_point_id = ''
                        FROM candidates
                        WHERE target.entity_id = candidates.entity_id
                        RETURNING target.entity_id, target.content_with_weight, target.entity_type
                        """
                    ),
                    {"limit": limit},
                ).mappings()
                return [
                    MissingVectorRecord(
                        id=f"kg_entity:{row['entity_id']}",
                        text=row["content_with_weight"],
                        content_type="kg_entity",
                        entity_id=row["entity_id"],
                        label=row["entity_type"],
                    )
                    for row in rows
                ]
        with self._claim_lock:
            return self._claim_entity_vectors_unlocked(limit)

    def _claim_entity_vectors_unlocked(self, limit: int) -> list[MissingVectorRecord]:
        with self.engine.begin() as connection:
            rows = list(
                connection.execute(
                    select(
                        retrieval_kg_entities_table.c.entity_id,
                        retrieval_kg_entities_table.c.content_with_weight,
                        retrieval_kg_entities_table.c.entity_type,
                    )
                    .where(retrieval_kg_entities_table.c.available_int == 1)
                    .where(retrieval_kg_entities_table.c.vector_status == "missing")
                    .order_by(retrieval_kg_entities_table.c.entity_name)
                    .limit(limit)
                )
            )
            entity_ids = [row.entity_id for row in rows]
            if entity_ids:
                connection.execute(
                    retrieval_kg_entities_table.update()
                    .where(retrieval_kg_entities_table.c.entity_id.in_(entity_ids))
                    .where(retrieval_kg_entities_table.c.vector_status == "missing")
                    .values(vector_status="queued", vector_point_id="")
                )
            return [
                MissingVectorRecord(
                    id=f"kg_entity:{row.entity_id}",
                    text=row.content_with_weight,
                    content_type="kg_entity",
                    entity_id=row.entity_id,
                    label=row.entity_type,
                )
                for row in rows
            ]

    def _claim_relation_vectors(self, limit: int) -> list[MissingVectorRecord]:
        if self.engine.dialect.name == "postgresql":
            with self.engine.begin() as connection:
                rows = connection.execute(
                    text(
                        """
                        WITH candidates AS (
                            SELECT relation_id
                            FROM retrieval_kg_relations
                            WHERE available_int = 1 AND vector_status = 'missing'
                            ORDER BY from_entity_kwd
                            FOR UPDATE SKIP LOCKED
                            LIMIT :limit
                        )
                        UPDATE retrieval_kg_relations AS target
                        SET vector_status = 'queued', vector_point_id = ''
                        FROM candidates
                        WHERE target.relation_id = candidates.relation_id
                        RETURNING target.relation_id, target.content_with_weight, target.relation_type
                        """
                    ),
                    {"limit": limit},
                ).mappings()
                return [
                    MissingVectorRecord(
                        id=f"kg_relation:{row['relation_id']}",
                        text=row["content_with_weight"],
                        content_type="kg_relation",
                        relation_id=row["relation_id"],
                        label=row["relation_type"],
                    )
                    for row in rows
                ]
        with self._claim_lock:
            return self._claim_relation_vectors_unlocked(limit)

    def _claim_relation_vectors_unlocked(self, limit: int) -> list[MissingVectorRecord]:
        with self.engine.begin() as connection:
            rows = list(
                connection.execute(
                    select(
                        retrieval_kg_relations_table.c.relation_id,
                        retrieval_kg_relations_table.c.content_with_weight,
                        retrieval_kg_relations_table.c.relation_type,
                    )
                    .where(retrieval_kg_relations_table.c.available_int == 1)
                    .where(retrieval_kg_relations_table.c.vector_status == "missing")
                    .order_by(retrieval_kg_relations_table.c.from_entity_kwd)
                    .limit(limit)
                )
            )
            relation_ids = [row.relation_id for row in rows]
            if relation_ids:
                connection.execute(
                    retrieval_kg_relations_table.update()
                    .where(retrieval_kg_relations_table.c.relation_id.in_(relation_ids))
                    .where(retrieval_kg_relations_table.c.vector_status == "missing")
                    .values(vector_status="queued", vector_point_id="")
                )
            return [
                MissingVectorRecord(
                    id=f"kg_relation:{row.relation_id}",
                    text=row.content_with_weight,
                    content_type="kg_relation",
                    relation_id=row.relation_id,
                    label=row.relation_type,
                )
                for row in rows
            ]


def _row(value) -> dict[str, Any]:
    return asdict(value)


def _graphrag_build_run_from_row(row) -> RetrievalGraphRagBuildRun:
    return RetrievalGraphRagBuildRun(
        run_id=row["sync_key"],
        status=row["status"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        cursor=row["cursor"],
        total=row["total"],
        processed=row["processed"],
        failed=row["failed"],
        metadata=row["metadata"],
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _count(connection, table) -> int:
    return int(connection.execute(select(func.count()).select_from(table)).scalar_one())


def _count_where(connection, table, predicate) -> int:
    return int(
        connection.execute(select(func.count()).select_from(table).where(predicate)).scalar_one()
    )


def _ratio(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(part / total, 6)


def _coverage(embedded: int, total: int, *, failed: int = 0) -> dict[str, float | int]:
    return {
        "embedded": embedded,
        "failed": failed,
        "total": total,
        "ratio": _ratio(embedded, total),
    }


def _evidence_coverage(with_evidence: int, total: int) -> dict[str, float | int]:
    return {
        "with_evidence": with_evidence,
        "total": total,
        "ratio": _ratio(with_evidence, total),
    }


def _vector_sync_plan(
    *,
    chunks: tuple[int, int, float],
    kg_entities: tuple[int, int, float],
    kg_relations: tuple[int, int, float],
) -> dict[str, Any]:
    plan = {
        "chunks": _sync_target(*chunks),
        "kg_entities": _sync_target(*kg_entities),
        "kg_relations": _sync_target(*kg_relations),
    }
    content_type_by_plan_key = {
        "chunks": "chunk",
        "kg_entities": "kg_entity",
        "kg_relations": "kg_relation",
    }
    pending_content_types = [
        content_type_by_plan_key[key]
        for key in ["chunks", "kg_entities", "kg_relations"]
        if int(plan[key]["remaining"]) > 0
    ]
    balanced_limit = sum(int(plan[key]["remaining"]) for key in plan)
    plan["balanced_limit"] = balanced_limit
    plan["content_types"] = pending_content_types
    plan["recommended_command"] = _vector_sync_command(
        limit=balanced_limit,
        content_types=pending_content_types,
    )
    return plan


def _vector_sync_command(*, limit: int, content_types: list[str]) -> str:
    if limit <= 0 or not content_types:
        return ""
    base = "python ../scripts/sync_ragflow_retrieval_vectors.py "
    content_types_arg = ",".join(content_types)
    if len(content_types) == 1:
        return f"{base}--limit {limit} --content-types {content_types_arg}"
    return f"{base}--balanced --limit {limit} --content-types {content_types_arg}"


def _sync_target(embedded: int, total: int, target_ratio: float) -> dict[str, float | int]:
    target_embedded = min(total, math.ceil(total * target_ratio))
    return {
        "target_ratio": target_ratio,
        "target_embedded": target_embedded,
        "current_embedded": embedded,
        "remaining": max(target_embedded - embedded, 0),
    }


def _candidate_keywords(keywords: list[str]) -> list[str]:
    seen = set()
    result = []
    for keyword in keywords:
        clean = keyword.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
