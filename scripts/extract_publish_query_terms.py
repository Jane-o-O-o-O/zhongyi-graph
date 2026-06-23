from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

from sqlalchemy import create_engine, text

PROJECT_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(PROJECT_BACKEND) not in sys.path:
    sys.path.insert(0, str(PROJECT_BACKEND))

from app.core.config import get_settings
from app.models.ingestion import KnowledgeBundle
from app.services.graph_extractor import GraphExtractor
from app.services.ingestion_repository import IngestionRepository
from app.services.knowledge_publisher import KnowledgePublisher
from app.services.model_clients import StructuredExtractionClient
from app.services.neo4j_publisher import Neo4jPublisher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-extract graph candidates for chunks matching terms.")
    parser.add_argument("terms", nargs="+")
    parser.add_argument("--postgres-dsn", default="")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--source-id", default="")
    parser.add_argument("--chunk-id", action="append", default=[])
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    dsn = args.postgres_dsn or settings.postgres_dsn
    repository = IngestionRepository(create_engine(dsn, future=True, pool_pre_ping=True))
    chunks = find_chunks(
        dsn=dsn,
        terms=args.terms,
        limit=args.limit,
        source_id=args.source_id,
        chunk_ids=args.chunk_id,
    )
    print(f"found chunks={len(chunks)}", flush=True)
    extractor = GraphExtractor(
        llm_extractor=StructuredExtractionClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    )
    by_source = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source_id].append(chunk)

    source_ids = []
    total_entities = 0
    total_relations = 0
    for source_id, source_chunks in by_source.items():
        print(f"extracting source={source_id} chunks={len(source_chunks)}", flush=True)
        entities, relations = extractor.extract(source_chunks, hint_terms=args.terms)
        print(
            f"extracted source={source_id} entities={len(entities)} relations={len(relations)}",
            flush=True,
        )
        if not entities and not relations:
            continue
        print(f"merging source={source_id}", flush=True)
        repository.merge_candidates(source_id, entities, relations)
        source_ids.append(source_id)
        total_entities += len(entities)
        total_relations += len(relations)
        print(
            f"source={source_id} chunks={len(source_chunks)} "
            f"entities={len(entities)} relations={len(relations)}",
            flush=True,
        )

    if args.publish and source_ids:
        print(f"building artifact for sources={len(source_ids)}", flush=True)
        bundles = [repository.get_bundle(source_id) for source_id in source_ids]
        artifact = KnowledgePublisher().build_artifact(bundles)
        print(
            f"publishing artifact nodes={len(artifact.nodes)} edges={len(artifact.edges)}",
            flush=True,
        )
        Neo4jPublisher.from_config(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        ).publish(artifact)
        for bundle in bundles:
            source = bundle.source
            source.status = "published"
            repository.upsert_source(source)
        print(
            f"published sources={len(source_ids)} nodes={len(artifact.nodes)} "
            f"edges={len(artifact.edges)} evidence={len(artifact.evidence)}",
            flush=True,
        )

    print(
        f"matched_chunks={len(chunks)} sources={len(source_ids)} "
        f"entities={total_entities} relations={total_relations}",
        flush=True,
    )


def find_chunks(
    dsn: str,
    terms: list[str],
    limit: int,
    source_id: str = "",
    chunk_ids: list[str] | None = None,
):
    repository = IngestionRepository(create_engine(dsn, future=True, pool_pre_ping=True))
    if chunk_ids:
        chunks = [repository.get_chunk(chunk_id) for chunk_id in chunk_ids]
        return [chunk for chunk in chunks if chunk is not None]

    clauses = [f"content LIKE :term{index}" for index, _term in enumerate(terms)]
    if source_id:
        clauses.append("source_id = :source_id")
    statement = text(
        f"""
        SELECT chunk_id
        FROM document_chunks
        WHERE {" AND ".join(clauses)}
        ORDER BY char_length(content) ASC
        LIMIT :limit
        """
    )
    params = {f"term{index}": f"%{term}%" for index, term in enumerate(terms)}
    params["limit"] = limit
    if source_id:
        params["source_id"] = source_id
    with repository.engine.begin() as connection:
        chunk_ids = [row.chunk_id for row in connection.execute(statement, params)]
    chunks = []
    for chunk_id in chunk_ids:
        chunk = repository.get_chunk(chunk_id)
        if chunk:
            chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    main()
