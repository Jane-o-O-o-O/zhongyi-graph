# Complete GraphRAG Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing TCM graph demo into a complete local GraphRAG pipeline with file storage, document chunks, vector indexing, graph publishing, and Graph + Chunk retrieval.

**Architecture:** Add MinIO for raw files and PostgreSQL for sources/pages/chunks/candidates/publish state while keeping Qdrant for vectors and Neo4j for the published graph. Ingestion lives in the backend as a separated module with explicit interfaces so it can later become its own worker service. Query retrieval uses Neo4j-style graph data plus Qdrant hits hydrated from PostgreSQL chunks.

**Tech Stack:** FastAPI, Pydantic, SQLAlchemy, PostgreSQL, MinIO, Qdrant, Neo4j, SiliconFlow-compatible LLM/OCR/Embedding/Rerank APIs, pytest.

---

### Task 1: Storage Configuration And Compose

**Files:**
- Modify: `compose.yml`
- Modify: `.env.example`
- Modify: `backend/app/core/config.py`
- Modify: `backend/pyproject.toml`
- Test: `backend/tests/test_contracts.py`

- [ ] Add `tcm-postgres` and `tcm-minio` services with persistent volumes.
- [ ] Add config values for PostgreSQL, MinIO, object bucket, and OCR model `deepseek-ai/DeepSeek-OCR`.
- [ ] Add Python dependencies: `sqlalchemy`, `psycopg[binary]`, `minio`, `python-multipart`, `pypdf`, `python-docx`.
- [ ] Test that settings expose the new values.

### Task 2: Canonical Ingestion Models

**Files:**
- Modify: `backend/app/models/ingestion.py`
- Test: `backend/tests/test_ingestion_models.py`

- [ ] Add `DocumentPage`, `DocumentChunk`, `EntityCandidate`, `RelationCandidate`, `PublishBatch`, and `KnowledgeBundle` Pydantic models.
- [ ] Validate chunk/source/evidence IDs are stable strings.
- [ ] Test model serialization for a source with chunks and graph candidates.

### Task 3: PostgreSQL Repository Layer

**Files:**
- Create: `backend/app/services/ingestion_repository.py`
- Test: `backend/tests/test_ingestion_repository.py`

- [ ] Implement SQLAlchemy table metadata for sources, pages, chunks, entity candidates, relation candidates, and publish batches.
- [ ] Provide `IngestionRepository.in_memory()` with SQLite for tests.
- [ ] Provide `upsert_source`, `replace_chunks`, `list_chunks`, `save_candidates`, `get_bundle`, and `record_publish_batch`.
- [ ] Test idempotent source/chunk writes and candidate hydration.

### Task 4: Object Storage Layer

**Files:**
- Create: `backend/app/services/object_storage.py`
- Test: `backend/tests/test_object_storage.py`

- [ ] Implement local filesystem object storage for tests and development fallback.
- [ ] Implement MinIO-backed adapter using config.
- [ ] Expose `put_bytes`, `get_bytes`, and deterministic object key generation.
- [ ] Test content-addressed object keys.

### Task 5: OCR And Parsing Layer

**Files:**
- Create: `backend/app/services/ocr_client.py`
- Create: `backend/app/services/document_parser.py`
- Test: `backend/tests/test_ocr_client.py`
- Test: `backend/tests/test_document_parser.py`

- [ ] Add SiliconFlow OCR client configured with `deepseek-ai/DeepSeek-OCR`.
- [ ] Parse `.txt`, `.md`, `.csv`, `.json`, `.docx`, and text PDFs locally.
- [ ] If a PDF page has no text, call OCR adapter and mark source as OCR-backed.
- [ ] Chunk parsed text by section and character budget with page/source metadata.
- [ ] Test TXT parsing, CSV/table chunking, and OCR fallback with a fake OCR client.

### Task 6: Graph Extraction Layer

**Files:**
- Create: `backend/app/services/graph_extractor.py`
- Test: `backend/tests/test_graph_extractor.py`

- [ ] Implement deterministic heuristic extraction for TCM labels and relations.
- [ ] Add LLM extraction client path with strict JSON output contract.
- [ ] Keep extraction candidates bound to chunk IDs.
- [ ] Test extraction from a chunk containing symptom, syndrome, treatment, formula, and herbs.

### Task 7: Publishing Layer

**Files:**
- Create: `backend/app/services/knowledge_publisher.py`
- Modify: `backend/app/services/vector_service.py`
- Modify: `scripts/import_seed_graph.py`
- Test: `backend/tests/test_knowledge_publisher.py`

- [ ] Publish candidates into the in-process graph service and Neo4j-compatible bundle.
- [ ] Index chunks, entities, and evidence into Qdrant payloads.
- [ ] Persist evidence cards from chunks.
- [ ] Test that publishing a bundle yields graph nodes/edges and vector payloads.

### Task 8: Ingestion API

**Files:**
- Modify: `backend/app/services/ingestion_service.py`
- Modify: `backend/app/api/routes.py`
- Test: `backend/tests/test_api.py`

- [ ] Add file upload endpoint `POST /api/ingestion/upload`.
- [ ] Add parse/run endpoint `POST /api/ingestion/jobs/{job_id}/run`.
- [ ] Add publish endpoint `POST /api/ingestion/publish`.
- [ ] Add source/chunk listing endpoints.
- [ ] Test upload, parse, chunk persistence, and publish response counts.

### Task 9: Query GraphRAG Retrieval

**Files:**
- Modify: `backend/app/services/question_service.py`
- Modify: `backend/app/services/hybrid_retriever.py`
- Create: `backend/app/services/chunk_retriever.py`
- Test: `backend/tests/test_question_service.py`

- [ ] Add chunk retrieval from Qdrant hits hydrated through PostgreSQL.
- [ ] Merge graph evidence and chunk evidence before LLM synthesis.
- [ ] Keep graph primary in frontend response.
- [ ] Test that a query can answer from a newly published chunk.

### Task 10: Docker Verification

**Files:**
- Modify: `README.md`

- [ ] Start Docker services.
- [ ] Upload a small document.
- [ ] Run parse and publish.
- [ ] Verify PostgreSQL chunk count, Qdrant point count, Neo4j node/relationship count, and `/api/query` response.
- [ ] Document exact commands for the demo import flow.

