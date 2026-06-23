# TCM Knowledge Graph Platform Design

## Goal

Build `tcm-kg-platform`, a polished Chinese medicine knowledge graph demo platform for leadership presentations. The demo must support live user questions, prioritize graph visualization, and combine local Chinese medicine data, a Neo4j knowledge graph, document retrieval, selected authoritative external sources, and a configurable large-model API.

The first version should look like a real fundable product, not a generic chatbot or raw open-source admin console.

## Product Positioning

The platform is a Chinese medicine knowledge graph intelligent workbench. Its primary interaction is:

1. A leader enters a natural-language question.
2. The system identifies entities, intent, and keywords.
3. The backend queries the knowledge graph for related paths.
4. The backend retrieves local evidence from structured data and documents.
5. When useful, the backend checks authoritative external sources from a whitelist.
6. A configurable OpenAI-compatible large-model API synthesizes the answer.
7. The frontend presents the result with the graph as the main visual, supported by answer summaries and evidence cards.

The system should feel confident and smooth in demo mode. The main answer area should not emphasize uncertainty. Evidence and source cards provide credibility without turning the presentation into a risk disclaimer.

## Visual Direction

The confirmed visual direction is a light classical Chinese medicine workbench:

- Background: restrained rice-white and paper-like tones.
- Accent colors: cinnabar red, ink black, herb green, warm gold, and muted blue-gray for external sources.
- Layout: modern workbench with refined cards, thin borders, compact spacing, and low-radius surfaces.
- Graph interaction: modern, animated, and technology-forward.
- Avoid: generic dark dashboards, heavy decorative traditional motifs, and chatbot-first layouts.

## Main User Experience

### Primary Page: Question-Driven Graph Workbench

The primary page layout is:

- Top: platform name, question input, source status, and quick controls.
- Left: concise answer summary, parsed question, and suggested follow-up questions.
- Center: large interactive knowledge graph canvas.
- Right: evidence chain, selected-node details, and source snippets.
- Bottom: highlighted relationship path and current retrieval summary.

After a question is submitted:

1. The graph expands from the most relevant entities.
2. Related paths are highlighted.
3. A concise answer appears beside the graph.
4. Evidence cards slide in on the right.
5. Clicking a node reveals node details and allows expansion.
6. Clicking evidence shows source snippets and local file/source metadata.

### Supported Story Lines

The first version must support all three story lines through live questions:

- Disease/symptom questions: symptom -> syndrome -> treatment method -> formula -> herbs -> evidence.
- Formula questions: formula -> composition -> dosage -> functions -> indications -> source.
- Herb questions: herb -> aliases -> properties/flavors -> channel tropism -> effects -> indications -> related formulas -> source.

These are not fixed templates. They are graph and retrieval paths that the system should surface based on live questions.

## Frontend Design

### Stack

- React
- TypeScript
- Vite
- Ant Design
- AntV G6
- ECharts

### Key Components

- `QuestionInput`: captures the leader's question and displays submit/loading state.
- `GraphCanvas`: renders the main AntV G6 graph and supports node expansion, path highlighting, and layout transitions.
- `AnswerPanel`: shows a concise synthesized answer and parsed intent/entities.
- `EvidencePanel`: lists local and external evidence cards with source metadata.
- `NodeDetailPanel`: shows details for selected graph nodes.
- `AssetOverview`: displays counts for herbs, formulas, symptoms, documents, and relationships.
- `SourceStatus`: shows whether local graph, local documents, model API, and external source checks are available.

### Node Types

- Disease or symptom: cinnabar red.
- Syndrome: amber.
- Treatment method: blue-green.
- Formula: ink black.
- Herb: herb green.
- Classical text or local document: warm gold.
- External authoritative source: muted blue-gray.

### Interaction Rules

- The graph is always the visual center.
- Answers should stay concise and confident.
- Evidence cards should be visible but secondary.
- Node expansion should be animated and stable.
- Highlighted paths should clearly show the reasoning chain.
- If results are sparse, the frontend should still show related entities and exploration paths.

## Backend Design

### Stack

- FastAPI
- Python
- Neo4j driver
- OpenAI-compatible model API client
- Local parsers and import scripts for CSV, JSON, Markdown, and GB18030 text files

### Core Services

- `QuestionService`: receives questions and coordinates the full answer flow.
- `EntityExtractionService`: calls the model API to identify entities, keywords, and intent.
- `GraphQueryService`: queries Neo4j for relevant nodes, relationships, and paths.
- `EvidenceRetrievalService`: retrieves snippets from local structured and document sources.
- `ExternalSourceService`: checks authoritative whitelisted external sources when enabled.
- `AnswerSynthesisService`: calls the model API to generate the final answer from graph and evidence context.
- `GraphImportService`: imports cleaned graph data into Neo4j.
- `IngestionService`: registers uploaded/local sources and coordinates parsing, normalization, extraction, validation, and publishing.
- `DocumentParsingService`: extracts text, tables, page locations, and metadata from uploaded PDF, Word, Markdown, text, CSV, and JSON files.

### Knowledge Ingestion Service

The platform needs a stable upstream parsing and ingestion service, not one-off scripts. This service is responsible for turning both existing local materials and newly uploaded user documents into canonical graph and retrieval artifacts.

The ingestion service should expose:

- `POST /api/ingestion/sources`: register or upload a source document.
- `POST /api/ingestion/jobs`: start an ingestion job for one or more sources.
- `GET /api/ingestion/jobs/{job_id}`: inspect job status, errors, and generated artifacts.
- `GET /api/ingestion/sources`: list registered sources and versions.
- `POST /api/ingestion/publish`: publish validated artifacts into Neo4j and the document index.

The ingestion flow is:

1. Register source file with path, type, checksum, owner, upload time, and version.
2. Parse content into canonical text blocks and tables.
3. Normalize content into `DocumentChunk` records with stable source locations.
4. Extract candidate entities and relationships with rules and model assistance.
5. Bind each extracted relation to one or more evidence chunks.
6. Validate artifacts against the graph schema and quality rules.
7. Publish approved entities, relations, evidence, and document chunks to downstream stores.

For the first version, ingestion can run as a FastAPI-triggered background worker inside the same backend container. The design should allow moving it to a dedicated worker service later.

### Uploaded Document Parsing

User-added documents may be PDF, Word, Markdown, text, CSV, or JSON. The parser layer should hide file differences and produce the same canonical output shape.

Supported first-version parsing strategy:

- `.pdf`: use a PDF text parser for selectable text and preserve page numbers. If a PDF has no extractable text, mark it as requiring OCR rather than silently producing empty content.
- `.docx`: parse paragraphs, headings, and tables with a Word document parser.
- `.doc`: convert to `.docx` or PDF through headless LibreOffice, then parse the converted file.
- `.md` / `.txt`: decode text, detect GB18030 when needed, preserve headings and paragraph boundaries.
- `.csv` / `.json`: parse with structured readers and map fields through source-specific adapters.
- Scanned PDFs and image-only documents: reserve an OCR path for a later version. The first version should record them as accepted sources with `requires_ocr` status if OCR is not configured.

Canonical parsing output:

```json
{
  "source_id": "source:uploaded:2026-06-23:abc123",
  "chunk_id": "chunk:source:abc123:00042",
  "content": "柴胡桂枝干姜汤，和解少阳，兼化痰饮。",
  "content_type": "paragraph",
  "location": {
    "page": 12,
    "heading": "方剂条目",
    "paragraph": 4
  },
  "metadata": {
    "filename": "方剂资料.pdf",
    "mime_type": "application/pdf",
    "checksum": "..."
  }
}
```

The graph extractor should only consume canonical chunks, never raw files directly. This keeps Neo4j import, RAGFlow indexing, and evidence display stable even as parser implementations evolve.

### API Shape

The first version should expose:

- `POST /api/query`: submit a natural-language question and receive answer, graph, evidence, and metadata.
- `GET /api/graph/node/{node_id}`: fetch node details.
- `POST /api/graph/expand`: expand a selected node.
- `GET /api/assets/summary`: fetch data asset counts.
- `GET /api/health`: check frontend-facing service status.
- `POST /api/ingestion/sources`: register or upload a document source.
- `POST /api/ingestion/jobs`: start parsing and extraction for selected sources.
- `GET /api/ingestion/jobs/{job_id}`: inspect ingestion status and generated artifacts.

## Data Design

### Canonical Ingestion Artifacts

The ingestion service produces stable intermediate artifacts before anything is imported into Neo4j or indexed for retrieval:

- `SourceManifest`: registered source identity, file path, checksum, type, version, and parser status.
- `DocumentChunk`: normalized text or table chunk with location metadata.
- `EntityCandidate`: extracted entity with type, name, aliases, source references, and confidence metadata.
- `RelationCandidate`: extracted relationship with source entity, relation type, target entity, and evidence references.
- `Evidence`: source snippet tied to an entity, relation, or answer.
- `ImportBatch`: a versioned batch of validated entities and relations published into Neo4j and retrieval indexes.

These artifacts should be stored under `data/artifacts/` in the first version and can later move to a database-backed registry.

### Local Sources

Use the existing local data under `/Users/jinzhangzheng/中医/中医文献资料`:

- `TCM-DB-master/data/中药库/zcy.csv`
- `TCM-DB-master/data/中药方剂库/zyfj.csv`
- `Knowlegde_Graph_TCM-main` herb and formula graph files
- `TCM_KG-main` triple sample files
- `TCM-Note-master` Markdown notes, including herbs, acupuncture points, Jingyue Quanshu, and Huangdi Neijing Suwen
- `TCM-Ancient-Books-master` classical text files, converted from GB18030 as needed
- `ChatMed_TCM-v0.2.json` as later evaluation/reference material, not as a primary fact source in the first version

### Graph Model

Initial node labels:

- `Symptom`
- `Syndrome`
- `Treatment`
- `Formula`
- `Prescription`
- `Herb`
- `Dosage`
- `Function`
- `Indication`
- `Channel`
- `Property`
- `Flavor`
- `TextSource`
- `Evidence`
- `ExternalSource`

Initial relationship types:

- `TREATS`
- `MANIFESTS_AS`
- `RECOMMENDS_TREATMENT`
- `RECOMMENDS_FORMULA`
- `COMPOSED_OF`
- `HAS_DOSAGE`
- `HAS_FUNCTION`
- `HAS_INDICATION`
- `HAS_ALIAS`
- `HAS_PROPERTY`
- `HAS_FLAVOR`
- `ENTERS_CHANNEL`
- `SOURCED_FROM`
- `SUPPORTED_BY`
- `RELATED_TO`

### Data Cleaning Rules

- Deduplicate mirrored directories such as `TCM-Note-master` and `TCM-Note-master 2`.
- Decode CSV and classical text files with GB18030 where needed.
- Normalize relationship names to a fixed English enum for Neo4j while displaying Chinese labels in the frontend.
- Preserve original source metadata for every imported relation when available.
- Do not treat ChatMed generated answers as authoritative facts in the first version.

## Retrieval And Answer Flow

For each submitted question:

1. Parse intent and candidate entities with the model API.
2. Match candidate entities against known graph aliases.
3. Query Neo4j for relevant paths based on intent.
4. Retrieve evidence snippets from local structured data and documents.
5. Optionally check authoritative external sources from a whitelist.
6. Synthesize an answer using only the assembled graph, evidence, and source context plus model language ability.
7. Return a structured response containing:
   - answer summary
   - parsed entities and intent
   - graph nodes and edges
   - highlighted paths
   - evidence cards
   - source metadata

## External Source Policy

External sources are supplemental and should not displace local graph evidence. The first version should design for a whitelist strategy:

- Official medicine, health, pharmacopeia, or regulatory sources.
- University or research institution sources.
- PubMed and other paper-index sources where accessible.
- Authoritative medical encyclopedia or herb databases.

If external retrieval is not configured, the system should still work from local graph and local documents.

## Deployment

Use one project directory and one Docker Compose environment:

```text
/Users/jinzhangzheng/中医/tcm-kg-platform
  frontend/
  backend/
  data/
  scripts/
  docker/
  docs/
  compose.yml
  .env.example
```

Expected services:

- `tcm-web`: React frontend
- `tcm-api`: FastAPI backend
- `tcm-neo4j`: Neo4j Community
- `ragflow`: RAGFlow service
- RAGFlow dependencies such as MySQL, Redis/Valkey, MinIO, and Elasticsearch or Infinity

Expected local URLs:

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- Neo4j Browser: `http://localhost:7474`
- RAGFlow: `http://localhost:8088`

Configuration must use `.env` values, including:

```env
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=
RAGFLOW_BASE_URL=
RAGFLOW_API_KEY=
```

## First-Version Scope

The first implementation should produce a runnable demo skeleton with:

- A polished frontend workbench matching the confirmed visual direction.
- FastAPI query endpoints with real request/response contracts.
- Neo4j service in Docker Compose.
- Initial graph import path for selected local data.
- Real model API integration through `.env`.
- Structured answer, graph, and evidence response format.
- RAGFlow included in the unified environment or prepared as part of the same Compose setup, depending on local resource limits.

The first version may use a small cleaned subset of local data, but the question flow must be real and must not rely on prewritten answer templates.

## Non-Goals For First Version

- Full clinical decision support.
- Full ingestion of all 700+ classical texts.
- Expert annotation workflow.
- User account and permission management.
- Manual graph editor.
- Complete medical safety review.
- Fixed fake demos or prewritten answer templates.

## Success Criteria

- A leader can enter a new Chinese medicine question in the UI.
- The system returns a confident answer, graph nodes/edges, highlighted relationships, and evidence cards.
- The graph is the main visual focus.
- The visual style reads as a high-quality Chinese medicine knowledge platform.
- The project runs from a single code directory with Docker Compose.
- Model, Neo4j, and RAG settings are configurable without code changes.
