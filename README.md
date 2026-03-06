# AtomETL

Note: This README is intentionally concise and scannable for quick review.

AtomETL is a PDF knowledge system with:
- FastAPI API for ingestion and querying
- async ETL pipeline for text/tables/visuals
- PostgreSQL for structured data
- Qdrant for vector search
- LangGraph agent for grounded answers with citations
- Langsmith for tracing

## Tech Stack

- API: FastAPI, Pydantic
- Agent: LangGraph, LangChain
- LLM/Embeddings: OpenAI (`gpt-4.1`, `text-embedding-3-small`)
- Database: PostgreSQL + SQLAlchemy (async)
- Vector Store: Qdrant
- Background Jobs: Dramatiq + Redis
- Migrations: Alembic
- PDF/ETL: PyMuPDF, pdfplumber

## Project Structure

```text
src/
├── core/            # config, logging, events, router
├── db/              # SQLAlchemy base/session
├── modules/
│   ├── documents/   # ingest + status APIs, models
│   ├── query/       # query API + orchestration
│   ├── agent/       # LangGraph flow, tools, citations
│   └── etl/         # extraction/chunking/enrichment pipeline
├── vectorstore/     # Qdrant integration
├── providers/       # AI provider abstraction/implementations
└── workers/         # Dramatiq broker + ingestion tasks

tests/               # module-level tests
migrations/          # Alembic migrations
```

## Python Version

Use **Python 3.12** (same as Docker image).

## API Endpoints

- `GET /` — root metadata
- `GET /health` — health check
- `POST /api/v1/documents/ingest` — upload a PDF and start ingestion
- `GET /api/v1/documents/{document_id}/status` — ingestion status
- `POST /api/v1/query` — ask questions over ingested data

## Environment Variables (`.env` only)

All runtime/config values must be declared in `.env`.
Use this file for Docker Compose loading and do not rely on any other env file.
For the full list of available variables, refer to [`.env-example`](.env-example).

Required variables:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=atom

DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/atom

REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=document_chunks

OPENAI_API_KEY=your-openai-api-key

ENVIRONMENT=development
DEBUG=false
```


## Local Run (Docker, single flow)

1. Create `.env` in the project root.
2. Start everything:

```bash
docker compose up --build -d
```

3. Check service status:

```bash
docker compose ps
```

4. Verify API:

```bash
curl http://localhost:8000/health
```

Swagger UI (interactive testing):

```text
http://localhost:8000/docs
```

5. Stop services:

```bash
docker compose down
```

Notes:
- Database migrations run automatically in the `app` container startup command.
- The worker service starts with the same `.env` file.
- `/docs` is available when `ENVIRONMENT=development`.

## Ingestion & Query Approach

- Ingestion is asynchronous: `POST /api/v1/documents/ingest` stores the file, creates a document record, and dispatches a Dramatiq task.
- The ETL pipeline extracts text, tables, and visuals from PDF pages, then enriches/store results for retrieval.
- Storage is split by access pattern: PostgreSQL for structured/tabular data and metadata, Qdrant for semantic vector search over content chunks.
- Querying uses an agent workflow: retrieve relevant context (vector + SQL paths), run deterministic tools when needed (for calculations), and return a grounded answer with citations.
- This hybrid approach is intentional: semantic retrieval for narrative understanding, SQL for precise numbers, and tool-assisted computation for reliable math.

## Database Tables (PostgreSQL)

- `documents` — uploaded file record and ingestion lifecycle status.
- `document_tables` — metadata for each extracted table (name, page, description).
- `table_rows` — normalized table cells in EAV form (`row_index`, `column_name`, `value`).
- `document_visuals` — metadata for extracted charts/visuals (type, title, page).
- `visual_data` — structured datapoints extracted from visuals.

All tables inherit common base columns: `id`, `created_at`, `updated_at`.
