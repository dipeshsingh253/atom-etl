# Atom — Autonomous Knowledge System for PDF Documents

A RAG (Retrieval-Augmented Generation) system that transforms PDF documents into a structured knowledge base. It extracts text, tables, and visuals from PDFs, stores them in PostgreSQL and Qdrant, and answers natural-language questions using a LangGraph agent with citations.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  POST /ingest│────▶│ Dramatiq     │────▶│ ETL Pipeline    │
│  (FastAPI)   │     │ Worker       │     │                 │
└─────────────┘     └──────────────┘     │ Text → Chunks   │
                                          │ Tables → Rows   │
                                          │ Images → Vision  │
                                          └────┬───────┬────┘
                                               │       │
                                          ┌────▼──┐ ┌──▼──────┐
                                          │Qdrant │ │PostgreSQL│
                                          │(vectors)│(structured)│
                                          └────┬──┘ └──┬──────┘
                                               │       │
┌─────────────┐     ┌──────────────┐     ┌────▼───────▼────┐
│  POST /query │────▶│ LangGraph    │────▶│ Tools           │
│  (FastAPI)   │     │ Agent        │     │ · vector search │
└─────────────┘     └──────────────┘     │ · SQL queries   │
                                          │ · math calcs    │
                                          └─────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Pydantic |
| Database | PostgreSQL (asyncpg) + SQLAlchemy |
| Vector Store | Qdrant |
| AI/LLM | OpenAI GPT-4o, text-embedding-3-small |
| Agent Framework | LangGraph + LangChain |
| PDF Processing | PyMuPDF, pdfplumber |
| Background Tasks | Dramatiq + Redis |
| Migrations | Alembic |

## Project Structure

```
src/
├── core/               # Config, logging, events, router, schemas
├── db/                 # SQLAlchemy base & async session
├── exceptions/         # Global error handlers
├── middlewares/         # CORS, request-id, security headers
├── providers/          # Abstract AI provider layer
│   ├── base.py         # ABCs: BaseLLMProvider, BaseEmbeddingProvider, BaseVisionProvider
│   ├── openai_provider.py  # OpenAI concrete implementations
│   └── factory.py      # Provider factory with caching
├── vectorstore/
│   └── qdrant.py       # Qdrant client wrapper (store/search/delete)
├── modules/
│   ├── documents/      # Document model, ingest endpoint, status endpoint
│   ├── etl/            # PDF processing pipeline
│   │   ├── text_extractor.py    # PyMuPDF text extraction
│   │   ├── text_chunker.py      # Token-based chunking (tiktoken)
│   │   ├── table_extractor.py   # pdfplumber table extraction
│   │   ├── visual_extractor.py  # Image/chart detection
│   │   ├── visual_analyzer.py   # GPT-4o vision analysis
│   │   └── pipeline.py          # Full ETL orchestrator
│   ├── agent/          # LangGraph agent
│   │   ├── tools/      # retrieval, sql, math tools
│   │   ├── prompts.py  # System prompt & query classification
│   │   └── graph.py    # StateGraph workflow
│   └── query/          # Query endpoint
└── workers/
    └── tasks/
        └── ingestion_tasks.py  # Dramatiq actor for async ETL
```

## Prerequisites

- Python 3.12+
- Docker & Docker Compose
- OpenAI API key

## Setup

### 1. Environment

```bash
cp .env-example .env
# Edit .env and set your OPENAI_API_KEY
```

### 2. Virtual Environment

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Start Infrastructure

```bash
docker compose up -d postgres redis qdrant
```

### 4. Run Migrations

```bash
alembic upgrade head
```

### 5. Start the Application

```bash
# API server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Worker (separate terminal)
dramatiq src.workers.tasks
```

Or run everything with Docker:

```bash
docker compose up --build
```

## API Endpoints

### Ingest a Document

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -F "file=@report.pdf"
```

Response:
```json
{
  "success": true,
  "data": {
    "status": "ingestion_started",
    "document_id": "abc123-..."
  }
}
```

### Check Ingestion Status

```bash
curl http://localhost:8000/api/v1/documents/{document_id}/status
```

### Query the Knowledge Base

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the revenue growth in 2023?"}'
```

Optional: scope to a specific document:
```json
{
  "query": "What are the key findings?",
  "document_id": "abc123-..."
}
```

Response:
```json
{
  "success": true,
  "data": {
    "answer": "Revenue grew 15% in 2023...",
    "citations": [
      {"source": "Page 12", "content": "...relevant excerpt..."}
    ],
    "confidence": 0.92
  }
}
```

## Abstract Provider Pattern

The AI layer is decoupled from OpenAI via abstract base classes:

```python
from src.providers.factory import get_llm_provider, get_embedding_provider

llm = get_llm_provider()          # Returns BaseLLMProvider
embedder = get_embedding_provider() # Returns BaseEmbeddingProvider
```

To add a new provider (e.g., Anthropic), implement the ABCs in `src/providers/` and register it in `factory.py`.

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing
```

Tests use SQLite (via aiosqlite) as a lightweight test database and mock external services (OpenAI, Qdrant, Dramatiq workers).

## Configuration

Key environment variables (see `.env-example` for full list):

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL async connection string | `postgresql+asyncpg://...` |
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `OPENAI_MODEL` | LLM model name | `gpt-4o` |
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `REDIS_URL` | Redis URL for Dramatiq | `redis://localhost:6379/0` |

## Architecture Justification

### Why This ETL Strategy?

**Dual-store approach (PostgreSQL + Qdrant):** PDF documents contain two fundamentally different types of information — narrative text and structured data (tables, charts). Narrative text is best searched via embeddings (semantic similarity), while table data demands exact lookups (SQL). Storing extracted tables in an EAV (Entity-Attribute-Value) schema in PostgreSQL allows the agent to run precise SQL queries like "What is the total employment?" whereas vector search alone would lose numerical precision. This mirrors how a human analyst would work: scan the text for context, but look at the actual tables for numbers.

**EAV for table storage:** Tables extracted from PDFs have wildly varying schemas (different column counts, merged cells, inconsistent headers). An EAV model (`table_rows` with `row_index`, `column_name`, `value`) normalizes all tables into a single schema, making them queryable without per-table DDL.

**Token-based chunking with tiktoken:** Chunk size is critical for retrieval quality. Using tiktoken (the same tokenizer as GPT-4o) ensures chunks align with the model's context window, avoiding mid-sentence splits that hurt retrieval relevance.

### Why LangGraph?

LangGraph was chosen over alternatives (e.g., plain LangChain agents, CrewAI, AutoGen) for several reasons:

1. **Explicit control flow:** The StateGraph lets us define exactly when the agent reasons, when it calls tools, and when it stops — preventing runaway iterations while still allowing multi-step reasoning.
2. **State management:** The typed `AgentState` carries `messages`, `citations`, `execution_trace`, and `iteration_count` through the workflow, making the agent's behavior debuggable and traceable.
3. **Conditional routing:** The `_should_continue` function provides a clean decision point — continue calling tools or generate the final answer — with a hard iteration cap for safety.
4. **Production-ready:** LangGraph compiles to an async-compatible graph that integrates cleanly with FastAPI's async request handling.

### Why These Tools?

| Tool | Rationale |
|---|---|
| `retrieve_documents` | Semantic search for finding narrative context, methodology, qualitative information |
| `run_sql_query` | Exact numerical lookups from extracted tables — no hallucination risk on numbers |
| `calculate_cagr` | Deterministic math — the LLM cannot compute CAGR reliably in its head |
| `calculate_percentage` / `calculate_percentage_change` | Same principle: offload arithmetic to deterministic code |
| `calculate_arithmetic` | Safe `eval` for arbitrary arithmetic expressions |

### Why Background Ingestion (Dramatiq)?

PDF processing (text extraction, table extraction, vision analysis, embedding generation) can take 30-120 seconds for a large document. Running this synchronously in a request handler would cause timeouts. Dramatiq + Redis provides reliable async task execution with retry semantics, while the status endpoint lets clients poll for completion.

## Execution Trace & Agent Reasoning

Every `/query` response includes an `execution_trace` array that logs each step of the agent's thought process:

```json
{
  "execution_trace": [
    {"step": 0, "node": "start", "query": "What is the total number of jobs?"},
    {"step": 1, "node": "agent", "action": "tool_calls", "tools": ["run_sql_query"]},
    {"step": 1, "node": "tool_result", "tool_name": "run_sql_query", "has_data": true},
    {"step": 2, "node": "agent", "action": "tool_calls", "tools": ["run_sql_query"]},
    {"step": 2, "node": "tool_result", "tool_name": "run_sql_query", "has_data": true},
    {"step": 3, "node": "agent", "action": "final_response"},
    {"step": 4, "node": "generate_final_answer", "citations_count": 2},
    {"step": 5, "node": "complete", "total_time_seconds": 8.42, "total_iterations": 3}
  ]
}
```

This enables:
- **Debugging:** See exactly which tools were called, in what order, and what they returned
- **Evaluation:** Verify the agent used the right tools for the right data
- **Auditability:** Full provenance trail from question to answer

## Evaluation: Moments of Truth

The system is validated against three test scenarios:

### 1. Verification — "What is the total number of jobs reported?"
- **Expected:** 7,351 (Page 19)
- **Agent behavior:** Queries SQL → finds `NUMBER OF FIRMS EMPLOYMENT` table → returns exact value with page citation

### 2. Data Synthesis — "Compare Pure-Play firms in South-West vs National Average"
- **Expected:** Cork (South-West) has 7 businesses per 10K population vs Ireland's national average of 1.5
- **Agent behavior:** Discovers tables → queries `REGION` table on Page 15 → compares Cork vs Ireland rows → synthesizes comparison

### 3. Forecasting — "What CAGR is needed from 2022 baseline to 2030 target?"
- **Expected:** ~9-11% CAGR (7,351 → 15,000-17,000 over 8 years)
- **Agent behavior:** SQL query for baseline (7,351) → retrieval for 2030 target → `calculate_cagr` tool → deterministic result

## Limitations

1. **Table extraction quality:** pdfplumber struggles with complex merged cells, multi-level headers, and tables that span page breaks. Some extracted tables have garbled column names (e.g., `column_0`, `column_1` instead of meaningful headers). This is a fundamental limitation of PDF table extraction.

2. **Vision analysis coverage:** Chart/image extraction depends on detecting embedded images in the PDF. Some charts rendered as vector graphics (not raster images) may not be detected by the current PyMuPDF-based extractor.

3. **EAV schema limitations:** The Entity-Attribute-Value model trades query simplicity for schema flexibility. Complex joins across tables or aggregations spanning multiple extracted tables can be slow or require sophisticated SQL that the agent may not always generate correctly.

4. **LLM non-determinism:** Despite `temperature=0`, OpenAI models can produce slightly different tool-calling sequences on repeated runs. The same query may use 2 tool calls on one run and 4 on another. The answers converge but the paths may differ.

5. **Single-document scope:** The current system ingests documents independently. Cross-document queries (e.g., comparing data across two different PDFs) are not yet supported.

6. **No incremental re-ingestion:** Re-ingesting the same PDF creates duplicate entries. A deduplication or versioning mechanism would be needed for production use.

7. **Token budget:** Very large PDFs (100+ pages) may produce thousands of chunks. The agent's context window limits how many retrieval results it can process in a single reasoning cycle.

8. **Citation granularity:** Citations reference page numbers, not specific paragraphs or table cells. Finer-grained citations would require storing character offsets during ETL.
