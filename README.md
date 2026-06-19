# Ferret

> ArXiv RAG chatbot with two chat modes — **Deep Dive** (single paper, strict grounding) and **Ask** (cross-corpus, grows over time) — built on Corrective RAG, hybrid retrieval, and streaming SSE.

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What is Ferret?

**Deep Dive** — enter an arXiv paper ID and chat strictly about that paper. Answers are grounded in the ingested full text; when context is insufficient, the system surfaces related papers to explore instead of hallucinating.

**Ask** — chat freely across every paper ingested so far. When retrieval comes up short, the system searches arXiv, ingests the most relevant candidates on the fly, and retries before answering.

Both modes share the same CRAG backbone: hybrid vector retrieval (dense + sparse with server-side RRF) → rerank → small-to-big context expansion → sufficiency grading → conditional generation, all streamed over SSE.

---

## Architecture

See [`docs/architecture.excalidraw`](docs/architecture.excalidraw) for the full interactive diagram (open in [excalidraw.com](https://excalidraw.com) or the VS Code extension).

```
Browser
  └─ Express frontend (vanilla JS, SSE consumer)
       └─ FastAPI backend
            ├─ Ingestion Graph (LangGraph)
            │    └─ arXiv API → HTML/PDF parser → filter → chunk
            │         → Voyage embed → Qdrant upsert + SQLite write
            └─ CRAG Chat Graph (LangGraph)
                 Retrieve (hybrid RRF)
                   → Rerank (Voyage)
                     → Expand small-to-big (SQLite)
                       → Grade (2-threshold + LLM judge)
                         ├─ sufficient  → Generate (Groq) → SSE stream
                         └─ insufficient
                              ├─ Deep Dive: cite related papers (no ingest)
                              └─ Ask: search arXiv → ingest → re-retrieve → Generate
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Backend runtime |
| Node.js 18+ | Frontend server |
| [Groq](https://console.groq.com) API key | LLM generation + grading (free tier) |
| [Voyage AI](https://www.voyageai.com) API key | Embeddings + reranking (free tier) |
| [Qdrant Cloud](https://cloud.qdrant.io) cluster | Vector store (free tier, 1 GB RAM) |

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd ferret

# 2. Copy env template and fill in your API keys
cp .env.example .env

# 3. Install Python package (exposes the `ferret` CLI)
pip install -e ".[dev]"

# 4. Install frontend dependencies
cd frontend && npm install && cd ..

# 5. Initialise SQLite schema + Qdrant collection (idempotent)
ferret init

# 6. Start backend + frontend together
ferret dev
```

The frontend is at `http://localhost:3000` and the API at `http://localhost:8000`.

---

## CLI Reference

All commands are exposed via the `ferret` CLI (installed by `pip install -e .`).

| Command | Description |
|---|---|
| `ferret init` | Create SQLite schema and ensure the Qdrant collection exists (safe to re-run) |
| `ferret serve` | Start the FastAPI backend (`--host`, `--port`, `--reload`) |
| `ferret web` | Start the Express frontend (`--port`) |
| `ferret dev` | Start both backend and frontend; streams combined logs; `Ctrl-C` tears both down |
| `ferret ingest <arxiv_id>` | Manually ingest a paper (e.g. `ferret ingest 2305.10601`) |
| `ferret chat --mode ask` | Terminal REPL in Ask mode |
| `ferret chat --mode deep_dive --paper-id <id>` | Terminal REPL grounded to a single paper |
| `ferret eval --paper-id <id>` | Run the full RAG evaluation suite against an ingested paper |

---

## Configuration

Copy `.env.example` to `.env`. The variables below are the ones you're most likely to tune.

### Groq (LLM)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required |
| `GRADING_MODEL` | `llama-3.1-8b-instant` | Small model for grading, query gen, candidate checks |
| `GENERATION_MODEL` | `llama-3.3-70b-versatile` | Larger model for final user-facing answers |

### Voyage AI (Embeddings + Reranking)

| Variable | Default | Description |
|---|---|---|
| `VOYAGE_API_KEY` | — | Required |
| `VOYAGE_EMBED_MODEL` | `voyage-4-lite` | Dense embedding model — don't change after first ingest |
| `VOYAGE_EMBED_DIM` | `1024` | Vector dimension |
| `VOYAGE_RERANK_MODEL` | `rerank-2.5-lite` | Reranker |
| `VOYAGE_MAX_CONCURRENCY` | `2` | Bounded concurrency for free-tier rate limits |

### Qdrant

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | — | Your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | — | Required |
| `QDRANT_COLLECTION_NAME` | `arxiv_chunks` | Collection name |
| `SPARSE_MODEL` | `Qdrant/bm25` | BM25 sparse encoder |

### Ingestion

| Variable | Default | Description |
|---|---|---|
| `ARXIV_PREFER_HTML` | `true` | Use arXiv HTML rendering (better section boundaries); falls back to PDF |
| `CHILD_CHUNK_TOKENS` | `512` | Child chunk size (retrieval unit) |
| `CHILD_CHUNK_OVERLAP` | `64` | Overlap between consecutive child chunks |
| `SECTION_DROP_LIST` | `references,bibliography,...` | Comma-separated section names to drop before chunking |

### CRAG Chat Flow

| Variable | Default | Description |
|---|---|---|
| `GRADE_HIGH_THRESHOLD` | `0.6` | Above this → sufficient (skip LLM judge) |
| `GRADE_LOW_THRESHOLD` | `0.35` | Below this → insufficient (skip LLM judge) |
| `RERANK_CANDIDATE_COUNT` | `30` | Candidates sent to reranker |
| `RERANK_TOP_K` | `5` | Kept after reranking |
| `GENERATION_MAX_PARENTS` | `5` | Max parent sections expanded into generation context |
| `CRAG_MAX_RETRIES` | `2` | Ask-mode corrective loop cap |
| `ASK_ARXIV_QUERY_COUNT` | `3` | Queries generated per corrective step |
| `ASK_INGEST_CANDIDATES` | `3` | Papers ingested per corrective step |

### Server

| Variable | Default | Description |
|---|---|---|
| `BACKEND_HOST` | `127.0.0.1` | FastAPI host |
| `BACKEND_PORT` | `8000` | FastAPI port |
| `FRONTEND_PORT` | `3000` | Express port |

---

## Testing

Tests live in `tests/` (backend) and `frontend/tests/`. External APIs are mocked in unit tests.

### Unit tests (default — no real API calls)

```bash
pytest
```

### Integration tests (real API calls — requires `.env`)

```bash
pytest -m integration
```

### RAG evaluation suite (requires an ingested paper)

```bash
ferret eval --paper-id 2305.10601
# or
pytest -m eval
# or
python evals/run_all.py --paper-id 2305.10601
```

Reports are written to `evals/reports/`. The first run saves a baseline; subsequent runs flag regressions.

### Frontend tests

```bash
cd frontend && npm test
```

### Testing strategy

- **Unit**: every backend module has a corresponding `tests/test_<module>.py`. Groq, Voyage, Qdrant, and arXiv calls are patched with `pytest-mock`. Parser, chunker, filter, and grader logic are exercised with fixture inputs.
- **Integration** (`@pytest.mark.integration`): hit real endpoints to verify the full ingestion and chat flow end-to-end. Skipped in CI by default.
- **Eval** (`@pytest.mark.eval`): 5-layer RAG quality suite run against a real ingested paper. See below.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, LangGraph, SQLAlchemy (async), aiosqlite, Typer |
| Frontend | Node.js, Express, vanilla JS (ES6), HTML5/CSS3, Vitest |
| Vector store | Qdrant Cloud — dense (Voyage) + sparse (BM25) named vectors, server-side RRF |
| Embeddings | Voyage AI — `voyage-4-lite` (1024-dim dense), `rerank-2.5-lite` |
| LLM | Groq — `llama-3.1-8b-instant` (grading), `llama-3.3-70b-versatile` (generation) |
| Document parsing | BeautifulSoup4 (HTML), PyMuPDF (PDF fallback) |
| Eval | pytest, RAGAS, ranx, Datasets |

---

## Project Structure

```
ferret/
├── backend/
│   ├── api/            # FastAPI app, routers (papers, sessions), Pydantic schemas
│   ├── cli/            # Typer CLI commands
│   ├── db/             # SQLAlchemy models + async session factory
│   ├── graph/          # LangGraph graphs, state, nodes (retrieve/rerank/grade/generate/…)
│   ├── ingestion/      # arXiv client, HTML+PDF parser, filter, chunker, embedder
│   └── vectorstore/    # Qdrant client, hybrid search, BM25 sparse encoder
├── frontend/
│   ├── public/         # index.html
│   ├── static/         # app.js, style.css
│   └── server.js       # Express app
├── evals/              # 5-layer RAG evaluation suite
├── tests/              # pytest unit + integration tests
├── docs/               # Architecture diagram
├── PRD.md              # Authoritative architecture & design doc
├── CLAUDE.md           # AI coding conventions
└── .env.example        # All environment variables with descriptions
```
