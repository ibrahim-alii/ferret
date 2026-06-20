# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project conventions for Claude Code working in this repo. See `PRD.md` for full architecture and design decisions.

## Project Summary

ArXiv RAG chatbot with two modes: Deep Dive (single-paper, strict grounding) and Ask (cross-corpus, grows over time). LangGraph-based CRAG flow, Qdrant Cloud hybrid retrieval, OpenAI embeddings + Voyage AI rerank, Groq LLMs, FastAPI + SSE backend, Express + vanilla JS frontend, SQLite for state.

## Repo Structure

```
backend/      FastAPI app, LangGraph graphs, ingestion, vectorstore, db, CLI (ferret)
frontend/     Express server, static HTML/CSS/JS
evals/        Standalone RAG evaluation suite (module 6)
.claude/      ECC plugin config, skills, worktrees
PRD.md        Source of truth for architecture/design
CLAUDE.md     This file
.env.example  Required environment variables
```

## Commands

```bash
pip install -e .                          # exposes `ferret` CLI
ferret init                               # create SQLite schema + Qdrant collection (idempotent)
ferret serve                              # FastAPI backend
ferret web                                # Express frontend
ferret dev                                # both together, Ctrl-C tears down both
ferret ingest <arxiv_id>
ferret chat --mode ask
ferret chat --mode deep_dive --paper-id <id>
ferret eval --paper-id <id>

pytest                                    # unit tests (external APIs mocked)
pytest -m integration                     # real API calls (skipped by default)
pytest -m eval                            # full RAG eval suite (skipped by default)
pytest path/to/test_foo.py::test_name     # single test
```

## Environment

Model routing is via `GRADING_MODEL` and `GENERATION_MODEL` env vars, not hardcoded model names in code, since Groq's free-tier rate limits differ sharply by model.

## Code Conventions

- Python: async/await throughout the backend (FastAPI, ingestion, graph nodes). Use `asyncio.gather` for concurrent I/O (concurrent arxiv fetches, the metadata+download pair, the parallel arxiv searches in Ask's corrective branch) per PRD Sections 5-6.
- Bounded concurrency for rate-limited providers: wrap OpenAI embedding (`OPENAI_MAX_CONCURRENCY`), Voyage rerank (`VOYAGE_MAX_CONCURRENCY`), and Groq calls in an `asyncio.Semaphore` plus retry-with-backoff. Free tiers are RPM-limited, so prefer a few large embedding batches over many small parallel ones. Async is for not blocking the event loop and overlapping independent calls, not for hammering a rate-limited API.
- Retrieval uses Qdrant's server-side RRF fusion (Universal Query API `prefetch`), not a client-side fusion function. Sparse (BM25) generation lives only in module 2, for both upsert and query.
- Only child chunks are embedded and stored in Qdrant. Parent sections live in SQLite and are expanded into the generation context (small-to-big).
- No local models, no GPU code paths. All embedding/reranking/generation calls go to hosted APIs (OpenAI embeddings, Voyage rerank, Groq). Embeddings have an OpenAI/Gemini toggle (`USE_LOCAL_EMBEDDINGS`) for local/dev testing; both providers share one dim (`OPENAI_EMBED_DIM`) and are centralized in `backend/ingestion/embedder.py` (`embed_chunks` for documents, `embed_query` for queries).
- SQLAlchemy (async, over aiosqlite) for SQLite access, to keep a Postgres migration path open.
- Table write ownership is a hard contract: `papers`/`chunks` -> module 1; `sessions`/`messages`/`cited_papers` -> module 4; module 3 reads chunk text read-only and writes nothing. Never add a second writer to a table in another worktree.
- Type hints required on all function signatures.
- Use Context7 MCP for up-to-date library docs (LangGraph, Qdrant client, FastAPI, OpenAI SDK, Voyage SDK) instead of relying on memorized API shapes.

## Testing

- pytest for all backend modules. Unit tests mock external APIs (Groq, OpenAI, Voyage, Qdrant, arxiv); integration tests (marked `@pytest.mark.integration`) hit real endpoints and are skipped by default; eval tests (marked `@pytest.mark.eval`) run the full RAG evaluation suite against a real ingested paper and are also skipped by default.
- TDD workflow expected: write/adjust tests alongside implementation, not after.
- Eval suite: `pytest -m eval` or `python evals/run_all.py --paper-id <id>` (or `ferret eval --paper-id <id>`). Produces retrieval (ranx), grading accuracy, and generation (RAGAS) reports under `evals/reports/`.

---

## Behavioral Guidelines

These bias toward caution over speed. Use judgment on trivial tasks.

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

Every changed line should trace directly to the request.

### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
