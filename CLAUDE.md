# Module 1: Ingestion — Worktree Agent Instructions

You are implementing **Module 1 only**. Read `PLAN.md` for full scope.

## Key Rules
- **TDD**: write `tests/test_ingestion.py` first, then implement
- Owns: `backend/db/` (SQLAlchemy async models for `papers`+`chunks`), `backend/ingestion/`
- Stub module 2 at `backend/vectorstore/interface.py` — tests mock it
- Never embed parent chunks; only child chunks go to Voyage + Qdrant
- `ingest_paper(arxiv_id) -> PaperRecord` must be idempotent
- Use `asyncio.gather` for concurrent metadata + PDF download
- Wrap Voyage calls in `asyncio.Semaphore(VOYAGE_MAX_CONCURRENCY)` + retry/backoff
- `pip install -e ".[dev]"` installs deps; `pytest tests/` runs unit tests
- Mock all external APIs (arxiv, Voyage, Qdrant) in unit tests
- Integration tests: mark `@pytest.mark.integration`, skip by default

## Skills
- Run `/ecc:tdd-workflow` before writing any implementation code
- Run `/ecc:python-review` after implementing each file
- Run `/ecc:security-review` after implementing `embedder.py` (handles external API calls)

## Files to Create
- `tests/test_ingestion.py`
- `backend/db/models.py` — SQLAlchemy async models
- `backend/db/session.py` — async engine + session factory
- `backend/vectorstore/interface.py` — stub `upsert_chunks`
- `backend/ingestion/arxiv_client.py`
- `backend/ingestion/parser.py`
- `backend/ingestion/filter.py`
- `backend/ingestion/chunker.py`
- `backend/ingestion/embedder.py`
- `backend/ingestion/ingest.py` — `ingest_paper` entrypoint
