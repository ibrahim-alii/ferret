# Module 4: FastAPI + SSE — Worktree Agent Instructions

You are implementing **Module 4 only**. Read `PLAN.md` for full scope.

## Key Rules
- **TDD**: write `tests/test_api.py` first, then implement
- Sole writer of `sessions`, `messages`, `cited_papers` SQLite tables
- Endpoints:
  - `POST /papers` -> queues `ingest_paper` as background task, returns immediately
  - `GET /papers/{arxiv_id}` -> polling target
  - `POST /sessions` -> `{session_id}`
  - `POST /sessions/{session_id}/messages` -> SSE stream
  - `GET /sessions/{session_id}/messages` -> history
- SSE over POST: clients use `fetch()` + `ReadableStream`, NOT native `EventSource`
- Stub modules 1+3 in unit tests
- `pip install -e ".[dev]"` installs deps; `pytest tests/` runs unit tests

## Skills
- Run `/ecc:tdd-workflow` before writing any implementation code
- Run `/ecc:fastapi-review` after implementing each router/endpoint
- Run `/ecc:security-review` after implementing auth and session handling

## Files to Create
- `tests/test_api.py`
- `backend/db/models.py` — SQLAlchemy models for sessions/messages/cited_papers
- `backend/db/session.py` — async engine + session factory
- `backend/api/app.py` — FastAPI app with lifespan, CORS
- `backend/api/routers/papers.py`
- `backend/api/routers/sessions.py`
- `backend/api/schemas.py` — Pydantic request/response models
- `backend/ingestion/ingest.py` — stub
- `backend/graph/entrypoint.py` — stub
