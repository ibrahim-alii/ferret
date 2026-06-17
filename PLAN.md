# Module 4: FastAPI + SSE

Ref: PRD Sections 3, 4, 6, 8, 9, 10 (module 4).

## Scope

HTTP API exposing: paper ingestion trigger (Deep Dive, run as a background task with polling), chat message endpoint (both modes) with SSE streaming, and session management. Wires module 3's `astream_chat` and module 1's `ingest_paper` to HTTP. Sole owner of `sessions`, `messages`, and `cited_papers` writes per PRD Section 4; reads chat history and passes it to module 3.

## Dependencies / Contracts

- Calls module 1's `ingest_paper(arxiv_id)` as a background task.
- Calls module 3's `astream_chat(mode, paper_id, user_message, chat_history)`. Reads the session's prior `messages` and passes them as `chat_history`.
- Sole writer of `sessions`, `messages`, and `cited_papers` (PRD Section 4): creates a session row with `mode` and `paper_id` (nullable for `ask`); persists the user message before streaming; persists the assistant message on `done`; persists `cited_papers` rows from `citation` events.
- Endpoints (exact paths finalized during implementation, but contract for module 5):
  - `POST /papers` `{arxiv_id}` -> `{arxiv_id, ingestion_status}` (queues `ingest_paper` as a background task, returns immediately with current status)
  - `GET /papers/{arxiv_id}` -> `{arxiv_id, ingestion_status, title, abstract}` (polling target for the frontend)
  - `POST /sessions` `{mode, paper_id?}` -> `{session_id}`
  - `POST /sessions/{session_id}/messages` `{content}` -> SSE stream of events matching module 3's `StreamEvent` types (`token`, `interim_message`, `citation`, `done`). Note: the response is an SSE body over POST; clients consume it via `fetch()` streaming, not native `EventSource` (which is GET-only and cannot send a body).
  - `GET /sessions/{session_id}/messages` -> chat history (for reload)

## TDD Test Checklist (write first)

- [ ] `test_post_papers_queues_background_ingestion_and_returns_status` (mock module 1, assert returns immediately)
- [ ] `test_post_papers_idempotent_for_already_ingested_paper`
- [ ] `test_post_papers_invalid_arxiv_id_returns_4xx`
- [ ] `test_get_papers_status_returns_current_ingestion_status`
- [ ] `test_post_sessions_deep_dive_requires_paper_id` (missing `paper_id` with `mode=deep_dive` -> 4xx)
- [ ] `test_post_sessions_ask_does_not_require_paper_id`
- [ ] `test_post_sessions_writes_session_row_to_sqlite`
- [ ] `test_post_message_persists_user_message_before_streaming_response`
- [ ] `test_post_message_passes_prior_messages_as_chat_history_to_graph` (mock `astream_chat`, assert `chat_history` arg)
- [ ] `test_post_message_streams_sse_events` (mock module 3's `astream_chat`, assert SSE format and event types)
- [ ] `test_post_message_persists_assistant_message_on_done`
- [ ] `test_post_message_persists_cited_papers_on_citation_event`
- [ ] `test_post_message_for_unknown_session_returns_404`
- [ ] `test_get_messages_returns_history_in_order`
- [ ] `test_sse_connection_handles_client_disconnect_gracefully` (no server error/crash)
- [ ] `test_cors_or_local_dev_config_allows_frontend_origin`

Integration tests (marked `@pytest.mark.integration`, skipped by default):
- [ ] `test_full_request_cycle_deep_dive_against_real_graph`
- [ ] `test_full_request_cycle_ask_against_real_graph`

## Implementation Tasks

1. FastAPI app setup: CORS config for frontend dev origin, lifespan startup/shutdown handlers for DB/Qdrant connection setup (module 2's `ensure_collection`)
2. Pydantic request/response models for all endpoints above
3. `POST /papers`: queues module 1's `ingest_paper` as a FastAPI background task and returns immediately with the current `ingestion_status`; the frontend polls `GET /papers/{arxiv_id}` until `full`/`abstract_only`/`failed`. (Chosen over synchronous because eager full-paper ingest can exceed request timeouts, especially under Voyage's free-tier rate limit.)
4. `POST /sessions`: validates mode/paper_id combination, writes `sessions` row, returns `session_id`
5. `POST /sessions/{session_id}/messages`: persists the user message, reads prior messages and passes them as `chat_history`, calls module 3's `astream_chat`, maps `StreamEvent`s to SSE `data:` frames (event type in `event:` field), persists the assistant message on `done`, and persists `cited_papers` rows from `citation` events
6. `GET /sessions/{session_id}/messages`: returns ordered history from SQLite
7. Error handling: typed exceptions from modules 1/3 mapped to appropriate HTTP status codes

## Done Criteria

- All unit tests pass with mocked modules 1 and 3
- SSE event format documented (event names, data shape) for module 5 to consume via `fetch()` streaming
- Background ingestion + polling flow documented in code comments/PR description
- Module 4 is the only writer of `sessions`, `messages`, and `cited_papers`; module 3 receives `chat_history` and writes nothing
