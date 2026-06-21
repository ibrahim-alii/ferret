# Deploying Ferret to Fly.io

Two apps, deployed independently from the repo root:

| App          | Dockerfile             | Fly config           | Public URL                     |
| ------------ | ---------------------- | -------------------- | ------------------------------ |
| `ferret-api` | `backend/Dockerfile`   | `backend/fly.toml`   | `https://ferret-api.fly.dev`   |
| `ferret-web` | `frontend/Dockerfile`  | `frontend/fly.toml`  | `https://ferret-web.fly.dev`   |

> Fly app names are **globally unique**. If `ferret-api` / `ferret-web` are taken,
> pick your own and update: `app` in both `fly.toml`s, `FRONTEND_ORIGIN` (backend),
> and `BACKEND_URL` (frontend).

## Prerequisites

```bash
# Install flyctl and log in
fly auth login
```

## 1. Backend (`ferret-api`)

```bash
# From the repo root.
fly apps create ferret-api

# Persistent disk for SQLite (papers, chunks, sessions, messages).
fly volumes create ferret_data --region iad --size 1 --config backend/fly.toml

# Secrets — never put these in fly.toml (they'd be committed). Only the providers
# you actually use are required (see backend choices below).
fly secrets set --config backend/fly.toml \
  GROQ_API_KEY=...        \
  OPENAI_API_KEY=...      \
  JINA_API_KEY=...        \
  QDRANT_URL=...          \
  QDRANT_API_KEY=...

# Deploy (build context = repo root).
fly deploy --config backend/fly.toml
```

Non-secret config (model names, thresholds, `SQLITE_DB_PATH`, `FRONTEND_ORIGIN`,
`ARXIV_USER_AGENT`) lives in `[env]` in `backend/fly.toml`. Edit there as needed —
**replace the placeholder `ARXIV_USER_AGENT` contact email** before launch.

Embeddings provider toggle: the default uses OpenAI. To use Gemini instead, set
`USE_LOCAL_EMBEDDINGS=true` and add `GEMINI_API_KEY` as a secret.

The container ensures the SQLite schema and the Qdrant collection on boot
(idempotent), then starts uvicorn — no separate `ferret init` step needed.

## 2. Frontend (`ferret-web`)

```bash
fly apps create ferret-web
# BACKEND_URL is set in frontend/fly.toml [env]; confirm it points at your backend.
fly deploy --config frontend/fly.toml
```

## 3. Wire the two together

- Backend `FRONTEND_ORIGIN` (in `backend/fly.toml`) must equal the frontend's URL.
- Frontend `BACKEND_URL` (in `frontend/fly.toml`) must equal the backend's URL.

Both are baked at deploy time, so after changing either, redeploy that app.

## Local Docker smoke test

```bash
# From the repo root.
docker build -f backend/Dockerfile  -t ferret-api .
docker build -f frontend/Dockerfile -t ferret-web .

docker run --rm -p 8000:8000 --env-file .env ferret-api
docker run --rm -p 3000:3000 -e BACKEND_URL=http://localhost:8000 ferret-web
```

## Security model & known limitations

This is an anonymous, no-accounts app. Isolation is **obscurity, not auth**:

- Each browser generates a UUID (`localStorage['ferret_client_id']`) sent as the
  `X-Client-ID` header. `GET /sessions` is scoped to that id, so users don't see
  each other's history in the sidebar.
- A session id is an unguessable UUIDv4. Anyone who *obtains* one (or sends another
  client's `X-Client-ID`) can read that session. Acceptable for a demo; for true
  isolation you'd need signed, server-issued client tokens (e.g. an HTTP-only cookie).
- Sessions created without an `X-Client-ID` header (e.g. raw `curl`) store
  `client_id = NULL` and are reachable by anyone with the session UUID. The
  frontend always sends the header, so normal users never hit this.

Cost controls in place:

- Rate limits (slowapi, keyed on the real client IP via `Fly-Client-IP`):
  `POST /papers` 10/min, `POST /sessions/{id}/messages` 30/min.
- `arxiv_id` is regex-validated before any external fetch (no SSRF / path traversal).
- CORS is restricted to `FRONTEND_ORIGIN`; `"*"` is rejected at startup.

No API keys are exposed to the browser — all provider calls happen in the backend.
