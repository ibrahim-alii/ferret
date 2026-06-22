"""FastAPI application factory with lifespan and CORS."""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env before importing modules that read config at import time (db.session
# binds the engine to SQLITE_DB_PATH on import).
load_dotenv()

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.api.limiter import limiter
from backend.api.routers import papers, sessions
from backend.db.session import close_db, init_db

# Comma-separated list of allowed origins for CORS. Falls back to local dev
# servers when FRONTEND_ORIGIN is unset.
_DEFAULT_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]


def _cors_origins() -> list[str]:
    raw = os.environ.get("FRONTEND_ORIGIN")
    if not raw:
        return _DEFAULT_ORIGINS
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    # allow_origins=["*"] with allow_credentials=True is rejected by browsers and
    # by Starlette; guard against the misconfiguration so it fails loudly here
    # rather than silently breaking every request in production.
    if "*" in origins:
        raise ValueError(
            "FRONTEND_ORIGIN must list explicit origins, not '*', "
            "because credentials are allowed."
        )
    return origins


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Ferret API",
        version="0.1.0",
        description="Research paper ingestion and SSE chat API.",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(papers.router)
    app.include_router(sessions.router)

    # Serve figure images extracted from PDF-only papers. Referenced as
    # /media/<arxiv_id>/<file> in chunk media_url and baked into answer markdown.
    media_dir = Path(os.environ.get("MEDIA_DIR", "./backend/data/media"))
    media_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(media_dir)), name="media")

    return app


# Entrypoint for `uvicorn backend.api.app:app`
app = create_app()
