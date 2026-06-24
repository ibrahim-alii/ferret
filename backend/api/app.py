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
from starlette.responses import JSONResponse

from backend.api.limiter import limiter
from backend.api.routers import papers, sessions
from backend.db.session import close_db, init_db

# Comma-separated list of allowed origins for CORS. Falls back to local dev
# servers when FRONTEND_ORIGIN is unset.
_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    # 127.0.0.1 is a distinct origin from localhost; allow both so the app works
    # regardless of which address the browser is pointed at in local dev.
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# Reject request bodies larger than this before they are buffered/parsed. The
# Pydantic per-field caps only apply after the full body is read, so without this
# a client could stream an arbitrarily large payload.
_DEFAULT_MAX_BODY_BYTES = 1024 * 1024  # 1 MiB


class BodySizeLimitMiddleware:
    """Pure-ASGI guard that 413s oversized requests by their Content-Length.

    Implemented at the ASGI layer (not BaseHTTPMiddleware) so it never wraps or
    buffers the response — the SSE StreamingResponse must stream untouched.
    """

    def __init__(self, app, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            for name, value in scope.get("headers", []):
                if name == b"content-length":
                    try:
                        length = int(value)
                    except ValueError:
                        break
                    if length > self.max_body_bytes:
                        response = JSONResponse(
                            {"detail": "Request body too large"}, status_code=413
                        )
                        await response(scope, receive, send)
                        return
                    break
        await self.app(scope, receive, send)


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

    max_body_bytes = int(os.environ.get("MAX_BODY_BYTES", str(_DEFAULT_MAX_BODY_BYTES)))
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=max_body_bytes)

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
