"""FastAPI application factory with lifespan and CORS."""

from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env before importing modules that read config at import time (db.session
# binds the engine to SQLITE_DB_PATH on import).
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers import papers, sessions
from backend.db.session import close_db, init_db


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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",  # Vite dev server
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(papers.router)
    app.include_router(sessions.router)

    return app


# Entrypoint for `uvicorn backend.api.app:app`
app = create_app()
