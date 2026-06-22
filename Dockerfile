# Backend image: FastAPI + LangGraph served by uvicorn.
# Single process per container — no npm here (the frontend has its own image).
FROM python:3.11-slim

# - PYTHONDONTWRITEBYTECODE: no .pyc clutter in the image/volume
# - PYTHONUNBUFFERED: logs stream out immediately (so `docker logs` is live)
# - PIP_NO_CACHE_DIR: smaller image, we never reinstall inside the container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy the project metadata first. If only source changes (not deps), Docker can
# reuse the cached dependency layer below. setuptools needs the package tree to
# build, so we copy it before installing.
COPY pyproject.toml ./
COPY backend ./backend
COPY evals ./evals

# Install the package and its runtime dependencies. Wheels exist on PyPI for the
# heavy deps (pymupdf, lxml, fastembed/onnxruntime), so no apt build toolchain
# is needed on slim.
RUN pip install .

# SQLite db + extracted figure images live here; mount a volume at this path
# (see docker-compose.yml) so data survives container rebuilds.
RUN mkdir -p /app/backend/data/media

EXPOSE 8000

# `ferret init` is idempotent: creates the SQLite schema AND the Qdrant
# collection (the FastAPI lifespan only does the former). Then serve, binding
# 0.0.0.0 so the port is reachable from the host — the browser connects
# directly for SSE. BACKEND_HOST/PORT can still override via compose env.
CMD ["sh", "-c", "ferret init && uvicorn backend.api.app:app --host ${BACKEND_HOST:-0.0.0.0} --port ${BACKEND_PORT:-8000}"]
