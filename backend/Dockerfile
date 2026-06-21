# Ferret FastAPI backend. Build context = repo root (needs pyproject.toml + backend/ + evals/).
#   docker build -f backend/Dockerfile -t ferret-api .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies from the project metadata. Copying source before install
# because setuptools builds the package (no editable install in the image).
COPY pyproject.toml ./
COPY backend ./backend
COPY evals ./evals
RUN pip install .

EXPOSE 8000

# On boot: ensure the SQLite schema (on the mounted volume) and the Qdrant
# collection exist, then start uvicorn. Both init calls are idempotent. We call
# the underlying functions directly rather than `ferret init` to avoid importing
# the CLI's eval-suite dependencies (ragas/ranx), which aren't installed here.
# --proxy-headers + --forwarded-allow-ips lets uvicorn trust Fly's proxy so
# request.client.host reflects the real client (belt-and-suspenders with the
# Fly-Client-IP-aware limiter key func).
CMD python -c "import asyncio; from backend.db.session import init_db; from backend.vectorstore.store import ensure_collection; asyncio.run(init_db()); asyncio.run(ensure_collection())" \
    && exec uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'
