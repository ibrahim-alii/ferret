"""ferret CLI — thin orchestration layer over modules 1-4 and 6."""
from __future__ import annotations

import asyncio
import enum
import os
import subprocess
import sys
import threading
from typing import Optional

import typer
import uvicorn
from dotenv import load_dotenv

# Load .env before importing modules that read config at import time.
load_dotenv()

from backend.db.session import init_db
from backend.vectorstore.store import ensure_collection
from backend.ingestion.ingest import ingest_paper
from backend.graph.entrypoint import astream_chat
from evals.run_all import run_all

app = typer.Typer(help="Ferret — research paper chat assistant.")


class ChatMode(str, enum.Enum):
    deep_dive = "deep_dive"
    ask = "ask"


# ---------------------------------------------------------------------------
# ferret init
# ---------------------------------------------------------------------------

@app.command()
def init() -> None:
    """Create SQLite schema and ensure the Qdrant collection exists."""
    asyncio.run(init_db())
    asyncio.run(ensure_collection())
    typer.echo("Initialized: schema and vector collection are ready.")


# ---------------------------------------------------------------------------
# ferret serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Bind host"),
    port: Optional[int] = typer.Option(None, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Run the FastAPI backend via uvicorn."""
    _host = host or os.environ.get("BACKEND_HOST", "127.0.0.1")
    _port = port or int(os.environ.get("BACKEND_PORT", "8000"))
    uvicorn.run("backend.api.app:app", host=_host, port=_port, reload=reload)


# ---------------------------------------------------------------------------
# ferret web
# ---------------------------------------------------------------------------

@app.command()
def web(
    port: Optional[int] = typer.Option(None, help="Frontend port"),
) -> None:
    """Run the Express frontend as a subprocess."""
    _port = port or int(os.environ.get("FRONTEND_PORT", "3000"))
    env = {**os.environ, "PORT": str(_port)}
    proc = subprocess.Popen(["npm", "start"], env=env)
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        returncode = 130
    raise typer.Exit(returncode)


# ---------------------------------------------------------------------------
# ferret dev
# ---------------------------------------------------------------------------

def _pipe_output(proc: subprocess.Popen, prefix: str) -> None:
    """Forward lines from a subprocess stdout to our stdout with a prefix."""
    if proc.stdout is None:
        return
    for line in proc.stdout:
        sys.stdout.write(f"[{prefix}] {line}")
        sys.stdout.flush()


@app.command()
def dev() -> None:
    """Run backend and frontend together, streaming both logs; Ctrl-C shuts both down."""
    backend_host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    backend_port = os.environ.get("BACKEND_PORT", "8000")
    frontend_port = os.environ.get("FRONTEND_PORT", "3000")

    backend_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "backend.api.app:app",
            "--host", backend_host, "--port", backend_port,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    frontend_proc = subprocess.Popen(
        ["npm", "start"],
        env={**os.environ, "PORT": frontend_port},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    threading.Thread(target=_pipe_output, args=(backend_proc, "backend"), daemon=True).start()
    threading.Thread(target=_pipe_output, args=(frontend_proc, "frontend"), daemon=True).start()

    try:
        returncode = backend_proc.wait()
    except KeyboardInterrupt:
        returncode = 130
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
    raise typer.Exit(returncode)


# ---------------------------------------------------------------------------
# ferret ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(arxiv_id: str = typer.Argument(..., help="arXiv paper ID")) -> None:
    """Ingest a paper from arXiv into the knowledge base."""
    result = asyncio.run(ingest_paper(arxiv_id))
    typer.echo(f"Status: {result.ingestion_status}")
    typer.echo(f"Title:  {result.title}")


# ---------------------------------------------------------------------------
# ferret chat
# ---------------------------------------------------------------------------

async def _run_chat_turn(
    mode: str,
    paper_id: Optional[str],
    user_message: str,
    chat_history: list,
) -> str:
    """Stream one chat turn; return the full assistant response."""
    full_response = ""
    async for event in astream_chat(mode, paper_id, user_message, chat_history):
        etype = event.get("type")
        content = event.get("content", "")
        if etype == "token":
            typer.echo(content, nl=False)
            full_response += content
        elif etype == "citation":
            typer.echo(f"\n[citation] {content}")
        elif etype == "interim_message":
            typer.echo(f"\n[…] {content}")
    typer.echo("")
    return full_response


@app.command()
def chat(
    mode: ChatMode = typer.Option(..., help="Chat mode"),
    paper_id: Optional[str] = typer.Option(None, help="Paper ID (required for deep_dive)"),
) -> None:
    """Terminal chat against the graph. History is held in memory only."""
    if mode == ChatMode.deep_dive and paper_id is None:
        typer.echo("Error: --paper-id is required for deep_dive mode.", err=True)
        raise typer.Exit(1)
    if mode == ChatMode.ask and paper_id is not None:
        typer.echo("Error: --paper-id must not be set for ask mode.", err=True)
        raise typer.Exit(1)

    chat_history: list[dict] = []

    while True:
        try:
            user_message = input("> ").strip()
        except EOFError:
            break
        if not user_message:
            continue

        response = asyncio.run(
            _run_chat_turn(mode.value, paper_id, user_message, chat_history)
        )
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# ferret eval
# ---------------------------------------------------------------------------

@app.command(name="eval")
def run_eval(
    paper_id: str = typer.Option(..., help="Paper ID to evaluate against"),
    mode: Optional[str] = typer.Option(None, help="Restrict to a single mode"),
) -> None:
    """Run the eval suite and print a summary table."""
    result = asyncio.run(run_all(paper_id=paper_id, mode=mode))
    typer.echo(f"Summary:     {result.get('summary', '')}")
    typer.echo(f"Report path: {result.get('report_path', '')}")
