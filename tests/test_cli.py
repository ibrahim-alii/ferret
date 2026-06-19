"""Unit tests for ferret CLI (Module 7).

All module dependencies (1-4, 6) are mocked — no real services required.
"""
from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from backend.cli.main import app
from backend.ingestion.ingest import PaperRecord

runner = CliRunner()


# ---------------------------------------------------------------------------
# ferret init
# ---------------------------------------------------------------------------

class TestFerretInit:
    @patch("backend.cli.main.ensure_collection", new_callable=AsyncMock)
    @patch("backend.cli.main.init_db", new_callable=AsyncMock)
    def test_init_calls_schema_create_all_and_ensure_collection(
        self, mock_init_db, mock_ensure
    ):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        mock_init_db.assert_awaited_once()
        mock_ensure.assert_awaited_once()

    @patch("backend.cli.main.ensure_collection", new_callable=AsyncMock)
    @patch("backend.cli.main.init_db", new_callable=AsyncMock)
    def test_init_is_idempotent(self, mock_init_db, mock_ensure):
        result1 = runner.invoke(app, ["init"])
        result2 = runner.invoke(app, ["init"])
        assert result1.exit_code == 0
        assert result2.exit_code == 0
        assert mock_init_db.await_count == 2
        assert mock_ensure.await_count == 2


# ---------------------------------------------------------------------------
# ferret serve
# ---------------------------------------------------------------------------

class TestFerretServe:
    @patch("backend.cli.main.uvicorn")
    def test_serve_uses_env_host_and_port(self, mock_uvicorn, monkeypatch):
        monkeypatch.setenv("BACKEND_HOST", "0.0.0.0")
        monkeypatch.setenv("BACKEND_PORT", "9000")
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once_with(
            "backend.api.app:app",
            host="0.0.0.0",
            port=9000,
            reload=False,
        )

    @patch("backend.cli.main.uvicorn")
    def test_serve_flags_override_env_defaults(self, mock_uvicorn, monkeypatch):
        monkeypatch.setenv("BACKEND_HOST", "0.0.0.0")
        monkeypatch.setenv("BACKEND_PORT", "9000")
        result = runner.invoke(
            app,
            ["serve", "--host", "127.0.0.1", "--port", "8888", "--reload"],
        )
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once_with(
            "backend.api.app:app",
            host="127.0.0.1",
            port=8888,
            reload=True,
        )


# ---------------------------------------------------------------------------
# ferret web
# ---------------------------------------------------------------------------

class TestFerretWeb:
    @patch("backend.cli.main.subprocess")
    def test_web_spawns_frontend_subprocess_with_port(self, mock_subprocess, monkeypatch):
        monkeypatch.setenv("FRONTEND_PORT", "3000")
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_subprocess.Popen.return_value = mock_proc

        result = runner.invoke(app, ["web"])
        assert result.exit_code == 0
        mock_subprocess.Popen.assert_called_once()
        env_passed = mock_subprocess.Popen.call_args.kwargs.get("env", {})
        assert env_passed.get("PORT") == "3000"

    @patch("backend.cli.main.subprocess")
    def test_web_port_flag_overrides_env(self, mock_subprocess, monkeypatch):
        monkeypatch.setenv("FRONTEND_PORT", "3000")
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_subprocess.Popen.return_value = mock_proc

        result = runner.invoke(app, ["web", "--port", "4000"])
        assert result.exit_code == 0
        mock_subprocess.Popen.assert_called_once()
        env_passed = mock_subprocess.Popen.call_args.kwargs.get("env", {})
        assert env_passed.get("PORT") == "4000"


# ---------------------------------------------------------------------------
# ferret dev
# ---------------------------------------------------------------------------

class TestFerretDev:
    @patch("backend.cli.main.subprocess")
    def test_dev_starts_both_backend_and_frontend(self, mock_subprocess):
        mock_backend = MagicMock()
        mock_frontend = MagicMock()
        mock_backend.wait.side_effect = KeyboardInterrupt
        mock_subprocess.Popen.side_effect = [mock_backend, mock_frontend]

        runner.invoke(app, ["dev"])
        assert mock_subprocess.Popen.call_count == 2

    @patch("backend.cli.main.subprocess")
    def test_dev_shuts_down_both_on_interrupt(self, mock_subprocess):
        mock_backend = MagicMock()
        mock_frontend = MagicMock()
        mock_backend.wait.side_effect = [KeyboardInterrupt, None]
        mock_subprocess.Popen.side_effect = [mock_backend, mock_frontend]

        runner.invoke(app, ["dev"])
        mock_backend.terminate.assert_called()
        mock_frontend.terminate.assert_called()


# ---------------------------------------------------------------------------
# ferret ingest
# ---------------------------------------------------------------------------

class TestFerretIngest:
    @patch("backend.cli.main.ingest_paper", new_callable=AsyncMock)
    def test_ingest_calls_ingest_paper_with_arxiv_id(self, mock_ingest):
        mock_ingest.return_value = PaperRecord(
            arxiv_id="2401.00001", title="Test Paper", abstract="", ingestion_status="full"
        )
        result = runner.invoke(app, ["ingest", "2401.00001"])
        assert result.exit_code == 0
        mock_ingest.assert_awaited_once_with("2401.00001")

    @patch("backend.cli.main.ingest_paper", new_callable=AsyncMock)
    def test_ingest_prints_status_and_title(self, mock_ingest):
        mock_ingest.return_value = PaperRecord(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            abstract="",
            ingestion_status="full",
        )
        result = runner.invoke(app, ["ingest", "1706.03762"])
        assert result.exit_code == 0
        assert "full" in result.output
        assert "Attention Is All You Need" in result.output


# ---------------------------------------------------------------------------
# ferret chat
# ---------------------------------------------------------------------------

class TestFerretChat:
    def test_chat_deep_dive_requires_paper_id(self):
        result = runner.invoke(app, ["chat", "--mode", "deep_dive"])
        assert result.exit_code != 0

    def test_chat_ask_rejects_paper_id(self):
        result = runner.invoke(app, ["chat", "--mode", "ask", "--paper-id", "abc"])
        assert result.exit_code != 0

    @patch("backend.cli.main.astream_chat")
    def test_chat_streams_token_events_to_stdout(self, mock_astream):
        async def _gen(*_, **__) -> AsyncIterator[dict]:
            yield {"type": "token", "content": "Hello"}
            yield {"type": "token", "content": " world"}

        mock_astream.side_effect = _gen
        result = runner.invoke(app, ["chat", "--mode", "ask"], input="hi\n")
        assert "Hello" in result.output
        assert "world" in result.output

    @patch("backend.cli.main.astream_chat")
    def test_chat_renders_citation_events_distinctly(self, mock_astream):
        async def _gen(*_, **__) -> AsyncIterator[dict]:
            yield {"type": "token", "content": "answer"}
            yield {"type": "citation", "content": "Paper XYZ, p.5"}

        mock_astream.side_effect = _gen
        result = runner.invoke(app, ["chat", "--mode", "ask"], input="question\n")
        assert result.exit_code == 0
        assert "Paper XYZ" in result.output

    @patch("backend.cli.main.astream_chat")
    def test_chat_maintains_in_memory_history_across_turns(self, mock_astream):
        calls: list = []

        async def _gen(mode, paper_id, user_message, chat_history) -> AsyncIterator[dict]:
            calls.append((user_message, list(chat_history)))
            yield {"type": "token", "content": "response"}

        mock_astream.side_effect = _gen
        result = runner.invoke(
            app,
            ["chat", "--mode", "ask"],
            input="first question\nsecond question\n",
        )
        assert result.exit_code == 0
        assert len(calls) == 2
        second_history = calls[1][1]
        assert len(second_history) >= 1
        assert any("first question" in str(h) for h in second_history)


# ---------------------------------------------------------------------------
# ferret eval
# ---------------------------------------------------------------------------

class TestFerretEval:
    @patch("backend.cli.main.run_all", new_callable=AsyncMock)
    def test_eval_calls_run_all_with_paper_id_and_mode(self, mock_run_all):
        mock_run_all.return_value = {
            "summary": "passed",
            "report_path": "/tmp/report.json",
        }
        result = runner.invoke(app, ["eval", "--paper-id", "abc123", "--mode", "ask"])
        assert result.exit_code == 0
        mock_run_all.assert_awaited_once_with(paper_id="abc123", mode="ask")

    @patch("backend.cli.main.run_all", new_callable=AsyncMock)
    def test_eval_prints_summary_and_report_path(self, mock_run_all):
        mock_run_all.return_value = {
            "summary": "all passed",
            "report_path": "/tmp/report.json",
        }
        result = runner.invoke(app, ["eval", "--paper-id", "abc123"])
        assert result.exit_code == 0
        assert "all passed" in result.output or "/tmp/report.json" in result.output


# ---------------------------------------------------------------------------
# Invalid command
# ---------------------------------------------------------------------------

def test_invalid_command_returns_nonzero_exit():
    result = runner.invoke(app, ["nonexistent-command"])
    assert result.exit_code != 0
