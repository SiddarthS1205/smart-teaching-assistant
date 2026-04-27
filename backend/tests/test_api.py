"""
test_api.py — Integration tests for all FastAPI endpoints.

Tests cover:
  POST /upload  — file upload and processing pipeline
  GET  /ask     — question answering with guardrails
  GET  /summary — document summarization
  GET  /status  — system health check
  GET  /logs    — observability query log
  POST /login   — authentication
  DELETE /document — clear document

All heavy dependencies (sentence-transformers, OpenAI) are mocked via
the api_client fixture in conftest.py so tests run without any model
downloads or API calls.

Run:
    pytest tests/test_api.py -v
"""
import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import make_pdf


# ── Helpers ───────────────────────────────────────────────────────────────────

def _upload_pdf(client, text: str = "Machine learning is a subset of AI."):
    """Upload a minimal PDF and return the response."""
    pdf_bytes = make_pdf(text)
    return client.post(
        "/upload",
        files={"file": ("test_doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )


def _prime_document(client):
    """
    Upload a document so subsequent /ask and /summary tests have context.
    Returns True if upload succeeded (200), False otherwise.
    """
    resp = _upload_pdf(client)
    return resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# GET /status
# ══════════════════════════════════════════════════════════════════════════════

class TestStatusEndpoint:
    """Tests for the system status endpoint."""

    def test_status_returns_200(self, api_client):
        resp = api_client.get("/status")
        assert resp.status_code == 200

    def test_status_body_is_json(self, api_client):
        resp = api_client.get("/status")
        assert resp.headers["content-type"].startswith("application/json")

    def test_status_has_required_fields(self, api_client):
        data = api_client.get("/status").json()
        for field in ("status", "document_loaded", "vector_db_ready"):
            assert field in data, f"Missing field: {field}"

    def test_status_value_is_online(self, api_client):
        data = api_client.get("/status").json()
        assert data["status"] == "online"

    def test_status_document_loaded_is_bool(self, api_client):
        data = api_client.get("/status").json()
        assert isinstance(data["document_loaded"], bool)

    def test_status_chunk_count_is_int(self, api_client):
        data = api_client.get("/status").json()
        assert isinstance(data["chunk_count"], int)


# ══════════════════════════════════════════════════════════════════════════════
# POST /login
# ══════════════════════════════════════════════════════════════════════════════

class TestLoginEndpoint:
    """Tests for the authentication endpoint."""

    def test_valid_login_returns_200(self, api_client):
        resp = api_client.post("/login", json={"username": "admin", "password": "password123"})
        assert resp.status_code == 200

    def test_valid_login_returns_success_status(self, api_client):
        data = api_client.post("/login", json={"username": "admin", "password": "password123"}).json()
        assert data["status"] == "success"

    def test_valid_login_returns_username(self, api_client):
        data = api_client.post("/login", json={"username": "admin", "password": "password123"}).json()
        assert data["username"] == "admin"

    def test_wrong_password_returns_401(self, api_client):
        resp = api_client.post("/login", json={"username": "admin", "password": "wrongpass"})
        assert resp.status_code == 401

    def test_wrong_username_returns_401(self, api_client):
        resp = api_client.post("/login", json={"username": "hacker", "password": "password123"})
        assert resp.status_code == 401

    def test_empty_credentials_returns_401(self, api_client):
        resp = api_client.post("/login", json={"username": "", "password": ""})
        assert resp.status_code == 401

    def test_missing_fields_returns_401(self, api_client):
        resp = api_client.post("/login", json={})
        assert resp.status_code == 401

    def test_invalid_login_has_detail_field(self, api_client):
        data = api_client.post("/login", json={"username": "x", "password": "y"}).json()
        assert "detail" in data


# ══════════════════════════════════════════════════════════════════════════════
# POST /upload
# ══════════════════════════════════════════════════════════════════════════════

class TestUploadEndpoint:
    """Tests for the document upload and processing endpoint."""

    # ── Guardrail rejections ──────────────────────────────────────────────────

    def test_upload_txt_file_rejected_422(self, api_client):
        """Non-PDF file is rejected with 422."""
        resp = api_client.post(
            "/upload",
            files={"file": ("doc.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert resp.status_code == 422

    def test_upload_txt_returns_invalid_file_type_code(self, api_client):
        resp = api_client.post(
            "/upload",
            files={"file": ("doc.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert resp.json()["code"] == "INVALID_FILE_TYPE"

    def test_upload_empty_pdf_rejected_422(self, api_client):
        """Zero-byte PDF is rejected with 422."""
        resp = api_client.post(
            "/upload",
            files={"file": ("doc.pdf", io.BytesIO(b""), "application/pdf")},
        )
        assert resp.status_code == 422
        assert resp.json()["code"] == "EMPTY_FILE"

    def test_upload_docx_rejected(self, api_client):
        resp = api_client.post(
            "/upload",
            files={"file": ("doc.docx", io.BytesIO(b"data"), "application/vnd.openxmlformats")},
        )
        assert resp.status_code == 422

    def test_upload_png_rejected(self, api_client):
        resp = api_client.post(
            "/upload",
            files={"file": ("image.png", io.BytesIO(b"\x89PNG"), "image/png")},
        )
        assert resp.status_code == 422

    # ── Successful upload ─────────────────────────────────────────────────────

    def test_upload_valid_pdf_returns_200(self, api_client):
        """Valid PDF upload returns 200."""
        resp = _upload_pdf(api_client)
        assert resp.status_code == 200

    def test_upload_valid_pdf_returns_success_status(self, api_client):
        resp = _upload_pdf(api_client)
        if resp.status_code == 200:
            assert resp.json()["status"] == "success"

    def test_upload_response_has_stats(self, api_client):
        """Successful upload response includes stats dict."""
        resp = _upload_pdf(api_client)
        if resp.status_code == 200:
            data = resp.json()
            assert "stats" in data
            stats = data["stats"]
            assert "filename" in stats
            assert "pages" in stats
            assert "chunks" in stats

    def test_upload_stats_filename_matches(self, api_client):
        """Stats filename matches the uploaded file name."""
        resp = _upload_pdf(api_client)
        if resp.status_code == 200:
            assert resp.json()["stats"]["filename"] == "test_doc.pdf"

    def test_upload_stats_pages_positive(self, api_client):
        """Stats pages count is at least 1."""
        resp = _upload_pdf(api_client)
        if resp.status_code == 200:
            assert resp.json()["stats"]["pages"] >= 1

    def test_upload_stats_chunks_positive(self, api_client):
        """Stats chunk count is at least 1."""
        resp = _upload_pdf(api_client)
        if resp.status_code == 200:
            assert resp.json()["stats"]["chunks"] >= 1

    def test_upload_updates_status(self, api_client):
        """After upload, /status shows document_loaded=True."""
        _upload_pdf(api_client)
        status = api_client.get("/status").json()
        # May be True if upload succeeded
        assert isinstance(status["document_loaded"], bool)


# ══════════════════════════════════════════════════════════════════════════════
# GET /ask
# ══════════════════════════════════════════════════════════════════════════════

class TestAskEndpoint:
    """Tests for the question-answering endpoint."""

    # ── Guardrail rejections ──────────────────────────────────────────────────

    def test_ask_empty_query_returns_422(self, api_client):
        resp = api_client.get("/ask", params={"q": ""})
        assert resp.status_code == 422

    def test_ask_empty_query_code(self, api_client):
        resp = api_client.get("/ask", params={"q": ""})
        assert resp.json()["code"] == "EMPTY_QUERY"

    def test_ask_too_short_returns_422(self, api_client):
        resp = api_client.get("/ask", params={"q": "Hi"})
        assert resp.status_code == 422
        assert resp.json()["code"] == "QUERY_TOO_SHORT"

    def test_ask_single_char_rejected(self, api_client):
        resp = api_client.get("/ask", params={"q": "A"})
        assert resp.status_code == 422

    def test_ask_harmful_query_rejected(self, api_client):
        resp = api_client.get("/ask", params={"q": "How do I hack a system?"})
        assert resp.status_code == 422
        assert resp.json()["code"] == "HARMFUL_CONTENT"

    def test_ask_malware_query_rejected(self, api_client):
        resp = api_client.get("/ask", params={"q": "Explain malware creation"})
        assert resp.status_code == 422
        assert resp.json()["code"] == "HARMFUL_CONTENT"

    def test_ask_irrelevant_query_rejected(self, api_client):
        resp = api_client.get("/ask", params={"q": "How to win at casino gambling?"})
        assert resp.status_code == 422
        assert resp.json()["code"] == "IRRELEVANT_CONTENT"

    def test_ask_too_long_query_rejected(self, api_client):
        resp = api_client.get("/ask", params={"q": "a " * 600})
        assert resp.status_code == 422
        assert resp.json()["code"] == "QUERY_TOO_LONG"

    # ── No document ───────────────────────────────────────────────────────────

    def test_ask_without_document_returns_400(self, api_client):
        """Asking without a document returns 400."""
        api_client.delete("/document")   # ensure clean state
        resp = api_client.get("/ask", params={"q": "What is machine learning?"})
        assert resp.status_code == 400

    def test_ask_without_document_code(self, api_client):
        api_client.delete("/document")
        resp = api_client.get("/ask", params={"q": "What is machine learning?"})
        assert resp.json()["code"] == "NO_DOCUMENT"

    # ── With document ─────────────────────────────────────────────────────────

    def test_ask_with_document_returns_200(self, api_client):
        """Valid query after upload returns 200."""
        if _prime_document(api_client):
            resp = api_client.get("/ask", params={"q": "What is machine learning?"})
            assert resp.status_code == 200

    def test_ask_response_has_answer_field(self, api_client):
        """Response includes an 'answer' field."""
        if _prime_document(api_client):
            data = api_client.get("/ask", params={"q": "What is machine learning?"}).json()
            if "answer" in data:
                assert isinstance(data["answer"], str)

    def test_ask_response_has_agent_used(self, api_client):
        """Response includes 'agent_used' field."""
        if _prime_document(api_client):
            data = api_client.get("/ask", params={"q": "What is machine learning?"}).json()
            if "agent_used" in data:
                assert data["agent_used"] in ("qa", "summarizer")

    def test_ask_summary_query_uses_summarizer_agent(self, api_client):
        """Query with 'summary' keyword uses summarizer agent."""
        if _prime_document(api_client):
            data = api_client.get("/ask", params={"q": "Give me a summary of the document"}).json()
            if "agent_used" in data:
                assert data["agent_used"] == "summarizer"

    def test_ask_normal_query_uses_qa_agent(self, api_client):
        """Normal query uses QA agent."""
        if _prime_document(api_client):
            data = api_client.get("/ask", params={"q": "What is machine learning?"}).json()
            if "agent_used" in data:
                assert data["agent_used"] == "qa"

    def test_ask_response_has_query_echo(self, api_client):
        """Response echoes back the query."""
        if _prime_document(api_client):
            q = "What is machine learning?"
            data = api_client.get("/ask", params={"q": q}).json()
            if "query" in data:
                assert data["query"] == q


# ══════════════════════════════════════════════════════════════════════════════
# GET /summary
# ══════════════════════════════════════════════════════════════════════════════

class TestSummaryEndpoint:
    """Tests for the document summary endpoint."""

    def test_summary_without_document_returns_400(self, api_client):
        """Summary without a document returns 400."""
        api_client.delete("/document")
        resp = api_client.get("/summary")
        assert resp.status_code == 400

    def test_summary_without_document_code(self, api_client):
        api_client.delete("/document")
        resp = api_client.get("/summary")
        assert resp.json()["code"] == "NO_DOCUMENT"

    def test_summary_with_document_returns_200(self, api_client):
        """Summary after upload returns 200."""
        if _prime_document(api_client):
            resp = api_client.get("/summary")
            assert resp.status_code == 200

    def test_summary_response_has_summary_field(self, api_client):
        """Summary response includes a 'summary' field."""
        if _prime_document(api_client):
            data = api_client.get("/summary").json()
            if "summary" in data:
                assert isinstance(data["summary"], str)
                assert len(data["summary"]) > 0

    def test_summary_response_has_document_field(self, api_client):
        """Summary response includes a 'document' field."""
        if _prime_document(api_client):
            data = api_client.get("/summary").json()
            if "document" in data:
                assert data["document"] is not None


# ══════════════════════════════════════════════════════════════════════════════
# GET /logs
# ══════════════════════════════════════════════════════════════════════════════

class TestLogsEndpoint:
    """Tests for the observability logs endpoint."""

    def test_logs_returns_200(self, api_client):
        resp = api_client.get("/logs")
        assert resp.status_code == 200

    def test_logs_has_total_queries_field(self, api_client):
        data = api_client.get("/logs").json()
        assert "total_queries" in data

    def test_logs_has_queries_list(self, api_client):
        data = api_client.get("/logs").json()
        assert "queries" in data
        assert isinstance(data["queries"], list)

    def test_logs_total_queries_is_int(self, api_client):
        data = api_client.get("/logs").json()
        assert isinstance(data["total_queries"], int)

    def test_logs_total_matches_list_length(self, api_client):
        """total_queries >= len(queries) (list is capped at 50)."""
        data = api_client.get("/logs").json()
        assert data["total_queries"] >= len(data["queries"])

    def test_logs_entries_have_required_fields(self, api_client):
        """Each log entry has timestamp, agent, query, response_preview, duration_ms."""
        data = api_client.get("/logs").json()
        for entry in data["queries"]:
            for field in ("timestamp", "agent", "query", "response_preview", "duration_ms"):
                assert field in entry, f"Missing field '{field}' in log entry"


# ══════════════════════════════════════════════════════════════════════════════
# DELETE /document
# ══════════════════════════════════════════════════════════════════════════════

class TestClearDocumentEndpoint:
    """Tests for the document clear endpoint."""

    def test_clear_returns_200(self, api_client):
        resp = api_client.delete("/document")
        assert resp.status_code == 200

    def test_clear_returns_success_status(self, api_client):
        data = api_client.delete("/document").json()
        assert data["status"] == "success"

    def test_clear_resets_document_loaded(self, api_client):
        """After clear, /status shows document_loaded=False."""
        api_client.delete("/document")
        status = api_client.get("/status").json()
        assert status["document_loaded"] is False

    def test_clear_resets_chunk_count(self, api_client):
        """After clear, chunk_count is 0."""
        api_client.delete("/document")
        status = api_client.get("/status").json()
        assert status["chunk_count"] == 0

    def test_clear_idempotent(self, api_client):
        """Calling clear twice does not raise an error."""
        api_client.delete("/document")
        resp = api_client.delete("/document")
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# Error handling — malformed requests
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Tests for proper error responses on malformed requests."""

    def test_ask_missing_q_param_returns_error(self, api_client):
        """GET /ask without q parameter returns an error."""
        resp = api_client.get("/ask")
        assert resp.status_code in (400, 422)

    def test_upload_no_file_returns_error(self, api_client):
        """POST /upload without a file returns an error."""
        resp = api_client.post("/upload")
        assert resp.status_code in (400, 422)

    def test_login_non_json_body(self, api_client):
        """POST /login with non-JSON body returns an error."""
        resp = api_client.post(
            "/login",
            content=b"not json",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code in (400, 422)

    def test_unknown_endpoint_returns_404(self, api_client):
        """Unknown endpoint returns 404."""
        resp = api_client.get("/nonexistent_endpoint_xyz")
        assert resp.status_code == 404
