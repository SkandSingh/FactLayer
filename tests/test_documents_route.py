"""Tests for the /documents upload route: end-to-end pipeline wiring.

Run: python -m pytest tests/test_documents_route.py -v

No real Gemini calls are made -- the `get_llm_client` dependency is
overridden with a FakeLLMClient test double (same pattern as
tests/test_extraction.py, copied here rather than imported so this file
stands alone). The real factlayer.db is never touched -- `temp_db`
monkeypatches `config.FACTLAYER_DB_PATH` at a per-test temp file, same
approach as tests/test_store.py.
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

from app import config, store
from app.llm.base import LLMClient
from app.main import app
from app.routes import documents as documents_routes

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")


class FakeLLMClient(LLMClient):
    """Test double for LLMClient: always returns the same canned JSON array
    of facts, regardless of prompt content, and counts how many times it
    was called (== number of sections the document was split into)."""

    def __init__(self, response: str):
        self._response = response
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self._response


def _canned_fact_dict() -> dict:
    return {
        "entity_name": "Acme Corp",
        "entity_id": None,
        "attribute": "revenue_from_operations",
        "raw_value": "100",
        "unit": "Cr",
        "time_scope": {
            "period_type": "point_in_time",
            "start": None,
            "end": None,
            "label": "FY24",
            "vintage": None,
        },
        "entity_scope": "standalone",
        "verbatim_quote": "some quote that may or may not appear verbatim",
        "confidence": 0.9,
    }


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Isolate every test in this file from the real factlayer.db."""
    db_path = tmp_path / "test_factlayer.db"
    monkeypatch.setattr(config, "FACTLAYER_DB_PATH", str(db_path))
    yield db_path


@pytest.fixture
def client():
    return TestClient(app)


def test_upload_pdf_end_to_end_persists_normalized_facts(client):
    canned_response = json.dumps([_canned_fact_dict()])
    fake_client = FakeLLMClient(response=canned_response)
    app.dependency_overrides[documents_routes.get_llm_client] = lambda: fake_client

    try:
        with open(FIXTURE_PATH, "rb") as f:
            resp = client.post(
                "/documents",
                files={"file": ("sample.pdf", f, "application/pdf")},
            )
    finally:
        app.dependency_overrides.pop(documents_routes.get_llm_client, None)

    assert resp.status_code == 200, resp.text
    body = resp.json()

    # One canned fact returned per LLM call, one call per detected section.
    assert fake_client.call_count > 0
    assert body["facts_extracted"] == fake_client.call_count
    assert body["section_count"] > 0
    assert body["filename"] == "sample.pdf"
    assert isinstance(body["document_id"], int)
    assert body["entity_name"] == "Acme Corp"
    assert len(body["fact_ids"]) == body["facts_extracted"]

    # The facts must actually be persisted, with normalization applied:
    # raw_value "100" + unit "Cr" -> 100 * 1e7 = 1_000_000_000.0
    persisted = store.list_facts(document_id=body["document_id"])
    assert len(persisted) == body["facts_extracted"]
    for fact in persisted:
        assert fact.raw_value == "100"
        assert fact.unit == "Cr"
        assert fact.normalized_value == pytest.approx(1_000_000_000.0)
        assert fact.normalized_unit == "Cr"
        # Best-effort fiscal-year fill: label "FY24" -> Apr 2023 - Mar 2024.
        assert fact.time_scope.start == "2023-04-01"
        assert fact.time_scope.end == "2024-03-31"


def test_upload_non_pdf_returns_400_not_500(client):
    fake_client = FakeLLMClient(response="[]")
    app.dependency_overrides[documents_routes.get_llm_client] = lambda: fake_client

    try:
        resp = client.post(
            "/documents",
            files={"file": ("not_a_pdf.txt", b"this is plain text, not a pdf", "text/plain")},
        )
    finally:
        app.dependency_overrides.pop(documents_routes.get_llm_client, None)

    assert resp.status_code == 400


def test_upload_corrupt_pdf_extension_returns_400_not_500(client):
    """A file named *.pdf whose bytes are garbage should still 400, not 500."""
    fake_client = FakeLLMClient(response="[]")
    app.dependency_overrides[documents_routes.get_llm_client] = lambda: fake_client

    try:
        resp = client.post(
            "/documents",
            files={"file": ("corrupt.pdf", b"%PDF-not-actually-a-real-pdf-body", "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(documents_routes.get_llm_client, None)

    assert resp.status_code == 400


def test_upload_over_size_limit_returns_400_or_413(client, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_MAX_BYTES", 100)

    fake_client = FakeLLMClient(response="[]")
    app.dependency_overrides[documents_routes.get_llm_client] = lambda: fake_client

    try:
        with open(FIXTURE_PATH, "rb") as f:
            contents = f.read()
        assert len(contents) > 100  # sanity check on the fixture itself

        resp = client.post(
            "/documents",
            files={"file": ("sample.pdf", contents, "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(documents_routes.get_llm_client, None)

    assert resp.status_code in (400, 413)
