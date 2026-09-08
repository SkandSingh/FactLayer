"""Tests for the server-rendered /ui routes: index, fact detail, relationship detail.

Run: python -m pytest tests/test_ui_routes.py -v

Uses the same temp_db isolation pattern as test_store.py / test_facts_route.py /
test_relationships_route.py: monkeypatch config.FACTLAYER_DB_PATH to a tmp_path
file, then insert test data directly via store functions before hitting the
HTTP endpoints through TestClient.
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

from app import config, store
from app.llm.base import LLMClient
from app.main import app
from app.models import ExtractedFact, TimeScope
from app.routes import documents as documents_routes

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")


class FakeLLMClient(LLMClient):
    """Test double for LLMClient (same pattern as test_documents_route.py)."""

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


def make_extracted_fact(**overrides) -> ExtractedFact:
    """Factory for creating ExtractedFact test data."""
    defaults = dict(
        entity_name="Delhivery Limited",
        entity_id="L63090HR2011PLC044150",
        attribute="revenue_from_operations",
        raw_value="2,067.6",
        unit="Rs Million",
        time_scope=TimeScope(
            period_type="span",
            start="2023-04-01",
            end="2024-03-31",
            label="FY24",
            vintage=None,
        ),
        entity_scope="consolidated",
        verbatim_quote="Revenue from operations was Rs 2,067.6 Million in FY24.",
        confidence=0.95,
    )
    defaults.update(overrides)
    return ExtractedFact(**defaults)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Isolate every test in this file from the real factlayer.db."""
    db_path = tmp_path / "test_factlayer.db"
    monkeypatch.setattr(config, "FACTLAYER_DB_PATH", str(db_path))
    yield db_path


@pytest.fixture
def client():
    return TestClient(app)


def test_index_returns_200_and_contains_seeded_data(client):
    """GET /ui/ returns 200 and shows seeded document + fact data."""
    doc_id = store.insert_document("delhivery_fy24.pdf", "Delhivery Limited", 3)
    store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(attribute="revenue_from_operations"),
        section_path="Financial Statements > Revenue",
        char_offset=120,
        page_number=4,
    )

    resp = client.get("/ui/")
    assert resp.status_code == 200
    body = resp.text
    assert "Delhivery Limited" in body
    assert "revenue_from_operations" in body


def test_fact_detail_returns_200_and_contains_verbatim_quote(client):
    """GET /ui/facts/{id} for an existing fact returns 200 with its verbatim_quote."""
    doc_id = store.insert_document("delhivery_fy24.pdf", "Delhivery Limited", 3)
    fact_id = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(),
        section_path="Financial Statements > Revenue",
        char_offset=120,
        page_number=4,
    )

    resp = client.get(f"/ui/facts/{fact_id}")
    assert resp.status_code == 200
    assert "Revenue from operations was Rs 2,067.6 Million in FY24." in resp.text


def test_fact_detail_for_nonexistent_id_returns_404(client):
    """GET /ui/facts/{id} for a nonexistent id returns 404."""
    resp = client.get("/ui/facts/9999")
    assert resp.status_code == 404


def test_relationship_detail_returns_200_and_contains_both_facts(client):
    """GET /ui/relationships/{id} shows both facts' entity_name and verbatim_quote."""
    doc_id = store.insert_document("delhivery_fy24.pdf", "Delhivery Limited", 3)
    fact_a = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(
            entity_name="Delhivery Limited",
            attribute="revenue_from_operations",
            verbatim_quote="Revenue from operations was Rs 2,067.6 Million in FY24.",
        ),
        section_path="Financial Statements > Revenue",
        char_offset=120,
        page_number=4,
    )
    fact_b = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(
            entity_name="Zomato Limited",
            attribute="total_revenue",
            verbatim_quote="Total revenue reported was Rs 2,060 Million for FY24.",
        ),
        section_path="Notes to Accounts > Revenue",
        char_offset=340,
        page_number=7,
    )
    rel_id = store.insert_relationship(
        fact_id_a=fact_a,
        fact_id_b=fact_b,
        relation_type="corroborates",
        reconciled_dimension=None,
        reasoning_text="Both figures are consistent within rounding.",
        confidence=0.88,
    )

    resp = client.get(f"/ui/relationships/{rel_id}")
    assert resp.status_code == 200
    body = resp.text
    assert "Delhivery Limited" in body
    assert "Zomato Limited" in body
    assert "Revenue from operations was Rs 2,067.6 Million in FY24." in body
    assert "Total revenue reported was Rs 2,060 Million for FY24." in body


def test_relationship_detail_for_nonexistent_id_returns_404(client):
    """GET /ui/relationships/{id} for a nonexistent id returns 404."""
    resp = client.get("/ui/relationships/9999")
    assert resp.status_code == 404


def test_upload_page_get_returns_200_with_form(client):
    """GET /ui/upload shows the upload form."""
    resp = client.get("/ui/upload")
    assert resp.status_code == 200
    assert "form" in resp.text.lower()
    assert 'name="files"' in resp.text


def test_upload_page_post_processes_pdf_and_shows_results(client):
    """POST /ui/upload runs the real pipeline (via a FakeLLMClient) and
    renders a result page with the extracted facts summary."""
    canned_response = json.dumps([_canned_fact_dict()])
    fake_client = FakeLLMClient(response=canned_response)
    app.dependency_overrides[documents_routes.get_llm_client] = lambda: fake_client

    try:
        with open(FIXTURE_PATH, "rb") as f:
            resp = client.post(
                "/ui/upload",
                files={"files": ("sample.pdf", f, "application/pdf")},
            )
    finally:
        app.dependency_overrides.pop(documents_routes.get_llm_client, None)

    assert resp.status_code == 200
    body = resp.text
    assert "sample.pdf" in body
    assert "Acme Corp" in body
    assert fake_client.call_count > 0

    persisted_docs = store.list_documents()
    assert len(persisted_docs) == 1
    assert persisted_docs[0].filename == "sample.pdf"


def test_upload_page_post_rejects_non_pdf_without_crashing(client):
    """A non-PDF upload through the UI form shows an error, not a 500."""
    resp = client.post(
        "/ui/upload",
        files={"files": ("notes.txt", b"not a pdf", "text/plain")},
    )

    assert resp.status_code == 200
    assert "must be a PDF" in resp.text
    assert store.list_documents() == []
