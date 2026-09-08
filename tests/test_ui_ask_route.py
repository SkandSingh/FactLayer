"""Tests for the server-rendered /ui/ask routes.

Run: python -m pytest tests/test_ui_ask_route.py -v

Uses the same temp_db isolation pattern as test_ui_routes.py: monkeypatch
config.FACTLAYER_DB_PATH to a tmp_path file, then insert test data
directly via store functions before hitting the HTTP endpoints through
TestClient.
"""
import pytest
from fastapi.testclient import TestClient

from app import config, store
from app.main import app
from app.models import ExtractedFact, TimeScope


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


def test_ask_page_get_returns_200_with_search_form(client):
    """GET /ui/ask returns 200 with a search form."""
    resp = client.get("/ui/ask")
    assert resp.status_code == 200
    body = resp.text
    assert "<form" in body
    assert 'name="question"' in body


def test_ask_page_post_returns_matched_fact_quote(client):
    """POST /ui/ask (form-encoded) with a matching question returns 200
    and the response HTML contains the matched fact's verbatim_quote text."""
    doc_id = store.insert_document("delhivery_fy24.pdf", "Delhivery Limited", 2)
    store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(
            entity_name="Delhivery Limited",
            attribute="revenue_from_operations",
            verbatim_quote="Revenue from operations was Rs 2,067.6 Million in FY24.",
        ),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )

    resp = client.post("/ui/ask", data={"question": "Delhivery revenue from operations"})
    assert resp.status_code == 200
    assert "Revenue from operations was Rs 2,067.6 Million in FY24." in resp.text
