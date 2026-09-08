"""Tests for the retrieval-only POST /ask route.

Run: python -m pytest tests/test_ask_route.py -v
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


def test_ask_matches_entity_name_and_excludes_unrelated(client):
    """A question closely matching one document's entity_name returns
    that fact, and does not return an obviously unrelated fact."""
    doc_id = store.insert_document("delhivery_fy24.pdf", "Delhivery Limited", 2)
    fact_id = store.insert_fact(
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
    unrelated_id = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(
            entity_name="Zebra Trading Co",
            attribute="employee_headcount",
            verbatim_quote="The company had 42 employees at year end.",
        ),
        section_path="s2",
        char_offset=10,
        page_number=2,
    )

    resp = client.post("/ask", json={"question": "What is Delhivery's revenue from operations?"})
    assert resp.status_code == 200
    body = resp.json()

    matched_ids = {m["fact"]["id"] for m in body["matched_facts"]}
    assert fact_id in matched_ids
    assert unrelated_id not in matched_ids


def test_ask_matches_across_two_documents(client):
    """A question matching facts from two different documents returns
    both, each with the correct document_filename."""
    doc1 = store.insert_document("delhivery_fy24.pdf", "Delhivery Limited", 2)
    doc2 = store.insert_document("zomato_fy24.pdf", "Zomato Limited", 2)

    fact1 = store.insert_fact(
        document_id=doc1,
        fact=make_extracted_fact(
            entity_name="Delhivery Limited",
            attribute="revenue_from_operations",
            verbatim_quote="Delhivery reported revenue from operations of Rs 2,067.6 Million.",
        ),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    fact2 = store.insert_fact(
        document_id=doc2,
        fact=make_extracted_fact(
            entity_name="Zomato Limited",
            attribute="revenue_from_operations",
            verbatim_quote="Zomato reported revenue from operations of Rs 4,000 Million.",
        ),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )

    resp = client.post("/ask", json={"question": "revenue from operations"})
    assert resp.status_code == 200
    body = resp.json()

    matched_by_id = {m["fact"]["id"]: m for m in body["matched_facts"]}
    assert fact1 in matched_by_id
    assert fact2 in matched_by_id
    assert matched_by_id[fact1]["document_filename"] == "delhivery_fy24.pdf"
    assert matched_by_id[fact2]["document_filename"] == "zomato_fy24.pdf"


def test_ask_returns_relationships_for_matched_fact(client):
    """A returned fact that has a relationship shows it in its
    relationships list."""
    doc_id = store.insert_document("delhivery_fy24.pdf", "Delhivery Limited", 2)
    fact1 = store.insert_fact(
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
    fact2 = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(
            entity_name="Delhivery Limited",
            attribute="total_revenue",
            verbatim_quote="Total revenue was Rs 2,060 Million in FY24.",
        ),
        section_path="s2",
        char_offset=10,
        page_number=2,
    )
    rel_id = store.insert_relationship(
        fact_id_a=fact1,
        fact_id_b=fact2,
        relation_type="corroborates",
        reconciled_dimension=None,
        reasoning_text="Both figures are consistent within rounding.",
        confidence=0.9,
    )

    resp = client.post("/ask", json={"question": "Delhivery revenue from operations"})
    assert resp.status_code == 200
    body = resp.json()

    matched_by_id = {m["fact"]["id"]: m for m in body["matched_facts"]}
    assert fact1 in matched_by_id
    relationships = matched_by_id[fact1]["relationships"]
    assert len(relationships) == 1
    assert relationships[0]["id"] == rel_id


def test_ask_with_no_matches_returns_empty_result(client):
    """A question matching nothing returns matched_facts: [],
    total_matches: 0, status 200."""
    doc_id = store.insert_document("delhivery_fy24.pdf", "Delhivery Limited", 1)
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

    resp = client.post("/ask", json={"question": "xyzzy quux nonexistent gibberish term"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["matched_facts"] == []
    assert body["total_matches"] == 0


def test_ask_respects_limit(client):
    """limit is respected: seed more matching facts than limit, assert
    the response has exactly limit entries but total_matches reflects
    the true (larger) count."""
    doc_id = store.insert_document("delhivery_fy24.pdf", "Delhivery Limited", 1)
    for i in range(5):
        store.insert_fact(
            document_id=doc_id,
            fact=make_extracted_fact(
                entity_name="Delhivery Limited",
                attribute=f"revenue_from_operations_{i}",
                verbatim_quote=f"Revenue from operations figure number {i} for Delhivery.",
            ),
            section_path=f"s{i}",
            char_offset=i,
            page_number=1,
        )

    resp = client.post("/ask", json={"question": "Delhivery revenue from operations", "limit": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["matched_facts"]) == 2
    assert body["total_matches"] == 5
