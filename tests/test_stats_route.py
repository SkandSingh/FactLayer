"""Tests for the /stats endpoint.

Run: python -m pytest tests/test_stats_route.py -v
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


def test_stats_returns_zero_counts_on_empty_db(client):
    """GET /stats returns zero counts when database is empty."""
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_documents"] == 0
    assert body["total_facts"] == 0
    assert body["total_relationships"] == 0
    assert body["relationships_by_type"] == {
        "corroborates": 0,
        "contradicts": 0,
        "reconciled_context": 0,
    }


def test_stats_counts_documents_and_facts(client):
    """GET /stats correctly counts documents and facts."""
    # Setup: insert 2 documents and 3 facts
    doc1 = store.insert_document("doc1.pdf", "Delhivery Limited", 2)
    doc2 = store.insert_document("doc2.pdf", "Zomato Limited", 2)

    fact1 = store.insert_fact(
        document_id=doc1,
        fact=make_extracted_fact(entity_name="Delhivery Limited", attribute="revenue"),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    fact2 = store.insert_fact(
        document_id=doc1,
        fact=make_extracted_fact(entity_name="Delhivery Limited", attribute="profit"),
        section_path="s2",
        char_offset=10,
        page_number=2,
    )
    fact3 = store.insert_fact(
        document_id=doc2,
        fact=make_extracted_fact(entity_name="Zomato Limited", attribute="revenue"),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )

    # Test
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_documents"] == 2
    assert body["total_facts"] == 3
    assert body["total_relationships"] == 0
    # All relationship types should appear even with 0 count
    assert body["relationships_by_type"] == {
        "corroborates": 0,
        "contradicts": 0,
        "reconciled_context": 0,
    }


def test_stats_counts_relationships_by_type(client):
    """GET /stats correctly counts relationships and breaks down by type."""
    # Setup: create facts and relationships of different types
    doc_id = store.insert_document("doc.pdf", "Delhivery Limited", 3)

    fact1 = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(attribute="revenue_from_operations"),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    fact2 = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(attribute="total_revenue"),
        section_path="s2",
        char_offset=10,
        page_number=2,
    )
    fact3 = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(attribute="net_profit"),
        section_path="s3",
        char_offset=20,
        page_number=3,
    )

    # Insert relationships of different types
    rel1 = store.insert_relationship(
        fact_id_a=fact1,
        fact_id_b=fact2,
        relation_type="corroborates",
        reconciled_dimension=None,
        reasoning_text="Both are revenue figures.",
        confidence=0.9,
    )
    rel2 = store.insert_relationship(
        fact_id_a=fact2,
        fact_id_b=fact3,
        relation_type="contradicts",
        reconciled_dimension=None,
        reasoning_text="Conflicting numbers.",
        confidence=0.7,
    )
    rel3 = store.insert_relationship(
        fact_id_a=fact1,
        fact_id_b=fact3,
        relation_type="corroborates",
        reconciled_dimension=None,
        reasoning_text="Supporting evidence.",
        confidence=0.8,
    )

    # Test
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_documents"] == 1
    assert body["total_facts"] == 3
    assert body["total_relationships"] == 3
    assert body["relationships_by_type"] == {
        "corroborates": 2,
        "contradicts": 1,
        "reconciled_context": 0,
    }


def test_stats_includes_all_relationship_types_even_if_zero(client):
    """GET /stats includes all known relationship types even if count is 0."""
    # Setup: create facts and relationships of only one type
    doc_id = store.insert_document("doc.pdf", "Company", 2)

    fact1 = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(attribute="attr1"),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    fact2 = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(attribute="attr2"),
        section_path="s2",
        char_offset=10,
        page_number=2,
    )

    # Insert only "reconciled_context" type
    store.insert_relationship(
        fact_id_a=fact1,
        fact_id_b=fact2,
        relation_type="reconciled_context",
        reconciled_dimension="time",
        reasoning_text="Context reconciliation.",
        confidence=0.85,
    )

    # Test
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_relationships"] == 1
    # All types should appear, with 0 for the unused ones
    assert body["relationships_by_type"] == {
        "corroborates": 0,
        "contradicts": 0,
        "reconciled_context": 1,
    }


def test_stats_multiple_docs_and_relationships(client):
    """GET /stats aggregates correctly across multiple documents."""
    # Setup: create 3 documents with facts and various relationships
    doc1 = store.insert_document("doc1.pdf", "Company A", 2)
    doc2 = store.insert_document("doc2.pdf", "Company B", 2)
    doc3 = store.insert_document("doc3.pdf", "Company C", 2)

    # Insert facts
    fact1 = store.insert_fact(
        document_id=doc1,
        fact=make_extracted_fact(entity_name="Company A"),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    fact2 = store.insert_fact(
        document_id=doc2,
        fact=make_extracted_fact(entity_name="Company B"),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    fact3 = store.insert_fact(
        document_id=doc3,
        fact=make_extracted_fact(entity_name="Company C"),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    fact4 = store.insert_fact(
        document_id=doc1,
        fact=make_extracted_fact(entity_name="Company A", attribute="other_attr"),
        section_path="s2",
        char_offset=10,
        page_number=2,
    )

    # Insert relationships
    store.insert_relationship(
        fact_id_a=fact1,
        fact_id_b=fact2,
        relation_type="corroborates",
        reconciled_dimension=None,
        reasoning_text="Match.",
        confidence=0.9,
    )
    store.insert_relationship(
        fact_id_a=fact2,
        fact_id_b=fact3,
        relation_type="contradicts",
        reconciled_dimension=None,
        reasoning_text="Conflict.",
        confidence=0.7,
    )
    store.insert_relationship(
        fact_id_a=fact3,
        fact_id_b=fact4,
        relation_type="corroborates",
        reconciled_dimension=None,
        reasoning_text="Support.",
        confidence=0.85,
    )
    store.insert_relationship(
        fact_id_a=fact1,
        fact_id_b=fact4,
        relation_type="reconciled_context",
        reconciled_dimension="scope",
        reasoning_text="Context.",
        confidence=0.8,
    )

    # Test
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_documents"] == 3
    assert body["total_facts"] == 4
    assert body["total_relationships"] == 4
    assert body["relationships_by_type"] == {
        "corroborates": 2,
        "contradicts": 1,
        "reconciled_context": 1,
    }
