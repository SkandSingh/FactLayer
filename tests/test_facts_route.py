"""Tests for the /facts listing and detail routes.

Run: python -m pytest tests/test_facts_route.py -v
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


def test_get_facts_returns_all_inserted_facts(client):
    """GET /facts returns all inserted facts."""
    # Setup: insert a couple of facts
    doc_id = store.insert_document("doc.pdf", "Delhivery Limited", 2)
    fact1 = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(entity_name="Delhivery Limited", attribute="revenue"),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    fact2 = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(entity_name="Delhivery Limited", attribute="profit"),
        section_path="s2",
        char_offset=10,
        page_number=2,
    )

    # Test
    resp = client.get("/facts")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 2
    fact_ids = {f["id"] for f in facts}
    assert fact_ids == {fact1, fact2}


def test_get_facts_filters_by_document_id(client):
    """GET /facts?document_id=X filters correctly."""
    # Setup: two documents with facts
    doc1 = store.insert_document("doc1.pdf", "Delhivery Limited", 2)
    doc2 = store.insert_document("doc2.pdf", "Zomato Limited", 2)

    fact1 = store.insert_fact(
        document_id=doc1,
        fact=make_extracted_fact(entity_name="Delhivery Limited"),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    fact2 = store.insert_fact(
        document_id=doc2,
        fact=make_extracted_fact(entity_name="Zomato Limited"),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )

    # Test
    resp = client.get(f"/facts?document_id={doc1}")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 1
    assert facts[0]["id"] == fact1

    resp = client.get(f"/facts?document_id={doc2}")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 1
    assert facts[0]["id"] == fact2


def test_get_facts_filters_by_entity_name_substring(client):
    """GET /facts?entity_name=... performs case-insensitive substring match."""
    # Setup
    doc_id = store.insert_document("doc.pdf", "Company Names", 2)
    fact1 = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(entity_name="Delhivery Limited"),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    fact2 = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(entity_name="Zomato Limited"),
        section_path="s2",
        char_offset=10,
        page_number=2,
    )

    # Test: exact case
    resp = client.get("/facts?entity_name=Delhivery")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 1
    assert facts[0]["id"] == fact1

    # Test: lowercase
    resp = client.get("/facts?entity_name=delhivery")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 1
    assert facts[0]["id"] == fact1

    # Test: substring
    resp = client.get("/facts?entity_name=hiver")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 1
    assert facts[0]["id"] == fact1

    # Test: substring that matches nothing
    resp = client.get("/facts?entity_name=NotACompany")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 0


def test_get_facts_filters_by_attribute_substring(client):
    """GET /facts?attribute=... performs case-insensitive substring match."""
    # Setup
    doc_id = store.insert_document("doc.pdf", "Delhivery Limited", 2)
    fact1 = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(attribute="revenue_from_operations"),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    fact2 = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(attribute="net_profit_margin"),
        section_path="s2",
        char_offset=10,
        page_number=2,
    )

    # Test: exact case
    resp = client.get("/facts?attribute=revenue_from_operations")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 1
    assert facts[0]["id"] == fact1

    # Test: lowercase
    resp = client.get("/facts?attribute=revenue_from_operations".lower())
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 1

    # Test: substring
    resp = client.get("/facts?attribute=revenue")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 1
    assert facts[0]["id"] == fact1

    # Test: another substring
    resp = client.get("/facts?attribute=profit")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 1
    assert facts[0]["id"] == fact2


def test_get_facts_with_multiple_filters(client):
    """Combining multiple filters works correctly."""
    # Setup: facts with different combinations
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
        section_path="s3",
        char_offset=20,
        page_number=3,
    )

    # Test: filter by document_id AND entity_name
    resp = client.get(f"/facts?document_id={doc1}&entity_name=Delhivery")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 2
    assert {f["id"] for f in facts} == {fact1, fact2}

    # Test: filter by document_id AND attribute
    resp = client.get(f"/facts?document_id={doc1}&attribute=revenue")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 1
    assert facts[0]["id"] == fact1

    # Test: filter by entity_name AND attribute (cross document)
    resp = client.get("/facts?entity_name=Delhivery&attribute=profit")
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) == 1
    assert facts[0]["id"] == fact2


def test_get_fact_detail_returns_fact_with_relationships(client):
    """GET /facts/{id} for an existing id returns 200 with fact + relationships."""
    # Setup: create facts and a relationship
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
    rel_id = store.insert_relationship(
        fact_id_a=fact1,
        fact_id_b=fact2,
        relation_type="corroborates",
        reconciled_dimension=None,
        reasoning_text="Both report similar revenue figures.",
        confidence=0.9,
    )

    # Test
    resp = client.get(f"/facts/{fact1}")
    assert resp.status_code == 200
    body = resp.json()

    # Should have both 'fact' and 'relationships' keys
    assert "fact" in body
    assert "relationships" in body

    fact = body["fact"]
    assert fact["id"] == fact1
    assert fact["document_id"] == doc_id
    assert fact["entity_name"] == "Delhivery Limited"
    assert fact["attribute"] == "revenue_from_operations"

    # Should include the relationship touching this fact
    relationships = body["relationships"]
    assert len(relationships) == 1
    assert relationships[0]["id"] == rel_id
    assert relationships[0]["fact_id_a"] == fact1
    assert relationships[0]["fact_id_b"] == fact2


def test_get_fact_detail_for_nonexistent_id_returns_404(client):
    """GET /facts/{id} for a nonexistent id returns 404."""
    resp = client.get("/facts/9999")
    assert resp.status_code == 404
    body = resp.json()
    assert "not found" in body["detail"].lower()


def test_get_fact_detail_with_no_relationships(client):
    """GET /facts/{id} with no relationships returns empty list."""
    # Setup: create a fact with no relationships
    doc_id = store.insert_document("doc.pdf", "Delhivery Limited", 1)
    fact_id = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )

    # Test
    resp = client.get(f"/facts/{fact_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["relationships"] == []
