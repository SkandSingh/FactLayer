"""Tests for the /relationships route: list and detail endpoints.

Run: python -m pytest tests/test_relationships_route.py -v

Uses the same temp_db isolation pattern as test_store.py and test_documents_route.py.
Inserts test data directly via store functions, then tests the HTTP endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from app import config, store
from app.main import app
from app.models import ExtractedFact, TimeScope


def make_extracted_fact(**overrides) -> ExtractedFact:
    """Helper to create an ExtractedFact with reasonable defaults."""
    defaults = dict(
        entity_name="Test Company",
        entity_id="TEST123",
        attribute="revenue",
        raw_value="1000",
        unit="Rs Million",
        time_scope=TimeScope(
            period_type="span",
            start="2023-04-01",
            end="2024-03-31",
            label="FY24",
            vintage=None,
        ),
        entity_scope="consolidated",
        verbatim_quote="Test quote for revenue",
        confidence=0.9,
    )
    defaults.update(overrides)
    return ExtractedFact(**defaults)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point config.FACTLAYER_DB_PATH at a per-test temp file.

    Isolates each test from the real factlayer.db.
    """
    db_path = tmp_path / "test_factlayer.db"
    monkeypatch.setattr(config, "FACTLAYER_DB_PATH", str(db_path))
    yield db_path


@pytest.fixture
def client():
    """Return a TestClient for the FastAPI app."""
    return TestClient(app)


def test_list_relationships_empty(client):
    """GET /relationships returns empty list when no relationships exist."""
    resp = client.get("/relationships")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_relationships_returns_inserted_relationship(client):
    """GET /relationships returns relationships that were inserted."""
    # Create test facts
    doc_id = store.insert_document("test.pdf", "Test Company", 1)
    fact_a = make_extracted_fact(attribute="revenue_a", raw_value="1000")
    fact_b = make_extracted_fact(attribute="revenue_b", raw_value="1100")

    fact_id_a = store.insert_fact(doc_id, fact_a)
    fact_id_b = store.insert_fact(doc_id, fact_b)

    # Create a relationship
    rel_id = store.insert_relationship(
        fact_id_a=fact_id_a,
        fact_id_b=fact_id_b,
        relation_type="corroborates",
        reconciled_dimension="time",
        reasoning_text="Both facts are about revenue in FY24.",
        confidence=0.95,
    )

    resp = client.get("/relationships")
    assert resp.status_code == 200
    rels = resp.json()
    assert len(rels) == 1
    assert rels[0]["id"] == rel_id
    assert rels[0]["fact_id_a"] == fact_id_a
    assert rels[0]["fact_id_b"] == fact_id_b
    assert rels[0]["relation_type"] == "corroborates"
    assert rels[0]["confidence"] == 0.95


def test_list_relationships_filter_by_fact_id(client):
    """GET /relationships?fact_id=X filters to relationships touching fact X."""
    doc_id = store.insert_document("test.pdf", "Test Company", 1)
    fact_a = make_extracted_fact(attribute="revenue_a", raw_value="1000")
    fact_b = make_extracted_fact(attribute="revenue_b", raw_value="1100")
    fact_c = make_extracted_fact(attribute="revenue_c", raw_value="1200")

    fact_id_a = store.insert_fact(doc_id, fact_a)
    fact_id_b = store.insert_fact(doc_id, fact_b)
    fact_id_c = store.insert_fact(doc_id, fact_c)

    # Create relationships: a-b and b-c
    rel_ab_id = store.insert_relationship(
        fact_id_a=fact_id_a,
        fact_id_b=fact_id_b,
        relation_type="corroborates",
        reconciled_dimension=None,
        reasoning_text="a corroborates b",
        confidence=0.9,
    )
    rel_bc_id = store.insert_relationship(
        fact_id_a=fact_id_b,
        fact_id_b=fact_id_c,
        relation_type="contradicts",
        reconciled_dimension=None,
        reasoning_text="b contradicts c",
        confidence=0.8,
    )

    # Filter by fact_id_a: should only get rel_ab
    resp = client.get(f"/relationships?fact_id={fact_id_a}")
    assert resp.status_code == 200
    rels = resp.json()
    assert len(rels) == 1
    assert rels[0]["id"] == rel_ab_id

    # Filter by fact_id_b: should get both rel_ab and rel_bc
    resp = client.get(f"/relationships?fact_id={fact_id_b}")
    assert resp.status_code == 200
    rels = resp.json()
    assert len(rels) == 2
    rel_ids = {r["id"] for r in rels}
    assert rel_ids == {rel_ab_id, rel_bc_id}

    # Filter by fact_id_c: should only get rel_bc
    resp = client.get(f"/relationships?fact_id={fact_id_c}")
    assert resp.status_code == 200
    rels = resp.json()
    assert len(rels) == 1
    assert rels[0]["id"] == rel_bc_id

    # Filter by nonexistent fact: should return empty list
    resp = client.get("/relationships?fact_id=99999")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_relationships_filter_by_relation_type(client):
    """GET /relationships?relation_type=X filters to relationships of type X."""
    doc_id = store.insert_document("test.pdf", "Test Company", 1)
    fact_a = make_extracted_fact(attribute="revenue_a", raw_value="1000")
    fact_b = make_extracted_fact(attribute="revenue_b", raw_value="1100")
    fact_c = make_extracted_fact(attribute="revenue_c", raw_value="1200")

    fact_id_a = store.insert_fact(doc_id, fact_a)
    fact_id_b = store.insert_fact(doc_id, fact_b)
    fact_id_c = store.insert_fact(doc_id, fact_c)

    # Create relationships of different types
    corr_id = store.insert_relationship(
        fact_id_a=fact_id_a,
        fact_id_b=fact_id_b,
        relation_type="corroborates",
        reconciled_dimension=None,
        reasoning_text="corroborates",
        confidence=0.9,
    )
    contra_id = store.insert_relationship(
        fact_id_a=fact_id_b,
        fact_id_b=fact_id_c,
        relation_type="contradicts",
        reconciled_dimension=None,
        reasoning_text="contradicts",
        confidence=0.8,
    )

    # Filter by "corroborates": should only get corr_id
    resp = client.get("/relationships?relation_type=corroborates")
    assert resp.status_code == 200
    rels = resp.json()
    assert len(rels) == 1
    assert rels[0]["id"] == corr_id
    assert rels[0]["relation_type"] == "corroborates"

    # Filter by "contradicts": should only get contra_id
    resp = client.get("/relationships?relation_type=contradicts")
    assert resp.status_code == 200
    rels = resp.json()
    assert len(rels) == 1
    assert rels[0]["id"] == contra_id
    assert rels[0]["relation_type"] == "contradicts"

    # Filter by nonexistent type: should return empty list
    resp = client.get("/relationships?relation_type=reconciled_context")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_relationship_detail_success(client):
    """GET /relationships/{id} returns relationship with both facts and evidence."""
    doc_id = store.insert_document("test.pdf", "Test Company", 1)
    fact_a = make_extracted_fact(
        attribute="revenue_a",
        raw_value="1000",
        verbatim_quote="Revenue was 1000 in FY24",
    )
    fact_b = make_extracted_fact(
        attribute="revenue_b",
        raw_value="1100",
        verbatim_quote="Revenue was 1100 in FY24",
    )

    fact_id_a = store.insert_fact(
        doc_id,
        fact_a,
        section_path="section_1",
        char_offset=10,
        page_number=1,
    )
    fact_id_b = store.insert_fact(
        doc_id,
        fact_b,
        section_path="section_2",
        char_offset=50,
        page_number=2,
    )

    rel_id = store.insert_relationship(
        fact_id_a=fact_id_a,
        fact_id_b=fact_id_b,
        relation_type="corroborates",
        reconciled_dimension="time",
        reasoning_text="Both facts about revenue in FY24.",
        confidence=0.95,
    )

    resp = client.get(f"/relationships/{rel_id}")
    assert resp.status_code == 200
    body = resp.json()

    # Check relationship data
    assert "relationship" in body
    rel = body["relationship"]
    assert rel["id"] == rel_id
    assert rel["fact_id_a"] == fact_id_a
    assert rel["fact_id_b"] == fact_id_b
    assert rel["relation_type"] == "corroborates"
    assert rel["reconciled_dimension"] == "time"
    assert rel["reasoning_text"] == "Both facts about revenue in FY24."
    assert rel["confidence"] == 0.95

    # Check fact_a data (with evidence pointers)
    assert "fact_a" in body
    fact_a_data = body["fact_a"]
    assert fact_a_data["id"] == fact_id_a
    assert fact_a_data["verbatim_quote"] == "Revenue was 1000 in FY24"
    assert fact_a_data["section_path"] == "section_1"
    assert fact_a_data["char_offset"] == 10
    assert fact_a_data["page_number"] == 1
    assert fact_a_data["raw_value"] == "1000"
    assert fact_a_data["entity_name"] == "Test Company"

    # Check fact_b data (with evidence pointers)
    assert "fact_b" in body
    fact_b_data = body["fact_b"]
    assert fact_b_data["id"] == fact_id_b
    assert fact_b_data["verbatim_quote"] == "Revenue was 1100 in FY24"
    assert fact_b_data["section_path"] == "section_2"
    assert fact_b_data["char_offset"] == 50
    assert fact_b_data["page_number"] == 2
    assert fact_b_data["raw_value"] == "1100"
    assert fact_b_data["entity_name"] == "Test Company"


def test_get_relationship_detail_not_found(client):
    """GET /relationships/{id} returns 404 for nonexistent relationship."""
    resp = client.get("/relationships/99999")
    assert resp.status_code == 404
    body = resp.json()
    assert "detail" in body
    assert "Relationship not found" in body["detail"]


def test_list_relationships_with_both_filters(client):
    """GET /relationships?fact_id=X&relation_type=Y applies both filters."""
    doc_id = store.insert_document("test.pdf", "Test Company", 1)
    fact_a = make_extracted_fact(attribute="revenue_a", raw_value="1000")
    fact_b = make_extracted_fact(attribute="revenue_b", raw_value="1100")
    fact_c = make_extracted_fact(attribute="revenue_c", raw_value="1200")

    fact_id_a = store.insert_fact(doc_id, fact_a)
    fact_id_b = store.insert_fact(doc_id, fact_b)
    fact_id_c = store.insert_fact(doc_id, fact_c)

    # Create: a-b corroborates, b-c contradicts
    rel_ab_id = store.insert_relationship(
        fact_id_a=fact_id_a,
        fact_id_b=fact_id_b,
        relation_type="corroborates",
        reconciled_dimension=None,
        reasoning_text="a corroborates b",
        confidence=0.9,
    )
    rel_bc_id = store.insert_relationship(
        fact_id_a=fact_id_b,
        fact_id_b=fact_id_c,
        relation_type="contradicts",
        reconciled_dimension=None,
        reasoning_text="b contradicts c",
        confidence=0.8,
    )

    # fact_id_b with relation_type=corroborates: only rel_ab
    resp = client.get(f"/relationships?fact_id={fact_id_b}&relation_type=corroborates")
    assert resp.status_code == 200
    rels = resp.json()
    assert len(rels) == 1
    assert rels[0]["id"] == rel_ab_id

    # fact_id_b with relation_type=contradicts: only rel_bc
    resp = client.get(f"/relationships?fact_id={fact_id_b}&relation_type=contradicts")
    assert resp.status_code == 200
    rels = resp.json()
    assert len(rels) == 1
    assert rels[0]["id"] == rel_bc_id

    # fact_id_a with relation_type=contradicts: empty (a touches only corroborates)
    resp = client.get(f"/relationships?fact_id={fact_id_a}&relation_type=contradicts")
    assert resp.status_code == 200
    assert resp.json() == []
