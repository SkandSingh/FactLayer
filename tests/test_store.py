"""Tests for app.store against a temp SQLite file (isolated from factlayer.db).

Run: python -m pytest tests/test_store.py -v
"""
import pytest

from app import config, store
from app.models import ExtractedFact, TimeScope


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point config.FACTLAYER_DB_PATH at a per-test temp file.

    store.py reads config.FACTLAYER_DB_PATH at call time (via
    db.get_connection()), so monkeypatching the module attribute here
    is enough to isolate every test from the real factlayer.db.
    """
    db_path = tmp_path / "test_factlayer.db"
    monkeypatch.setattr(config, "FACTLAYER_DB_PATH", str(db_path))
    yield db_path


def make_extracted_fact(**overrides) -> ExtractedFact:
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


def test_insert_and_fetch_document():
    doc_id = store.insert_document("annual_report.pdf", "Delhivery Limited", 12)
    assert isinstance(doc_id, int)

    doc = store.get_document(doc_id)
    assert doc is not None
    assert doc.id == doc_id
    assert doc.filename == "annual_report.pdf"
    assert doc.entity_name == "Delhivery Limited"
    assert doc.section_count == 12
    assert doc.uploaded_at  # non-empty ISO timestamp


def test_insert_and_fetch_fact_round_trips_time_scope():
    doc_id = store.insert_document("doc.pdf", "Delhivery Limited", 3)
    extracted = make_extracted_fact()

    fact_id = store.insert_fact(
        document_id=doc_id,
        fact=extracted,
        normalized_value=2067.6,
        normalized_unit="INR_million",
        section_path="Financials > Revenue",
        char_offset=1234,
        page_number=5,
    )
    assert isinstance(fact_id, int)

    fact = store.get_fact(fact_id)
    assert fact is not None
    assert fact.id == fact_id
    assert fact.document_id == doc_id
    assert fact.entity_name == extracted.entity_name
    assert fact.entity_id == extracted.entity_id
    assert fact.attribute == extracted.attribute
    assert fact.raw_value == extracted.raw_value
    assert fact.unit == extracted.unit
    assert fact.normalized_value == 2067.6
    assert fact.normalized_unit == "INR_million"
    assert fact.entity_scope == extracted.entity_scope
    assert fact.verbatim_quote == extracted.verbatim_quote
    assert fact.section_path == "Financials > Revenue"
    assert fact.char_offset == 1234
    assert fact.page_number == 5
    assert fact.confidence == extracted.confidence
    assert fact.created_at

    # time_scope must round-trip exactly, as a TimeScope instance.
    assert isinstance(fact.time_scope, TimeScope)
    assert fact.time_scope == extracted.time_scope


def test_insert_and_fetch_relationship():
    doc_id = store.insert_document("doc.pdf", "Delhivery Limited", 3)
    fact_a = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    fact_b = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(raw_value="2,067.6 Mn"),
        section_path="s2",
        char_offset=10,
        page_number=2,
    )

    rel_id = store.insert_relationship(
        fact_id_a=fact_a,
        fact_id_b=fact_b,
        relation_type="corroborates",
        reconciled_dimension=None,
        reasoning_text="Both report the same FY24 revenue figure.",
        confidence=0.9,
    )
    assert isinstance(rel_id, int)

    rel = store.get_relationship(rel_id)
    assert rel is not None
    assert rel.id == rel_id
    assert rel.fact_id_a == fact_a
    assert rel.fact_id_b == fact_b
    assert rel.relation_type == "corroborates"
    assert rel.reconciled_dimension is None
    assert rel.reasoning_text == "Both report the same FY24 revenue figure."
    assert rel.confidence == 0.9
    assert rel.created_at

    # list_relationships filters to relationships touching a given fact.
    touching_a = store.list_relationships(fact_id=fact_a)
    assert [r.id for r in touching_a] == [rel_id]

    unrelated_doc = store.insert_document("other.pdf", "Other Co", 1)
    unrelated_fact = store.insert_fact(
        document_id=unrelated_doc,
        fact=make_extracted_fact(entity_name="Other Co"),
        section_path="s3",
        char_offset=0,
        page_number=1,
    )
    assert store.list_relationships(fact_id=unrelated_fact) == []


def test_list_facts_filters_by_document_id():
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

    doc1_facts = store.list_facts(document_id=doc1)
    assert [f.id for f in doc1_facts] == [fact1]
    assert doc1_facts[0].document_id == doc1

    doc2_facts = store.list_facts(document_id=doc2)
    assert [f.id for f in doc2_facts] == [fact2]

    all_facts = store.list_facts()
    assert {f.id for f in all_facts} == {fact1, fact2}


def test_get_all_facts_across_two_documents():
    doc1 = store.insert_document("doc1.pdf", "Delhivery Limited", 2)
    doc2 = store.insert_document("doc2.pdf", "Zomato Limited", 2)

    fact_ids = set()
    for i, doc_id in enumerate([doc1, doc1, doc2]):
        fid = store.insert_fact(
            document_id=doc_id,
            fact=make_extracted_fact(attribute=f"attr_{i}"),
            section_path=f"s{i}",
            char_offset=i,
            page_number=1,
        )
        fact_ids.add(fid)

    all_facts = store.get_all_facts()
    assert {f.id for f in all_facts} == fact_ids
    assert len(all_facts) == 3
