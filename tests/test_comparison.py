"""Tests for app.comparison: digest-style LLM fact comparison.

Run: python -m pytest tests/test_comparison.py -v

Facts are created through the real `app.store` functions against a
per-test temp SQLite file (same isolation pattern as tests/test_store.py),
so they carry real, distinct ids -- exactly what `compare_new_document_facts`
expects (it runs after facts have already been inserted). No real Gemini
calls are made -- a FakeLLMClient test double stands in for the real
GeminiClient, returning canned JSON strings (same pattern as
tests/test_extraction.py).
"""
import asyncio
import json

import pytest

from app import config, store
from app.comparison import compare_new_document_facts
from app.llm.base import LLMClient
from app.models import ExtractedFact, TimeScope


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point config.FACTLAYER_DB_PATH at a per-test temp file, isolating
    every test from the real factlayer.db (see tests/test_store.py)."""
    db_path = tmp_path / "test_factlayer.db"
    monkeypatch.setattr(config, "FACTLAYER_DB_PATH", str(db_path))
    yield db_path


class FakeLLMClient(LLMClient):
    """Test double for LLMClient. Returns a fixed canned response string
    and tracks how many times `generate` was called."""

    def __init__(self, response: str | None = None):
        self._response = response
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        if self._response is None:
            raise AssertionError("FakeLLMClient: no canned response configured")
        return self._response


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


def insert_fact(**overrides) -> "store.Fact":  # type: ignore[name-defined]
    """Insert a document + fact through the real store and return the
    resulting Fact (with a real id)."""
    doc_id = store.insert_document("doc.pdf", overrides.get("entity_name", "Delhivery Limited"), 1)
    fact_id = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(**overrides),
        section_path="s1",
        char_offset=0,
        page_number=1,
    )
    return store.get_fact(fact_id)


def test_related_facts_get_a_valid_relationship_inserted():
    # An existing fact already in the store...
    existing = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")
    # ...and a new fact that shares the same entity_id, so `find_candidates`
    # will surface `existing` as a candidate.
    new_fact = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")

    response = json.dumps(
        [
            {
                "fact_id_a": new_fact.id,
                "fact_id_b": existing.id,
                "relation_type": "corroborates",
                "reconciled_dimension": None,
                "reasoning_text": "Both report the same FY24 revenue figure.",
                "confidence": 0.9,
            }
        ]
    )
    fake_client = FakeLLMClient(response=response)

    results = asyncio.run(compare_new_document_facts([new_fact], fake_client))

    assert fake_client.call_count == 1
    assert len(results) == 1
    inserted = results[0]
    assert inserted.fact_id_a == new_fact.id
    assert inserted.fact_id_b == existing.id
    assert inserted.relation_type == "corroborates"
    assert inserted.reconciled_dimension is None
    assert inserted.reasoning_text == "Both report the same FY24 revenue figure."
    assert inserted.confidence == 0.9

    # Also visible via the store directly.
    touching = store.list_relationships(fact_id=new_fact.id)
    assert [r.id for r in touching] == [inserted.id]


def test_hallucinated_fact_id_is_skipped_without_crashing():
    existing = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")
    new_fact = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")

    hallucinated_id = max(existing.id, new_fact.id) + 999

    response = json.dumps(
        [
            {
                "fact_id_a": new_fact.id,
                "fact_id_b": hallucinated_id,
                "relation_type": "corroborates",
                "reconciled_dimension": None,
                "reasoning_text": "Fabricated relationship referencing an id never sent.",
                "confidence": 0.8,
            }
        ]
    )
    fake_client = FakeLLMClient(response=response)

    results = asyncio.run(compare_new_document_facts([new_fact], fake_client))

    assert fake_client.call_count == 1
    assert results == []
    assert store.list_relationships() == []


def test_too_small_digest_skips_llm_call_entirely():
    # A single new fact with nothing else in the store at all: no
    # candidates found, and the new fact alone isn't enough to compare.
    new_fact = insert_fact()

    fake_client = FakeLLMClient(response="should never be used")

    results = asyncio.run(compare_new_document_facts([new_fact], fake_client))

    assert fake_client.call_count == 0
    assert results == []
    assert store.list_relationships() == []


def test_malformed_json_response_returns_empty_list_without_raising():
    existing = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")
    new_fact = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")

    fake_client = FakeLLMClient(response="this is not json at all {{{ ???")

    results = asyncio.run(compare_new_document_facts([new_fact], fake_client))

    assert fake_client.call_count == 1
    assert results == []
    assert store.list_relationships() == []
