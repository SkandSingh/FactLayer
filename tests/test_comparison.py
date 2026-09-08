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
from app import comparison as comparison_module


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


class _KeyedLLMClient(LLMClient):
    """Test double whose response depends on which fact ids appear in the
    prompt it received -- lets a test assert distinct chunks got distinct
    (correctly-scoped) digests, the same pattern tests/test_extraction.py
    uses for its multi-batch coverage."""

    def __init__(self, responses_by_marker: dict):
        # {marker_substring: canned_json_response}
        self._responses_by_marker = responses_by_marker
        self.prompts_received: list = []

    async def generate(self, prompt: str) -> str:
        self.prompts_received.append(prompt)
        for marker, response in self._responses_by_marker.items():
            if marker in prompt:
                return response
        raise AssertionError(f"no canned response matched this prompt: {prompt[:200]}")


def test_large_candidate_set_is_chunked_across_multiple_calls(monkeypatch):
    # Force a tiny chunk budget so a handful of candidates already spans
    # multiple chunks, without needing thousands of real facts in the test.
    monkeypatch.setattr(comparison_module, "MAX_DIGEST_CHARS", 900)

    new_fact = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")
    # Several same-entity_id candidates -- each one is its own document so
    # find_candidates matches all of them via the exact entity_id path.
    candidates = [
        insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")
        for _ in range(6)
    ]

    # One canned response per possible candidate id -- whichever chunk a
    # candidate lands in, the fake client recognizes it by that id's
    # presence in the prompt and returns a relationship for that pair.
    responses_by_marker = {
        f'"id": {c.id}': json.dumps(
            [
                {
                    "fact_id_a": new_fact.id,
                    "fact_id_b": c.id,
                    "relation_type": "corroborates",
                    "reconciled_dimension": None,
                    "reasoning_text": f"Matches candidate {c.id}.",
                    "confidence": 0.9,
                }
            ]
        )
        for c in candidates
    }
    fake_client = _KeyedLLMClient(responses_by_marker)

    results = asyncio.run(compare_new_document_facts([new_fact], fake_client))

    # A tiny budget with 6 candidates forces more than one call.
    assert len(fake_client.prompts_received) > 1
    # Every candidate still got a relationship recorded despite being
    # spread across multiple chunked calls.
    assert {r.fact_id_b for r in results} == {c.id for c in candidates}


def test_one_failed_chunk_does_not_block_other_chunks(monkeypatch):
    monkeypatch.setattr(comparison_module, "MAX_DIGEST_CHARS", 900)

    new_fact = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")
    good_candidate = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")
    bad_candidate = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")

    class _OneChunkFailsClient(LLMClient):
        def __init__(self):
            self.call_count = 0

        async def generate(self, prompt: str) -> str:
            self.call_count += 1
            if f'"id": {bad_candidate.id}' in prompt:
                raise RuntimeError("simulated provider failure for this chunk")
            return json.dumps(
                [
                    {
                        "fact_id_a": new_fact.id,
                        "fact_id_b": good_candidate.id,
                        "relation_type": "corroborates",
                        "reconciled_dimension": None,
                        "reasoning_text": "Matches the good candidate.",
                        "confidence": 0.9,
                    }
                ]
            )

    fake_client = _OneChunkFailsClient()

    results = asyncio.run(compare_new_document_facts([new_fact], fake_client))

    # The chunk containing bad_candidate failed and was skipped, but the
    # chunk containing good_candidate still succeeded -- the whole
    # document's comparison isn't sunk by one bad chunk.
    assert fake_client.call_count > 1
    assert len(results) == 1
    assert results[0].fact_id_b == good_candidate.id


def test_relationship_between_two_new_facts_is_not_duplicated_across_chunks(monkeypatch):
    # new_facts are resent in full in every chunk, so a relationship
    # between two new facts could plausibly be (re)found in more than one
    # chunk -- must be inserted only once.
    monkeypatch.setattr(comparison_module, "MAX_DIGEST_CHARS", 900)

    new_fact_a = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")
    new_fact_b = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")
    # Enough filler candidates (all irrelevant/no-match responses) to force
    # multiple chunks even though the "real" relationship is only between
    # the two new facts.
    fillers = [
        insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")
        for _ in range(4)
    ]

    same_pair_relationship = json.dumps(
        [
            {
                "fact_id_a": new_fact_a.id,
                "fact_id_b": new_fact_b.id,
                "relation_type": "corroborates",
                "reconciled_dimension": None,
                "reasoning_text": "Both new facts agree.",
                "confidence": 0.9,
            }
        ]
    )
    fake_client = FakeLLMClient(response=same_pair_relationship)

    results = asyncio.run(compare_new_document_facts([new_fact_a, new_fact_b], fake_client))

    assert fake_client.call_count > 1  # confirms multiple chunks actually ran
    # Despite every chunk "finding" the same pair, it's inserted only once.
    assert len(results) == 1
    assert {results[0].fact_id_a, results[0].fact_id_b} == {new_fact_a.id, new_fact_b.id}
    assert len(store.list_relationships()) == 1


def test_new_facts_too_large_to_repeat_falls_back_to_interleaving(monkeypatch):
    # Regression test for the real bug this was built to fix: a document
    # can have thousands of facts OF ITS OWN (observed directly: ~2,300),
    # already exceeding the digest budget before a single candidate is
    # even considered. The "repeat new_facts in full in every chunk"
    # strategy silently assumes new_facts is small -- verify the fallback
    # (interleave new_facts and candidates, chunk the combined pool)
    # actually engages instead of producing one gigantic unchunked call.
    monkeypatch.setattr(comparison_module, "MAX_DIGEST_CHARS", 900)

    # 6 new facts alone comfortably exceed a 900-char budget (each fact
    # serializes to ~400+ chars), forcing the interleave fallback.
    new_facts = [
        insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")
        for _ in range(6)
    ]
    candidate = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")

    response = json.dumps(
        [
            {
                "fact_id_a": new_facts[0].id,
                "fact_id_b": candidate.id,
                "relation_type": "corroborates",
                "reconciled_dimension": None,
                "reasoning_text": "Matches the candidate.",
                "confidence": 0.9,
            }
        ]
    )
    fake_client = FakeLLMClient(response=response)

    results = asyncio.run(compare_new_document_facts(new_facts, fake_client))

    # The oversized new_facts set had to be split across multiple calls
    # rather than repeated whole in one (which would itself have exceeded
    # the budget).
    assert fake_client.call_count > 1
    # No single call's digest ever exceeded the budget -- the whole point
    # of chunking in the first place.
    assert len(results) == 1
    assert results[0].fact_id_a == new_facts[0].id
    assert results[0].fact_id_b == candidate.id


def test_relationship_already_recorded_by_an_earlier_call_is_not_reinserted():
    # Regression test for a real duplication bug: the same fact pair,
    # independently rediscovered by two SEPARATE compare_new_document_facts
    # calls (e.g. document B's comparison pass, then document C's), used to
    # be inserted twice -- the in-call `seen_pairs` dedup only covers a
    # single call, not the store's actual prior state.
    fact_a = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")
    fact_b = insert_fact(entity_id="L63090HR2011PLC044150", attribute="revenue_from_operations")

    response = json.dumps(
        [
            {
                "fact_id_a": fact_a.id,
                "fact_id_b": fact_b.id,
                "relation_type": "corroborates",
                "reconciled_dimension": None,
                "reasoning_text": "Same figure both times.",
                "confidence": 0.9,
            }
        ]
    )

    # First call: genuinely new, gets inserted.
    first_results = asyncio.run(compare_new_document_facts([fact_a], FakeLLMClient(response=response)))
    assert len(first_results) == 1
    assert len(store.list_relationships()) == 1

    # Second call (simulating a later document's comparison pass that
    # independently rediscovers the same pair, possibly with fact_id_a/b
    # swapped): must not insert a second row for the same pair.
    swapped_response = json.dumps(
        [
            {
                "fact_id_a": fact_b.id,
                "fact_id_b": fact_a.id,
                "relation_type": "corroborates",
                "reconciled_dimension": None,
                "reasoning_text": "Same figure both times.",
                "confidence": 0.9,
            }
        ]
    )
    second_results = asyncio.run(compare_new_document_facts([fact_b], FakeLLMClient(response=swapped_response)))

    assert second_results == []
    assert len(store.list_relationships()) == 1
