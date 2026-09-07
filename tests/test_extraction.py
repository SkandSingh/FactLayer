"""Tests for app.extraction: per-section LLM extraction orchestration and
evidence-location.

Run: python -m pytest tests/test_extraction.py -v

No real Gemini calls are made here (no API key is configured in this
environment) -- a FakeLLMClient test double stands in for the real
GeminiClient, returning canned JSON strings.

Tests use plain `def` test functions that drive the async code under test
via `asyncio.run(...)`, since no pytest-asyncio plugin is installed in
this environment.
"""
import asyncio
import json

from app.extraction import LocatedFact, extract_facts_from_document
from app.llm.base import LLMClient
from app.pdf_ingest import PageContent
from app.sectioning import Section
from app import config


class FakeLLMClient(LLMClient):
    """Test double for LLMClient.

    Responses can be provided either as a single fixed string (`response`)
    or as a mapping from a substring-of-the-prompt "key" to a response
    string (`responses_by_marker`), so different sections (whose section
    text is embedded verbatim in the prompt) can be made to return
    different canned responses in the same `extract_facts_from_document`
    call.
    """

    def __init__(self, response: str | None = None, responses_by_marker: dict | None = None):
        self._response = response
        self._responses_by_marker = responses_by_marker or {}
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        for marker, response in self._responses_by_marker.items():
            if marker in prompt:
                return response
        if self._response is not None:
            return self._response
        raise AssertionError(f"FakeLLMClient: no canned response matched prompt: {prompt!r}")


def _make_page(page_number: int, text: str) -> PageContent:
    return PageContent(page_number=page_number, text=text, spans=[])


def _make_section(section_path: str, pages: list) -> Section:
    return Section(
        section_path=section_path,
        text="\n\n".join(p.text for p in pages),
        start_page=pages[0].page_number,
        end_page=pages[-1].page_number,
        pages=pages,
    )


def _valid_fact_dict(quote: str, entity_name: str = "Acme Corp") -> dict:
    return {
        "entity_name": entity_name,
        "entity_id": None,
        "attribute": "revenue_from_operations",
        "raw_value": "1234",
        "unit": "Rs Million",
        "time_scope": {
            "period_type": "point_in_time",
            "start": None,
            "end": None,
            "label": "FY24",
            "vintage": None,
        },
        "entity_scope": "standalone",
        "verbatim_quote": quote,
        "confidence": 0.95,
    }


def test_normal_case_locates_quote_and_returns_located_fact():
    quote = "Revenue from operations was Rs 1,234 Million in FY24."
    page_text = f"Some preamble text.\n{quote}\nSome trailing text."
    page = _make_page(3, page_text)
    section = _make_section("Financial Highlights", [page])

    expected_offset = page_text.find(quote)
    assert expected_offset != -1  # sanity check on the fixture itself

    response = json.dumps([_valid_fact_dict(quote)])
    fake_client = FakeLLMClient(response=response)

    results = asyncio.run(extract_facts_from_document([section], fake_client))

    assert len(results) == 1
    located = results[0]
    assert isinstance(located, LocatedFact)
    assert located.section_path == "Financial Highlights"
    assert located.page_number == 3
    assert located.char_offset == expected_offset
    assert located.fact.entity_name == "Acme Corp"
    assert located.fact.verbatim_quote == quote


def test_markdown_code_fenced_response_parses_correctly():
    quote = "Net profit rose to Rs 500 Million."
    page = _make_page(1, quote)
    section = _make_section("Section A", [page])

    raw = json.dumps([_valid_fact_dict(quote)])
    fenced_response = f"```json\n{raw}\n```"
    fake_client = FakeLLMClient(response=fenced_response)

    results = asyncio.run(extract_facts_from_document([section], fake_client))

    assert len(results) == 1
    assert results[0].fact.verbatim_quote == quote
    assert results[0].char_offset == 0


def test_plain_code_fence_without_json_marker_also_parses():
    quote = "Total assets stood at Rs 900 Million."
    page = _make_page(1, quote)
    section = _make_section("Section A", [page])

    raw = json.dumps([_valid_fact_dict(quote)])
    fenced_response = f"```\n{raw}\n```"
    fake_client = FakeLLMClient(response=fenced_response)

    results = asyncio.run(extract_facts_from_document([section], fake_client))

    assert len(results) == 1
    assert results[0].fact.verbatim_quote == quote


def test_malformed_item_in_array_is_skipped_but_valid_items_survive():
    quote_good = "Revenue was Rs 100 Million."
    page = _make_page(1, quote_good)
    section = _make_section("Section A", [page])

    good_item = _valid_fact_dict(quote_good)
    bad_item = _valid_fact_dict("some other quote", entity_name="Broken Co")
    del bad_item["attribute"]  # required field missing -> should fail validation

    response = json.dumps([good_item, bad_item])
    fake_client = FakeLLMClient(response=response)

    results = asyncio.run(extract_facts_from_document([section], fake_client))

    assert len(results) == 1
    assert results[0].fact.entity_name == "Acme Corp"
    assert results[0].fact.verbatim_quote == quote_good


def test_non_json_response_for_one_section_does_not_block_other_sections():
    quote_b = "GDP growth was 7 per cent in FY24."
    page_a = _make_page(1, "Section A page text, nothing quotable found here.")
    page_b = _make_page(1, quote_b)

    section_a = _make_section("Section A marker", [page_a])
    section_b = _make_section("Section B marker", [page_b])

    responses_by_marker = {
        "Section A page text": "this is not json at all {{{ ???",
        quote_b: json.dumps([_valid_fact_dict(quote_b, entity_name="Econ")]),
    }
    fake_client = FakeLLMClient(responses_by_marker=responses_by_marker)

    results = asyncio.run(
        extract_facts_from_document([section_a, section_b], fake_client)
    )

    assert len(results) == 1
    assert results[0].fact.entity_name == "Econ"
    assert results[0].section_path == "Section B marker"


def test_quote_not_found_in_any_page_still_returns_fact_with_degraded_location():
    page = _make_page(5, "This page does not contain the quoted text at all.")
    section = _make_section("Section A", [page])

    missing_quote = "This exact sentence never appears anywhere in the section."
    response = json.dumps([_valid_fact_dict(missing_quote)])
    fake_client = FakeLLMClient(response=response)

    results = asyncio.run(extract_facts_from_document([section], fake_client))

    assert len(results) == 1
    located = results[0]
    assert located.char_offset == -1
    # Degraded fallback: first page of the section.
    assert located.page_number == 5
    assert located.fact.verbatim_quote == missing_quote


def test_fuzzy_whitespace_normalized_quote_is_located():
    # The page text has the sentence split across a line break (as PDFs
    # commonly reflow text), while the LLM quotes it as one normalized line.
    page_text = "Total revenue for the year\nwas Rs 2,000 Million in FY24."
    page = _make_page(2, page_text)
    section = _make_section("Section A", [page])

    quote = "Total revenue for the year was Rs 2,000 Million in FY24."
    response = json.dumps([_valid_fact_dict(quote)])
    fake_client = FakeLLMClient(response=response)

    results = asyncio.run(extract_facts_from_document([section], fake_client))

    assert len(results) == 1
    located = results[0]
    assert located.page_number == 2
    assert located.char_offset == 0


def test_bounded_concurrency_correctness_at_scale(monkeypatch):
    # Exercise correctness with more sections than MAX_CONCURRENT_LLM_CALLS
    # allows to run at once; this doesn't prove true concurrency limiting
    # via timing, but confirms results stay correct and complete when the
    # semaphore forces batching.
    monkeypatch.setattr(config, "MAX_CONCURRENT_LLM_CALLS", 2)

    num_sections = 7
    sections = []
    responses_by_marker = {}
    for i in range(num_sections):
        marker = f"unique-marker-{i}"
        quote = f"Fact number {i} was reported as {i * 10} units."
        page_text = f"{marker}\n{quote}"
        page = _make_page(i + 1, page_text)
        sections.append(_make_section(f"Section {i}", [page]))
        responses_by_marker[marker] = json.dumps(
            [_valid_fact_dict(quote, entity_name=f"Entity{i}")]
        )

    fake_client = FakeLLMClient(responses_by_marker=responses_by_marker)

    results = asyncio.run(extract_facts_from_document(sections, fake_client))

    assert fake_client.call_count == num_sections
    assert len(results) == num_sections
    entity_names = {located.fact.entity_name for located in results}
    assert entity_names == {f"Entity{i}" for i in range(num_sections)}
    for located in results:
        assert located.char_offset != -1
