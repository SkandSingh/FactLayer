"""Tests for app.extraction: batched LLM extraction orchestration and
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

from app.extraction import LocatedFact, MAX_BATCH_CHARS, extract_facts_from_document
from app.llm.base import LLMClient
from app.pdf_ingest import PageContent
from app.sectioning import Section
from app import config


class FakeLLMClient(LLMClient):
    """Test double for LLMClient.

    Responses can be provided either as a single fixed string (`response`)
    or as a mapping from a substring-of-the-prompt "key" to a response
    string (`responses_by_marker`), so different batches (whose sections'
    text/labels are embedded verbatim in the prompt) can be made to
    return different canned responses in the same
    `extract_facts_from_document` call.
    """

    def __init__(self, response: str | None = None, responses_by_marker: dict | None = None):
        self._response = response
        self._responses_by_marker = responses_by_marker or {}
        self.call_count = 0
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
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


def _valid_fact_dict(quote: str, section_path: str, entity_name: str = "Acme Corp") -> dict:
    return {
        "section_path": section_path,
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


def test_normal_case_multiple_sections_fit_in_one_batch():
    # Three small sections, well under MAX_BATCH_CHARS combined -> one batch,
    # one LLM call, one JSON array response covering facts from all three.
    quote_a = "Revenue from operations was Rs 1,234 Million in FY24."
    quote_b = "Net profit rose to Rs 500 Million in FY24."
    quote_c = "GDP growth was 7 per cent in FY24."

    page_a = _make_page(3, f"Preamble.\n{quote_a}\nTrailing.")
    page_b = _make_page(7, f"Preamble.\n{quote_b}\nTrailing.")
    page_c = _make_page(11, f"Preamble.\n{quote_c}\nTrailing.")

    section_a = _make_section("Financial Highlights", [page_a])
    section_b = _make_section("Profitability", [page_b])
    section_c = _make_section("Macro Overview", [page_c])

    response = json.dumps(
        [
            _valid_fact_dict(quote_a, "Financial Highlights", entity_name="Acme Corp"),
            _valid_fact_dict(quote_b, "Profitability", entity_name="Acme Corp"),
            _valid_fact_dict(quote_c, "Macro Overview", entity_name="Economy"),
        ]
    )
    fake_client = FakeLLMClient(response=response)

    results = asyncio.run(
        extract_facts_from_document([section_a, section_b, section_c], fake_client)
    )

    # All three sections fit comfortably under MAX_BATCH_CHARS -> single call.
    assert fake_client.call_count == 1
    assert len(results) == 3

    by_entity = {r.fact.entity_name: r for r in results if r.fact.entity_name == "Economy"}

    # There are two Acme Corp facts (a and b); look them up by section_path instead.
    by_section = {r.section_path: r for r in results if r.fact.entity_name != "Economy"}
    financial = by_section["Financial Highlights"]
    profitability = by_section["Profitability"]
    macro = by_entity["Economy"]

    assert financial.page_number == 3
    assert financial.char_offset == page_a.text.find(quote_a)
    assert financial.fact.verbatim_quote == quote_a

    assert profitability.page_number == 7
    assert profitability.char_offset == page_b.text.find(quote_b)
    assert profitability.fact.verbatim_quote == quote_b

    assert macro.section_path == "Macro Overview"
    assert macro.page_number == 11
    assert macro.char_offset == page_c.text.find(quote_c)


def test_sections_exceeding_budget_are_split_into_multiple_batches():
    # Build two sections whose combined text exceeds MAX_BATCH_CHARS, so
    # they must land in separate batches (separate LLM calls).
    quote_1 = "Revenue from operations was Rs 1,234 Million in FY24."
    quote_2 = "Net profit rose to Rs 500 Million in FY24."

    filler_1 = "x" * (MAX_BATCH_CHARS - 50)
    filler_2 = "y" * (MAX_BATCH_CHARS - 50)

    page_1 = _make_page(1, f"{filler_1}\n{quote_1}")
    page_2 = _make_page(2, f"{filler_2}\n{quote_2}")

    section_1 = _make_section("Section One", [page_1])
    section_2 = _make_section("Section Two", [page_2])

    responses_by_marker = {
        "=== SECTION: Section One ===": json.dumps(
            [_valid_fact_dict(quote_1, "Section One", entity_name="Batch1Entity")]
        ),
        "=== SECTION: Section Two ===": json.dumps(
            [_valid_fact_dict(quote_2, "Section Two", entity_name="Batch2Entity")]
        ),
    }
    fake_client = FakeLLMClient(responses_by_marker=responses_by_marker)

    results = asyncio.run(
        extract_facts_from_document([section_1, section_2], fake_client)
    )

    assert fake_client.call_count == 2
    assert len(results) == 2

    by_entity = {r.fact.entity_name: r for r in results}

    r1 = by_entity["Batch1Entity"]
    assert r1.section_path == "Section One"
    assert r1.page_number == 1
    assert r1.char_offset == page_1.text.find(quote_1)

    r2 = by_entity["Batch2Entity"]
    assert r2.section_path == "Section Two"
    assert r2.page_number == 2
    assert r2.char_offset == page_2.text.find(quote_2)


def test_markdown_code_fenced_response_parses_correctly():
    quote = "Net profit rose to Rs 500 Million."
    page = _make_page(1, quote)
    section = _make_section("Section A", [page])

    raw = json.dumps([_valid_fact_dict(quote, "Section A")])
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

    raw = json.dumps([_valid_fact_dict(quote, "Section A")])
    fenced_response = f"```\n{raw}\n```"
    fake_client = FakeLLMClient(response=fenced_response)

    results = asyncio.run(extract_facts_from_document([section], fake_client))

    assert len(results) == 1
    assert results[0].fact.verbatim_quote == quote


def test_malformed_item_in_array_is_skipped_but_siblings_survive():
    # Malformed item lives in a batch alongside a valid item from a
    # DIFFERENT section, to prove the failure is scoped to just that item,
    # not the section or the batch.
    quote_good_a = "Revenue was Rs 100 Million."
    quote_good_b = "GDP growth was 7 per cent."
    page_a = _make_page(1, quote_good_a)
    page_b = _make_page(2, quote_good_b)
    section_a = _make_section("Section A", [page_a])
    section_b = _make_section("Section B", [page_b])

    good_item_a = _valid_fact_dict(quote_good_a, "Section A")
    good_item_b = _valid_fact_dict(quote_good_b, "Section B", entity_name="Econ")
    bad_item = _valid_fact_dict("some other quote", "Section A", entity_name="Broken Co")
    del bad_item["attribute"]  # required field missing -> should fail validation

    response = json.dumps([good_item_a, bad_item, good_item_b])
    fake_client = FakeLLMClient(response=response)

    results = asyncio.run(
        extract_facts_from_document([section_a, section_b], fake_client)
    )

    assert len(results) == 2
    entity_names = {r.fact.entity_name for r in results}
    assert entity_names == {"Acme Corp", "Econ"}


def test_totally_invalid_response_for_one_batch_does_not_block_another_batch():
    quote_1 = "Revenue from operations was Rs 1,234 Million in FY24."
    quote_2 = "GDP growth was 7 per cent in FY24."

    filler_1 = "x" * (MAX_BATCH_CHARS - 50)
    filler_2 = "y" * (MAX_BATCH_CHARS - 50)

    page_1 = _make_page(1, f"{filler_1}\n{quote_1}")
    page_2 = _make_page(2, f"{filler_2}\n{quote_2}")

    section_1 = _make_section("Section One", [page_1])
    section_2 = _make_section("Section Two", [page_2])

    responses_by_marker = {
        "=== SECTION: Section One ===": "this is not json at all {{{ ???",
        "=== SECTION: Section Two ===": json.dumps(
            [_valid_fact_dict(quote_2, "Section Two", entity_name="Econ")]
        ),
    }
    fake_client = FakeLLMClient(responses_by_marker=responses_by_marker)

    results = asyncio.run(
        extract_facts_from_document([section_1, section_2], fake_client)
    )

    assert len(results) == 1
    assert results[0].fact.entity_name == "Econ"
    assert results[0].section_path == "Section Two"


def test_quote_not_found_in_any_page_still_returns_fact_with_degraded_location():
    page = _make_page(5, "This page does not contain the quoted text at all.")
    section = _make_section("Section A", [page])

    missing_quote = "This exact sentence never appears anywhere in the section."
    response = json.dumps([_valid_fact_dict(missing_quote, "Section A")])
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
    response = json.dumps([_valid_fact_dict(quote, "Section A")])
    fake_client = FakeLLMClient(response=response)

    results = asyncio.run(extract_facts_from_document([section], fake_client))

    assert len(results) == 1
    located = results[0]
    assert located.page_number == 2
    assert located.char_offset == 0


def test_single_oversized_section_becomes_its_own_batch():
    # A single section whose own text already exceeds MAX_BATCH_CHARS must
    # still be processed correctly as its own one-section batch, not
    # dropped or truncated.
    quote = "Revenue from operations was Rs 1,234 Million in FY24."
    filler = "x" * (MAX_BATCH_CHARS + 500)
    page = _make_page(1, f"{filler}\n{quote}")
    section = _make_section("Giant Section", [page])

    assert len(section.text) > MAX_BATCH_CHARS  # sanity check on the fixture

    response = json.dumps([_valid_fact_dict(quote, "Giant Section")])
    fake_client = FakeLLMClient(response=response)

    results = asyncio.run(extract_facts_from_document([section], fake_client))

    assert fake_client.call_count == 1
    assert len(results) == 1
    assert results[0].section_path == "Giant Section"
    assert results[0].page_number == 1
    assert results[0].char_offset == page.text.find(quote)


def test_hallucinated_section_path_falls_back_to_searching_batch_sections():
    # The LLM tags a fact with a section_path that doesn't match any real
    # section in the batch. Since both sections fit in one batch, the
    # fallback search across all of the batch's sections should still find
    # the quote and locate it correctly.
    quote = "Net profit rose to Rs 500 Million in FY24."
    page_a = _make_page(1, "Some unrelated text about Section A.")
    page_b = _make_page(2, f"Preamble.\n{quote}\nTrailing.")

    section_a = _make_section("Section A", [page_a])
    section_b = _make_section("Section B", [page_b])

    bad_fact = _valid_fact_dict(quote, "Nonexistent Section", entity_name="Ghost Co")
    response = json.dumps([bad_fact])
    fake_client = FakeLLMClient(response=response)

    results = asyncio.run(
        extract_facts_from_document([section_a, section_b], fake_client)
    )

    assert len(results) == 1
    located = results[0]
    assert located.fact.entity_name == "Ghost Co"
    assert located.section_path == "Section B"
    assert located.page_number == 2
    assert located.char_offset == page_b.text.find(quote)


def test_bounded_concurrency_correctness_at_scale(monkeypatch):
    # Exercise correctness with more batches than MAX_CONCURRENT_LLM_CALLS
    # allows to run at once; this doesn't prove true concurrency limiting
    # via timing, but confirms results stay correct and complete when the
    # semaphore forces batching. Each section is padded to force its own
    # batch, so num batches == num sections here.
    monkeypatch.setattr(config, "MAX_CONCURRENT_LLM_CALLS", 2)

    num_sections = 7
    sections = []
    responses_by_marker = {}
    for i in range(num_sections):
        marker = f"=== SECTION: Section {i} ==="
        quote = f"Fact number {i} was reported as {i * 10} units."
        filler = chr(ord("a") + i) * (MAX_BATCH_CHARS - 50)
        page_text = f"{filler}\n{quote}"
        page = _make_page(i + 1, page_text)
        sections.append(_make_section(f"Section {i}", [page]))
        responses_by_marker[marker] = json.dumps(
            [_valid_fact_dict(quote, f"Section {i}", entity_name=f"Entity{i}")]
        )

    fake_client = FakeLLMClient(responses_by_marker=responses_by_marker)

    results = asyncio.run(extract_facts_from_document(sections, fake_client))

    assert fake_client.call_count == num_sections
    assert len(results) == num_sections
    entity_names = {located.fact.entity_name for located in results}
    assert entity_names == {f"Entity{i}" for i in range(num_sections)}
    for located in results:
        assert located.char_offset != -1
