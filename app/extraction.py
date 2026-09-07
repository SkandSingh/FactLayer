"""Per-section LLM fact extraction, orchestrated with bounded concurrency.

This module is stage 3 of the pipeline described in docs/ARCHITECTURE.md:
it takes the ``list[Section]`` produced by ``app.sectioning`` and, for each
section, asks an ``LLMClient`` to extract structured facts (see
``app.llm.prompts.build_extraction_prompt`` and ``app.models.ExtractedFact``).

Two concerns are layered on top of the raw LLM call:

* **Resilience** -- a single section's LLM call failing (timeout, API
  error, malformed/non-JSON response, one bad item in an otherwise valid
  array) must never abort extraction for the rest of the document. Each
  failure mode is caught at the narrowest possible scope and logged as a
  warning; the caller always gets back whatever succeeded.
* **Evidence location** -- every ``ExtractedFact`` carries a
  ``verbatim_quote`` the LLM claims came from the section, but not *where*
  in the source PDF that text lives. We search the section's own
  ``PageContent.text`` (the exact text the LLM was shown, reading order)
  to pin down a ``(page_number, char_offset)`` for it, so a fact can later
  be traced back to an exact spot in the document. This is inherently
  best-effort: LLMs sometimes normalize whitespace/line-breaks when they
  quote, so an exact substring search is tried first and a whitespace-
  tolerant regex search is tried second. If neither finds it, we still
  keep the fact (dropping it just because provenance search failed would
  throw away a real extraction) but degrade its location to the section's
  first page with ``char_offset = -1`` to signal "couldn't pinpoint it".
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from app import config
from app.llm.prompts import build_extraction_prompt
from app.models import ExtractedFact

logger = logging.getLogger(__name__)

# Matches a leading ```/```json fence and/or a trailing ``` fence around an
# LLM response. LLMs asked to "respond with JSON" frequently wrap the
# answer in a markdown code block anyway; we strip that before json.loads
# rather than relying on the prompt alone to prevent it.
_LEADING_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*[ \t]*\r?\n?")
_TRAILING_FENCE_RE = re.compile(r"\r?\n?```[ \t]*$")


@dataclass
class LocatedFact:
    fact: object  # ExtractedFact
    section_path: str
    page_number: int  # best-effort page where the verbatim_quote was found
    char_offset: int  # best-effort char offset within that page's text; -1 if not confidently located


def _strip_code_fences(text: str) -> str:
    """Strip a single leading/trailing markdown code fence, if present.

    Handles both ```` ```json ... ``` ```` and plain ```` ``` ... ``` ````.
    Leaves the text untouched if it isn't fenced at all.
    """
    stripped = text.strip()
    stripped = _LEADING_FENCE_RE.sub("", stripped, count=1)
    stripped = _TRAILING_FENCE_RE.sub("", stripped, count=1)
    return stripped.strip()


def _parse_facts(raw_response: str, section_path: str) -> list[ExtractedFact]:
    """Parse an LLM response into validated ExtractedFact objects.

    A response that isn't valid JSON at all (or isn't a JSON array) yields
    an empty list for the whole section. Within a valid array, each item
    is validated independently -- one malformed item is skipped (and
    logged) without discarding the rest of the array.
    """
    cleaned = _strip_code_fences(raw_response)

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning(
            "Section %r: LLM response was not valid JSON; skipping section.",
            section_path,
        )
        return []

    if not isinstance(data, list):
        logger.warning(
            "Section %r: LLM response JSON was not a list; skipping section.",
            section_path,
        )
        return []

    facts: list[ExtractedFact] = []
    for item in data:
        try:
            facts.append(ExtractedFact.model_validate(item))
        except (ValidationError, TypeError) as exc:
            logger.warning(
                "Section %r: skipping malformed fact item: %s", section_path, exc
            )
    return facts


def _fuzzy_pattern(quote: str) -> Optional[re.Pattern]:
    """Build a regex that matches `quote` with whitespace runs relaxed.

    Splits the quote on whitespace, escapes each literal chunk, and joins
    them with `\\s+`. This lets a quote that was copied with normalized
    (or reflowed) whitespace/line-breaks still match against the source
    page text, which uses the PDF's original spacing.
    """
    chunks = [chunk for chunk in quote.split() if chunk]
    if not chunks:
        return None
    pattern = r"\s+".join(re.escape(chunk) for chunk in chunks)
    return re.compile(pattern)


def _locate_quote(quote: str, pages: list) -> tuple[Optional[int], int]:
    """Search `pages` (in order) for `quote`. Returns (page_number, char_offset),
    or (None, -1) if it can't be confidently located in any page.
    """
    if not quote:
        return None, -1

    # 1) Exact substring search, page by page, in reading order.
    for page in pages:
        idx = page.text.find(quote)
        if idx != -1:
            return page.page_number, idx

    # 2) Whitespace-tolerant fuzzy search, for quotes the LLM reflowed.
    pattern = _fuzzy_pattern(quote)
    if pattern is not None:
        for page in pages:
            match = pattern.search(page.text)
            if match is not None:
                return page.page_number, match.start()

    return None, -1


async def _extract_section(section, llm_client) -> list[LocatedFact]:
    """Run extraction for a single section. Never raises -- any failure
    (LLM call error, unparseable response, bad items) is logged and
    results in fewer (possibly zero) facts for this section, not an
    exception propagating to the caller.
    """
    try:
        prompt = build_extraction_prompt(section.text)
        raw_response = await llm_client.generate(prompt)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any provider/network failure
        logger.warning(
            "Section %r: LLM call failed: %s", section.section_path, exc
        )
        return []

    facts = _parse_facts(raw_response, section.section_path)

    located: list[LocatedFact] = []
    for fact in facts:
        page_number, char_offset = _locate_quote(fact.verbatim_quote, section.pages)
        if page_number is None:
            page_number = (
                section.pages[0].page_number if section.pages else section.start_page
            )
            char_offset = -1
        located.append(
            LocatedFact(
                fact=fact,
                section_path=section.section_path,
                page_number=page_number,
                char_offset=char_offset,
            )
        )
    return located


async def extract_facts_from_document(sections: list, llm_client) -> list:
    """Extract facts from every section of a document.

    Args:
        sections: list[Section] (see app.sectioning).
        llm_client: an LLMClient (duck-typed: any object with an async
            `generate(prompt: str) -> str` method).

    Returns:
        list[LocatedFact], for whichever sections succeeded. A section
        whose LLM call fails, or whose response can't be parsed at all,
        contributes zero facts rather than aborting the whole document.

    Concurrency: all sections are dispatched concurrently, bounded by
    `config.MAX_CONCURRENT_LLM_CALLS` via a semaphore, so a document with
    many sections doesn't fire an unbounded number of simultaneous LLM
    calls.
    """
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_LLM_CALLS)

    async def _bounded(section):
        async with semaphore:
            return await _extract_section(section, llm_client)

    results = await asyncio.gather(
        *(_bounded(section) for section in sections), return_exceptions=True
    )

    located_facts: list[LocatedFact] = []
    for section, result in zip(sections, results):
        if isinstance(result, Exception):
            # _extract_section already catches its own failures, so this
            # is a last-resort safety net for anything unexpected.
            logger.warning(
                "Section %r: extraction failed unexpectedly: %s",
                section.section_path,
                result,
            )
            continue
        located_facts.extend(result)

    return located_facts
