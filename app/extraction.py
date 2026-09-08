"""Batched LLM fact extraction, orchestrated with bounded concurrency.

This module is stage 3 of the pipeline described in docs/ARCHITECTURE.md:
it takes the ``list[Section]`` produced by ``app.sectioning`` and asks an
``LLMClient`` to extract structured facts (see
``app.llm.prompts.build_batch_extraction_prompt`` and
``app.models.ExtractedFact``).

Sections are packed into batches (see ``MAX_BATCH_CHARS`` below) and each
batch is sent to the LLM in a single call, rather than one call per
section. This matters because this project's real Gemini API key is
rate-limited to a handful of requests per minute -- a document with dozens
of sections previously meant dozens of calls (and, under retries, minutes
of wall-clock time); batching cuts the call count by roughly the average
batch size (typically 5-10x for documents with many small-to-medium
sections).

Two concerns are layered on top of the raw LLM call:

* **Resilience** -- a single batch's LLM call failing (timeout, API
  error, malformed/non-JSON response, one bad item in an otherwise valid
  array) must never abort extraction for the rest of the document. Each
  failure mode is caught at the narrowest possible scope and logged as a
  warning; the caller always gets back whatever succeeded.
* **Evidence location** -- every ``ExtractedFact`` carries a
  ``verbatim_quote`` the LLM claims came from some section, but not
  *where* in the source PDF that text lives. We use the fact's returned
  ``section_path`` to look up the original ``Section`` object (so we
  search the correct section's pages even though a batch spans several),
  then search that section's own ``PageContent.text`` (the exact text the
  LLM was shown, reading order) to pin down a ``(page_number,
  char_offset)`` for it. This is inherently best-effort: LLMs sometimes
  normalize whitespace/line-breaks when they quote, so an exact substring
  search is tried first and a whitespace-tolerant regex search is tried
  second. If neither finds it, we still keep the fact (dropping it just
  because provenance search failed would throw away a real extraction)
  but degrade its location to the section's first page with
  ``char_offset = -1`` to signal "couldn't pinpoint it".
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
from app.llm.prompts import build_batch_extraction_prompt
from app.models import ExtractedFact

logger = logging.getLogger(__name__)

# Matches a leading ```/```json fence and/or a trailing ``` fence around an
# LLM response. LLMs asked to "respond with JSON" frequently wrap the
# answer in a markdown code block anyway; we strip that before json.loads
# rather than relying on the prompt alone to prevent it.
_LEADING_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*[ \t]*\r?\n?")
_TRAILING_FENCE_RE = re.compile(r"\r?\n?```[ \t]*$")

# Greedy packing budget (in characters of section .text) for how much goes
# into a single batched extraction prompt. Chosen to comfortably cut the
# number of LLM calls (this project's real Gemini key is rate-limited to
# 5 requests/minute, so fewer, larger calls matter a lot) while staying
# well inside typical LLM context windows even after the per-section
# prompt scaffolding (section headers, instructions, JSON schema) and
# response tokens are accounted for -- 12000 chars of source text is
# roughly 3-4k tokens, leaving ample room. A single section whose own
# text already exceeds this budget is not split; it simply becomes its
# own one-section batch (same behavior as the old one-call-per-section
# path for that section).
MAX_BATCH_CHARS = 12000


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


def _batch_sections(sections: list) -> list[list]:
    """Greedily pack sections (in order) into batches whose combined
    ``.text`` length stays under ``MAX_BATCH_CHARS``.

    A section is added to the current batch if doing so keeps the batch's
    total text length under the budget; otherwise the current batch is
    closed and a new one is started with that section. A single section
    longer than the budget on its own still becomes its own one-section
    batch (it is never split or dropped).
    """
    batches: list[list] = []
    current: list = []
    current_len = 0

    for section in sections:
        section_len = len(section.text)
        if current and current_len + section_len > MAX_BATCH_CHARS:
            batches.append(current)
            current = []
            current_len = 0
        current.append(section)
        current_len += section_len

    if current:
        batches.append(current)

    return batches


def _parse_facts(raw_response: str, batch_label: str) -> list[dict]:
    """Parse an LLM response into raw fact dicts (not yet validated as
    ExtractedFact, since each dict also carries a ``section_path`` field
    used only for routing to the right Section, not part of the
    ExtractedFact schema itself).

    A response that isn't valid JSON at all (or isn't a JSON array) yields
    an empty list for the whole batch. Within a valid array, each item is
    handled independently downstream -- one malformed item is skipped
    (and logged) without discarding the rest of the array.
    """
    cleaned = _strip_code_fences(raw_response)

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning(
            "Batch %r: LLM response was not valid JSON; skipping batch.",
            batch_label,
        )
        return []

    if not isinstance(data, list):
        logger.warning(
            "Batch %r: LLM response JSON was not a list; skipping batch.",
            batch_label,
        )
        return []

    return [item for item in data if isinstance(item, dict)]


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


def _locate_fact(item: dict, batch: list, section_by_path: dict) -> Optional[LocatedFact]:
    """Validate one raw fact dict and locate its evidence within the
    correct section's pages.

    The fact's ``section_path`` is used to look up the originating
    ``Section`` (from the full document, not just this batch) so evidence
    search happens against the right pages even though a batch spans
    multiple sections. If ``section_path`` is missing or doesn't match
    any real section in this batch -- a hallucinated or mismatched label
    -- we fall back to searching every section in the batch (in order)
    for the quote, rather than dropping the fact outright: a wrong label
    doesn't mean the extraction itself is wrong, and the quote search
    itself is exact-match-first so a false positive across sections is
    unlikely. Only if the fact fails ExtractedFact validation entirely
    is it skipped.
    """
    item = dict(item)  # avoid mutating the caller's dict
    section_path = item.pop("section_path", None)

    try:
        fact = ExtractedFact.model_validate(item)
    except (ValidationError, TypeError) as exc:
        logger.warning("Batch: skipping malformed fact item: %s", exc)
        return None

    section = section_by_path.get(section_path) if section_path else None
    if section is not None and section in batch:
        candidate_sections = [section]
        resolved_section_path = section.section_path
    else:
        if section_path:
            logger.warning(
                "Batch: fact declared section_path %r which doesn't match "
                "any section in its batch; falling back to searching all "
                "sections in the batch for its evidence quote.",
                section_path,
            )
        candidate_sections = batch
        resolved_section_path = section_path or (batch[0].section_path if batch else "")

    page_number = None
    char_offset = -1
    located_section = None
    for candidate in candidate_sections:
        page_number, char_offset = _locate_quote(fact.verbatim_quote, candidate.pages)
        if page_number is not None:
            located_section = candidate
            break

    if page_number is None:
        fallback_section = candidate_sections[0] if candidate_sections else None
        logger.warning(
            "Batch: could not locate verbatim_quote %r in any candidate "
            "section's source text; falling back to section start with "
            "char_offset=-1.",
            fact.verbatim_quote,
        )
        if fallback_section is not None:
            page_number = (
                fallback_section.pages[0].page_number
                if fallback_section.pages
                else fallback_section.start_page
            )
            resolved_section_path = fallback_section.section_path
        else:
            page_number = 0
        char_offset = -1
    else:
        resolved_section_path = located_section.section_path

    return LocatedFact(
        fact=fact,
        section_path=resolved_section_path,
        page_number=page_number,
        char_offset=char_offset,
    )


async def _extract_batch(batch: list, llm_client, section_by_path: dict) -> list[LocatedFact]:
    """Run extraction for a single batch of sections. Never raises -- any
    failure (LLM call error, unparseable response, bad items) is logged
    and results in fewer (possibly zero) facts for this batch, not an
    exception propagating to the caller.
    """
    batch_label = ", ".join(s.section_path for s in batch)
    try:
        prompt = build_batch_extraction_prompt([(s.section_path, s.text) for s in batch])
        raw_response = await llm_client.generate(prompt)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any provider/network failure
        logger.warning("Batch %r: LLM call failed: %s", batch_label, exc)
        return []

    items = _parse_facts(raw_response, batch_label)

    located: list[LocatedFact] = []
    for item in items:
        result = _locate_fact(item, batch, section_by_path)
        if result is not None:
            located.append(result)
    return located


async def extract_facts_from_document(sections: list, llm_client) -> list:
    """Extract facts from every section of a document.

    Args:
        sections: list[Section] (see app.sectioning).
        llm_client: an LLMClient (duck-typed: any object with an async
            `generate(prompt: str) -> str` method).

    Returns:
        list[LocatedFact], for whichever batches succeeded. A batch whose
        LLM call fails, or whose response can't be parsed at all,
        contributes zero facts rather than aborting the whole document.

    Sections are first packed into batches (see MAX_BATCH_CHARS) so that
    several sections can be extracted in a single LLM call. Concurrency
    is then bounded across BATCHES (not individual sections), by
    `config.MAX_CONCURRENT_LLM_CALLS` via a semaphore, so a document with
    many batches doesn't fire an unbounded number of simultaneous LLM
    calls.
    """
    if not sections:
        return []

    section_by_path = {s.section_path: s for s in sections}
    batches = _batch_sections(sections)

    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_LLM_CALLS)

    async def _bounded(batch):
        async with semaphore:
            return await _extract_batch(batch, llm_client, section_by_path)

    results = await asyncio.gather(
        *(_bounded(batch) for batch in batches), return_exceptions=True
    )

    located_facts: list[LocatedFact] = []
    for batch, result in zip(batches, results):
        if isinstance(result, Exception):
            # _extract_batch already catches its own failures, so this
            # is a last-resort safety net for anything unexpected.
            batch_label = ", ".join(s.section_path for s in batch)
            logger.warning(
                "Batch %r: extraction failed unexpectedly: %s",
                batch_label,
                result,
            )
            continue
        located_facts.extend(result)

    return located_facts
