"""Digest-style LLM fact comparison, stage 4 of the pipeline described in
docs/ARCHITECTURE.md.

This module runs *after* a new document's facts have already been
extracted and inserted into the store (so every new fact has a real
``id``). For each new fact it asks ``app.matching.find_candidates`` for a
short list of plausibly-related existing facts (pure code, no LLM), unions
those candidate sets across the whole new-document batch, and sends one
combined digest -- the new facts themselves plus the deduped candidates --
to the LLM in a single call via ``app.llm.prompts.build_comparison_prompt``.

The new document's own facts are included in the digest alongside the
cross-document candidates so that within-document consistency (e.g. two
sections of the same filing quoting different figures for the same
metric) gets checked too, not just cross-document corroboration.

Resilience follows the same pattern as ``app.extraction``: a failed or
unparseable LLM call degrades to an empty result rather than raising, and
an individual malformed/hallucinated relationship item is skipped (and
logged) without discarding the rest of the batch.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from app import store
from app.llm.prompts import build_comparison_prompt
from app.matching import find_candidates
from app.models import Relationship

logger = logging.getLogger(__name__)

# Same fence-stripping approach as app.extraction._strip_code_fences --
# duplicated locally (rather than imported) so this module doesn't take a
# dependency on another module's private helpers.
_LEADING_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*[ \t]*\r?\n?")
_TRAILING_FENCE_RE = re.compile(r"\r?\n?```[ \t]*$")

_VALID_RELATION_TYPES = {"corroborates", "contradicts", "reconciled_context"}

# Below this many total facts in the digest, there is nothing to compare
# (a single fact can't corroborate or contradict itself), so the LLM call
# is skipped entirely.
_MIN_DIGEST_SIZE = 2

# Upper bound (JSON-serialized characters) on one comparison call's digest.
# Observed directly on real data: a single document with ~2,300 facts
# produced a digest large enough to blow every pooled provider's per-minute
# input-token quota, so the ENTIRE comparison call failed and the document
# got zero relationships -- not because nothing was related, but because
# the one giant call never had a chance to succeed. Chunking the candidate
# set (see `_chunk_candidates`) keeps each call's digest to a size that
# reliably fits within free-tier provider quotas, at the cost of the new
# document's own facts being resent in every chunk (harmless overhead,
# deduped by `compare_new_document_facts` before insert).
MAX_DIGEST_CHARS = 10000


def _strip_code_fences(text: str) -> str:
    """Strip a single leading/trailing markdown code fence, if present."""
    stripped = text.strip()
    stripped = _LEADING_FENCE_RE.sub("", stripped, count=1)
    stripped = _TRAILING_FENCE_RE.sub("", stripped, count=1)
    return stripped.strip()


def _fact_to_digest_dict(fact) -> dict:
    """Serialize a Fact to the compact dict shape sent to the LLM."""
    return {
        "id": fact.id,
        "entity_name": fact.entity_name,
        "entity_id": fact.entity_id,
        "attribute": fact.attribute,
        "raw_value": fact.raw_value,
        "unit": fact.unit,
        "time_scope": fact.time_scope.model_dump(),
        "entity_scope": fact.entity_scope,
        "verbatim_quote": fact.verbatim_quote,
        "confidence": fact.confidence,
    }


def _collect_candidates(new_facts: list, existing_facts: list) -> list:
    """Union candidate sets across all new facts, deduped by id, excluding
    anything already among the new facts themselves (those are added to
    every chunk's digest separately by the caller, so this avoids
    double-counting).
    """
    new_ids = {f.id for f in new_facts}
    deduped: dict = {}
    for new_fact in new_facts:
        for candidate in find_candidates(new_fact, existing_facts):
            candidate_id = getattr(candidate, "id", None)
            if candidate_id is None or candidate_id in new_ids:
                continue
            deduped.setdefault(candidate_id, candidate)
    return list(deduped.values())


def _chunk_candidates(new_facts: list, candidates: list, max_chars: int) -> list:
    """Split `candidates` into groups such that `new_facts + group`,
    JSON-serialized, stays under `max_chars` per group.

    `new_facts` are NOT chunked -- every chunk carries the full new-facts
    set (so within-new-document consistency is still checked in every
    call) plus as many candidates as fit. Always yields at least one
    chunk (possibly with zero candidates) so a document with no
    candidates at all still gets its own facts checked against each
    other.
    """
    base_size = len(json.dumps([_fact_to_digest_dict(f) for f in new_facts]))
    chunks: list[list] = []
    current: list = []
    current_size = base_size

    for candidate in candidates:
        candidate_size = len(json.dumps(_fact_to_digest_dict(candidate)))
        if current and current_size + candidate_size > max_chars:
            chunks.append(current)
            current = []
            current_size = base_size
        current.append(candidate)
        current_size += candidate_size

    if current or not chunks:
        chunks.append(current)

    return chunks


def _parse_relationship_items(raw_response: str) -> list:
    """Parse an LLM response into a list of raw dict items.

    Returns [] if the response isn't valid JSON at all, or isn't a JSON
    array -- callers treat that as "nothing to insert" rather than an
    error to propagate.
    """
    cleaned = _strip_code_fences(raw_response)

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Comparison: LLM response was not valid JSON; skipping.")
        return []

    if not isinstance(data, list):
        logger.warning("Comparison: LLM response JSON was not a list; skipping.")
        return []

    return data


def _validate_relationship_item(item, valid_ids: set) -> Optional[dict]:
    """Validate one raw relationship item against the digest's own fact
    ids. Returns a normalized dict ready for `store.insert_relationship`,
    or None (with a logged warning) if the item is malformed or
    references an id that wasn't actually in the digest sent to the LLM
    (guards against a hallucinated id).
    """
    if not isinstance(item, dict):
        logger.warning("Comparison: skipping non-dict relationship item: %r", item)
        return None

    try:
        fact_id_a = item["fact_id_a"]
        fact_id_b = item["fact_id_b"]
        relation_type = item["relation_type"]
        reasoning_text = item["reasoning_text"]
        confidence = item["confidence"]
    except KeyError as exc:
        logger.warning(
            "Comparison: skipping relationship item missing field %s: %r", exc, item
        )
        return None

    reconciled_dimension = item.get("reconciled_dimension")

    if relation_type not in _VALID_RELATION_TYPES:
        logger.warning(
            "Comparison: skipping relationship item with invalid relation_type %r: %r",
            relation_type,
            item,
        )
        return None

    if not isinstance(fact_id_a, int) or not isinstance(fact_id_b, int):
        logger.warning(
            "Comparison: skipping relationship item with non-integer fact id(s): %r",
            item,
        )
        return None

    if fact_id_a == fact_id_b:
        logger.warning(
            "Comparison: skipping relationship item comparing a fact to itself: %r",
            item,
        )
        return None

    if fact_id_a not in valid_ids or fact_id_b not in valid_ids:
        logger.warning(
            "Comparison: skipping relationship item referencing id(s) not in the "
            "digest sent to the LLM (likely hallucinated): %r",
            item,
        )
        return None

    return {
        "fact_id_a": fact_id_a,
        "fact_id_b": fact_id_b,
        "relation_type": relation_type,
        "reconciled_dimension": reconciled_dimension,
        "reasoning_text": reasoning_text,
        "confidence": confidence,
    }


async def compare_new_document_facts(new_facts: list, llm_client) -> List[Relationship]:
    """Compare a newly-inserted document's facts against the existing
    store (and against each other) and persist any relationships the LLM
    finds.

    Args:
        new_facts: list[Fact] -- the newly-inserted facts for one
            document, already carrying real `.id` values from the store.
        llm_client: an LLMClient (duck-typed: any object with an async
            `generate(prompt: str) -> str` method).

    Returns:
        list[Relationship] -- the fully-populated Relationship objects
        that were inserted (fetched back from the store after insert, so
        they carry the real id/created_at). Empty if there was nothing to
        compare, the LLM call failed, the response was unparseable, or
        every candidate relationship failed validation.

    Never raises: any LLM call failure or malformed response degrades that
    one chunk to zero relationships rather than propagating -- and, since
    the digest is chunked (see `_chunk_candidates`), one chunk's failure
    no longer costs the WHOLE document its comparison pass the way a
    single oversized call used to (observed directly on real data: one
    document's full-digest call blew every pooled provider's token quota
    and the document got zero relationships even though it plausibly had
    real ones).
    """
    if not new_facts:
        return []

    existing_facts = store.get_all_facts()
    candidates = _collect_candidates(new_facts, existing_facts)

    if len(new_facts) + len(candidates) < _MIN_DIGEST_SIZE:
        return []

    chunks = _chunk_candidates(new_facts, candidates, MAX_DIGEST_CHARS)

    # Every chunk resends the full `new_facts` set, so a relationship
    # between two new facts could be independently (re)found in more than
    # one chunk -- dedupe by the unordered fact-id pair before inserting.
    seen_pairs: set = set()
    inserted: List[Relationship] = []

    for chunk_candidates in chunks:
        digest_facts = list(new_facts) + chunk_candidates
        if len(digest_facts) < _MIN_DIGEST_SIZE:
            continue

        digest_json = json.dumps([_fact_to_digest_dict(f) for f in digest_facts])
        prompt = build_comparison_prompt(digest_json)

        try:
            raw_response = await llm_client.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any provider/network failure
            logger.warning(
                "Comparison: LLM call failed for one digest chunk (%d facts); "
                "skipping this chunk, other chunks still proceed: %s",
                len(digest_facts),
                exc,
            )
            continue

        items = _parse_relationship_items(raw_response)
        if not items:
            continue

        valid_ids = {f.id for f in digest_facts}

        for item in items:
            validated = _validate_relationship_item(item, valid_ids)
            if validated is None:
                continue

            pair_key = tuple(sorted((validated["fact_id_a"], validated["fact_id_b"])))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            relationship_id = store.insert_relationship(
                fact_id_a=validated["fact_id_a"],
                fact_id_b=validated["fact_id_b"],
                relation_type=validated["relation_type"],
                reconciled_dimension=validated["reconciled_dimension"],
                reasoning_text=validated["reasoning_text"],
                confidence=validated["confidence"],
            )
            relationship = store.get_relationship(relationship_id)
            if relationship is not None:
                inserted.append(relationship)

    return inserted
