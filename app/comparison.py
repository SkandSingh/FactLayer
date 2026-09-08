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
# Observed directly on real data: a single document with ~2,300 facts of
# ITS OWN produced a digest large enough to blow every pooled provider's
# per-minute input-token quota, so the ENTIRE comparison call failed and
# the document got zero relationships -- not because nothing was related,
# but because the one giant call never had a chance to succeed. Note this
# means `new_facts` alone can already exceed this budget; chunking must
# cover `new_facts`, not just the candidate set (see `_chunk_digest`).
#
# 25,000 chars is deliberately larger than the smallest pooled provider's
# comfortable per-call budget (observed: Groq's free tier caps around
# 8,000 tokens/minute per key, roughly 30,000 chars) -- a chunk this size
# occasionally overflowing one provider is fine, since `RoundRobinLLMClient`
# already fails over to the next pooled client on any failure. Sized to
# cut total call count substantially versus a more conservative budget,
# while staying well under Gemini's much larger 250,000-token/minute cap.
MAX_DIGEST_CHARS = 25000


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


def _chunk_digest(new_facts: list, candidates: list, max_chars: int) -> list:
    """Split the comparison pool into digest-sized groups (each group is a
    complete `list[Fact]` ready to serialize and send as one call).

    Two regimes, chosen by whether `new_facts` alone fits the budget:

    * **Common case -- `new_facts` fits in `max_chars` on its own**
      (true for almost every upload: a document contributes some new
      facts, checked against a potentially much larger existing store).
      `new_facts` is repeated in full in every chunk, each paired with a
      different slice of `candidates`. This is thorough -- every new fact
      is compared against every candidate, just spread across multiple
      calls -- and cheap, since `new_facts` is the small side here.

    * **Fallback -- `new_facts` itself exceeds `max_chars`** (observed
      directly on real data: one document had ~2,300 facts of its own).
      Repeating an already-oversized list in full per chunk is impossible.
      `new_facts` and `candidates` are interleaved into one pool and THAT
      pool is chunked, so most chunks still carry a mix of both. This is
      best-effort, not exhaustive: two facts landing in different chunks
      aren't directly compared in this pass (though they may still be
      compared later, when a subsequent document's own comparison pass
      pulls both in as candidates). Exhaustive pairing here would need
      `len(new_chunks) * len(candidate_chunks)` calls -- thousands, for a
      document this large -- trading one scaling problem for a worse one.
      This mirrors the same "batch what fits, accept partial coverage
      over a call that can never succeed" trade-off `app.extraction`
      already makes for oversized documents.
    """
    new_facts_size = len(json.dumps([_fact_to_digest_dict(f) for f in new_facts]))

    if new_facts_size < max_chars:
        chunks: list[list] = []
        current: list = []
        current_size = new_facts_size

        for candidate in candidates:
            candidate_size = len(json.dumps(_fact_to_digest_dict(candidate)))
            if current and current_size + candidate_size > max_chars:
                chunks.append(list(new_facts) + current)
                current = []
                current_size = new_facts_size
            current.append(candidate)
            current_size += candidate_size

        if current or not chunks:
            chunks.append(list(new_facts) + current)

        return chunks

    interleaved: list = []
    i = j = 0
    while i < len(new_facts) or j < len(candidates):
        if i < len(new_facts):
            interleaved.append(new_facts[i])
            i += 1
        if j < len(candidates):
            interleaved.append(candidates[j])
            j += 1

    chunks = []
    current = []
    current_size = 0

    for fact in interleaved:
        fact_size = len(json.dumps(_fact_to_digest_dict(fact)))
        if current and current_size + fact_size > max_chars:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(fact)
        current_size += fact_size

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
    the digest is chunked (see `_chunk_digest`), one chunk's failure no
    longer costs the WHOLE document its comparison pass the way a single
    oversized call used to (observed directly on real data: one
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

    chunks = _chunk_digest(new_facts, candidates, MAX_DIGEST_CHARS)

    # A given fact pair is only found once (each chunk is a disjoint
    # subset), but dedupe anyway -- cheap, and guards against the LLM
    # itself ever reporting the same pair twice within one response.
    seen_pairs: set = set()
    inserted: List[Relationship] = []

    for digest_facts in chunks:
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

            # Guards across separate `compare_new_document_facts` calls,
            # not just within this one: a candidate pair surfaced here may
            # already have been recorded by an earlier document's
            # comparison pass (observed directly on real data: the same
            # pair rediscovered and re-inserted three times across three
            # documents' separate comparison calls).
            if store.relationship_exists(validated["fact_id_a"], validated["fact_id_b"]):
                continue

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
