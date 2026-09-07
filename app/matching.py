"""Pure-code candidate prefilter for cross-document fact matching.

No LLM calls, no I/O. `find_candidates` narrows the entire existing fact
store down to a short list of facts that are *plausibly* related to a
new fact, before the expensive LLM comparison step decides whether each
pair actually corroborates, contradicts, or is unrelated.

Recall over precision, deliberately: it is fine (expected, even) for
this prefilter to pass through some false-positive candidates -- the
downstream LLM comparison step will correctly say "not related" for
those. What is NOT fine is silently dropping a real match, because that
pair would then never even be shown to the LLM. Every criterion below is
therefore intentionally loose.

This module exists because exact-identifier-only prefiltering (matching
solely on `entity_id`) silently drops real corroborations whenever two
documents word the same entity differently -- e.g. "Delhivery Limited"
in one filing vs "Delhivery Ltd." in another, with no shared entity_id
at all. That is the same class of problem as "differently written
addresses may refer to the same place" and is a documented limitation
of a purely exact-match prefilter.

A candidate `existing_fact` is included if ANY of the following hold
(never against itself, when `new_fact.id` is set):

1. Exact identifier match (via `normalize_identifier`, which already
   collapses CIN listing-status prefixes like the U/L distinction).
2. Attribute keyword overlap: tokenize both `attribute` strings on
   non-alphanumeric characters, lowercase, drop a small stoplist of
   trivial connector words, and require at least one shared token.
3. Fuzzy entity name match: normalize both names (lowercase, strip
   common corporate suffixes, strip punctuation, collapse whitespace)
   and compare with `difflib.SequenceMatcher.ratio()` against a
   threshold.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List

from app.normalization import normalize_identifier

# ---------------------------------------------------------------------------
# Criterion 2: attribute keyword overlap
# ---------------------------------------------------------------------------

# Trivial connector words that appear inside attribute names (e.g.
# "revenue_from_operations") but carry no discriminating meaning on
# their own. A shared token has to be something more specific than one
# of these to count as overlap -- otherwise "revenue_from_operations"
# and "profit_from_sale" would falsely match on "from" alone.
_ATTRIBUTE_STOPWORDS = {"from", "of", "the", "a", "an", "in", "on", "for", "to", "and", "or"}

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenize_attribute(attribute: str) -> set:
    """Split an attribute string on non-alphanumeric characters, lowercase,
    and drop empty pieces and stopwords. e.g. "revenue_from_operations"
    -> {"revenue", "operations"}."""
    if not attribute:
        return set()
    tokens = _TOKEN_SPLIT_RE.split(attribute.lower())
    return {t for t in tokens if t and t not in _ATTRIBUTE_STOPWORDS}


def _attribute_overlap(attr_a: str, attr_b: str) -> bool:
    tokens_a = _tokenize_attribute(attr_a)
    tokens_b = _tokenize_attribute(attr_b)
    return bool(tokens_a & tokens_b)


# ---------------------------------------------------------------------------
# Criterion 3: fuzzy entity name match
# ---------------------------------------------------------------------------

# Common corporate-entity suffixes seen across Indian and international
# filings. Stripped before comparison so "Delhivery Limited" and
# "Delhivery Ltd." normalize to the same core name "delhivery".
_CORPORATE_SUFFIXES = [
    "limited",
    "ltd",
    "incorporated",
    "inc",
    "corporation",
    "corp",
    "company",
    "co",
    "plc",
    "llp",
    "pvt",
    "private",
]

_SUFFIX_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in _CORPORATE_SUFFIXES) + r")\b\.?",
    re.IGNORECASE,
)
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# Similarity threshold for SequenceMatcher.ratio() on normalized entity
# names. Chosen at 0.8 (rather than a looser value like 0.6-0.7) because
# entity names are short proper nouns -- at that length, character-level
# edit-similarity is a good, cheap proxy for "same name, different
# spelling/formatting" (e.g. "delhivery" vs "delhivery" post-suffix-strip
# is a perfect match; genuine near-duplicates like transliteration or
# abbreviation differences still land at 0.8-0.95), while staying well
# above the ratio two genuinely different short company names tend to
# share by chance (two unrelated 8-10 character names rarely exceed
# ~0.5-0.6 on SequenceMatcher). SequenceMatcher is preferred over a
# token-set Jaccard ratio here specifically because company names are
# often ONE token after suffix-stripping (e.g. "delhivery"), where a
# token-set comparison degenerates to exact-match-or-nothing and can't
# express partial similarity; character-level ratio still gracefully
# handles minor spelling/spacing differences within that single token.
_FUZZY_NAME_THRESHOLD = 0.8


def _normalize_entity_name(name: str) -> str:
    """Lowercase, strip common corporate suffixes, strip punctuation, and
    collapse whitespace, so "Delhivery Limited" and "Delhivery Ltd."
    both normalize to "delhivery"."""
    if not name:
        return ""
    text = name.lower()
    text = _SUFFIX_RE.sub(" ", text)
    text = _NON_ALNUM_SPACE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _fuzzy_name_match(name_a: str, name_b: str) -> bool:
    norm_a = _normalize_entity_name(name_a)
    norm_b = _normalize_entity_name(name_b)
    if not norm_a or not norm_b:
        return False
    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    return ratio >= _FUZZY_NAME_THRESHOLD


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def find_candidates(new_fact, existing_facts: List) -> List:
    """Return the subset of `existing_facts` plausibly related to
    `new_fact`, for the downstream LLM comparison step.

    `new_fact` may be a `Fact` or an `ExtractedFact`-like object -- only
    `entity_name`, `entity_id`, and `attribute` are read. If `new_fact`
    has an `id` attribute, an `existing_fact` with the same `id` is
    skipped (never match a fact against itself).

    This is a prefilter, not a final decision: recall matters far more
    than precision here, so a candidate is included if ANY of the three
    documented criteria hold (see module docstring).
    """
    new_id = getattr(new_fact, "id", None)
    new_identifier = normalize_identifier(new_fact.entity_id)

    candidates = []
    for existing_fact in existing_facts:
        if new_id is not None and getattr(existing_fact, "id", None) == new_id:
            continue

        # Criterion 1: exact identifier match.
        existing_identifier = normalize_identifier(existing_fact.entity_id)
        if new_identifier is not None and new_identifier == existing_identifier:
            candidates.append(existing_fact)
            continue

        # Criterion 2: attribute keyword overlap.
        if _attribute_overlap(new_fact.attribute, existing_fact.attribute):
            candidates.append(existing_fact)
            continue

        # Criterion 3: fuzzy entity name match.
        if _fuzzy_name_match(new_fact.entity_name, existing_fact.entity_name):
            candidates.append(existing_fact)
            continue

    return candidates
