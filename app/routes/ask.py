"""Retrieval-only question answering over the fact store.

No LLM call: given a natural-language question, score every fact in
`store.get_all_facts()` with a small keyword + fuzzy-name heuristic and
return the top matches with their document filename and relationships,
so a user can see "here's what different PDFs say that's relevant, and
how those facts relate to each other."

Scoring weights (deliberately simple, documented here since there's no
single "correct" answer):

- entity_name token match: 3 points per question token found in the
  tokenized entity_name. entity_name is the strongest signal of
  relevance -- if the question is literally about the entity, matching
  facts should float to the top.
- attribute token match: 2 points per question token found in the
  tokenized attribute. Attribute is a precise field name and a good
  signal, but slightly weaker than entity_name because attributes are
  short and share connector-ish tokens more easily.
- verbatim_quote token match: 1 point per question token found in the
  tokenized quote. The quote is free text, so overlap is a much weaker
  (higher-recall, lower-precision) signal than the structured fields.
- fuzzy entity-name boost: up to +2 points, scaled by
  `difflib.SequenceMatcher.ratio()` between the question and
  entity_name (only applied above a 0.6 ratio threshold), so that
  slightly-different wording ("Delhivery Ltd" vs "Delhivery Limited")
  still surfaces the fact even if tokenization missed an exact word
  match. This mirrors the fuzzy-match approach in `app.matching`
  (SequenceMatcher ratio, corporate-suffix-aware normalization) without
  importing its private helpers.

Facts that score 0 are dropped entirely -- we'd rather show nothing
than noise.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from app import store

router = APIRouter()

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "what", "which", "how", "does", "do", "did",
    "has", "have",
}

# Same corporate-suffix stripping idea as app.matching, kept local so we
# don't reach into that module's private helpers.
_CORPORATE_SUFFIXES = [
    "limited", "ltd", "incorporated", "inc", "corporation", "corp",
    "company", "co", "plc", "llp", "pvt", "private",
]
_SUFFIX_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in _CORPORATE_SUFFIXES) + r")\b\.?",
    re.IGNORECASE,
)
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")

_FUZZY_NAME_THRESHOLD = 0.6
_FUZZY_NAME_MAX_BONUS = 2.0

# Scoring weights, see module docstring for rationale.
_ENTITY_NAME_TOKEN_WEIGHT = 3
_ATTRIBUTE_TOKEN_WEIGHT = 2
_QUOTE_TOKEN_WEIGHT = 1


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric characters, drop stopwords
    and empty pieces."""
    if not text:
        return []
    tokens = _TOKEN_SPLIT_RE.split(text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS]


def _normalize_name(name: str) -> str:
    if not name:
        return ""
    text = name.lower()
    text = _SUFFIX_RE.sub(" ", text)
    text = _NON_ALNUM_SPACE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _score_fact(question_tokens: List[str], question_raw: str, fact) -> float:
    score = 0.0

    entity_tokens = set(_tokenize(fact.entity_name))
    attribute_tokens = set(_tokenize(fact.attribute))
    quote_tokens = set(_tokenize(fact.verbatim_quote))

    for token in question_tokens:
        if token in entity_tokens:
            score += _ENTITY_NAME_TOKEN_WEIGHT
        if token in attribute_tokens:
            score += _ATTRIBUTE_TOKEN_WEIGHT
        if token in quote_tokens:
            score += _QUOTE_TOKEN_WEIGHT

    # Fuzzy entity-name boost: catches near-matches that exact
    # tokenization misses (e.g. slightly different wording/spelling).
    norm_question = _normalize_name(question_raw)
    norm_entity = _normalize_name(fact.entity_name)
    if norm_question and norm_entity:
        ratio = SequenceMatcher(None, norm_question, norm_entity).ratio()
        if ratio >= _FUZZY_NAME_THRESHOLD:
            score += ratio * _FUZZY_NAME_MAX_BONUS

    return score


def find_relevant_facts(question: str, limit: int = 10) -> dict:
    """Score every fact in the store against `question` and return the
    top `limit` matches, each enriched with its document filename and
    relationships.

    Returns a dict shaped like:
        {"matched_facts": [...], "total_matches": <int>}
    where `total_matches` is the count of facts with a non-zero score
    BEFORE the limit is applied.
    """
    question_tokens = _tokenize(question)
    facts = store.get_all_facts()

    scored = []
    for fact in facts:
        score = _score_fact(question_tokens, question, fact)
        if score > 0:
            scored.append((score, fact))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    total_matches = len(scored)

    top = scored[:limit]

    matched_facts = []
    for _score, fact in top:
        document = store.get_document(fact.document_id)
        relationships = store.list_relationships(fact_id=fact.id)
        matched_facts.append(
            {
                "fact": fact,
                "document_filename": document.filename if document else None,
                "relationships": relationships,
            }
        )

    return {"matched_facts": matched_facts, "total_matches": total_matches}


class AskRequest(BaseModel):
    question: str
    limit: int = 10  # max facts to return


@router.post("/ask")
async def ask(request: AskRequest):
    result = find_relevant_facts(request.question, limit=request.limit)
    return {
        "question": request.question,
        "matched_facts": result["matched_facts"],
        "total_matches": result["total_matches"],
    }
