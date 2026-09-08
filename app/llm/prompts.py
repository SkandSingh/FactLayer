"""Prompt templates for LLM-driven fact extraction and comparison."""

_EXTRACTION_PROMPT_TEMPLATE = """You are extracting factual claims from one section of a document for a
fact-checking system. Read the section text below and list every
meaningful numerical or semantic fact it states — a figure, a status, a
role, an identifier, a date, anything a reader might later want to verify
or compare against another document.

For each fact, output an object with these fields:
- entity_name: who/what the fact is about (a company, person, place, or
  economy — use the name as the document states it)
- entity_id: a canonical identifier the document itself provides for this
  entity, if any (e.g. a registration number, DIN, CIN, ISIN, PAN, ticker
  symbol). Use null if the document doesn't give one. Do not invent one.
- attribute: a short label for what kind of fact this is (e.g.
  "revenue_from_operations", "board_status", "registered_office",
  "real_gdp_growth"). Invent a new label if none of your prior labels fit
  — do not force-fit facts into categories that don't describe them.
- raw_value: the value exactly as stated (numbers, names, or text)
- unit: the unit as stated (e.g. "Rs Million", "per cent", "DIN"), or null
- time_scope: {{period_type: "point_in_time" | "span" | "unknown", start,
  end (ISO dates if you can tell), label (verbatim, e.g. "FY24"), vintage
  (e.g. "first advance estimate", "provisional", "final", or null)}}
- entity_scope: e.g. "standalone", "consolidated", or null if not
  applicable
- verbatim_quote: the exact sentence or table cell the fact came from —
  copy it precisely, do not paraphrase
- confidence: 0.0-1.0, your own confidence that this is a real,
  correctly-read fact (not a guess about whether it's true)

Only extract what the text actually states. Do not infer facts the
section doesn't say. If a number's meaning is ambiguous (e.g. a footnote
marker glued onto it), lower the confidence rather than silently
resolving the ambiguity.

Section text:
\"\"\"
{section_text}
\"\"\"

Respond with a JSON array of fact objects only.
"""

_COMPARISON_PROMPT_TEMPLATE = """You are comparing facts extracted from different documents (or different
parts of the same document) to decide how they relate. For each pair of
facts below that plausibly describe the same real-world thing, decide:

- "corroborates" — same fact, consistent with each other (allow for
  rounding/unit differences already normalized away)
- "contradicts" — genuinely conflicting values for the same entity,
  attribute, time period, and scope, with no reconciling explanation
- "reconciled_context" — apparently conflicting, but explained by a
  difference in time period, entity scope (e.g. standalone vs.
  consolidated), units, or estimate vintage (e.g. first advance estimate
  vs. final) — name which dimension explains it

For each relationship found, output: fact_id_a, fact_id_b, relation_type,
reconciled_dimension (only for reconciled_context: "time" | "scope" |
"units" | "estimate_vintage" | "other"), reasoning_text (plain-language
explanation citing both facts' verbatim quotes), confidence (0.0-1.0).

Do not force a relationship between facts that aren't actually about the
same thing — most pairs in a real document set aren't related, and
omitting them is the correct answer.

Facts:
{fact_digest_json}

Respond with a JSON array of relationship objects only.
"""


_BATCH_EXTRACTION_PROMPT_TEMPLATE = """You are extracting factual claims from MULTIPLE sections of a document for a
fact-checking system. Each section is delimited below with a header line of
the form "=== SECTION: <section_path> ===" followed by that section's text.
Read every section and list every meaningful numerical or semantic fact each
one states — a figure, a status, a role, an identifier, a date, anything a
reader might later want to verify or compare against another document.

For each fact, output an object with these fields:
- section_path: the exact section_path (copied verbatim from that section's
  "=== SECTION: ... ===" header) that this fact came from. This must exactly
  match one of the section labels given below — do not invent, paraphrase,
  or alter it.
- entity_name: who/what the fact is about (a company, person, place, or
  economy — use the name as the document states it)
- entity_id: a canonical identifier the document itself provides for this
  entity, if any (e.g. a registration number, DIN, CIN, ISIN, PAN, ticker
  symbol). Use null if the document doesn't give one. Do not invent one.
- attribute: a short label for what kind of fact this is (e.g.
  "revenue_from_operations", "board_status", "registered_office",
  "real_gdp_growth"). Invent a new label if none of your prior labels fit
  — do not force-fit facts into categories that don't describe them.
- raw_value: the value exactly as stated (numbers, names, or text)
- unit: the unit as stated (e.g. "Rs Million", "per cent", "DIN"), or null
- time_scope: {{period_type: "point_in_time" | "span" | "unknown", start,
  end (ISO dates if you can tell), label (verbatim, e.g. "FY24"), vintage
  (e.g. "first advance estimate", "provisional", "final", or null)}}
- entity_scope: e.g. "standalone", "consolidated", or null if not
  applicable
- verbatim_quote: the exact sentence or table cell the fact came from —
  copy it precisely, do not paraphrase
- confidence: 0.0-1.0, your own confidence that this is a real,
  correctly-read fact (not a guess about whether it's true)

Only extract what the text actually states. Do not infer facts a section
doesn't say. If a number's meaning is ambiguous (e.g. a footnote marker
glued onto it), lower the confidence rather than silently resolving the
ambiguity.

Treat each section as an independent source: never combine, cross-reference,
or attribute a fact from one section's text to a different section's
section_path. If a section has no extractable facts at all, it is correct
and expected to omit it entirely from the output — do not invent facts to
give every section an entry.

Sections:
{sections_block}

Respond with a single JSON array of fact objects (covering all sections
together) only.
"""


def build_extraction_prompt(section_text: str) -> str:
    """Build the fact-extraction prompt for a single document section."""
    return _EXTRACTION_PROMPT_TEMPLATE.format(section_text=section_text)


def build_batch_extraction_prompt(sections: list) -> str:
    """Build the fact-extraction prompt covering multiple document sections
    in a single call.

    Args:
        sections: list of (section_path, section_text) tuples. Each is
            rendered as its own clearly delimited, labeled block so the
            model can attribute facts back to the right section via the
            required `section_path` field on each returned fact.
    """
    blocks = [
        f"=== SECTION: {section_path} ===\n{section_text}\n"
        for section_path, section_text in sections
    ]
    sections_block = "\n".join(blocks)
    return _BATCH_EXTRACTION_PROMPT_TEMPLATE.format(sections_block=sections_block)


def build_comparison_prompt(fact_digest_json: str) -> str:
    """Build the fact-comparison prompt for a digest of extracted facts."""
    return _COMPARISON_PROMPT_TEMPLATE.format(fact_digest_json=fact_digest_json)
