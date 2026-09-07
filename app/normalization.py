"""Pure-Python normalization helpers for the FactLayer pipeline.

No LLM calls, no I/O. These functions turn raw extracted strings (as
pulled out of Indian corporate/economic filings) into values that can be
safely compared across documents: a single float base unit for
quantities, ISO date strings for fiscal years and point-in-time dates,
and a canonical string for entity identifiers.

Design choices (documented per function below):

- `normalize_quantity` returns values in *absolute base units* — plain
  rupees (or whatever currency unit was already implicit in the raw
  number) for money, and a plain float (not divided by 100) for
  percentages. e.g. "8,142" with unit "Cr" -> 81,420,000,000.0 (absolute
  rupees), and "6.5" with unit "per cent" -> 6.5 (not 0.065). This keeps
  same-unit-family comparisons (money vs money, percent vs percent)
  trivial, while cross-family comparisons were never meaningful anyway.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from dateutil import parser as dateutil_parser

# ---------------------------------------------------------------------------
# normalize_quantity
# ---------------------------------------------------------------------------

# Multipliers for Indian and international magnitude words. Matched
# case-insensitively against whatever text is available -- either
# glued onto the raw number's tail (e.g. "8142Cr", no space) or living
# in the separate `unit` string (e.g. "Rs Million"). We deliberately use
# letter-only lookaround (not `\b`) as the word boundary: `\b` treats
# digits and letters as the same "word" character class, so it would
# NOT match "cr" in "8142cr" (no transition it recognizes between "2"
# and "c"). Since unit words glued directly onto digits are exactly the
# case we need to support, the boundary is defined relative to
# surrounding LETTERS only -- a digit immediately before/after the word
# still counts as a valid boundary, but another letter does not (so
# "cr" won't spuriously match inside "crore", and "mn" won't match
# inside a longer alphabetic token).
_MULTIPLIER_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"(?<![a-zA-Z])crores?(?![a-zA-Z])", re.IGNORECASE), 1e7),
    (re.compile(r"(?<![a-zA-Z])cr(?![a-zA-Z])", re.IGNORECASE), 1e7),
    (re.compile(r"(?<![a-zA-Z])lakhs?(?![a-zA-Z])", re.IGNORECASE), 1e5),
    (re.compile(r"(?<![a-zA-Z])lacs?(?![a-zA-Z])", re.IGNORECASE), 1e5),
    (re.compile(r"(?<![a-zA-Z])billions?(?![a-zA-Z])", re.IGNORECASE), 1e9),
    (re.compile(r"(?<![a-zA-Z])bn(?![a-zA-Z])", re.IGNORECASE), 1e9),
    (re.compile(r"(?<![a-zA-Z])millions?(?![a-zA-Z])", re.IGNORECASE), 1e6),
    (re.compile(r"(?<![a-zA-Z])mn(?![a-zA-Z])", re.IGNORECASE), 1e6),
]

_PERCENT_PATTERN = re.compile(r"per\s*cent|percent|%", re.IGNORECASE)

# A "plausible number" after stripping commas/parens/sign/unit-words:
# optional digits, optional decimal point, optional digits.
_NUMBER_RE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")


def _find_multiplier(*texts: Optional[str]) -> float:
    """Search the given strings (raw_value and/or unit) for a magnitude
    word and return its multiplier, or 1.0 if none is found."""
    for text in texts:
        if not text:
            continue
        for pattern, mult in _MULTIPLIER_PATTERNS:
            if pattern.search(text):
                return mult
    return 1.0


def _is_percent(*texts: Optional[str]) -> bool:
    for text in texts:
        if text and _PERCENT_PATTERN.search(text):
            return True
    return False


def _strip_footnote_marker(numeric_str: str) -> str:
    """Heuristic: detect and strip a footnote-marker digit glued onto the
    end of a number by the upstream text extraction (e.g. a superscript
    "4" rendered inline as "6.5" + "4" -> "6.54" in the extracted text
    layer, with no separating space).

    Heuristic used (conservative by design):
      A number with a decimal point and MORE THAN 2 digits after the
      decimal point is treated as "decimal point + value + one glued-on
      footnote digit", and the LAST digit is stripped. e.g. "6.54" as
      three digits after the point would trigger this ("6.541" -> "6.54"),
      but plain "6.54" (exactly two decimal digits) is left alone.

    Why more-than-2 and not >=1: Indian financial figures very commonly
    carry exactly two decimal places (paise-level precision, or percent
    figures like "6.54 per cent" meaning six-point-five-four). Stripping
    on ANY decimal-digit count would silently corrupt every genuine
    two-decimal number (a far more common case than a glued footnote
    marker), which is an unacceptable false-positive rate. Requiring
    three-or-more decimal digits is a defensible middle ground: Indian
    filings essentially never report a base metric to 3+ decimal places,
    so 3+ digits after the point is itself already a strong signal that
    something is glued on, independent of the footnote-specific example
    in this docstring.

    Known limitation (documented, not fixed): this heuristic CANNOT catch
    the single most treacherous case explicitly called out in this
    project's docs -- "6.5" + footnote "4" glued together as "6.54" --
    because "6.54" has exactly two digits after the decimal point and is
    indistinguishable, by digit-shape alone, from a genuine value of
    six-point-five-four. There is no safe general fix for that specific
    shape without either an out-of-band signal (e.g. the original
    superscript/unicode footnote glyph, or cross-document corroboration
    showing "6.5" elsewhere) or accepting a much higher false-positive
    rate against real two-decimal numbers. This is intentionally left as
    an honest, documented limitation -- see the test for the exact
    scenario and the assertion of the actual (unfixed) behavior.
    """
    if "." in numeric_str:
        int_part, _, frac_part = numeric_str.partition(".")
        if len(frac_part) > 2:
            return f"{int_part}.{frac_part[:-1]}"
    return numeric_str


def normalize_quantity(raw_value: str, unit: str | None) -> float | None:
    """Parse an Indian-financial-document number + unit into a single
    base-unit float.

    Base unit convention (see module docstring): absolute currency units
    for money (i.e. a stated multiplier like "Cr" or "Million" IS
    applied), and a plain percent-as-float (i.e. NOT divided by 100) for
    percentages.

    Handles: Indian-style comma grouping ("81,415.38"), parenthesized
    negatives ("(1,234.5)" -> -1234.5), magnitude words attached to the
    number or living separately in `unit` ("Cr"/"Crore", "Lakh"/"Lac",
    "Million"/"Mn", "Billion"/"Bn", case-insensitive), and plain
    percentages (unit contains "per cent"/"%"/"percent" -> no
    multiplier). Also applies a conservative footnote-marker-stripping
    heuristic -- see `_strip_footnote_marker` for exactly what it does
    and does not catch.

    Returns None if the string cannot be confidently parsed as a number.
    """
    if raw_value is None:
        return None

    text = raw_value.strip()
    if not text:
        return None

    # Parenthesized negative, e.g. "(1,234.5)" -> negative.
    is_negative = False
    match = re.match(r"^\((.*)\)$", text)
    if match:
        is_negative = True
        text = match.group(1).strip()

    # An explicit leading sign (rare, but handle it before we strip
    # anything else).
    if text.startswith("-"):
        is_negative = True
        text = text[1:].strip()
    elif text.startswith("+"):
        text = text[1:].strip()

    # Determine multiplier / percent-ness from the combined raw text and
    # the separate unit string BEFORE we strip words out of `text`,
    # since the multiplier word may be glued onto the number itself
    # (e.g. "8142Cr").
    percent = _is_percent(text, unit)
    multiplier = 1.0 if percent else _find_multiplier(text, unit)

    # Strip currency symbols, magnitude words, percent signs, and any
    # remaining letters/whitespace from the numeric text, leaving just
    # digits, comma separators, and an optional decimal point.
    numeric_candidate = re.sub(r"[^\d.,]", "", text)
    if not numeric_candidate:
        return None

    # Remove Indian/international comma grouping.
    numeric_candidate = numeric_candidate.replace(",", "")

    # Guard against multiple decimal points or otherwise malformed
    # leftovers (e.g. "1.2.3") -- not confidently parseable.
    if numeric_candidate.count(".") > 1:
        return None

    if not _NUMBER_RE.match(numeric_candidate):
        return None

    numeric_candidate = _strip_footnote_marker(numeric_candidate)

    try:
        value = float(numeric_candidate)
    except ValueError:
        return None

    if is_negative:
        value = -value

    return value * multiplier


# ---------------------------------------------------------------------------
# parse_fiscal_year
# ---------------------------------------------------------------------------


@dataclass
class FiscalYear:
    """India's Apr-Mar fiscal year. `start`/`end` are ISO date strings.

    FY24 (or "FY 2023-24") means the fiscal year ENDING March 2024, i.e.
    1 Apr 2023 - 31 Mar 2024.
    """

    label: str
    start: str
    end: str


_FY_2DIGIT_RE = re.compile(r"^FY\s*['’]?\s*(\d{2})$", re.IGNORECASE)
_FY_4DIGIT_RE = re.compile(r"^FY\s*(\d{4})$", re.IGNORECASE)
_FY_RANGE_RE = re.compile(
    r"^(?:FY\s*)?(\d{4})\s*[-/–—]\s*(\d{2,4})$", re.IGNORECASE
)


def _year_from_2digit(yy: int) -> int:
    """Map a 2-digit year to a 4-digit one. Pivot at 50: 00-49 -> 2000s,
    50-99 -> 1900s. This is a standard, if arbitrary, windowing choice;
    India's fiscal filings in scope for this project are all firmly in
    the 2000s so the pivot point itself rarely matters in practice."""
    return 2000 + yy if yy < 50 else 1900 + yy


def parse_fiscal_year(label: str) -> Optional[dict]:
    """Parse an Indian fiscal-year label into start/end ISO dates.

    Accepts:
      - "FY24"          (2-digit, ending-year form)
      - "FY2024"        (4-digit, ending-year form)
      - "FY 2023-24"    (hyphenated range, end year 2-digit)
      - "FY2023-2024"   (hyphenated range, end year 4-digit)
      - "2023-24"       (bare hyphenated range, no "FY" prefix)

    Returns a dict {"label": <original>, "start": "YYYY-04-01",
    "end": "YYYY-03-31"} where the end year is the fiscal year's naming
    year (FY24 -> ends in 2024). Returns None if the label cannot be
    confidently parsed.
    """
    if not label:
        return None

    text = label.strip()

    match = _FY_RANGE_RE.match(text)
    if match:
        start_year_raw, end_year_raw = match.groups()
        start_year = int(start_year_raw)
        if len(end_year_raw) == 2:
            end_year = _year_from_2digit(int(end_year_raw))
        else:
            end_year = int(end_year_raw)
        # Sanity check: range should be consecutive years (FY convention).
        if end_year != start_year + 1:
            return None
        return _build_fy(text, end_year)

    match = _FY_2DIGIT_RE.match(text)
    if match:
        end_year = _year_from_2digit(int(match.group(1)))
        return _build_fy(text, end_year)

    match = _FY_4DIGIT_RE.match(text)
    if match:
        end_year = int(match.group(1))
        return _build_fy(text, end_year)

    return None


def _build_fy(label: str, end_year: int) -> dict:
    start_year = end_year - 1
    return {
        "label": label,
        "start": f"{start_year:04d}-04-01",
        "end": f"{end_year:04d}-03-31",
    }


# ---------------------------------------------------------------------------
# parse_point_in_time
# ---------------------------------------------------------------------------

_PLAUSIBLE_YEAR_MIN = 1900
_PLAUSIBLE_YEAR_MAX = 2100


def parse_point_in_time(text: str) -> Optional[str]:
    """Parse a date string into ISO format (YYYY-MM-DD) via
    `dateutil.parser`. Returns None (rather than guessing) if dateutil
    cannot parse it, or if the parsed year falls outside a plausible
    range for this project's documents.
    """
    if not text or not text.strip():
        return None

    try:
        parsed = dateutil_parser.parse(text.strip(), dayfirst=True, fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None

    if not (_PLAUSIBLE_YEAR_MIN <= parsed.year <= _PLAUSIBLE_YEAR_MAX):
        return None

    return parsed.date().isoformat()


# ---------------------------------------------------------------------------
# normalize_identifier
# ---------------------------------------------------------------------------

# CIN structure: [U or L] + 5 digits + 2 letters + 4 digits + 3 letters
# + 6 digits = 21 chars total. The leading U/L only encodes listing
# status (U = unlisted, L = listed) at the time the CIN was issued /
# last updated, and can change when a company lists or delists -- it is
# not part of the company's stable identity for cross-document matching.
_CIN_RE = re.compile(r"^[UL](\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})$")


def normalize_identifier(entity_id: str | None) -> Optional[str]:
    """Normalize an identifier for equality comparison: uppercase, strip
    surrounding whitespace, and collapse CIN listing-status prefixes (U
    vs L) to a single canonical form so a company's unlisted-era CIN and
    listed-era CIN compare equal.

    Returns None if input is None.
    """
    if entity_id is None:
        return None

    text = entity_id.strip().upper()

    match = _CIN_RE.match(text)
    if match:
        # Canonicalize on the 20-digit/letter core only, dropping the
        # U/L listing-status prefix entirely.
        return match.group(1)

    return text
