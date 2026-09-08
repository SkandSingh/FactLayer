"""Layout-based section detection.

Turns the flat ``list[PageContent]`` produced by ``app.pdf_ingest`` into a
list of ``Section`` objects -- the unit of text we hand to the extraction
LLM one call per section (see docs/ARCHITECTURE.md, stage 2/3).

Two modes:

* **Layout mode** -- when the document has enough heading-like spans
  (large font and/or bold+short, see ``_find_heading_candidates``) to
  trust that its own formatting tells us where sections start, we build
  one ``Section`` per detected heading, named after that heading's own
  text. This generalizes to any PDF's own heading vocabulary -- nothing
  here hardcodes a document-specific heading string.
* **Fallback mode** -- when a document has no (or too little) detectable
  heading structure -- e.g. a scan-quality PDF, a document typeset with
  uniform formatting throughout -- we fall back to fixed-size page-group
  chunking so extraction still has document-sized inputs instead of one
  giant blob.

Important invariant carried over from ``pdf_ingest``: the input
``list[PageContent]`` is *reading order*, but ``PageContent.page_number``
is not guaranteed to be a contiguous integer sequence -- some documents in
this project's dataset are curated excerpts whose page numbers jump (e.g.
1, 2, 3, 47, 48; see docs/FOUR_REQUIRED_CASES.md). Everywhere in this
module that decides "what comes next" or "how many pages does this
section span", we key off position in the input list, never off
arithmetic on ``page_number`` values (e.g. we never do
``range(start_page, end_page + 1)``). ``start_page``/``end_page`` on a
``Section`` are just the real ``page_number`` of its first/last page,
recorded for display -- not used to reconstruct which pages exist.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from app.pdf_ingest import PageContent, Span

# Minimum number of detected heading candidates required to trust
# layout-based sectioning ("layout mode"). A tiny handful of large/bold
# spans -- a title page, a single stray bold word -- isn't strong enough
# evidence that the *whole* document has a real, consistent heading
# structure we can chunk by. Trusting layout mode on too little evidence
# risks a document split into one enormous "section" plus a couple of
# slivers, which is worse for extraction than plain fixed-size chunking.
# Three is a pragmatic floor: it takes at least three consistent signals
# spread through the document before we prefer layout mode over the
# safer, document-agnostic page-group fallback.
MIN_HEADINGS_FOR_LAYOUT_MODE = 3

# Fallback chunk size (in pages) when no reliable heading structure is
# detected. Chosen from the middle of the "~8-10 pages" range: large
# enough to keep the number of LLM calls in check for a 100+ page
# document, small enough that one section's concatenated text stays well
# within typical LLM prompt budgets.
FALLBACK_PAGE_GROUP_SIZE = 10

# A span's font size must be at least this multiple of the document's
# baseline (median) body-text font size to count as a "large font"
# heading candidate. 1.3x comfortably separates a genuine heading (often
# 1.5-2x body size) from normal font-size noise between body paragraphs
# and captions/footnotes without being so aggressive that a slightly
# larger sub-heading gets missed.
_LARGE_FONT_RATIO = 1.3

# Bold spans longer than this are almost certainly running body text
# (a bold word or phrase inside a normal paragraph), not a heading.
_MAX_HEADING_CHARS = 80

# Bold spans shorter than this are usually decorative bullets, page
# numbers, or stray single glyphs, not real headings.
_MIN_HEADING_CHARS = 3


@dataclass
class Section:
    section_path: str  # heading text (layout mode) or "Pages N-M" (fallback)
    text: str  # concatenated plain text of this section's pages
    start_page: int
    end_page: int
    pages: list = field(default_factory=list)  # list[PageContent], original objects


def _line_key(span: Span) -> float:
    """Cluster spans into approximate visual "lines" by bbox top (y0).

    Spans that sit on the same visual line in the source PDF report
    near-identical y0 values; rounding absorbs the sub-pixel jitter
    PyMuPDF sometimes reports between spans that are visually aligned.
    This is a heuristic, not a real layout/line-detection algorithm --
    good enough to tell "this bold span is alone on its line" from "this
    bold span sits next to other text", which is all we need it for.
    """
    y0 = span.bbox[1] if span.bbox else 0.0
    return round(y0, 0)


def _median_font_size(pages: list) -> float | None:
    """Rough body-text font-size baseline: the median span font size
    across the whole document. Median (rather than a strict mode) is
    robust to the small float variations real PDFs produce for what is
    visually "the same" body font size, while still reflecting whatever
    size the bulk of the document's text actually is.
    """
    sizes = [
        s.font_size
        for page in pages
        for s in page.spans
        if s.text.strip() and s.font_size > 0
    ]
    if not sizes:
        return None
    return statistics.median(sizes)


def _find_heading_candidates(pages: list) -> list[tuple[int, Span]]:
    """Scan every span on every page and return heading candidates as
    ``(page_list_index, span)`` pairs, in document order.

    ``page_list_index`` is the span's page's position in the *input
    list* -- not ``span.page_number`` -- because the input list order is
    the only thing guaranteed to be reading order (page numbers may be
    non-contiguous for curated excerpts; see module docstring).

    A span qualifies if either:
      * its font size is notably larger than the document's body-text
        baseline (``_LARGE_FONT_RATIO``), or
      * it is bold, a plausible heading length, and effectively alone on
        its visual line (no other non-blank span shares its line) --
        this is what keeps us from flagging every bold word inside a
        normal paragraph as a heading.
    """
    baseline = _median_font_size(pages)
    candidates: list[tuple[int, Span]] = []

    for list_index, page in enumerate(pages):
        lines: dict[float, list[Span]] = {}
        for s in page.spans:
            lines.setdefault(_line_key(s), []).append(s)

        for s in page.spans:
            stripped = s.text.strip()
            if not stripped:
                continue

            is_large_font = (
                baseline is not None and s.font_size > _LARGE_FONT_RATIO * baseline
            )

            is_bold_heading = False
            if s.bold and _MIN_HEADING_CHARS <= len(stripped) <= _MAX_HEADING_CHARS:
                line_group = lines[_line_key(s)]
                other_nonblank = [o for o in line_group if o is not s and o.text.strip()]
                is_bold_heading = not other_nonblank

            if is_large_font or is_bold_heading:
                candidates.append((list_index, s))

    return candidates


def _layout_mode_sections(pages: list, candidates: list) -> list:
    """Build one Section per heading candidate, from that heading's page
    to just before the next heading's page (or end of document).

    A section's page RANGE is still computed in whole pages -- precise
    sub-page positioning for evidence-location purposes lives downstream,
    via ``pages[*].spans`` char_offsets, not here. But a section's *first*
    page is sliced starting at its own heading's char_offset, and stops at
    the next heading's char_offset when that next heading lands on the
    SAME page. Real documents make "two headings on one page" the common
    case, not the rare one -- a dense summary/TOC page can carry five or
    more short sub-headings -- and without this slice, every one of those
    headings' sections would carry a full duplicate copy of that entire
    page's text, multiplying both LLM cost and duplicate-extracted facts
    (observed directly on a real prospectus: 5 sections sharing one page,
    each carrying that page's full ~4,000 characters -- a 3.6x content
    amplification across the whole document before this fix). Every page
    AFTER a section's first page is guaranteed (by the end_index
    computation below) not to contain another heading before this
    section's boundary, so those pages are used in full, unsliced.
    """
    sections: list[Section] = []
    n = len(candidates)

    for i, (list_index, span) in enumerate(candidates):
        next_index: int | None = None
        next_span: Span | None = None
        if i + 1 < n:
            next_index, next_span = candidates[i + 1]
            # -1 because the next heading's page belongs to the next
            # section; max(...) guards the same-page case (next_index-1
            # would be < list_index) so this section still gets its own
            # page rather than an empty range.
            end_index = max(list_index, next_index - 1)
        else:
            end_index = len(pages) - 1

        section_pages = pages[list_index : end_index + 1]
        first_page = section_pages[0]

        if next_index == list_index:
            # Next heading shares this same page -- stop this section's
            # slice of the page right where the next heading begins, so
            # the two sections partition the page instead of both
            # claiming all of it.
            first_page_text = first_page.text[span.char_offset : next_span.char_offset]
        else:
            first_page_text = first_page.text[span.char_offset :]

        text = "\n\n".join([first_page_text] + [p.text for p in section_pages[1:]])

        sections.append(
            Section(
                section_path=span.text.strip(),
                text=text,
                start_page=section_pages[0].page_number,
                end_page=section_pages[-1].page_number,
                pages=section_pages,
            )
        )

    return sections


def _fallback_mode_sections(pages: list) -> list:
    """Fixed-size page-group chunking for documents with no detectable
    heading structure. Groups are formed purely by position in the input
    list (``FALLBACK_PAGE_GROUP_SIZE`` pages at a time) -- never by
    ``page_number`` arithmetic -- so this is safe even for a document
    whose page numbers jump between retained sections.
    """
    sections: list[Section] = []
    for start in range(0, len(pages), FALLBACK_PAGE_GROUP_SIZE):
        group = pages[start : start + FALLBACK_PAGE_GROUP_SIZE]
        sections.append(
            Section(
                section_path=f"Pages {group[0].page_number}-{group[-1].page_number}",
                text="\n\n".join(p.text for p in group),
                start_page=group[0].page_number,
                end_page=group[-1].page_number,
                pages=group,
            )
        )
    return sections


def detect_sections(pages: list) -> list:
    """Detect sections in a parsed document.

    Args:
        pages: list[PageContent] in reading order (as returned by
            ``app.pdf_ingest.parse_pdf``).

    Returns:
        list[Section].
    """
    if not pages:
        return []

    candidates = _find_heading_candidates(pages)

    if len(candidates) >= MIN_HEADINGS_FOR_LAYOUT_MODE:
        return _layout_mode_sections(pages, candidates)

    return _fallback_mode_sections(pages)
