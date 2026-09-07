"""Tests for app.sectioning against synthetic PageContent/Span fixtures.

No real PDFs needed here -- pdf_ingest's own fixture-based tests already
cover PyMuPDF extraction; this module only needs to exercise the
layout/fallback decision logic and section-boundary math, which only
depends on the PageContent/Span dataclass shapes.

Run: python -m pytest tests/test_sectioning.py -v
"""
from app.pdf_ingest import PageContent, Span
from app.sectioning import (
    FALLBACK_PAGE_GROUP_SIZE,
    MIN_HEADINGS_FOR_LAYOUT_MODE,
    Section,
    detect_sections,
)

BODY_SIZE = 11.0
HEADING_SIZE = 18.0


def _body_span(page_number, text, char_offset, y0=100.0):
    return Span(
        page_number=page_number,
        text=text,
        char_offset=char_offset,
        font_size=BODY_SIZE,
        bold=False,
        bbox=(72.0, y0, 500.0, y0 + 12.0),
    )


def _heading_span(page_number, text, char_offset, font_size=HEADING_SIZE, bold=True, y0=50.0):
    return Span(
        page_number=page_number,
        text=text,
        char_offset=char_offset,
        font_size=font_size,
        bold=bold,
        bbox=(72.0, y0, 72.0 + 10 * len(text), y0 + 20.0),
    )


def _page_with_heading(page_number, heading_text, body_text="Some body text about the topic."):
    """A page whose first line is a large, bold, isolated heading span,
    followed by a separate body paragraph span on its own line.
    """
    heading = heading_text + "\n"
    body = body_text
    full_text = heading + body
    spans = [
        _heading_span(page_number, heading_text, char_offset=0, y0=50.0),
        _body_span(page_number, body, char_offset=len(heading), y0=100.0),
    ]
    return PageContent(page_number=page_number, text=full_text, spans=spans)


def _plain_page(page_number, body_text="Uniform body text with nothing special about it at all."):
    """A page with only uniform, non-bold, body-sized text -- no heading
    signal whatsoever.
    """
    spans = [_body_span(page_number, body_text, char_offset=0, y0=100.0)]
    return PageContent(page_number=page_number, text=body_text, spans=spans)


# ---------------------------------------------------------------------------
# Layout mode
# ---------------------------------------------------------------------------


def test_layout_mode_triggers_with_enough_headings_and_builds_correct_sections():
    pages = [
        _page_with_heading(1, "Introduction"),
        _page_with_heading(2, "Financial Highlights"),
        _page_with_heading(3, "Risk Factors"),
        _page_with_heading(4, "Conclusion"),
    ]
    assert MIN_HEADINGS_FOR_LAYOUT_MODE <= 4  # sanity: fixture exercises layout mode

    sections = detect_sections(pages)

    assert len(sections) == 4
    assert [s.section_path for s in sections] == [
        "Introduction",
        "Financial Highlights",
        "Risk Factors",
        "Conclusion",
    ]
    # Each heading is on its own page here, so each section is exactly
    # one page wide.
    for section, page in zip(sections, pages):
        assert isinstance(section, Section)
        assert section.start_page == page.page_number
        assert section.end_page == page.page_number
        assert section.pages == [page]
        # The section's text is the page's full text (whole-page
        # simplification), and must be the *same* PageContent object,
        # not a copy.
        assert section.pages[0] is page
        assert page.text in section.text


def test_layout_mode_section_spans_multiple_pages_until_next_heading():
    # "Chapter One" covers pages 1-3 (no heading on 2 or 3), "Chapter Two"
    # covers page 4 onward, plus two more headings to clear the
    # layout-mode threshold.
    p1 = _page_with_heading(1, "Chapter One")
    p2 = _plain_page(2, "continuation of chapter one, page two.")
    p3 = _plain_page(3, "continuation of chapter one, page three.")
    p4 = _page_with_heading(4, "Chapter Two")
    p5 = _page_with_heading(5, "Chapter Three")
    p6 = _page_with_heading(6, "Chapter Four")
    pages = [p1, p2, p3, p4, p5, p6]

    sections = detect_sections(pages)

    assert [s.section_path for s in sections] == [
        "Chapter One",
        "Chapter Two",
        "Chapter Three",
        "Chapter Four",
    ]

    chapter_one = sections[0]
    assert chapter_one.start_page == 1
    assert chapter_one.end_page == 3
    assert chapter_one.pages == [p1, p2, p3]
    # Original objects, not re-derived copies.
    assert all(a is b for a, b in zip(chapter_one.pages, [p1, p2, p3]))
    assert "chapter one, page two" in chapter_one.text
    assert "chapter one, page three" in chapter_one.text

    chapter_two = sections[1]
    assert chapter_two.start_page == 4
    assert chapter_two.end_page == 4
    assert chapter_two.pages == [p4]


def test_bold_word_inside_paragraph_is_not_flagged_as_heading():
    # A bold span that shares its line with other non-blank spans (i.e.
    # bold emphasis inside running text) must not count as a heading
    # candidate, even though it's bold and short.
    page_number = 1
    lead = "This report shows "
    bold_word = "strong"
    tail = " growth this year."
    full_text = lead + bold_word + tail
    spans = [
        _body_span(page_number, lead, char_offset=0, y0=100.0),
        Span(
            page_number=page_number,
            text=bold_word,
            char_offset=len(lead),
            font_size=BODY_SIZE,
            bold=True,
            bbox=(150.0, 100.0, 190.0, 112.0),  # same line (same y0) as lead/tail
        ),
        _body_span(page_number, tail, char_offset=len(lead) + len(bold_word), y0=100.0),
    ]
    page = PageContent(page_number=page_number, text=full_text, spans=spans)

    # Only one plain page repeated -- nowhere near the layout-mode
    # threshold, and specifically the bold-in-paragraph word must not be
    # counted.
    pages = [page, _plain_page(2), _plain_page(3)]
    sections = detect_sections(pages)

    # Falls back to page-group chunking, proving the bold inline word
    # wasn't treated as a heading (that would have pushed toward layout
    # mode with a wrong heading like "strong").
    assert all(s.section_path.startswith("Pages ") for s in sections)
    assert all(s.section_path != "strong" for s in sections)


# ---------------------------------------------------------------------------
# Fallback mode
# ---------------------------------------------------------------------------


def test_fallback_mode_used_when_no_heading_signal():
    pages = [_plain_page(n) for n in range(1, 26)]  # 25 uniform pages

    sections = detect_sections(pages)

    expected_groups = -(-25 // FALLBACK_PAGE_GROUP_SIZE)  # ceil division
    assert len(sections) == expected_groups
    for s in sections:
        assert s.section_path.startswith("Pages ")
        assert isinstance(s, Section)

    # First and last group cover the expected page ranges.
    assert sections[0].start_page == 1
    assert sections[0].end_page == FALLBACK_PAGE_GROUP_SIZE
    assert sections[0].section_path == f"Pages 1-{FALLBACK_PAGE_GROUP_SIZE}"
    assert sections[-1].end_page == 25

    # All 25 pages are accounted for across the groups, each exactly once.
    all_pages = [p for s in sections for p in s.pages]
    assert len(all_pages) == 25
    assert [p.page_number for p in all_pages] == list(range(1, 26))


def test_fallback_mode_with_fewer_headings_than_threshold():
    # Exactly one heading-like page among many plain ones -- not enough
    # to clear MIN_HEADINGS_FOR_LAYOUT_MODE (assumed >= 2 here), so this
    # should still fall back to page-group chunking rather than building
    # a single lopsided "section" out of one heading.
    assert MIN_HEADINGS_FOR_LAYOUT_MODE >= 2
    pages = [_page_with_heading(1, "Only Heading")] + [
        _plain_page(n) for n in range(2, 13)
    ]

    sections = detect_sections(pages)

    assert all(s.section_path.startswith("Pages ") for s in sections)


def test_empty_document_returns_no_sections():
    assert detect_sections([]) == []


# ---------------------------------------------------------------------------
# Non-contiguous page numbers (curated excerpt simulation)
# ---------------------------------------------------------------------------


def test_layout_mode_does_not_stitch_across_page_number_gap():
    # Simulate a curated excerpt: page_number jumps from 3 straight to 47.
    # detect_sections must key off *list position*, not page_number
    # arithmetic, so the heading on "page 47" starts a fresh section that
    # does NOT pull in page 3's tail content, and the "page 3" section
    # does NOT expand to (wrongly) claim it covers pages up to 46.
    p1 = _page_with_heading(1, "Front Matter")
    p2 = _plain_page(2, "front matter continued.")
    p3 = _plain_page(3, "front matter tail end, last retained page before the gap.")
    p47 = _page_with_heading(47, "Main Analysis")
    p48 = _plain_page(48, "main analysis continued after the gap.")
    pages = [p1, p2, p3, p47, p48]

    # Add two more headings so we clear the layout-mode threshold while
    # keeping the gap scenario central.
    p49 = _page_with_heading(49, "Appendix A")
    p50 = _page_with_heading(50, "Appendix B")
    pages = [p1, p2, p3, p47, p48, p49, p50]

    sections = detect_sections(pages)
    assert [s.section_path for s in sections] == [
        "Front Matter",
        "Main Analysis",
        "Appendix A",
        "Appendix B",
    ]

    front_matter = sections[0]
    main_analysis = sections[1]

    # Front Matter must end at page_number 3, not stretch to 46/47.
    assert front_matter.start_page == 1
    assert front_matter.end_page == 3
    assert front_matter.pages == [p1, p2, p3]

    # Main Analysis starts at page_number 47 and must NOT contain any of
    # page 3's text (the "stitching" bug this test guards against).
    assert main_analysis.start_page == 47
    assert main_analysis.end_page == 48
    assert main_analysis.pages == [p47, p48]
    assert "front matter tail end" not in main_analysis.text
    assert "last retained page before the gap" not in main_analysis.text
    assert "main analysis continued after the gap" in main_analysis.text

    # And Front Matter must not contain any of page 47's text either.
    assert "Main Analysis" not in front_matter.text
    assert "main analysis continued" not in front_matter.text


def test_fallback_mode_page_groups_use_actual_page_numbers_not_computed_ranges():
    # Non-contiguous page numbers with NO heading signal at all -- must
    # fall back to page-group chunking, and the section_path labels must
    # reflect the real (possibly jumpy) page_number values of the group's
    # boundary pages, never an assumed contiguous range.
    pages = [_plain_page(1), _plain_page(2), _plain_page(3)] + [
        _plain_page(n) for n in range(47, 47 + 9)  # 47..55, 9 pages
    ]
    # 3 + 9 = 12 pages total, one group of FALLBACK_PAGE_GROUP_SIZE (10)
    # plus a remainder group of 2, given the default group size of 10.
    assert FALLBACK_PAGE_GROUP_SIZE == 10

    sections = detect_sections(pages)

    assert len(sections) == 2
    first, second = sections
    # First 10 pages in list order: 1, 2, 3, 47, 48, 49, 50, 51, 52, 53.
    assert first.start_page == 1
    assert first.end_page == 53
    assert first.section_path == "Pages 1-53"
    # Remainder: 54, 55.
    assert second.start_page == 54
    assert second.end_page == 55
    assert second.section_path == "Pages 54-55"

    # No page is duplicated or invented across groups.
    all_page_numbers = [p.page_number for s in sections for p in s.pages]
    assert all_page_numbers == [1, 2, 3, 47, 48, 49, 50, 51, 52, 53, 54, 55]


# ---------------------------------------------------------------------------
# Degenerate inputs: all-blank pages
# ---------------------------------------------------------------------------
# (The zero-pages case is already covered by test_empty_document_returns_no_sections above.)


def test_detect_sections_degrades_gracefully_for_all_blank_pages():
    """A document whose pages carry no extractable text/spans at all
    (e.g. every page came back blank from pdf_ingest) must not crash
    section detection -- with zero heading candidates it should simply
    fall back to page-group chunking, producing section(s) with empty
    text rather than raising.
    """
    blank_pages = [
        PageContent(page_number=1, text="", spans=[]),
        PageContent(page_number=2, text="", spans=[]),
    ]

    sections = detect_sections(blank_pages)

    assert len(sections) == 1
    assert sections[0].pages == blank_pages
    assert sections[0].start_page == 1
    assert sections[0].end_page == 2
