"""Tests for app.pdf_ingest against the synthetic fixture PDF.

Run: python -m pytest tests/test_pdf_ingest.py -v

If tests/fixtures/sample.pdf is missing, regenerate it with:
    python scripts/generate_fixtures.py
"""
import os

import pytest

from app.pdf_ingest import PageContent, PDFParseError, Span, parse_pdf

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")


@pytest.fixture(scope="module")
def pages():
    assert os.path.exists(FIXTURE_PATH), (
        f"Fixture not found at {FIXTURE_PATH!r}; run "
        "`python scripts/generate_fixtures.py` first."
    )
    return parse_pdf(FIXTURE_PATH)


def test_returns_list_of_page_content(pages):
    assert isinstance(pages, list)
    assert all(isinstance(p, PageContent) for p in pages)


def test_correct_page_count(pages):
    # The fixture generator writes exactly 3 pages.
    assert len(pages) == 3


def test_page_numbers_are_1_indexed_and_sequential(pages):
    assert [p.page_number for p in pages] == [1, 2, 3]


def test_heading_page_has_bold_large_font_span(pages):
    heading_page = pages[0]
    assert heading_page.spans, "Expected page 1 to have extracted spans."

    heading_spans = [s for s in heading_page.spans if s.bold and s.font_size >= 18]
    assert heading_spans, (
        "Expected at least one bold, large-font span on the heading page; "
        f"got spans: {heading_page.spans}"
    )

    heading = heading_spans[0]
    assert "Executive Summary" in heading.text
    assert isinstance(heading.bbox, tuple)
    assert len(heading.bbox) == 4


def test_body_pages_have_no_large_bold_heading_span(pages):
    # Pages 2 and 3 are plain body text (11pt, not bold) in this fixture,
    # giving later section-detection heuristics a clear contrast against
    # the heading on page 1.
    for page in pages[1:]:
        big_bold = [s for s in page.spans if s.bold and s.font_size >= 18]
        assert big_bold == []


def test_all_spans_are_span_instances(pages):
    for page in pages:
        for span in page.spans:
            assert isinstance(span, Span)
            assert span.page_number == page.page_number


def test_char_offset_invariant_holds_for_all_spans(pages):
    # The core evidence-tracing guarantee: every span's text must be
    # recoverable by slicing the page's plain text at char_offset.
    for page in pages:
        for span in page.spans:
            recovered = page.text[span.char_offset : span.char_offset + len(span.text)]
            assert recovered == span.text, (
                f"char_offset invariant broken on page {page.page_number}: "
                f"expected {span.text!r}, got {recovered!r} at offset {span.char_offset}"
            )


def test_char_offset_invariant_sampled_spans(pages):
    # Spot-check a handful of specific spans across different pages.
    page1_heading = pages[0].spans[0]
    assert pages[0].text[: len(page1_heading.text)] == page1_heading.text

    page2_first = pages[1].spans[0]
    assert (
        pages[1].text[page2_first.char_offset : page2_first.char_offset + len(page2_first.text)]
        == page2_first.text
    )

    page3_last = pages[2].spans[-1]
    assert (
        pages[2].text[page3_last.char_offset : page3_last.char_offset + len(page3_last.text)]
        == page3_last.text
    )


def test_spans_in_reading_order_have_increasing_offsets(pages):
    for page in pages:
        offsets = [s.char_offset for s in page.spans]
        assert offsets == sorted(offsets)


def test_parse_pdf_raises_clear_error_for_missing_file():
    with pytest.raises(PDFParseError):
        parse_pdf("tests/fixtures/does_not_exist.pdf")


def test_parse_pdf_raises_clear_error_for_corrupt_file(tmp_path):
    bogus = tmp_path / "not_a_pdf.pdf"
    bogus.write_bytes(b"this is not a valid pdf file")
    with pytest.raises(PDFParseError):
        parse_pdf(str(bogus))
