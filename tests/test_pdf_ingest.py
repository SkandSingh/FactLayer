"""Tests for app.pdf_ingest against the synthetic fixture PDF.

Run: python -m pytest tests/test_pdf_ingest.py -v

If tests/fixtures/sample.pdf is missing, regenerate it with:
    python scripts/generate_fixtures.py
"""
import os

import fitz  # PyMuPDF
import pytest

from app import pdf_ingest
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


def test_parse_pdf_raises_pdf_parse_error_for_encrypted_pdf(tmp_path):
    """A genuinely password-protected PDF must surface as a clean
    PDFParseError (-> HTTP 400 in the upload route), not let PyMuPDF's
    own encrypted-document state bubble up as an unhandled exception.

    PyMuPDF happily *opens* an encrypted file (fitz.open succeeds) but
    reports doc.is_encrypted=True; parse_pdf checks that flag explicitly
    and raises PDFParseError before attempting to read any page content.
    """
    encrypted_path = tmp_path / "encrypted.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "secret content")
    doc.save(
        str(encrypted_path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="owner",
    )
    doc.close()

    with pytest.raises(PDFParseError):
        parse_pdf(str(encrypted_path))


def test_parse_pdf_handles_page_with_no_extractable_text(tmp_path):
    """A page with no text content at all (e.g. a blank page, or a page
    that is purely a scanned image with no text layer) must not crash
    parse_pdf -- it should just yield a PageContent with empty text and
    no spans, same as any other page.
    """
    blank_path = tmp_path / "blank_pages.pdf"
    doc = fitz.open()
    doc.new_page()  # entirely blank, no text inserted
    doc.new_page()  # also blank
    doc.save(str(blank_path))
    doc.close()

    pages = parse_pdf(str(blank_path))

    assert len(pages) == 2
    for page in pages:
        assert page.text == ""
        assert page.spans == []


def test_parse_pdf_handles_zero_page_document(monkeypatch):
    """A PDF that opens successfully but reports zero pages (PyMuPDF
    itself refuses to *save* a zero-page file, so this can't be produced
    via a real fixture -- a fake fitz.Document stands in here) must
    degrade to an empty page list rather than raising or crashing.
    """

    class _FakeZeroPageDoc:
        is_encrypted = False

        def __iter__(self):
            return iter(())

        def close(self):
            pass

    monkeypatch.setattr(pdf_ingest.fitz, "open", lambda path: _FakeZeroPageDoc())

    pages = parse_pdf("irrelevant-path.pdf")

    assert pages == []
