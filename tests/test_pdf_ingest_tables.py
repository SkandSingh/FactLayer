"""Tests for the table-detection/markdown-splicing behavior in app.pdf_ingest.

These build small synthetic PDFs on the fly (via PyMuPDF) rather than
relying on a static fixture file, since we need to *verify empirically*
that ``page.find_tables()`` actually detects our fixture as a table before
trusting any assertions built on top of it.

Run: python -m pytest tests/test_pdf_ingest_tables.py -v
"""
import os

import fitz  # PyMuPDF
import pytest

from app import pdf_ingest
from app.pdf_ingest import PageContent, Span, parse_pdf

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")

# Cell values used to build the synthetic table fixture below. Kept as a
# module-level constant so tests can assert on exact cell values without
# duplicating the literal grid.
TABLE_ROWS = [
    ["", "FY24", "FY23"],
    ["Revenue", "81,415.38", "65,930.00"],
    ["Profit", "1,234.56", "(890.12)"],
]


def _build_table_pdf(path: str) -> None:
    """Write a 1-page PDF containing a real, ruled grid (so PyMuPDF's
    line-based table detector actually fires) plus a heading above it.

    Grid lines are drawn explicitly with `Shape.draw_line`, and each cell's
    text is inserted at a position aligned to that grid -- this mirrors how
    financial-report tables are typically rendered (ruled table with
    text placed in cells), and was verified empirically (see comment in
    test_table_fixture_is_actually_detected) to make
    ``page.find_tables()`` return a non-empty result.
    """
    doc = fitz.open()
    page = doc.new_page()

    page.insert_text((72, 60), "Financial Highlights", fontsize=16, fontname="hebo")

    x0, y0 = 72, 100
    col_widths = [150, 100, 100]
    row_height = 20
    n_rows = len(TABLE_ROWS)

    xs = [x0]
    for w in col_widths:
        xs.append(xs[-1] + w)
    ys = [y0 + i * row_height for i in range(n_rows + 1)]

    shape = page.new_shape()
    for x in xs:
        shape.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        shape.draw_line((xs[0], y), (xs[-1], y))
    shape.finish(color=(0, 0, 0), width=0.75)
    shape.commit()

    for r in range(n_rows):
        for c in range(len(col_widths)):
            cx = xs[c] + 5
            cy = ys[r] + row_height - 6
            page.insert_text((cx, cy), TABLE_ROWS[r][c], fontsize=10, fontname="helv")

    doc.save(path)
    doc.close()


@pytest.fixture()
def table_pdf_path(tmp_path):
    path = str(tmp_path / "table_sample.pdf")
    _build_table_pdf(path)
    return path


def test_table_fixture_is_actually_detected(table_pdf_path):
    """Sanity check the fixture itself, independent of app.pdf_ingest:
    confirm PyMuPDF's find_tables() really does detect our synthetic grid
    as a table, so the tests below aren't resting on an unverified
    assumption about the fixture.
    """
    doc = fitz.open(table_pdf_path)
    try:
        page = doc[0]
        table_finder = page.find_tables()
        tables = list(table_finder.tables)
        assert len(tables) >= 1, "Fixture PDF was not detected as containing a table"
        rows = tables[0].extract()
        assert rows == TABLE_ROWS
    finally:
        doc.close()


def test_table_page_produces_markdown_table_span(table_pdf_path):
    pages = parse_pdf(table_pdf_path)
    assert len(pages) == 1
    page = pages[0]

    # Find the synthetic table span: it's the one whose text contains the
    # markdown separator row.
    table_spans = [s for s in page.spans if "| --- |" in s.text]
    assert len(table_spans) == 1, f"Expected exactly one table span, got: {page.spans}"
    table_span = table_spans[0]

    # Correct cell values from the fixture must appear in the markdown.
    assert "Revenue" in table_span.text
    assert "81,415.38" in table_span.text
    assert "65,930.00" in table_span.text
    assert "Profit" in table_span.text
    assert "1,234.56" in table_span.text
    assert "(890.12)" in table_span.text

    # Basic markdown pipe-table shape: header row, separator row, 2 data rows.
    lines = table_span.text.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("|") and lines[0].endswith("|")
    assert lines[1].replace(" ", "").replace("|", "").strip("-") == ""

    # The markdown block must also appear verbatim in the page's full text,
    # at the span's recorded char_offset (this is also covered generically
    # by the char-offset invariant test below, but assert it directly here
    # too since it's the headline behavior this feature exists for).
    assert page.text[table_span.char_offset : table_span.char_offset + len(table_span.text)] == table_span.text

    # The heading above the table should still be extracted as an ordinary
    # (non-synthetic) span.
    heading_spans = [s for s in page.spans if "Financial Highlights" in s.text]
    assert len(heading_spans) == 1
    assert heading_spans[0].font_size > 0  # real span, not the 0.0 placeholder

    # The raw, unaligned per-cell text should NOT also appear as separate
    # ordinary spans (it should have been replaced by the single table span).
    assert not any(s.text == "Revenue" for s in page.spans)
    assert not any(s.text == "81,415.38" for s in page.spans)


def test_char_offset_invariant_holds_on_table_page(table_pdf_path):
    pages = parse_pdf(table_pdf_path)
    for page in pages:
        for span in page.spans:
            recovered = page.text[span.char_offset : span.char_offset + len(span.text)]
            assert recovered == span.text, (
                f"char_offset invariant broken on page {page.page_number}: "
                f"expected {span.text!r}, got {recovered!r} at offset {span.char_offset}"
            )


def test_table_span_is_a_span_instance_with_reasonable_metadata(table_pdf_path):
    pages = parse_pdf(table_pdf_path)
    page = pages[0]
    table_spans = [s for s in page.spans if "| --- |" in s.text]
    table_span = table_spans[0]

    assert isinstance(table_span, Span)
    assert table_span.page_number == 1
    assert table_span.bold is False
    assert table_span.font_size == 0.0  # documented synthetic placeholder
    assert isinstance(table_span.bbox, tuple)
    assert len(table_span.bbox) == 4


def test_tableless_page_behavior_is_unchanged():
    """Pages with no detected table must produce byte-for-byte identical
    output to the pre-table-support implementation. We reuse the existing
    sample.pdf fixture (heading + body pages, no tables) and confirm the
    same invariants the original test_pdf_ingest.py suite already checks.
    """
    assert os.path.exists(FIXTURE_PATH), (
        f"Fixture not found at {FIXTURE_PATH!r}; run "
        "`python scripts/generate_fixtures.py` first."
    )
    pages = parse_pdf(FIXTURE_PATH)

    assert len(pages) == 3
    for page in pages:
        assert isinstance(page, PageContent)
        # No table-shaped spans should have been introduced.
        assert not any("| --- |" in s.text for s in page.spans)
        for span in page.spans:
            recovered = page.text[span.char_offset : span.char_offset + len(span.text)]
            assert recovered == span.text

    heading_spans = [s for s in pages[0].spans if s.bold and s.font_size >= 18]
    assert heading_spans
    assert "Executive Summary" in heading_spans[0].text


def test_find_tables_exception_falls_back_to_plain_span_extraction(monkeypatch, table_pdf_path, caplog):
    """If find_tables() raises (simulating a malformed/weird page or a
    PyMuPDF internal error), parse_pdf must not crash -- it should fall
    back to ordinary flattened-span extraction for that page.
    """

    def _boom(self, **kwargs):
        raise RuntimeError("simulated find_tables failure")

    monkeypatch.setattr(fitz.Page, "find_tables", _boom, raising=True)

    with caplog.at_level("WARNING", logger="app.pdf_ingest"):
        pages = parse_pdf(table_pdf_path)

    assert len(pages) == 1
    page = pages[0]

    # No synthetic markdown table span -- detection was suppressed.
    assert not any("| --- |" in s.text for s in page.spans)

    # The table's cell text is still present, just flattened as ordinary
    # spans (old behavior), proving the fallback path actually ran.
    assert any(s.text == "Revenue" for s in page.spans)
    assert any(s.text == "81,415.38" for s in page.spans)

    # The char_offset invariant must still hold under the fallback path.
    for span in page.spans:
        recovered = page.text[span.char_offset : span.char_offset + len(span.text)]
        assert recovered == span.text

    assert any("Table detection failed" in rec.message for rec in caplog.records)


def test_find_tables_exception_does_not_affect_other_pages(monkeypatch, tmp_path):
    """A find_tables() failure on one page must be isolated to that page --
    it must not abort parsing of the whole document.
    """
    multi_path = str(tmp_path / "multi.pdf")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Page one, no table.")
    doc.new_page().insert_text((72, 72), "Page two, no table.")
    doc.save(multi_path)
    doc.close()

    def _boom(self, **kwargs):
        raise RuntimeError("simulated find_tables failure")

    monkeypatch.setattr(fitz.Page, "find_tables", _boom, raising=True)

    pages = parse_pdf(multi_path)
    assert len(pages) == 2
    assert "Page one" in pages[0].text
    assert "Page two" in pages[1].text
