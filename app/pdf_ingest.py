"""PDF ingestion: extract per-page plain text plus layout metadata for text spans.

This module is the entire "evidence" story for FactLayer: every span of text
pulled out of a PDF carries a pointer back to (page_number, char_offset) so
that any fact derived downstream can be traced back to an exact location in
the source document.

We use PyMuPDF (imported as ``fitz``) to walk each page's structured text
dict (blocks -> lines -> spans), reconstructing the page's plain text by
concatenating span text in reading order. Because we build ``PageContent.text``
ourselves from the very same spans we record offsets for, the invariant

    page.text[span.char_offset : span.char_offset + len(span.text)] == span.text

holds exactly for every span (see "Known limitations" below for the one
caveat around inserted newlines).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import fitz  # PyMuPDF


class PDFParseError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


# Bit flag on span["flags"] that PyMuPDF sets when a font is bold.
# See https://pymupdf.readthedocs.io/en/latest/textpage.html for the flag table.
_FONT_FLAG_BOLD = 1 << 4


@dataclass
class Span:
    page_number: int  # 1-indexed
    text: str
    char_offset: int  # offset of this span's text within page_text
    font_size: float
    bold: bool
    bbox: tuple  # (x0, y0, x1, y1)


@dataclass
class PageContent:
    page_number: int  # 1-indexed
    text: str  # full plain text of the page, spans' char_offset indexes into this
    spans: list = field(default_factory=list)  # list[Span], in reading order


def _is_bold(span: dict) -> bool:
    """Determine boldness from PyMuPDF's font flags bit and/or font name.

    PyMuPDF sets bit 4 (value 16) of ``span["flags"]`` when the font is bold,
    but this is not always reliable for every embedded font/subsetting
    scheme, so we also fall back to checking the font name for "bold" as a
    substring (case-insensitive) — a common convention such as
    "Helvetica-Bold" or "Arial,Bold".
    """
    flags = span.get("flags", 0)
    if flags & _FONT_FLAG_BOLD:
        return True
    font_name = span.get("font", "") or ""
    return "bold" in font_name.lower()


def _extract_page(page: "fitz.Page", page_number: int) -> PageContent:
    """Build a PageContent for a single page.

    We walk the page's structured text dict ourselves (rather than calling
    ``page.get_text("text")`` separately) so that the plain text we expose
    and the char_offset we record for each span are always derived from the
    exact same data and therefore always consistent with each other.
    """
    page_dict = page.get_text("dict")

    text_parts: list[str] = []
    spans: list[Span] = []
    offset = 0

    for block in page_dict.get("blocks", []):
        # Skip non-text blocks (e.g. images), which have no "lines" key.
        if block.get("type", 0) != 0:
            continue

        for line in block.get("lines", []):
            line_spans = line.get("spans", [])
            for span in line_spans:
                span_text = span.get("text", "")
                if span_text == "":
                    # Nothing to record an offset for; avoid emitting an
                    # empty/degenerate Span.
                    continue

                bbox = tuple(span.get("bbox", (0.0, 0.0, 0.0, 0.0)))
                spans.append(
                    Span(
                        page_number=page_number,
                        text=span_text,
                        char_offset=offset,
                        font_size=float(span.get("size", 0.0)),
                        bold=_is_bold(span),
                        bbox=bbox,
                    )
                )
                text_parts.append(span_text)
                offset += len(span_text)

            # PyMuPDF's own "text" extraction inserts a newline after each
            # line; we mirror that here so PageContent.text closely matches
            # page.get_text("text"). This newline is *not* part of any span
            # (matching the visual reality that spans don't include line
            # breaks), so it simply advances the offset counter without a
            # corresponding Span.
            if line_spans:
                text_parts.append("\n")
                offset += 1

        # Blank line between blocks, mirroring get_text("text") behavior.
        if block.get("lines"):
            text_parts.append("\n")
            offset += 1

    page_text = "".join(text_parts)
    return PageContent(page_number=page_number, text=page_text, spans=spans)


def parse_pdf(path: str) -> list:
    """Parse a PDF file into a list of PageContent, one per page.

    Raises:
        PDFParseError: if the file cannot be opened or is not a valid PDF.
    """
    try:
        doc = fitz.open(path)
    except Exception as exc:  # PyMuPDF raises its own exception types
        raise PDFParseError(f"Could not open PDF at {path!r}: {exc}") from exc

    try:
        if doc.is_encrypted:
            raise PDFParseError(f"PDF at {path!r} is encrypted and cannot be parsed.")

        pages: list[PageContent] = []
        try:
            for index, page in enumerate(doc):
                page_number = index + 1
                pages.append(_extract_page(page, page_number))
        except PDFParseError:
            raise
        except Exception as exc:
            raise PDFParseError(f"Failed to parse PDF at {path!r}: {exc}") from exc

        return pages
    finally:
        doc.close()
