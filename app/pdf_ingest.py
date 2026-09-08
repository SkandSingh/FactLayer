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

Tables: financial/economic source documents (company filings, macro
reports) are full of tables, and naively flattening a table's spans in
left-to-right/top-to-bottom reading order can scramble which number
belongs to which row/column label. To avoid that, we run PyMuPDF's
``page.find_tables()`` table detector on each page. For every block of
text that falls inside a detected table's bounding box, instead of
flattening its spans individually we splice in a single *synthetic* Span
whose text is the table serialized as a markdown pipe table (see
``_table_to_markdown``). This keeps row/column structure legible as plain
text while preserving the same char_offset invariant -- the synthetic
span's char_offset/text is computed and recorded exactly like any other
span. Pages with no detected tables (the common case) are completely
unaffected: the extraction loop takes the exact same code path it did
before table support existed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFParseError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


# Bit flag on span["flags"] that PyMuPDF sets when a font is bold.
# See https://pymupdf.readthedocs.io/en/latest/textpage.html for the flag table.
_FONT_FLAG_BOLD = 1 << 4

# Fraction of a text block's area that must fall inside a detected table's
# bbox for that block to be treated as part of the table (and therefore
# replaced by the table's synthetic markdown Span rather than emitted as
# ordinary flattened spans). Not 1.0 because PyMuPDF's detected table bbox
# is derived from cell content and can be very slightly tighter or looser
# than the exact bboxes of the text blocks that make it up.
_TABLE_OVERLAP_THRESHOLD = 0.5

# Placeholder font_size recorded on synthetic table Spans. A table Span
# does not correspond to a single real PyMuPDF text span -- it represents
# an entire markdown-serialized table spliced into the page text -- so
# there is no single "real" font size to report. 0.0 is used so that any
# downstream consumer inspecting Span.font_size can tell at a glance this
# is a synthetic span rather than a measured one.
_SYNTHETIC_TABLE_FONT_SIZE = 0.0


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


def _cell_str(cell) -> str:
    """Normalize a single extracted table cell to a markdown-safe string.

    ``Table.extract()`` can return ``None`` for empty cells (and for cells
    absorbed into a merged/spanned cell); pipe characters in cell text would
    otherwise break the markdown table's column structure.
    """
    if cell is None:
        return ""
    return str(cell).replace("|", "\\|").strip()


def _table_to_markdown(rows: list) -> str:
    """Serialize extracted table rows (list of list of cell values) as a
    markdown pipe table: the first row becomes the header row, followed by
    a ``---`` separator row, then the remaining rows as data.

    Ragged rows (a row with more/fewer cells than the header) are
    padded/truncated to the header's column count so the resulting
    markdown is always well-formed.
    """
    if not rows:
        return ""

    header_cells = [_cell_str(c) for c in rows[0]]
    ncols = len(header_cells)
    if ncols == 0:
        return ""

    lines = ["| " + " | ".join(header_cells) + " |"]
    lines.append("| " + " | ".join(["---"] * ncols) + " |")
    for row in rows[1:]:
        cells = [_cell_str(c) for c in row]
        if len(cells) < ncols:
            cells = cells + [""] * (ncols - len(cells))
        elif len(cells) > ncols:
            cells = cells[:ncols]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _rect_overlap_fraction(a: tuple, b: tuple) -> float:
    """Fraction of rect `a`'s area that overlaps with rect `b`.

    Rects are (x0, y0, x1, y1) tuples in PyMuPDF's page coordinate space.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter_area = inter_w * inter_h

    a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    if a_area <= 0:
        return 0.0
    return inter_area / a_area


def _find_page_tables(page: "fitz.Page") -> list:
    """Detect tables on `page` via PyMuPDF's ``find_tables()``.

    Returns a list of ``(bbox, rows)`` tuples -- one per detected table --
    where `bbox` is an (x0, y0, x1, y1) tuple and `rows` is the table's
    ``extract()``-ed list of rows (each row a list of cell strings/None).

    Defensive by design: table detection is a heuristic best-effort layer
    on top of ordinary span extraction, so any exception raised by
    ``find_tables()`` (or while pulling data out of a detected table, on a
    malformed/weird page) is logged and swallowed -- callers get back an
    empty list and fall back to plain span extraction for that page, rather
    than letting the whole `parse_pdf` call crash.
    """
    try:
        table_finder = page.find_tables()
        tables = list(table_finder.tables)
    except Exception as exc:
        logger.warning(
            "Table detection failed on page %s; falling back to plain span "
            "extraction for this page: %s",
            getattr(page, "number", "?"),
            exc,
        )
        return []

    results = []
    for table in tables:
        try:
            rows = table.extract()
            bbox = tuple(table.bbox)
        except Exception as exc:
            logger.warning(
                "Failed to extract a detected table on page %s; skipping "
                "that table: %s",
                getattr(page, "number", "?"),
                exc,
            )
            continue
        if not rows:
            continue
        results.append((bbox, rows))
    return results


def _map_blocks_to_tables(blocks: list, tables: list) -> list:
    """Return a list parallel to `blocks`: for each block, the index into
    `tables` it should be considered part of (by bbox overlap), or None.
    """
    mapping: list = [None] * len(blocks)
    for i, block in enumerate(blocks):
        if block.get("type", 0) != 0:
            continue
        bbox = block.get("bbox")
        if not bbox:
            continue
        block_bbox = tuple(bbox)
        for t_idx, (table_bbox, _rows) in enumerate(tables):
            if _rect_overlap_fraction(block_bbox, table_bbox) >= _TABLE_OVERLAP_THRESHOLD:
                mapping[i] = t_idx
                break
    return mapping


def _extract_page(page: "fitz.Page", page_number: int) -> PageContent:
    """Build a PageContent for a single page.

    We walk the page's structured text dict ourselves (rather than calling
    ``page.get_text("text")`` separately) so that the plain text we expose
    and the char_offset we record for each span are always derived from the
    exact same data and therefore always consistent with each other.

    Tables detected via ``page.find_tables()`` are handled specially: every
    block of text belonging to a detected table is replaced by a single
    synthetic markdown-table Span (see ``_table_to_markdown``) instead of
    being flattened span-by-span. When no tables are detected on this page
    (``tables`` is empty -- either because there genuinely are none, or
    because detection failed and was defensively suppressed), every block
    maps to `None` in `block_table_index` below and this function takes the
    exact same code path it did before table support existed.
    """
    page_dict = page.get_text("dict")
    blocks = page_dict.get("blocks", [])

    tables = _find_page_tables(page)
    block_table_index = _map_blocks_to_tables(blocks, tables) if tables else [None] * len(blocks)

    text_parts: list[str] = []
    spans: list[Span] = []
    offset = 0
    emitted_tables: set = set()

    for i, block in enumerate(blocks):
        table_idx = block_table_index[i]
        if table_idx is not None:
            if table_idx in emitted_tables:
                # Another block belonging to the same already-emitted
                # table (e.g. a different row); its content is already
                # represented by that single synthetic Span, so skip it
                # entirely rather than flattening it too.
                continue
            emitted_tables.add(table_idx)

            table_bbox, rows = tables[table_idx]
            markdown = _table_to_markdown(rows)
            if not markdown:
                continue

            spans.append(
                Span(
                    page_number=page_number,
                    text=markdown,
                    char_offset=offset,
                    font_size=_SYNTHETIC_TABLE_FONT_SIZE,
                    bold=False,
                    bbox=table_bbox,
                )
            )
            text_parts.append(markdown)
            offset += len(markdown)

            # Mirror the line-newline + block-separator blank line that an
            # ordinary single-line block would contribute (see below), so
            # the table reads as its own paragraph in the surrounding text.
            text_parts.append("\n\n")
            offset += 2
            continue

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
