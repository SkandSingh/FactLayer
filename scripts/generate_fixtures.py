"""Generate a small synthetic multi-page PDF for unit testing app.pdf_ingest.

Run directly:

    python scripts/generate_fixtures.py

This writes tests/fixtures/sample.pdf: a 3-page PDF where page 1 has a
large/bold heading followed by body text, and pages 2-3 are plain body text
paragraphs (so later section-detection heuristics have headings vs. body
text to key off of).
"""
from __future__ import annotations

import os

import fitz  # PyMuPDF

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures")
OUTPUT_PATH = os.path.join(FIXTURES_DIR, "sample.pdf")


def build_pdf() -> None:
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    doc = fitz.open()

    # --- Page 1: bold heading followed by body text -----------------------
    page1 = doc.new_page()
    heading_point = fitz.Point(72, 90)
    page1.insert_text(
        heading_point,
        "Executive Summary",
        fontsize=24,
        fontname="hebo",  # Helvetica-Bold
        color=(0, 0, 0),
    )
    body1 = (
        "This report summarizes the quarterly findings of the research team.\n"
        "Revenue grew steadily across all regions, driven by strong demand\n"
        "for the flagship product line. The following pages provide detail\n"
        "on methodology and regional breakdowns."
    )
    page1.insert_textbox(
        fitz.Rect(72, 130, 500, 300),
        body1,
        fontsize=11,
        fontname="helv",
        color=(0, 0, 0),
    )

    # --- Page 2: plain body text -------------------------------------------
    page2 = doc.new_page()
    body2 = (
        "Methodology\n\n"
        "Data was collected from internal sales systems and cross-referenced\n"
        "with third-party market reports. All figures are reported in\n"
        "constant currency to control for exchange rate fluctuations.\n"
        "Analysts reviewed each data point for consistency before inclusion\n"
        "in the final dataset used for this report."
    )
    page2.insert_textbox(
        fitz.Rect(72, 90, 500, 400),
        body2,
        fontsize=11,
        fontname="helv",
        color=(0, 0, 0),
    )

    # --- Page 3: plain body text --------------------------------------------
    page3 = doc.new_page()
    body3 = (
        "Regional Breakdown\n\n"
        "The North American region led growth with a twelve percent increase\n"
        "year over year. European markets grew more modestly, at four percent,\n"
        "while Asia-Pacific markets showed the fastest expansion at eighteen\n"
        "percent, largely attributed to new distribution partnerships signed\n"
        "earlier in the year."
    )
    page3.insert_textbox(
        fitz.Rect(72, 90, 500, 400),
        body3,
        fontsize=11,
        fontname="helv",
        color=(0, 0, 0),
    )

    doc.save(OUTPUT_PATH)
    doc.close()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
