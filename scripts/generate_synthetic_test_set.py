"""Generates two small synthetic PDF sets (3 documents each, 3-4 pages
each) for cheaply and repeatably testing the four required cases without
running the full pipeline against the large real starter dataset.

Each set is an independent fictional company with three filings that,
between them, deliberately contain:
  - Case 1 (corroboration): the same FY revenue figure stated in two
    documents in two different units (Million vs Cr).
  - Case 2 (contradiction): the same headcount date stated with two
    meaningfully different numbers, with no reconciling context given.
  - Case 3 (reconciled by context): a director/officer listed as active
    in one document and resigned in a later one (reconciled by time), and
    standalone vs consolidated revenue within the same document
    (reconciled by scope).
  - Case 4 (extraction failure): a footnote marker digit glued directly
    onto a number with no separating space (the same real failure mode
    documented in app/normalization.py's footnote-marker handling).

Not part of the graded submission -- a local testing aid, mirroring how
tests/fixtures/sample.pdf is a small synthetic PDF built for testing
rather than a real document. Run: python scripts/generate_synthetic_test_set.py
"""
import os

import fitz

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic-test-set")

BODY_SIZE = 11
HEADING_SIZE = 18
TITLE_SIZE = 20
MARGIN = 54
PAGE_WIDTH, PAGE_HEIGHT = fitz.paper_size("letter")


def new_doc():
    return fitz.open()


def add_page(doc, title, sections, is_first_page=False):
    """sections: list of (heading, paragraphs: list[str])"""
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = MARGIN

    if is_first_page:
        page.insert_text((MARGIN, y + TITLE_SIZE), title, fontsize=TITLE_SIZE, fontname="hebo")
        y += TITLE_SIZE + 30
    elif title:
        page.insert_text((MARGIN, y + HEADING_SIZE * 0.6), title, fontsize=HEADING_SIZE - 2, fontname="hebo")
        y += HEADING_SIZE + 16

    for heading, paragraphs in sections:
        page.insert_text((MARGIN, y + HEADING_SIZE * 0.7), heading, fontsize=HEADING_SIZE, fontname="hebo")
        y += HEADING_SIZE + 14
        for para in paragraphs:
            rect = fitz.Rect(MARGIN, y, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN)
            used = page.insert_textbox(rect, para, fontsize=BODY_SIZE, fontname="helv", align=0)
            # insert_textbox returns leftover space (negative if overflow);
            # approximate consumed height from paragraph length instead of
            # relying on exact metrics, generous enough for short paragraphs.
            lines = max(1, len(para) // 95 + 1)
            y += lines * (BODY_SIZE + 6) + 14
    return page


def build_set_a():
    doc_dir = os.path.join(OUT_DIR, "set-a-northwind")
    os.makedirs(doc_dir, exist_ok=True)

    # --- A1: Prospectus 2022 ---
    doc = new_doc()
    add_page(
        doc,
        "NORTHWIND LOGISTICS LIMITED\nDRAFT PROSPECTUS 2022",
        [
            (
                "CORPORATE INFORMATION",
                [
                    "Corporate Identity Number: U63090KA2015PLC090123",
                    "Registered Office: Plot 14, Whitefield Industrial Layout, Bengaluru 560066, Karnataka, India.",
                ],
            )
        ],
        is_first_page=True,
    )
    add_page(
        doc,
        "BOARD OF DIRECTORS",
        [
            (
                "Board Composition",
                [
                    "Ravi Kumar Shah (DIN 02345678) is a Non-Executive Nominee Director, nominee of Sequoia Capital India.",
                    "Ananya Desai (DIN 03456789) is our Managing Director and Chief Executive Officer.",
                    "Suresh Nair (DIN 04567890) is an Independent Director and chairs the Audit Committee.",
                ],
            )
        ],
    )
    add_page(
        doc,
        "FINANCIAL HIGHLIGHTS",
        [
            (
                "Select Financial Information",
                [
                    "Revenue from Operations (Standalone) was Rs 298.50 million for Fiscal 2020, Rs 356.20 million for Fiscal 2021, and Rs 410.75 million for Fiscal 2022.",
                    "Restated loss for the period was Rs 45.10 million for Fiscal 2022.",
                ],
            )
        ],
    )
    add_page(
        doc,
        "OPERATIONAL METRICS",
        [
            (
                "Network Infrastructure",
                [
                    "Our network operated 45⁴ sortation centres as of March 31, 2022, across 12 states.",
                    "As of March 31, 2022, we had 3,200 full-time employees.",
                    "4. Includes 5 centres under trial operations, not yet fully commissioned.",
                ],
            )
        ],
    )
    doc.save(os.path.join(doc_dir, "01-northwind-prospectus-2022.pdf"))
    doc.close()

    # --- A2: Annual Report FY24 ---
    doc = new_doc()
    add_page(
        doc,
        "NORTHWIND LOGISTICS LIMITED\nANNUAL REPORT FY24",
        [
            (
                "CORPORATE INFORMATION",
                [
                    "CIN: L63090KA2015PLC090123",
                    "Registered Office: Plot 14, Whitefield Industrial Layout, Bengaluru 560066, Karnataka, India.",
                ],
            )
        ],
        is_first_page=True,
    )
    add_page(
        doc,
        "DIRECTORS' REPORT",
        [
            (
                "Changes in Directorate",
                [
                    "Ravi Kumar Shah, Non-Executive Director, resigned from the Board with effect from June 15, 2023.",
                    "Ananya Desai continues as Managing Director and Chief Executive Officer.",
                    "The Board placed on record its appreciation for the valuable contribution made by Mr. Shah during his tenure.",
                ],
            )
        ],
    )
    add_page(
        doc,
        "FINANCIAL STATEMENTS",
        [
            (
                "Revenue from Operations",
                [
                    "Revenue from Operations (Standalone) was Rs 450.75 million for FY24.",
                    "Revenue from Operations (Consolidated) was Rs 512.30 million for FY24, reflecting the results of our subsidiary Northwind Cross-Border Logistics Pte Ltd.",
                ],
            )
        ],
    )
    add_page(
        doc,
        "HUMAN RESOURCES",
        [
            (
                "Workforce",
                [
                    "As of March 31, 2024, Northwind Logistics Limited had 5,000 permanent employees.",
                    "The Company continues to invest in employee training and welfare programs.",
                ],
            )
        ],
    )
    doc.save(os.path.join(doc_dir, "02-northwind-annual-report-fy24.pdf"))
    doc.close()

    # --- A3: Q4 FY24 Earnings Presentation ---
    doc = new_doc()
    add_page(
        doc,
        "NORTHWIND LOGISTICS LIMITED\nQ4 & FY24 EARNINGS PRESENTATION",
        [
            (
                "Disclaimer",
                ["This presentation contains certain forward-looking statements about the Company's business and financial performance."],
            )
        ],
        is_first_page=True,
    )
    add_page(
        doc,
        "FY24 FINANCIAL SUMMARY",
        [
            (
                "Revenue",
                [
                    "Northwind Logistics Limited's revenue from customers was Rs 51.23 Cr for FY24, up from Rs 40.12 Cr in FY23.",
                    "EBITDA margin improved to 8.2% in FY24 from 6.9% in FY23.",
                ],
            )
        ],
    )
    add_page(
        doc,
        "WORKFORCE",
        [
            (
                "Headcount",
                [
                    "Northwind Logistics Limited's employee headcount as of March 31, 2024 stood at 3,800, reflecting continued operational efficiency initiatives.",
                    "Attrition remained stable at 18% on an annualized basis.",
                ],
            )
        ],
    )
    doc.save(os.path.join(doc_dir, "03-northwind-q4-fy24-earnings.pdf"))
    doc.close()


def build_set_b():
    doc_dir = os.path.join(OUT_DIR, "set-b-bluepeak")
    os.makedirs(doc_dir, exist_ok=True)

    # --- B1: Investor Update 2023 ---
    doc = new_doc()
    add_page(
        doc,
        "BLUEPEAK FREIGHT PRIVATE LIMITED\nINVESTOR UPDATE 2023",
        [
            (
                "CORPORATE INFORMATION",
                [
                    "Corporate Identity Number: U60300MH2018PTC310456",
                    "Registered Office: Unit 7, Andheri Logistics Park, Mumbai 400072, Maharashtra, India.",
                ],
            )
        ],
        is_first_page=True,
    )
    add_page(
        doc,
        "LEADERSHIP TEAM",
        [
            (
                "Board of Directors",
                [
                    "Meera Iyer (DIN 08765432) serves as an Independent Director on our Board.",
                    "Arjun Malhotra (DIN 07654321) is our Founder and Managing Director.",
                ],
            )
        ],
    )
    add_page(
        doc,
        "KEY METRICS FY23",
        [
            (
                "Financial and Operational Highlights",
                [
                    "Revenue from Operations was Rs 210.40 million for Fiscal 2023.",
                    "Fleet size stood at 1,150 vehicles as of December 31, 2023, operating across 8 states.",
                ],
            )
        ],
    )
    doc.save(os.path.join(doc_dir, "01-bluepeak-investor-update-2023.pdf"))
    doc.close()

    # --- B2: Annual Report FY25 ---
    doc = new_doc()
    add_page(
        doc,
        "BLUEPEAK FREIGHT PRIVATE LIMITED\nANNUAL REPORT FY25",
        [
            (
                "CORPORATE INFORMATION",
                [
                    "CIN: U60300MH2018PTC310456",
                    "Registered Office: Unit 7, Andheri Logistics Park, Mumbai 400072, Maharashtra, India.",
                ],
            )
        ],
        is_first_page=True,
    )
    add_page(
        doc,
        "CORPORATE GOVERNANCE",
        [
            (
                "Changes to the Board",
                [
                    "Meera Iyer, Independent Director, ceased to be a Director with effect from January 10, 2024, upon completion of her term of appointment.",
                    "Arjun Malhotra continues as Founder and Managing Director.",
                ],
            )
        ],
    )
    add_page(
        doc,
        "FINANCIAL RESULTS",
        [
            (
                "Revenue from Operations",
                [
                    "Revenue from Operations (Standalone) was Rs 289.60 million for FY25.",
                    "Revenue from Operations (Consolidated) was Rs 334.15 million for FY25, including the results of our subsidiary Bluepeak Cold Chain Solutions Private Limited.",
                ],
            )
        ],
    )
    add_page(
        doc,
        "FLEET & OPERATIONS",
        [
            (
                "Workforce and Fleet",
                [
                    "Total fleet strength stood at 1,480 vehicles⁵ as of March 31, 2025.",
                    "The Company employed 2,100 personnel as of March 31, 2025.",
                    "5. Excludes vehicles leased from third-party fleet partners on a short-term basis.",
                ],
            )
        ],
    )
    doc.save(os.path.join(doc_dir, "02-bluepeak-annual-report-fy25.pdf"))
    doc.close()

    # --- B3: Q4 FY25 Investor Presentation ---
    doc = new_doc()
    add_page(
        doc,
        "BLUEPEAK FREIGHT PRIVATE LIMITED\nQ4 & FY25 INVESTOR PRESENTATION",
        [
            (
                "Disclaimer",
                ["This presentation contains forward-looking statements and is not an offer to sell securities."],
            )
        ],
        is_first_page=True,
    )
    add_page(
        doc,
        "FY25 HIGHLIGHTS",
        [
            (
                "Revenue",
                [
                    "Bluepeak Freight Private Limited's revenue was Rs 33.4 Cr for FY25, representing year-on-year growth of 19%.",
                    "The Company continued to expand its cold chain logistics footprint during the year.",
                ],
            )
        ],
    )
    add_page(
        doc,
        "TEAM",
        [
            (
                "Headcount",
                [
                    "Bluepeak Freight Private Limited's headcount as of March 31, 2025 stood at 1,650 employees across all functions.",
                    "The Company added 6 new regional warehouses during FY25.",
                ],
            )
        ],
    )
    doc.save(os.path.join(doc_dir, "03-bluepeak-q4-fy25-presentation.pdf"))
    doc.close()


if __name__ == "__main__":
    build_set_a()
    build_set_b()
    for root, _, files in os.walk(OUT_DIR):
        for f in sorted(files):
            path = os.path.join(root, f)
            print(path, "-", fitz.open(path).page_count, "pages")
