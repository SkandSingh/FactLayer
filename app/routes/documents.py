"""Document upload endpoint: wires the full ingestion pipeline together.

``POST /documents`` takes a single uploaded PDF and, synchronously, runs it
through every stage described in docs/ARCHITECTURE.md:

    upload -> pdf_ingest.parse_pdf -> sectioning.detect_sections
           -> extraction.extract_facts_from_document (LLM)
           -> normalization.* (best-effort)
           -> store.insert_document / store.insert_fact

Normalization is deliberately best-effort: a fact whose quantity or time
scope can't be confidently normalized is still stored (with whatever
fields normalization *could* fill in) rather than dropped, since the raw
extraction is still valuable evidence even when normalization falls short.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from collections import Counter

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app import comparison, config, extraction, normalization, sectioning, store
from app.llm.gemini_client import GeminiClient
from app.pdf_ingest import PDFParseError, parse_pdf

logger = logging.getLogger(__name__)

router = APIRouter()

# Loose heuristic for "does this raw_value look date-like" -- used only to
# decide whether it's worth *trying* parse_point_in_time on a fact's
# raw_value (as opposed to its time_scope.label). Deliberately permissive:
# a false positive here just costs a wasted (harmless, exception-guarded)
# parse attempt, not a correctness issue.
_DATE_LIKE_RE = re.compile(
    r"\d{1,4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,4}"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE,
)


def _looks_date_like(text: str | None) -> bool:
    return bool(text) and bool(_DATE_LIKE_RE.search(text))


def get_llm_client():
    return GeminiClient()


def _is_pdf_upload(file: UploadFile) -> bool:
    filename = (file.filename or "").lower()
    if filename.endswith(".pdf"):
        return True
    content_type = (file.content_type or "").lower()
    return "pdf" in content_type


def _fill_time_scope(time_scope, raw_value: str) -> None:
    """Best-effort in-place fill of a TimeScope's start/end fields.

    Never raises: normalization helpers return None (or, in principle,
    could raise on unexpected input) when they can't confidently parse
    something, and a normalization miss must never prevent the fact
    itself from being stored.
    """
    if not time_scope.start or not time_scope.end:
        label = time_scope.label
        if label:
            try:
                fy = normalization.parse_fiscal_year(label)
            except Exception as exc:
                logger.warning(
                    "Normalization miss: parse_fiscal_year(%r) raised: %s",
                    label,
                    exc,
                )
                fy = None
            if fy:
                if not time_scope.start:
                    time_scope.start = fy["start"]
                if not time_scope.end:
                    time_scope.end = fy["end"]

    if time_scope.period_type == "point_in_time" and not time_scope.start:
        candidates = []
        if time_scope.label:
            candidates.append(time_scope.label)
        if _looks_date_like(raw_value):
            candidates.append(raw_value)

        for candidate in candidates:
            try:
                point = normalization.parse_point_in_time(candidate)
            except Exception as exc:
                logger.warning(
                    "Normalization miss: parse_point_in_time(%r) raised: %s",
                    candidate,
                    exc,
                )
                point = None
            if point:
                time_scope.start = point
                break


def _best_effort_entity_name(located_facts: list) -> str | None:
    names = [lf.fact.entity_name for lf in located_facts if lf.fact.entity_name]
    if not names:
        return None
    return Counter(names).most_common(1)[0][0]


@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    llm_client=Depends(get_llm_client),
):
    contents = await file.read()

    if len(contents) > config.UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {len(contents)} bytes exceeds the "
                f"{config.UPLOAD_MAX_BYTES}-byte limit."
            ),
        )

    if not _is_pdf_upload(file):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a PDF (.pdf extension or PDF content-type).",
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contents)
            temp_path = tmp.name

        try:
            pages = parse_pdf(temp_path)
        except PDFParseError as exc:
            logger.warning(
                "Rejecting upload %r: PDF could not be parsed: %s", file.filename, exc
            )
            raise HTTPException(status_code=400, detail=f"Could not parse PDF: {exc}") from exc
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)

    sections = sectioning.detect_sections(pages)
    located_facts = await extraction.extract_facts_from_document(sections, llm_client)

    for located_fact in located_facts:
        fact = located_fact.fact
        try:
            _fill_time_scope(fact.time_scope, fact.raw_value)
        except Exception as exc:
            # Normalization is best-effort; never let it stop a fact from
            # being stored -- but a raised exception here (as opposed to
            # a normal "couldn't confidently parse" miss) is unexpected
            # and worth a trace.
            logger.warning(
                "Normalization miss: _fill_time_scope raised for fact "
                "attribute=%r raw_value=%r: %s",
                fact.attribute,
                fact.raw_value,
                exc,
            )

    entity_name = _best_effort_entity_name(located_facts)
    document_id = store.insert_document(
        filename=file.filename,
        entity_name=entity_name,
        section_count=len(sections),
    )

    fact_ids = []
    for located_fact in located_facts:
        fact = located_fact.fact
        try:
            normalized_value = normalization.normalize_quantity(fact.raw_value, fact.unit)
        except Exception as exc:
            # normalize_quantity normally signals "couldn't confidently
            # parse" by returning None, not by raising -- an exception
            # here is unexpected and worth a trace, but must still never
            # prevent the fact itself from being stored.
            logger.warning(
                "Normalization miss: normalize_quantity(%r, %r) raised: %s",
                fact.raw_value,
                fact.unit,
                exc,
            )
            normalized_value = None
        normalized_unit = fact.unit

        fact_id = store.insert_fact(
            document_id,
            fact,
            normalized_value=normalized_value,
            normalized_unit=normalized_unit,
            section_path=located_fact.section_path,
            char_offset=located_fact.char_offset,
            page_number=located_fact.page_number,
        )
        fact_ids.append(fact_id)

    fetched = [(fact_id, store.get_fact(fact_id)) for fact_id in fact_ids]
    missing_ids = [fact_id for fact_id, f in fetched if f is None]
    if missing_ids:
        logger.warning(
            "Document %s: %d just-inserted fact id(s) could not be "
            "re-fetched from the store and will be excluded from "
            "comparison: %s",
            document_id,
            len(missing_ids),
            missing_ids,
        )
    new_facts = [f for _, f in fetched if f is not None]
    relationships = await comparison.compare_new_document_facts(new_facts, llm_client)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "section_count": len(sections),
        "facts_extracted": len(located_facts),
        "fact_ids": fact_ids,
        "entity_name": entity_name,
        "relationships_found": len(relationships),
    }
