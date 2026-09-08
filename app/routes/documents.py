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

import asyncio
import logging
import os
import re
import tempfile
from collections import Counter

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app import comparison, config, extraction, jobs, normalization, sectioning, store
from app.llm.factory import build_default_llm_client
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
    try:
        return build_default_llm_client()
    except ValueError as exc:
        raise HTTPException(
            status_code=500, detail=f"No LLM API keys configured: {exc}"
        ) from exc


def _is_pdf(filename: str | None, content_type: str | None) -> bool:
    fn = (filename or "").lower()
    if fn.endswith(".pdf"):
        return True
    ct = (content_type or "").lower()
    return "pdf" in ct


def _is_pdf_upload(file: UploadFile) -> bool:
    return _is_pdf(file.filename, file.content_type)


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


async def _process_one_document(contents: bytes, filename: str, llm_client, on_stage=None) -> dict:
    """Runs the full ingest -> section -> extract -> normalize -> store -> compare
    pipeline for one already-read PDF's bytes. Returns the same response dict
    shape upload_document currently returns for a single file.

    Raises HTTPException(400) if the PDF bytes can't be parsed -- callers
    that need to keep processing other files after one fails (e.g. the
    batch endpoint) should catch that themselves.

    ``on_stage``, if given, is a plain (non-async) callable invoked as
    ``on_stage(stage: str, detail: str | None = None)`` right before each
    major pipeline step, purely for progress reporting (e.g. an in-memory
    job tracker for the async batch endpoint / UI). It changes nothing
    about the pipeline's behavior or return shape, and is never invoked
    when omitted.
    """
    if on_stage:
        on_stage("parsing")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contents)
            temp_path = tmp.name

        try:
            pages = parse_pdf(temp_path)
        except PDFParseError as exc:
            logger.warning(
                "Rejecting upload %r: PDF could not be parsed: %s", filename, exc
            )
            raise HTTPException(status_code=400, detail=f"Could not parse PDF: {exc}") from exc
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)

    if on_stage:
        on_stage("sectioning")
    sections = sectioning.detect_sections(pages)

    if on_stage:
        on_stage("extracting")

    def _on_batch(batch_number: int, total_batches: int) -> None:
        on_stage("extracting", f"batch {batch_number}/{total_batches}")

    located_facts = await extraction.extract_facts_from_document(
        sections, llm_client, on_batch=_on_batch if on_stage else None
    )

    if on_stage:
        on_stage("normalizing")
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

    if on_stage:
        on_stage("storing")
    entity_name = _best_effort_entity_name(located_facts)
    document_id = store.insert_document(
        filename=filename,
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

    if on_stage:
        on_stage("comparing")
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
        "filename": filename,
        "section_count": len(sections),
        "facts_extracted": len(located_facts),
        "fact_ids": fact_ids,
        "entity_name": entity_name,
        "relationships_found": len(relationships),
    }


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

    return await _process_one_document(contents, file.filename, llm_client)


@router.post("/documents/batch")
async def upload_documents_batch(
    files: list[UploadFile] = File(...),
    llm_client=Depends(get_llm_client),
):
    """Batch variant of ``POST /documents``: runs the same pipeline for
    each file sequentially (not concurrently -- each file's own extraction
    already fans out internally via MAX_CONCURRENT_LLM_CALLS, and racing
    whole documents on top of that would just fight over the same
    rate-limited pool for no benefit).

    Per-file validation is identical to the single-upload path, but a bad
    file never aborts the batch: it's recorded as an error entry in
    ``results`` and processing continues with the next file. Because each
    document's facts are stored (via `_process_one_document` ->
    `compare_new_document_facts`) before the next document in the batch is
    processed, within-batch relationships (doc 2 vs doc 1, etc.) emerge
    naturally from the existing sequential design -- no special handling
    needed here.
    """
    results = []
    documents_processed = 0
    documents_failed = 0
    total_facts_extracted = 0
    total_relationships_found = 0

    for file in files:
        contents = await file.read()

        if len(contents) > config.UPLOAD_MAX_BYTES:
            documents_failed += 1
            results.append(
                {
                    "filename": file.filename,
                    "error": (
                        f"File too large: {len(contents)} bytes exceeds the "
                        f"{config.UPLOAD_MAX_BYTES}-byte limit."
                    ),
                }
            )
            continue

        if not _is_pdf_upload(file):
            documents_failed += 1
            results.append(
                {
                    "filename": file.filename,
                    "error": "Uploaded file must be a PDF (.pdf extension or PDF content-type).",
                }
            )
            continue

        try:
            result = await _process_one_document(contents, file.filename, llm_client)
        except HTTPException as exc:
            documents_failed += 1
            results.append({"filename": file.filename, "error": str(exc.detail)})
            continue

        documents_processed += 1
        total_facts_extracted += result["facts_extracted"]
        total_relationships_found += result["relationships_found"]
        results.append(result)

    return {
        "results": results,
        "documents_processed": documents_processed,
        "documents_failed": documents_failed,
        "total_facts_extracted": total_facts_extracted,
        "total_relationships_found": total_relationships_found,
    }


async def _run_batch_job(
    job_id: str,
    file_data: list[tuple[str, str | None, bytes]],
    llm_client,
) -> None:
    """Runs the same per-file pipeline as ``POST /documents/batch`` for an
    already-created job, reporting progress into ``app.jobs`` as it goes.

    Shared by both ``POST /documents/batch/async`` and the UI's
    ``POST /ui/upload`` so the two entry points can't drift apart -- the
    only difference between them is what happens after this returns
    (nothing, vs. a page render).

    ``file_data`` is a list of ``(filename, content_type, contents)``
    tuples -- callers must read each ``UploadFile``'s bytes (and note its
    filename/content_type) *before* calling this, since the request has
    already returned by the time this runs as a background task and an
    ``UploadFile``'s underlying stream isn't guaranteed to survive past
    the request handler.

    Must never leave the job stuck "running" forever: every branch that
    can raise is caught, and the outermost `except` is a last-resort net
    that marks the whole job failed on anything truly unexpected.
    """
    try:
        for i, (filename, content_type, contents) in enumerate(file_data):
            jobs.set_current_file(job_id, i)

            if len(contents) > config.UPLOAD_MAX_BYTES:
                jobs.set_file_error(
                    job_id,
                    i,
                    f"File too large: {len(contents)} bytes exceeds the "
                    f"{config.UPLOAD_MAX_BYTES}-byte limit.",
                )
                continue

            if not _is_pdf(filename, content_type):
                jobs.set_file_error(
                    job_id,
                    i,
                    "Uploaded file must be a PDF (.pdf extension or PDF content-type).",
                )
                continue

            def on_stage(stage: str, detail: str | None = None, _job_id=job_id, _i=i) -> None:
                jobs.set_file_stage(_job_id, _i, stage, detail)

            try:
                result = await _process_one_document(contents, filename, llm_client, on_stage=on_stage)
            except HTTPException as exc:
                jobs.set_file_error(job_id, i, str(exc.detail))
                continue

            jobs.set_file_result(job_id, i, result)

        jobs.mark_job_completed(job_id)
    except Exception as exc:  # noqa: BLE001 - a background task's exception has nowhere else to go; the job must not be left stuck "running" forever
        logger.exception("Batch job %s failed unexpectedly", job_id)
        jobs.mark_job_failed(job_id, str(exc))


@router.post("/documents/batch/async")
async def upload_documents_batch_async(
    files: list[UploadFile] = File(...),
    llm_client=Depends(get_llm_client),
):
    """Kicks off the same per-file pipeline as ``POST /documents/batch``,
    but in a background asyncio task, returning immediately with a job id
    to poll via ``GET /documents/batch/status/{job_id}``.

    ``POST /documents`` and ``POST /documents/batch`` are left exactly as
    they are for simple synchronous curl/API use; this is an additive,
    non-breaking alternative for callers (including the browser UI) that
    want progress visibility instead of a request that hangs until the
    whole batch finishes.
    """
    file_data = [(f.filename, f.content_type, await f.read()) for f in files]
    job = jobs.create_job([filename for filename, _, _ in file_data])

    asyncio.create_task(_run_batch_job(job.id, file_data, llm_client))

    return {"job_id": job.id}


@router.get("/documents/batch/status/{job_id}")
async def get_batch_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return jobs.job_status_payload(job)
