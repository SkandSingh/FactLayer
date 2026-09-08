"""Server-rendered HTML UI for browsing facts and relationships.

Mounted under `/ui` so it doesn't collide with the existing JSON API
routes at `/documents`, `/facts`, and `/relationships`. This is a thin
Jinja2 presentation layer over `app.store` -- no new business logic,
just enough structure to make the evidence-grounding story (verbatim
quote + section/page/offset provenance) visible for a demo.
"""
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.templating import Jinja2Templates

from app import config, store
from app.routes.documents import _is_pdf_upload, _process_one_document, get_llm_client

router = APIRouter(prefix="/ui")
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def index(request: Request, entity_name: Optional[str] = None):
    documents = store.list_documents()
    facts = store.get_all_facts()

    if entity_name:
        entity_name_lower = entity_name.lower()
        facts = [f for f in facts if entity_name_lower in f.entity_name.lower()]

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "documents": documents,
            "facts": facts,
            "filter_entity_name": entity_name,
        },
    )


@router.get("/facts/{fact_id}")
async def fact_detail(request: Request, fact_id: int):
    fact = store.get_fact(fact_id)
    if fact is None:
        raise HTTPException(status_code=404, detail=f"Fact {fact_id} not found")

    relationships = store.list_relationships(fact_id=fact_id)

    return templates.TemplateResponse(
        request,
        "fact_detail.html",
        {
            "fact": fact,
            "relationships": relationships,
        },
    )


@router.get("/relationships/{relationship_id}")
async def relationship_detail(request: Request, relationship_id: int):
    relationship = store.get_relationship(relationship_id)
    if relationship is None:
        raise HTTPException(status_code=404, detail=f"Relationship {relationship_id} not found")

    fact_a = store.get_fact(relationship.fact_id_a)
    fact_b = store.get_fact(relationship.fact_id_b)
    if fact_a is None or fact_b is None:
        raise HTTPException(
            status_code=404,
            detail=f"Relationship {relationship_id} references a missing fact",
        )

    return templates.TemplateResponse(
        request,
        "relationship_detail.html",
        {
            "relationship": relationship,
            "fact_a": fact_a,
            "fact_b": fact_b,
        },
    )


@router.get("/upload")
async def upload_page(request: Request):
    return templates.TemplateResponse(
        request,
        "upload.html",
        {"results": None, "documents_failed": 0},
    )


@router.post("/upload")
async def upload_page_submit(
    request: Request,
    files: list[UploadFile] = File(...),
    llm_client=Depends(get_llm_client),
):
    """Browser-facing upload form: accepts one or several PDFs, runs the
    same pipeline as `POST /documents`/`POST /documents/batch`, and renders
    a result page instead of returning raw JSON. No new business logic --
    this reuses `app.routes.documents`'s validation and per-document
    processing directly so the two entry points can never drift apart.
    """
    results = []
    documents_failed = 0

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

        results.append(result)

    return templates.TemplateResponse(
        request,
        "upload.html",
        {"results": results, "documents_failed": documents_failed},
    )
