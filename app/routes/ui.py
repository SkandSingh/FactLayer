"""Server-rendered HTML UI for browsing facts and relationships.

Mounted under `/ui` so it doesn't collide with the existing JSON API
routes at `/documents`, `/facts`, and `/relationships`. This is a thin
Jinja2 presentation layer over `app.store` -- no new business logic,
just enough structure to make the evidence-grounding story (verbatim
quote + section/page/offset provenance) visible for a demo.
"""
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app import jobs, store
from app.routes.documents import _run_batch_job, get_llm_client

router = APIRouter(prefix="/ui")
templates = Jinja2Templates(directory="app/templates")

# Plain-language labels for app.jobs' FileProgress.stage values, used on the
# upload status page.
STAGE_LABELS = {
    "queued": "Queued",
    "parsing": "Parsing PDF",
    "sectioning": "Detecting sections",
    "extracting": "Extracting facts",
    "normalizing": "Normalizing values",
    "storing": "Storing facts",
    "comparing": "Comparing against existing documents",
    "done": "Done",
    "failed": "Failed",
}


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
            "job_running": jobs.any_running(),
            "running_job_id": jobs.most_recent_running_job_id(),
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
    """Browser-facing upload form: accepts one or several PDFs and kicks off
    the same background job as `POST /documents/batch/async` (via the
    shared `_run_batch_job` helper, so the two entry points can never drift
    apart), then redirects to a status page to watch it run.

    Processing used to happen synchronously here, which meant the browser
    just hung until the whole batch finished with zero feedback. A 303
    redirect (rather than 302) is used so a page refresh on the status page
    re-fetches that page instead of resubmitting the upload form.
    """
    file_data = [(f.filename, f.content_type, await f.read()) for f in files]
    job = jobs.create_job([filename for filename, _, _ in file_data])

    asyncio.create_task(_run_batch_job(job.id, file_data, llm_client))

    return RedirectResponse(url=f"/ui/upload/status/{job.id}", status_code=303)


@router.get("/upload/status/{job_id}")
async def upload_status(request: Request, job_id: str):
    """Shows the live progress of one upload batch.

    Renders a full server-side snapshot of the job's current state (so it
    works with JS disabled -- just without live updates), and includes a
    small inline script that polls `GET /documents/batch/status/{job_id}`
    to update the page in place while the job is still running.
    """
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return templates.TemplateResponse(
        request,
        "upload_status.html",
        {
            "job": job,
            "job_id": job_id,
            "stage_labels": STAGE_LABELS,
        },
    )
