"""Server-rendered HTML UI for browsing facts and relationships.

Mounted under `/ui` so it doesn't collide with the existing JSON API
routes at `/documents`, `/facts`, and `/relationships`. This is a thin
Jinja2 presentation layer over `app.store` -- no new business logic,
just enough structure to make the evidence-grounding story (verbatim
quote + section/page/offset provenance) visible for a demo.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates

from app import store

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
