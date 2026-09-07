"""Facts listing and detail endpoints.

GET /facts — list facts with optional filtering by document_id, entity_name, attribute
GET /facts/{fact_id} — detail view with fact + relationships evidence payload
"""
from typing import Optional

from fastapi import APIRouter, HTTPException

from app import store

router = APIRouter()


@router.get("/facts")
def list_facts(
    document_id: Optional[int] = None,
    entity_name: Optional[str] = None,
    attribute: Optional[str] = None,
):
    """List facts with optional query-param filtering.

    - document_id: filter to a specific document
    - entity_name: case-insensitive substring match on entity_name
    - attribute: case-insensitive substring match on attribute
    """
    # Start with the base query
    if document_id is not None:
        facts = store.list_facts(document_id=document_id)
    else:
        facts = store.get_all_facts()

    # Apply optional filters in Python
    if entity_name is not None:
        entity_name_lower = entity_name.lower()
        facts = [f for f in facts if entity_name_lower in f.entity_name.lower()]

    if attribute is not None:
        attribute_lower = attribute.lower()
        facts = [f for f in facts if attribute_lower in f.attribute.lower()]

    return facts


@router.get("/facts/{fact_id}")
def get_fact_detail(fact_id: int):
    """Get fact detail with evidence payload and relationships.

    Returns 404 if fact not found.
    Structure: {"fact": {...}, "relationships": [...]}
    """
    fact = store.get_fact(fact_id)
    if fact is None:
        raise HTTPException(status_code=404, detail=f"Fact {fact_id} not found")

    relationships = store.list_relationships(fact_id=fact_id)

    return {
        "fact": fact,
        "relationships": relationships,
    }
