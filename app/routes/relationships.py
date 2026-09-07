"""Relationships query endpoints: list and detail views for fact relationships.

``GET /relationships`` lists relationships with optional filtering by fact_id
and relation_type. ``GET /relationships/{id}`` returns a single relationship
together with both related facts and their evidence pointers (verbatim_quote,
section_path, char_offset, page_number).
"""
from typing import Optional

from fastapi import APIRouter, HTTPException

from app import store

router = APIRouter(prefix="/relationships", tags=["relationships"])


@router.get("")
def list_relationships(
    fact_id: Optional[int] = None,
    relation_type: Optional[str] = None,
) -> list[dict]:
    """List relationships, optionally filtered by fact_id or relation_type.

    Args:
        fact_id: If provided, return only relationships touching this fact
                 (on either side: fact_id_a or fact_id_b).
        relation_type: If provided, return only relationships of this type
                      (exact match against "corroborates", "contradicts",
                      "reconciled_context").

    Returns:
        JSON list of relationship objects.
    """
    relationships = store.list_relationships(fact_id=fact_id)

    # Filter by relation_type if provided
    if relation_type is not None:
        relationships = [r for r in relationships if r.relation_type == relation_type]

    return [r.model_dump() for r in relationships]


@router.get("/{relationship_id}")
def get_relationship_detail(relationship_id: int) -> dict:
    """Retrieve a relationship with both related facts and their evidence.

    Returns a JSON object with keys:
    - relationship: The relationship metadata (type, reasoning, confidence, etc.)
    - fact_a: The full fact object for fact_id_a, including evidence pointers
    - fact_b: The full fact object for fact_id_b, including evidence pointers

    Args:
        relationship_id: The ID of the relationship to retrieve.

    Returns:
        JSON object with relationship and both facts.

    Raises:
        HTTPException(404): If the relationship does not exist.
    """
    relationship = store.get_relationship(relationship_id)
    if relationship is None:
        raise HTTPException(status_code=404, detail="Relationship not found")

    fact_a = store.get_fact(relationship.fact_id_a)
    fact_b = store.get_fact(relationship.fact_id_b)

    return {
        "relationship": relationship.model_dump(),
        "fact_a": fact_a.model_dump() if fact_a else None,
        "fact_b": fact_b.model_dump() if fact_b else None,
    }
