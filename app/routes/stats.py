"""Stats endpoint for aggregate counts across the knowledge layer.

GET /stats — returns counts of documents, facts, relationships, and breakdown by relation_type
"""
from collections import Counter

from fastapi import APIRouter

from app import store

router = APIRouter()


@router.get("/stats")
async def get_stats():
    """Get aggregate statistics for the knowledge layer.

    Returns:
    - total_documents: count of all documents
    - total_facts: count of all facts
    - total_relationships: count of all relationships
    - relationships_by_type: breakdown of relationships by relation_type (includes all known types)
    """
    documents = store.list_documents()
    facts = store.get_all_facts()
    relationships = store.list_relationships()

    # Count relationships by type
    relation_types = [r.relation_type for r in relationships]
    relation_counter = Counter(relation_types)

    # Ensure all known relationship types appear in the output
    known_types = {"corroborates", "contradicts", "reconciled_context"}
    relationships_by_type = {
        rtype: relation_counter.get(rtype, 0) for rtype in known_types
    }

    return {
        "total_documents": len(documents),
        "total_facts": len(facts),
        "total_relationships": len(relationships),
        "relationships_by_type": relationships_by_type,
    }
