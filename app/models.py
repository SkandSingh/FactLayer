"""Pydantic v2 data models shared across the FactLayer pipeline."""
from typing import Literal, Optional

from pydantic import BaseModel


class TimeScope(BaseModel):
    period_type: Literal["point_in_time", "span", "unknown"]
    start: Optional[str] = None      # ISO date string
    end: Optional[str] = None        # ISO date string
    label: Optional[str] = None      # verbatim label, e.g. "FY24"
    vintage: Optional[str] = None    # e.g. "first advance estimate", "provisional", "final"


class ExtractedFact(BaseModel):
    entity_name: str
    entity_id: Optional[str] = None
    attribute: str
    raw_value: str
    unit: Optional[str] = None
    time_scope: TimeScope
    entity_scope: Optional[str] = None   # e.g. "standalone", "consolidated"
    verbatim_quote: str
    confidence: float


class Fact(ExtractedFact):
    id: int
    document_id: int
    normalized_value: Optional[float] = None
    normalized_unit: Optional[str] = None
    section_path: str
    char_offset: int
    page_number: int
    created_at: str


class Relationship(BaseModel):
    id: int
    fact_id_a: int
    fact_id_b: int
    relation_type: Literal["corroborates", "contradicts", "reconciled_context"]
    reconciled_dimension: Optional[Literal["time", "scope", "units", "estimate_vintage", "other"]] = None
    reasoning_text: str
    confidence: float
    created_at: str


class Document(BaseModel):
    id: int
    filename: str
    uploaded_at: str
    entity_name: Optional[str] = None
    section_count: int
