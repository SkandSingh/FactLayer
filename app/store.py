"""Data-access layer for FactLayer's SQLite store.

Plain functions; each opens its own connection via `db.get_connection()`
and ensures the schema exists via `db.init_db()` before doing any work.
"""
from datetime import datetime, timezone
from typing import List, Optional

from app import db
from app.models import Document, ExtractedFact, Fact, Relationship, TimeScope


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_document(filename: str, entity_name: Optional[str], section_count: int) -> int:
    conn = db.get_connection()
    try:
        db.init_db(conn)
        cur = conn.execute(
            """
            INSERT INTO documents (filename, uploaded_at, entity_name, section_count)
            VALUES (?, ?, ?, ?)
            """,
            (filename, _now(), entity_name, section_count),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_document(document_id: int) -> Optional[Document]:
    conn = db.get_connection()
    try:
        db.init_db(conn)
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        return _row_to_document(row) if row is not None else None
    finally:
        conn.close()


def list_documents() -> List[Document]:
    conn = db.get_connection()
    try:
        db.init_db(conn)
        rows = conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
        return [_row_to_document(row) for row in rows]
    finally:
        conn.close()


def insert_fact(
    document_id: int,
    fact: ExtractedFact,
    normalized_value: Optional[float] = None,
    normalized_unit: Optional[str] = None,
    section_path: str = "",
    char_offset: int = 0,
    page_number: int = 0,
) -> int:
    """Insert a fact derived from an `ExtractedFact` plus pipeline metadata.

    `fact` supplies the LLM-extracted fields (entity_name, entity_id,
    attribute, raw_value, unit, time_scope, entity_scope,
    verbatim_quote, confidence); the remaining arguments are the
    deterministic fields added later in the pipeline (normalization,
    section/page/offset provenance).
    """
    conn = db.get_connection()
    try:
        db.init_db(conn)
        cur = conn.execute(
            """
            INSERT INTO facts (
                document_id, entity_name, entity_id, attribute, raw_value, unit,
                normalized_value, normalized_unit, time_scope_json, entity_scope,
                verbatim_quote, section_path, char_offset, page_number, confidence,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                fact.entity_name,
                fact.entity_id,
                fact.attribute,
                fact.raw_value,
                fact.unit,
                normalized_value,
                normalized_unit,
                fact.time_scope.model_dump_json(),
                fact.entity_scope,
                fact.verbatim_quote,
                section_path,
                char_offset,
                page_number,
                fact.confidence,
                _now(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_fact(fact_id: int) -> Optional[Fact]:
    conn = db.get_connection()
    try:
        db.init_db(conn)
        row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return _row_to_fact(row) if row is not None else None
    finally:
        conn.close()


def list_facts(document_id: Optional[int] = None) -> List[Fact]:
    conn = db.get_connection()
    try:
        db.init_db(conn)
        if document_id is None:
            rows = conn.execute("SELECT * FROM facts ORDER BY id").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM facts WHERE document_id = ? ORDER BY id",
                (document_id,),
            ).fetchall()
        return [_row_to_fact(row) for row in rows]
    finally:
        conn.close()


def get_all_facts() -> List[Fact]:
    """Every fact ever stored, used later for cross-document comparison."""
    return list_facts(document_id=None)


def insert_relationship(
    fact_id_a: int,
    fact_id_b: int,
    relation_type: str,
    reconciled_dimension: Optional[str],
    reasoning_text: str,
    confidence: float,
) -> int:
    conn = db.get_connection()
    try:
        db.init_db(conn)
        cur = conn.execute(
            """
            INSERT INTO relationships (
                fact_id_a, fact_id_b, relation_type, reconciled_dimension,
                reasoning_text, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact_id_a,
                fact_id_b,
                relation_type,
                reconciled_dimension,
                reasoning_text,
                confidence,
                _now(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def relationship_exists(fact_id_a: int, fact_id_b: int) -> bool:
    """Whether a relationship between these two facts is already recorded,
    in either order. Guards against the same pair being independently
    rediscovered and re-inserted across separate comparison passes (e.g.
    document B's comparison call surfaces a pair that document C's
    comparison call, run later, also surfaces against the whole store) --
    within a single `compare_new_document_facts` call this is already
    deduped, but nothing previously prevented it across separate calls.
    """
    conn = db.get_connection()
    try:
        db.init_db(conn)
        row = conn.execute(
            """
            SELECT 1 FROM relationships
            WHERE (fact_id_a = ? AND fact_id_b = ?)
               OR (fact_id_a = ? AND fact_id_b = ?)
            LIMIT 1
            """,
            (fact_id_a, fact_id_b, fact_id_b, fact_id_a),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def get_relationship(relationship_id: int) -> Optional[Relationship]:
    conn = db.get_connection()
    try:
        db.init_db(conn)
        row = conn.execute(
            "SELECT * FROM relationships WHERE id = ?", (relationship_id,)
        ).fetchone()
        return _row_to_relationship(row) if row is not None else None
    finally:
        conn.close()


def list_relationships(fact_id: Optional[int] = None) -> List[Relationship]:
    """List relationships, optionally filtered to those touching `fact_id`."""
    conn = db.get_connection()
    try:
        db.init_db(conn)
        if fact_id is None:
            rows = conn.execute("SELECT * FROM relationships ORDER BY id").fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM relationships
                WHERE fact_id_a = ? OR fact_id_b = ?
                ORDER BY id
                """,
                (fact_id, fact_id),
            ).fetchall()
        return [_row_to_relationship(row) for row in rows]
    finally:
        conn.close()


def _row_to_document(row) -> Document:
    return Document(
        id=row["id"],
        filename=row["filename"],
        uploaded_at=row["uploaded_at"],
        entity_name=row["entity_name"],
        section_count=row["section_count"],
    )


def _row_to_fact(row) -> Fact:
    time_scope = TimeScope.model_validate_json(row["time_scope_json"])
    return Fact(
        id=row["id"],
        document_id=row["document_id"],
        entity_name=row["entity_name"],
        entity_id=row["entity_id"],
        attribute=row["attribute"],
        raw_value=row["raw_value"],
        unit=row["unit"],
        normalized_value=row["normalized_value"],
        normalized_unit=row["normalized_unit"],
        time_scope=time_scope,
        entity_scope=row["entity_scope"],
        verbatim_quote=row["verbatim_quote"],
        section_path=row["section_path"],
        char_offset=row["char_offset"],
        page_number=row["page_number"],
        confidence=row["confidence"],
        created_at=row["created_at"],
    )


def _row_to_relationship(row) -> Relationship:
    return Relationship(
        id=row["id"],
        fact_id_a=row["fact_id_a"],
        fact_id_b=row["fact_id_b"],
        relation_type=row["relation_type"],
        reconciled_dimension=row["reconciled_dimension"],
        reasoning_text=row["reasoning_text"],
        confidence=row["confidence"],
        created_at=row["created_at"],
    )
