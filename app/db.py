"""SQLite connection handling and schema initialization.

Plain `sqlite3` from the standard library, no ORM.
"""
import sqlite3
from typing import Optional

from app import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    uploaded_at TEXT,
    entity_name TEXT,
    section_count INTEGER
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER,
    entity_name TEXT,
    entity_id TEXT,
    attribute TEXT,
    raw_value TEXT,
    unit TEXT,
    normalized_value REAL,
    normalized_unit TEXT,
    time_scope_json TEXT,
    entity_scope TEXT,
    verbatim_quote TEXT,
    section_path TEXT,
    char_offset INTEGER,
    page_number INTEGER,
    confidence REAL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id_a INTEGER,
    fact_id_b INTEGER,
    relation_type TEXT,
    reconciled_dimension TEXT,
    reasoning_text TEXT,
    confidence REAL,
    created_at TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    """Open a new connection to the FactLayer SQLite database.

    Reads the path from `config.FACTLAYER_DB_PATH` at call time (so
    monkeypatching `config.FACTLAYER_DB_PATH` in tests takes effect).
    """
    conn = sqlite3.connect(config.FACTLAYER_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: Optional[sqlite3.Connection] = None) -> None:
    """Create the documents/facts/relationships tables if they don't exist.

    Accepts an existing connection to reuse, or opens (and closes) its
    own connection when called with no arguments.
    """
    owns_conn = conn is None
    if conn is None:
        conn = get_connection()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        if owns_conn:
            conn.close()
