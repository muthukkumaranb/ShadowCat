"""
SHADOWCAT Backend - Persistent Reports Database (SQLite)
Durable storage for alert and prediction-lineage records, queryable by the
dashboard "Reports & History" view.

Design constraints:
- stdlib sqlite3 only (no new dependencies)
- A write failure must NEVER break predict() — always fail soft (log + continue)
- Safe to call init_db() repeatedly (idempotent schema creation)
- DB file lives at backend/data/shadowcat_reports.db (runtime-generated, gitignored)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Resolve DB path: backend/data/shadowcat_reports.db
_BACKEND_DIR = Path(__file__).resolve().parent
_DB_DIR = _BACKEND_DIR / "data"
_DB_PATH = _DB_DIR / "shadowcat_reports.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    window_id TEXT,
    window_start TEXT,
    severity TEXT,
    max_risk REAL,
    notarized_via TEXT,
    target_node TEXT,
    raw_data_hash TEXT,
    feature_hash TEXT,
    prediction_hash TEXT,
    model_id TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_reports_severity ON reports(severity);",
    "CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(record_type);",
]


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a connection with WAL journal mode and a 5-second busy timeout."""
    path = db_path or _DB_PATH
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> bool:
    """
    Create the DB file, directory, and schema if they don't already exist.
    Safe to call repeatedly — uses CREATE TABLE IF NOT EXISTS.

    Returns True on success, False on failure (logged, never raises).
    """
    path = db_path or _DB_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = _get_connection(path)
        try:
            conn.execute(_CREATE_TABLE_SQL)
            for idx_sql in _CREATE_INDEXES_SQL:
                conn.execute(idx_sql)
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception as e:
        logger.warning(f"reports_db.init_db failed: {e}")
        return False


def insert_report(
    record_type: str,
    record_dict: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> bool:
    """
    Insert one alert or lineage record into the reports table.

    Extracts known columns from record_dict; stores the full original dict
    as payload_json for anything not broken into a dedicated column.

    Returns True on success, False on failure (logged, never raises past the
    caller — this must never break predict()).
    """
    path = db_path or _DB_PATH
    try:
        # Ensure schema exists (cheap no-op if already created)
        init_db(path)

        # Determine the canonical record_id
        if record_type == "alert":
            record_id = record_dict.get("alert_hash", "unknown")
        elif record_type == "lineage":
            record_id = record_dict.get("lineage_id", "unknown")
        else:
            record_id = record_dict.get("alert_hash") or record_dict.get("lineage_id") or "unknown"

        # Extract known columns with safe .get() defaults
        window_id = record_dict.get("window_id")
        window_start = record_dict.get("window_start") or record_dict.get("timestamp")
        severity = record_dict.get("severity")
        max_risk_raw = record_dict.get("max_risk")
        max_risk = float(max_risk_raw) if max_risk_raw is not None else None
        notarized_via = record_dict.get("notarized_via")
        target_node = record_dict.get("target_node")
        raw_data_hash = record_dict.get("raw_data_hash")
        feature_hash = record_dict.get("feature_hash")
        prediction_hash = record_dict.get("prediction_hash")
        model_id = record_dict.get("model_id")

        # Full payload as JSON for anything not broken into a column
        payload_json = json.dumps(record_dict, default=str, sort_keys=True)

        created_at = datetime.now(timezone.utc).isoformat()

        conn = _get_connection(path)
        try:
            conn.execute(
                """
                INSERT INTO reports (
                    record_type, record_id, window_id, window_start, severity,
                    max_risk, notarized_via, target_node, raw_data_hash,
                    feature_hash, prediction_hash, model_id, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_type,
                    str(record_id),
                    str(window_id) if window_id is not None else None,
                    str(window_start) if window_start is not None else None,
                    severity,
                    max_risk,
                    notarized_via,
                    str(target_node) if target_node is not None else None,
                    raw_data_hash,
                    feature_hash,
                    prediction_hash,
                    model_id,
                    payload_json,
                    created_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info(f"reports_db: inserted {record_type} record '{record_id}'")
        return True

    except Exception as e:
        logger.warning(f"reports_db.insert_report failed ({record_type}): {e}")
        return False


def query_reports(
    severity: Optional[str] = None,
    record_type: Optional[str] = None,
    limit: int = 100,
    since: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Query reports from the database, newest first.

    Args:
        severity: Filter by severity level (e.g. 'HIGH', 'MEDIUM', 'LOW').
        record_type: Filter by record type ('alert' or 'lineage').
        limit: Maximum number of results (default 100).
        since: ISO timestamp — only return records created at or after this time.
        db_path: Override DB path (used in tests).

    Returns a list of dicts, newest first. Returns [] on any error or if the
    DB is empty/missing (never crashes the dashboard).
    """
    path = db_path or _DB_PATH
    try:
        if not path.exists():
            return []

        conn = _get_connection(path)
        try:
            clauses: list[str] = []
            params: list[Any] = []

            if severity is not None:
                clauses.append("severity = ?")
                params.append(severity)
            if record_type is not None:
                clauses.append("record_type = ?")
                params.append(record_type)
            if since is not None:
                clauses.append("created_at >= ?")
                params.append(since)

            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            sql = f"SELECT * FROM reports{where} ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                row_dict = dict(row)
                # Parse payload_json back to dict for convenience
                pj = row_dict.get("payload_json")
                if pj:
                    try:
                        row_dict["payload"] = json.loads(pj)
                    except (json.JSONDecodeError, TypeError):
                        row_dict["payload"] = {}
                results.append(row_dict)

            return results
        finally:
            conn.close()

    except Exception as e:
        logger.warning(f"reports_db.query_reports failed: {e}")
        return []
