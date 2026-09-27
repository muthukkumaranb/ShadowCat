"""
SHADOWCAT Backend - Test Suite for reports_db.py
Validates:
1. init_db() creates schema idempotently
2. insert_report() stores records correctly
3. query_reports() retrieves records newest-first
4. Filtering by severity and record_type works
5. A failed/malformed insert doesn't raise past the caller
6. query_reports() returns [] for a missing DB file
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add paths
TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
WORKSPACE_DIR = BACKEND_DIR.parent

sys.path.insert(0, str(WORKSPACE_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from backend.reports_db import init_db, insert_report, query_reports


class TestReportsDB(unittest.TestCase):
    """Test suite for the persistent reports SQLite database."""

    def setUp(self):
        """Create a fresh temp DB for each test."""
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test_reports.db"

    def tearDown(self):
        """Clean up temp DB after each test."""
        if self.db_path.exists():
            os.remove(self.db_path)
        if os.path.exists(self.tmp_dir):
            # Clean up WAL/SHM files if they exist
            for suffix in ["-wal", "-shm"]:
                wal = Path(str(self.db_path) + suffix)
                if wal.exists():
                    os.remove(wal)
            os.rmdir(self.tmp_dir)

    def test_init_db_creates_schema_idempotently(self):
        """init_db() creates DB file and schema, safe to call repeatedly."""
        self.assertFalse(self.db_path.exists())
        result1 = init_db(db_path=self.db_path)
        self.assertTrue(result1)
        self.assertTrue(self.db_path.exists())

        # Second call should succeed without error (idempotent)
        result2 = init_db(db_path=self.db_path)
        self.assertTrue(result2)

    def test_insert_and_query_alert(self):
        """insert_report stores an alert and query_reports retrieves it."""
        init_db(db_path=self.db_path)

        alert_record = {
            "alert_hash": "abc123def456",
            "window_id": "win_001",
            "window_start": "2025-01-15T12:00:00Z",
            "max_risk": 0.8742,
            "severity": "HIGH",
            "notarized_via": "sha256_fallback",
        }
        ok = insert_report("alert", alert_record, db_path=self.db_path)
        self.assertTrue(ok)

        results = query_reports(db_path=self.db_path)
        self.assertEqual(len(results), 1)

        row = results[0]
        self.assertEqual(row["record_type"], "alert")
        self.assertEqual(row["record_id"], "abc123def456")
        self.assertEqual(row["severity"], "HIGH")
        self.assertEqual(row["notarized_via"], "sha256_fallback")
        self.assertAlmostEqual(row["max_risk"], 0.8742, places=4)
        self.assertIn("created_at", row)
        # payload_json should parse back to the original dict
        self.assertIn("payload", row)
        self.assertEqual(row["payload"]["alert_hash"], "abc123def456")

    def test_insert_and_query_lineage(self):
        """insert_report stores a lineage record and query_reports retrieves it."""
        init_db(db_path=self.db_path)

        lineage_record = {
            "lineage_id": "lin_win001_abcd1234",
            "raw_data_hash": "aabbccdd" * 8,
            "feature_hash": "11223344" * 8,
            "model_id": "lstm_world_model_v4",
            "prediction_hash": "deadbeef" * 8,
            "severity": "MEDIUM",
            "target_node": "172.31.69.21",
            "timestamp": "2025-01-15T12:00:00Z",
            "notarized_via": "sha256_fallback",
        }
        ok = insert_report("lineage", lineage_record, db_path=self.db_path)
        self.assertTrue(ok)

        results = query_reports(db_path=self.db_path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["record_type"], "lineage")
        self.assertEqual(results[0]["record_id"], "lin_win001_abcd1234")
        self.assertEqual(results[0]["model_id"], "lstm_world_model_v4")

    def test_query_filters_by_severity(self):
        """query_reports(severity=...) correctly filters results."""
        init_db(db_path=self.db_path)

        insert_report("alert", {"alert_hash": "h1", "severity": "HIGH", "notarized_via": "sha256_fallback"}, db_path=self.db_path)
        insert_report("alert", {"alert_hash": "m1", "severity": "MEDIUM", "notarized_via": "sha256_fallback"}, db_path=self.db_path)
        insert_report("alert", {"alert_hash": "l1", "severity": "LOW", "notarized_via": "sha256_fallback"}, db_path=self.db_path)

        highs = query_reports(severity="HIGH", db_path=self.db_path)
        self.assertEqual(len(highs), 1)
        self.assertEqual(highs[0]["record_id"], "h1")

        mediums = query_reports(severity="MEDIUM", db_path=self.db_path)
        self.assertEqual(len(mediums), 1)

    def test_query_filters_by_record_type(self):
        """query_reports(record_type=...) correctly filters results."""
        init_db(db_path=self.db_path)

        insert_report("alert", {"alert_hash": "a1", "severity": "HIGH", "notarized_via": "sha256_fallback"}, db_path=self.db_path)
        insert_report("lineage", {"lineage_id": "l1", "severity": "MEDIUM", "notarized_via": "fabric"}, db_path=self.db_path)

        alerts = query_reports(record_type="alert", db_path=self.db_path)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["record_type"], "alert")

        lineages = query_reports(record_type="lineage", db_path=self.db_path)
        self.assertEqual(len(lineages), 1)
        self.assertEqual(lineages[0]["record_type"], "lineage")

    def test_query_combined_filters(self):
        """Combined severity + type filtering works."""
        init_db(db_path=self.db_path)

        insert_report("alert", {"alert_hash": "a1", "severity": "HIGH", "notarized_via": "sha256_fallback"}, db_path=self.db_path)
        insert_report("alert", {"alert_hash": "a2", "severity": "MEDIUM", "notarized_via": "sha256_fallback"}, db_path=self.db_path)
        insert_report("lineage", {"lineage_id": "l1", "severity": "HIGH", "notarized_via": "fabric"}, db_path=self.db_path)

        results = query_reports(severity="HIGH", record_type="alert", db_path=self.db_path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["record_id"], "a1")

    def test_query_returns_newest_first(self):
        """Results are ordered by created_at descending (newest first)."""
        init_db(db_path=self.db_path)

        insert_report("alert", {"alert_hash": "first", "severity": "LOW", "notarized_via": "sha256_fallback"}, db_path=self.db_path)
        insert_report("alert", {"alert_hash": "second", "severity": "LOW", "notarized_via": "sha256_fallback"}, db_path=self.db_path)
        insert_report("alert", {"alert_hash": "third", "severity": "LOW", "notarized_via": "sha256_fallback"}, db_path=self.db_path)

        results = query_reports(db_path=self.db_path)
        self.assertEqual(len(results), 3)
        # Newest (last inserted) should be first
        self.assertEqual(results[0]["record_id"], "third")
        self.assertEqual(results[2]["record_id"], "first")

    def test_query_respects_limit(self):
        """query_reports(limit=N) caps results at N."""
        init_db(db_path=self.db_path)

        for i in range(10):
            insert_report("alert", {"alert_hash": f"h{i}", "severity": "LOW", "notarized_via": "sha256_fallback"}, db_path=self.db_path)

        results = query_reports(limit=3, db_path=self.db_path)
        self.assertEqual(len(results), 3)

    def test_malformed_insert_does_not_raise(self):
        """A malformed record logs a warning but doesn't raise past the caller."""
        init_db(db_path=self.db_path)

        # None as record_dict — should not raise
        ok = insert_report("alert", None, db_path=self.db_path)
        self.assertFalse(ok)  # Should return False, not crash

    def test_query_missing_db_returns_empty(self):
        """query_reports on a non-existent DB file returns [] without error."""
        nonexistent = Path(self.tmp_dir) / "nonexistent.db"
        results = query_reports(db_path=nonexistent)
        self.assertEqual(results, [])

    def test_insert_preserves_full_payload_json(self):
        """The full original record dict is stored as payload_json and can be parsed back."""
        init_db(db_path=self.db_path)

        record = {
            "alert_hash": "test_payload",
            "severity": "HIGH",
            "notarized_via": "fabric",
            "extra_field_1": "should be preserved",
            "extra_nested": {"key": "value"},
        }
        insert_report("alert", record, db_path=self.db_path)
        results = query_reports(db_path=self.db_path)
        payload = results[0]["payload"]
        self.assertEqual(payload["extra_field_1"], "should be preserved")
        self.assertEqual(payload["extra_nested"]["key"], "value")


if __name__ == "__main__":
    unittest.main()
