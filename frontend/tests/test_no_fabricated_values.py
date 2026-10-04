"""
Unit and integration test verifying absence of fabricated / synthetic values
and validating that all Streamlit views load cleanly via AppTest without exceptions.
"""

import os
import sys
import pytest
from pathlib import Path

# Set up paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
VIEWS_DIR = FRONTEND_DIR / "views"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

# Prohibited fabricated strings that must never appear in the frontend codebase
BANNED_STRINGS = [
    "0.835",
    "Crossed at t-",
    "m remaining",
    "NOMINAL",
    "Conformal Bound",
    "p_onset*0.9",
    "q=0.12",
    "900 + i*150",
    "+15m",
    "mock_data",
]


def test_no_fabricated_values_in_frontend():
    """Grep test that fails on any occurrence of prohibited fabricated strings across frontend code."""
    violations = []
    this_file = Path(__file__).resolve()

    # Search all python files and relevant templates in frontend
    for root, dirs, files in os.walk(FRONTEND_DIR):
        # Exclude pycache, virtualenvs, or tests directory where banned strings are listed
        if "__pycache__" in root or ".pytest_cache" in root:
            continue

        for fname in files:
            fpath = Path(root) / fname
            if fpath.resolve() == this_file:
                continue
            if not (fname.endswith(".py") or fname.endswith(".md") or fname.endswith(".json")):
                continue
            # Skip historical / problem statement scrape references if any
            if fname == "problem-statement-scrape-data.json":
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for banned in BANNED_STRINGS:
                if banned in content:
                    violations.append(f"{fpath.relative_to(REPO_ROOT)} contains prohibited string '{banned}'")

    assert not violations, "Found prohibited fabricated strings:\n" + "\n".join(violations)


def test_streamlit_apptest_pages():
    """Streamlit AppTest loading every page in frontend/views/ asserting no exceptions."""
    from streamlit.testing.v1 import AppTest
    import data_provider

    # Warm up live prediction cache once so pages execute rapidly
    data_provider._get_live_prediction()

    view_files = sorted(VIEWS_DIR.glob("*.py"))
    assert len(view_files) > 0, "No views found in frontend/views/"

    failed_pages = []
    for vf in view_files:
        if vf.name.startswith("__"):
            continue
        try:
            at = AppTest.from_file(str(vf), default_timeout=60)
            at.run()
            if at.exception:
                failed_pages.append(f"{vf.name}: {at.exception}")
        except Exception as e:
            failed_pages.append(f"{vf.name} failed with runtime error: {e}")

    assert not failed_pages, "Streamlit AppTest failed on pages:\n" + "\n".join(failed_pages)
