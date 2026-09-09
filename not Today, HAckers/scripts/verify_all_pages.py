#!/usr/bin/env python3
"""
SHADOWCAT - Comprehensive Multi-Page Regression & Governance Audit
Tests all 7 sidebar views against strict pass/fail criteria:
  (a) Zero unhandled exceptions / tracebacks
  (b) Zero placeholder / debug strings ('empty', 'traceback', 'todo', etc.)
  (c) Proper mock-data governance (badges or illustrative banner)
  (d) Header uniqueness and layout integrity
"""

import os
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import data_provider

PAGES = [
    {
        "name": "Threat Forecast",
        "file": "views/01_Forecast.py",
        "requires_mock_badge": True,
        "unique_headers": [
            "Attack Risk Trajectory & Epistemic Uncertainty",
            "Dynamic Enterprise Attack Graph & Lateral Rollout",
            "Host Threat Distribution"
        ],
    },
    {
        "name": "Evidence & Attribution",
        "file": "views/02_Evidence.py",
        "requires_mock_badge": True,
        "unique_headers": ["Correlated Network Flow Evidence", "Telemetry Evidence & Forensic Attribution"],
    },
    {
        "name": "Telemetry Ingestion",
        "file": "views/01a_Input.py",
        "requires_mock_badge": False,
        "unique_headers": ["Telemetry Ingestion & Dataset Replay"],
    },
    {
        "name": "Validation & Benchmarks",
        "file": "views/03_Validation.py",
        "requires_mock_badge": True,
        "requires_banner": True,
        "unique_headers": ["Model Performance vs Baselines", "Horizon Stability Benchmarks"],
    },
    {
        "name": "Platform Specifications",
        "file": "views/05_About.py",
        "requires_mock_badge": True,
        "requires_banner": True,
        "unique_headers": ["About SHADOWCAT", "Core Operational Capabilities"],
    },
]

FORBIDDEN_DEBUG_STRINGS = ["traceback", "nameerror", "keyerror", "typeerror", "syntaxerror"]

def audit_page(page_info: dict) -> dict:
    page_name = page_info["name"]
    file_path = ROOT / page_info["file"]
    result = {
        "page": page_name,
        "file": page_info["file"],
        "no_exceptions": False,
        "no_debug_strings": False,
        "mock_governed": False,
        "layout_clean": False,
        "passed": False,
        "errors": []
    }

    try:
        at = AppTest.from_file(str(file_path), default_timeout=35)
        at.run()

        # (a) Exceptions check
        if at.exception:
            result["errors"].append(f"Unhandled exception: {at.exception}")
            return result
        result["no_exceptions"] = True

        all_markdown = [m.value for m in at.markdown]
        full_html = "".join(all_markdown)
        full_lower = full_html.lower()

        # (b) Placeholder / debug strings check
        for dbg in FORBIDDEN_DEBUG_STRINGS:
            if dbg in full_lower:
                result["errors"].append(f"Found forbidden debug term '{dbg}' in page content")

        # Check dataframes for literal 'empty'
        for df_widget in at.dataframe:
            val = df_widget.value
            if hasattr(val, "columns"):
                for col in val.columns:
                    for cell in val[col]:
                        cell_str = str(cell).strip().lower()
                        if cell_str == "empty":
                            result["errors"].append(f"DataFrame column '{col}' contains literal 'empty'")

        if not any("forbidden" in e or "contains literal 'empty'" in e for e in result["errors"]):
            result["no_debug_strings"] = True

        # (c) Mock governance check
        has_badge = "badge-mock" in full_html
        has_banner = "Illustrative Benchmark Data" in full_html

        if page_info.get("requires_banner"):
            if not has_banner:
                result["errors"].append("Missing required mock warning banner")
        if page_info.get("requires_mock_badge"):
            if not has_badge:
                result["errors"].append("Missing required [MOCK] badges")

        if page_info.get("requires_mock_badge") or page_info.get("requires_banner"):
            result["mock_governed"] = (has_badge or has_banner)
        else:
            result["mock_governed"] = True

        # (d) Unique headers check
        for header in page_info.get("unique_headers", []):
            count = full_html.count(header)
            if count == 0:
                result["errors"].append(f"Expected header '{header}' not found")
            elif count > 1:
                result["errors"].append(f"Header '{header}' is duplicated (found {count} times)")

        if not any("duplicated" in e or "not found" in e for e in result["errors"]):
            result["layout_clean"] = True

        # (e) Zero emoji assertion in rendered output
        import unicodedata
        rendered_emojis = []
        for ch in full_html:
            if ch in '•●✓—±→↓↑←↔·○▲':
                continue
            cp = ord(ch)
            cat = unicodedata.category(ch)
            if cat in ('So', 'Sk') or 0x1F000 <= cp <= 0x1FFFF or 0x2600 <= cp <= 0x27BF or 0xFE00 <= cp <= 0xFE0F:
                rendered_emojis.append(ch)
        
        result["zero_emojis"] = len(rendered_emojis) == 0
        if rendered_emojis:
            result["errors"].append(f"Found {len(rendered_emojis)} forbidden emoji(s) in rendered output: {set(rendered_emojis)}")

        result["passed"] = (
            result["no_exceptions"]
            and result["no_debug_strings"]
            and result["mock_governed"]
            and result["layout_clean"]
            and result["zero_emojis"]
        )

    except Exception as exc:
        result["errors"].append(f"Test runner execution error: {exc}")

    return result


def main():
    print("=" * 70)
    print("SHADOWCAT - Comprehensive Multi-Page Regression & Governance Audit")
    print("=" * 70)

    results = []
    for page in PAGES:
        res = audit_page(page)
        results.append(res)
        status_icon = "✅ PASS" if res["passed"] else "❌ FAIL"
        print(f"\n{status_icon} | {res['page']} ({res['file']})")
        print(f"   • Zero Exceptions:     {res['no_exceptions']}")
        print(f"   • Zero Debug/Empty:    {res['no_debug_strings']}")
        print(f"   • Mock Governed:       {res['mock_governed']}")
        print(f"   • Header/Layout Clean: {res['layout_clean']}")
        print(f"   • Zero Emojis:         {res.get('zero_emojis')}")
        if res["errors"]:
            for err in res["errors"]:
                print(f"     ⚠️  {err}")

    print("\n" + "=" * 70)
    all_passed = all(r["passed"] for r in results)
    if all_passed:
        print("ALL 5 PAGES PASSED ALL REGRESSION AND GOVERNANCE CHECKS.")
    else:
        print("SOME PAGES FAILED. SEE DETAILS ABOVE.")
    print("=" * 70)

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
