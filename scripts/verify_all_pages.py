#!/usr/bin/env python3
"""
SHADOWCAT - Comprehensive Multi-Page Regression & Governance Audit
Tests all 5 primary views against strict pass/fail criteria:
  (a) Zero unhandled exceptions / tracebacks
  (b) Zero placeholder / debug strings ('traceback', 'nameerror', 'keyerror', etc.)
  (c) Proper mock-data governance (sticky banner, offline validation banner, ZERO per-widget badges)
  (d) Header uniqueness and layout integrity
  (e) Zero emojis in rendered output
  (f) Revision G compliance: Assert ZERO call sites of get_mock_badge_html() outside header.py
"""

import os
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PAGES = [
    {
        "name": "Threat Forecast",
        "file": "views/01_Forecast.py",
        "requires_banner": True,
        "unique_headers": [
            "Attack Risk Trajectory & Epistemic Uncertainty",
            "MITRE ATT&CK Killchain Progression"
        ],
    },
    {
        "name": "Lateral Movement Graph",
        "file": "views/01b_AttackGraph.py",
        "requires_banner": True,
        "unique_headers": [
            "Lateral Movement Attack Graph",
            "Dynamic Enterprise Attack Graph & Lateral Rollout",
            "Host Threat Distribution"
        ],
    },
    {
        "name": "Evidence & Attribution",
        "file": "views/02_Evidence.py",
        "requires_banner": True,
        "unique_headers": [
            "Correlated Network Flow Evidence",
            "Telemetry Evidence & Forensic Attribution"
        ],
    },
    {
        "name": "Telemetry Ingestion",
        "file": "views/01a_Input.py",
        "requires_banner": True,
        "unique_headers": [
            "Telemetry Ingestion & Dataset Replay"
        ],
    },
    {
        "name": "Validation & Benchmarks",
        "file": "views/03_Validation.py",
        "requires_banner": True,
        "unique_headers": [
            "Model Performance vs Baselines",
            "Horizon Stability Benchmarks"
        ],
    },
    {
        "name": "Platform Specifications",
        "file": "views/05_About.py",
        "requires_banner": True,
        "unique_headers": [
            "Platform Specifications",
            "Core Operational Capabilities"
        ],
    },
]

FORBIDDEN_DEBUG_STRINGS = ["traceback", "nameerror", "keyerror", "typeerror", "syntaxerror"]

def check_revision_g_badge_call_sites() -> tuple[bool, list[str]]:
    """
    Revision G: Assert zero call sites of get_mock_badge_html() outside header.py across
    all views, components, and primary application files.
    """
    violations = []
    targets = ["views", "components", "app.py", "styles.py"]

    for t in targets:
        target_path = ROOT / t
        if target_path.is_file():
            files = [target_path]
        elif target_path.is_dir():
            files = list(target_path.rglob("*.py"))
        else:
            continue

        for f in files:
            # Skip header.py as allowed by Revision G specification
            if f.name == "header.py":
                continue

            content = f.read_text(encoding="utf-8", errors="ignore")
            for idx, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                # Ignore comments
                if stripped.startswith("#"):
                    continue
                if "get_mock_badge_html(" in line:
                    violations.append(
                        f"{f.relative_to(ROOT)}:{idx}: Unauthorized call site of get_mock_badge_html(): {stripped}"
                    )

    passed = (len(violations) == 0)
    return passed, violations


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
        "zero_emojis": False,
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

        # (c) Mock governance & badge deprecation check
        has_per_widget_badge = "badge-mock" in full_html
        if has_per_widget_badge:
            result["errors"].append("Deprecated per-widget [MOCK] badge found in rendered HTML")

        has_header_banner = (
            "Running on benchmark data" in full_html
            or "BENCHMARK MODE" in full_html
            or "● LIVE" in full_html
            or "Model validated offline" in full_html
        )

        if page_info.get("requires_banner"):
            if not has_header_banner:
                result["errors"].append("Missing required sticky benchmark or offline validation banner")
            result["mock_governed"] = (not has_per_widget_badge) and has_header_banner
        else:
            result["mock_governed"] = (not has_per_widget_badge)

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

    # 1. Revision G Gate: Assert zero call sites of get_mock_badge_html() outside header.py
    print("\n--- Revision G Code Inspection: Assert 0 Call Sites of get_mock_badge_html() ---")
    call_sites_ok, call_site_violations = check_revision_g_badge_call_sites()
    if call_sites_ok:
        print("[PASS] Revision G Verified: Exactly ZERO call sites of get_mock_badge_html() outside header.py.")
    else:
        print(f"[FAIL] Revision G Violation: Found {len(call_site_violations)} forbidden call sites:")
        for v in call_site_violations:
            print(f"   • {v}")

    # 2. Page-by-page audit
    print("\n--- Multi-Page Rendering & Governance Audit ---")
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
    all_passed = call_sites_ok and all(r["passed"] for r in results)
    if all_passed:
        print("ALL 6 PAGES AND REVISION G GOVERNANCE CHECKS PASSED.")
    else:
        print("SOME CHECKS FAILED. SEE DETAILS ABOVE.")
    print("=" * 70)

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
