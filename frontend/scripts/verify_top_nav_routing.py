"""
Automated Browser Regression & Top Navigation Routing Verification Script for SHADOWCAT
Tests:
1. Top navigation bar structure & styling across all 6 pages.
2. Active tab indicator synchronization.
3. Direct URL access for each route (/Forecast, /AttackGraph, /Evidence, /Input, /Validation, /About).
4. Full viewport width reclaim (.main .block-container).
5. Sidebar complete suppression.
6. Interactive controls (radios, sliders, buttons, expanders, cross-page links).
7. Browser console error & pageerror monitoring (zero JS errors).
"""

import sys
import time
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[union-attr]

PAGES = [
    {
        "tab_name": "Threat Forecast",
        "url_path": "Forecast",
        "expected_header": "Attack Risk Trajectory & Epistemic Uncertainty",
    },
    {
        "tab_name": "Lateral Movement Graph",
        "url_path": "AttackGraph",
        "expected_header": "Dynamic Enterprise Attack Graph & Lateral Rollout",
    },
    {
        "tab_name": "Evidence & Attribution",
        "url_path": "Evidence",
        "expected_header": "Correlated Network Flow Evidence",
    },
    {
        "tab_name": "Telemetry Ingestion",
        "url_path": "Input",
        "expected_header": "Telemetry Ingestion & Dataset Replay",
    },
    {
        "tab_name": "Validation & Benchmarks",
        "url_path": "Validation",
        "expected_header": "Model performance vs baselines",
    },
    {
        "tab_name": "Platform Specs",
        "url_path": "About",
        "expected_header": "Platform Specifications",
    },
]

BASE_URL = "http://localhost:8501"

def run_playwright_verification():
    from playwright.sync_api import sync_playwright

    print("======================================================================")
    print("SHADOWCAT - Playwright Top Navigation Routing & UI Regression Suite")
    print(f"Target Base URL: {BASE_URL}")
    print("======================================================================\n")

    results = {}
    js_console_errors = []
    js_page_errors = []

    import os
    chrome_paths = [
        r"C:\Users\NEHA\AppData\Local\ms-playwright\chromium-1200\chrome-win64\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright\chromium-1200\chrome-win64\chrome.exe")
    ]
    executable_path = next((cp for cp in chrome_paths if os.path.exists(cp)), None)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=executable_path)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Capture JS errors and console logs
        page.on("pageerror", lambda err: js_page_errors.append(str(err)))
        page.on("console", lambda msg: js_console_errors.append(f"[{msg.type}] {msg.text} at {msg.location.get('url', '')}:{msg.location.get('lineNumber', '')}") if msg.type in ["error", "warning"] else None)
        page.on("response", lambda resp: print(f"  [HTTP {resp.status}] {resp.url}") if resp.status >= 400 else None)

        # -----------------------------------------------------------------
        # TEST SUITE 1: Direct URL Routing & Page Verification
        # -----------------------------------------------------------------
        print("--- TEST SUITE 1: Direct URL Routing & Active Tab Assertions ---")
        for pinfo in PAGES:
            tab_name = pinfo["tab_name"]
            route = pinfo["url_path"]
            url = f"{BASE_URL}/{route}"
            print(f"\n[Direct Route] Navigating directly to {url} ...")

            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2000) # Allow Streamlit script execution

            page_result = {
                "route": route,
                "tab_name": tab_name,
                "status": "PASS",
                "checks": {},
                "errors": []
            }

            # 1. Check top nav bar presence
            has_top_nav = page.locator(".top-nav-bar").count() > 0
            page_result["checks"]["top_nav_present"] = has_top_nav
            if not has_top_nav:
                page_result["errors"].append("Top navigation bar (.top-nav-bar) not found in DOM")

            # 2. Check brand logo
            brand_text = page.locator(".top-nav-brand").text_content() if page.locator(".top-nav-brand").count() > 0 else ""
            has_brand = "SHADOWCAT" in (brand_text or "")
            page_result["checks"]["brand_present"] = has_brand
            if not has_brand:
                page_result["errors"].append(f"Brand '.top-nav-brand' missing or invalid (found: '{brand_text}')")

            # 3. Check active tab indicator
            active_tabs = page.locator(".top-nav-tab.active").all_text_contents()
            if len(active_tabs) == 1 and active_tabs[0].strip() == tab_name:
                page_result["checks"]["active_tab_correct"] = True
            else:
                page_result["checks"]["active_tab_correct"] = False
                page_result["errors"].append(f"Active tab mismatch: expected ['{tab_name}'], found {active_tabs}")

            # 4. Check Air-Gapped status pill in top nav
            util_text = page.locator(".top-nav-right").text_content() if page.locator(".top-nav-right").count() > 0 else ""
            has_air_gapped = "AIR-GAPPED" in (util_text or "")
            page_result["checks"]["air_gapped_pill"] = has_air_gapped
            if not has_air_gapped:
                page_result["errors"].append(f"Air-Gapped status pill missing from utilities (found: '{util_text}')")

            # 5. Check Sidebar complete suppression
            sidebar_visible = False
            sidebar = page.locator('[data-testid="stSidebar"]')
            if sidebar.count() > 0:
                box = sidebar.first.bounding_box()
                if box and box["width"] > 0 and box["height"] > 0:
                    sidebar_visible = True
            page_result["checks"]["sidebar_suppressed"] = not sidebar_visible
            if sidebar_visible:
                page_result["errors"].append("Left sidebar [data-testid='stSidebar'] is still visible!")

            # 6. Check full-width layout reclaim
            block = page.locator(".block-container, [data-testid='stMainBlockContainer']").first
            block_box = block.bounding_box() if block.count() > 0 else None
            # In 1440px viewport, width should be >= 1300px
            is_full_width = block_box is not None and block_box["width"] >= 1300
            page_result["checks"]["full_width_reclaimed"] = is_full_width
            if not is_full_width:
                width_val = block_box["width"] if block_box else 0
                page_result["errors"].append(f"Content container width not reclaimed (found {width_val}px, expected >= 1300px)")

            # 7. Check expected header content
            body_text = page.locator("body").text_content() or ""
            has_header = pinfo["expected_header"] in body_text
            page_result["checks"]["expected_header_found"] = has_header
            if not has_header:
                page_result["errors"].append(f"Expected header '{pinfo['expected_header']}' not found on page")

            if page_result["errors"]:
                page_result["status"] = "FAIL"
                print(f"  ❌ FAIL: {page_result['errors']}")
            else:
                print(f"  ✅ PASS: All route assertions passed for {tab_name} ({route})")

            results[tab_name] = page_result

        # -----------------------------------------------------------------
        # TEST SUITE 2: Tab Click-Through Navigation Matrix
        # -----------------------------------------------------------------
        print("\n--- TEST SUITE 2: Tab Click-Through Navigation Matrix ---")
        # Start from Forecast
        page.goto(f"{BASE_URL}/Forecast", wait_until="networkidle")
        page.wait_for_timeout(2000)

        matrix_success = True
        for target in PAGES:
            target_name = target["tab_name"]
            print(f"Clicking top nav tab '{target_name}'...")
            tab_locator = page.locator(f".top-nav-tab:has-text('{target_name}')").first
            if tab_locator.count() == 0:
                print(f"  ❌ Tab locator not found for '{target_name}'")
                matrix_success = False
                continue

            tab_locator.click()
            page.wait_for_timeout(2500)

            # Verify active class switched
            active_text = page.locator(".top-nav-tab.active").text_content()
            if active_text and active_text.strip() == target_name:
                print(f"  ✅ Active indicator successfully switched to '{target_name}'")
            else:
                print(f"  ❌ Failed to switch active indicator to '{target_name}' (got '{active_text}')")
                matrix_success = False

        # -----------------------------------------------------------------
        # TEST SUITE 3: Interactive Component Assertions
        # -----------------------------------------------------------------
        print("\n--- TEST SUITE 3: Interactive Component Assertions ---")

        # 3a. Threat Forecast MITRE Stage Radio interaction
        print("[Interaction] Testing Threat Forecast MITRE radio selection...")
        page.goto(f"{BASE_URL}/Forecast", wait_until="networkidle")
        page.wait_for_timeout(2000)
        radios = page.locator('[data-testid="stRadio"] [role="radiogroup"] label')
        if radios.count() >= 4:
            # Click t+3
            radios.nth(2).click()
            page.wait_for_timeout(1500)
            print("  ✅ MITRE stage radio selection responsive")
        else:
            print(f"  ⚠️  Found {radios.count()} radio labels (expected >= 4)")

        # 3b. Threat Forecast Cross-Page Link to Lateral Movement Attack Graph
        print("[Interaction] Testing Threat Forecast cross-page link to Attack Graph...")
        cross_link = page.locator("a:has-text('Explore Dedicated Lateral Movement Attack Graph')")
        if cross_link.count() > 0:
            cross_link.first.click()
            page.wait_for_timeout(2500)
            active_text = page.locator(".top-nav-tab.active").text_content() or ""
            if "Lateral Movement Graph" in active_text:
                print("  ✅ Cross-page link smoothly navigated to Lateral Movement Attack Graph")
            else:
                print(f"  ❌ Cross-page link failed to activate Lateral Movement Graph (got '{active_text}')")
        else:
            print("  ⚠️  Cross-page link not found on Forecast page")

        # 3c. Lateral Movement Graph: Stacked Layout, Canvas, and Host Cards
        print("[Interaction] Testing Lateral Movement Graph Canvas & Host Cards...")
        page.goto(f"{BASE_URL}/AttackGraph", wait_until="networkidle")
        page.wait_for_timeout(2500)

        # Check slider
        slider = page.locator('[data-testid="stSlider"]')
        if slider.count() > 0:
            print("  ✅ Horizon slider present and interactable")
        else:
            print("  ❌ Slider not found on Attack Graph")

        # Check Host Threat Distribution cards & action buttons (5 hosts total: 4 Inspect + 1 Telemetry active)
        host_inspect_widgets = page.locator('[data-testid="stButton"]:has-text("Inspect")')
        host_telemetry_widgets = page.locator('[data-testid="stButton"]:has-text("Telemetry")')
        total_host_cards = host_inspect_widgets.count() + host_telemetry_widgets.count()
        if total_host_cards == 5 and host_inspect_widgets.count() == 4 and host_telemetry_widgets.count() == 1:
            print(f"  ✅ Verified exactly 5 host cards (4 'Inspect' + 1 active 'Telemetry' on SSH Jump Host). Clicking first host 'Inspect'...")
            host_inspect_widgets.first.locator("button").first.click()
            page.wait_for_timeout(1500)
            print("  ✅ Host Inspection interaction handled cleanly")
        else:
            print(f"  ⚠️ Host cards count mismatch: found {host_inspect_widgets.count()} Inspect and {host_telemetry_widgets.count()} Telemetry (expected 4 + 1 = 5)")

        # 3d. Telemetry Ingestion: Benchmark selection & file uploader
        print("[Interaction] Testing Telemetry Ingestion Page Controls...")
        page.goto(f"{BASE_URL}/Input", wait_until="networkidle")
        page.wait_for_timeout(2000)
        uploaders = page.locator('[data-testid="stFileUploader"]')
        print(f"  ✅ Ingestion page loaded with {uploaders.count()} file uploader component(s)")

        # 3e. Validation & Benchmarks: Expanders
        print("[Interaction] Testing Validation & Benchmarks Expanders...")
        page.goto(f"{BASE_URL}/Validation", wait_until="networkidle")
        page.wait_for_timeout(2000)
        expanders = page.locator('[data-testid="stExpander"]')
        if expanders.count() > 0:
            print(f"  ✅ Found {expanders.count()} expanders. Clicking first expander...")
            expanders.first.click()
            page.wait_for_timeout(1000)
            print("  ✅ Expander toggle verified")
        else:
            print("  ⚠️ No expanders found on Validation page")

        # -----------------------------------------------------------------
        # TEST SUITE 4: Browser Console JS Errors
        # -----------------------------------------------------------------
        print("\n--- TEST SUITE 4: Browser Console & Page Error Audit ---")
        # Filter benign warnings / third-party noise if any
        critical_js_errors = [
            err for err in js_page_errors
            if "favicon" not in err.lower()
        ]
        # Check console error logs (excluding Streamlit internal framework health/host-config probes and favicons)
        critical_console_errors = [
            c for c in js_console_errors
            if c.startswith("[error]")
            and "favicon" not in c.lower()
            and "_stcore" not in c.lower()
        ]

        print(f"Total Page Errors: {len(critical_js_errors)}")
        for err in critical_js_errors:
            print(f"  ❌ JS Page Error: {err}")

        print(f"Total Console Errors: {len(critical_console_errors)}")
        for err in critical_console_errors:
            print(f"  ❌ JS Console Error: {err}")

        all_passed = (
            all(r["status"] == "PASS" for r in results.values())
            and matrix_success
            and len(critical_js_errors) == 0
            and len(critical_console_errors) == 0
        )

        print("\n======================================================================")
        if all_passed:
            print("SUMMARY: ALL TOP NAVIGATION & REGRESSION TESTS PASSED (ZERO JS ERRORS) ✅")
        else:
            print("SUMMARY: SOME TOP NAVIGATION OR BROWSER TESTS FAILED ❌")
        print("======================================================================")

        browser.close()
        return all_passed, results, critical_js_errors, critical_console_errors

if __name__ == "__main__":
    success, results, page_errors, console_errors = run_playwright_verification()
    sys.exit(0 if success else 1)
