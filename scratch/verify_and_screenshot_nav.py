import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\NEHA\.gemini\antigravity-ide\brain\ebba8ef6-e15d-42e6-862f-9e6c0d1ed938"
os.makedirs(artifact_dir, exist_ok=True)

chrome_paths = [
    r"C:\Users\NEHA\AppData\Local\ms-playwright\chromium-1200\chrome-win64\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright\chromium-1200\chrome-win64\chrome.exe")
]
executable_path = next((cp for cp in chrome_paths if os.path.exists(cp)), None)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=executable_path)
    # Create fresh context with cache disabled to guarantee hard-refresh behavior
    context = browser.new_context(
        viewport={"width": 1440, "height": 900},
        ignore_https_errors=True,
    )
    page = context.new_page()

    # Route to Forecast
    print("Navigating to http://localhost:8501/Forecast ...")
    page.goto("http://localhost:8501/Forecast", wait_until="networkidle")
    page.wait_for_timeout(2000)

    # Perform hard refresh (reload)
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(2000)

    # Inspect computed styles for all tabs
    tabs = page.locator(".top-nav-tab").all()
    print(f"Found {len(tabs)} tabs.")

    styles_report = []
    for i, tab in enumerate(tabs):
        txt = tab.text_content().strip()
        is_active = "active" in (tab.get_attribute("class") or "")
        eval_script = """
        el => {
            const cs = window.getComputedStyle(el);
            return {
                textDecorationLine: cs.textDecorationLine,
                textDecoration: cs.textDecoration,
                borderBottomStyle: cs.borderBottomStyle,
                borderBottomWidth: cs.borderBottomWidth,
                borderBottomColor: cs.borderBottomColor,
                color: cs.color
            };
        }
        """
        style = tab.evaluate(eval_script)
        styles_report.append((txt, is_active, style))
        print(f"\nTab: '{txt}' (active={is_active})")
        print(f"  textDecorationLine: {style['textDecorationLine']}")
        print(f"  borderBottom: {style['borderBottomWidth']} {style['borderBottomStyle']} {style['borderBottomColor']}")

    # Capture dedicated close-up of top nav bar
    nav_bar = page.locator(".top-nav-bar")
    nav_screenshot_path = os.path.join(artifact_dir, "top_nav_bar_closeup.png")
    nav_bar.screenshot(path=nav_screenshot_path)
    print(f"\nSaved nav bar close-up: {nav_screenshot_path}")

    # Capture full header and page view
    full_screenshot_path = os.path.join(artifact_dir, "01_Threat_Forecast_verified_nav.png")
    page.screenshot(path=full_screenshot_path, full_page=False)
    print(f"Saved full page view: {full_screenshot_path}")

    # Also capture Lateral Movement Graph page to verify active switch
    page.goto("http://localhost:8501/AttackGraph", wait_until="networkidle")
    page.wait_for_timeout(2000)
    ag_nav_path = os.path.join(artifact_dir, "attack_graph_nav_closeup.png")
    page.locator(".top-nav-bar").screenshot(path=ag_nav_path)
    print(f"Saved Attack Graph nav bar close-up: {ag_nav_path}")

    browser.close()

print("\nAll verification and screenshot tasks completed successfully!")
