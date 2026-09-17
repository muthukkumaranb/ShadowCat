import os
import sys
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

artifact_dir = r"C:\Users\NEHA\.gemini\antigravity-ide\brain\ebba8ef6-e15d-42e6-862f-9e6c0d1ed938"
executable_path = r"C:\Users\NEHA\AppData\Local\ms-playwright\chromium-1200\chrome-win64\chrome.exe"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=executable_path)
    context = browser.new_context(viewport={"width": 1440, "height": 950})
    page = context.new_page()

    page.goto("http://localhost:8501/AttackGraph", wait_until="networkidle")
    page.wait_for_timeout(3000)

    # Filter by search "SSH"
    search_input = page.locator('input[placeholder*="Search by IP or host role"]')
    search_input.fill("SSH")
    search_input.press("Enter")
    page.wait_for_timeout(2000)

    # Scroll so the search bar, counter, and filtered card are cleanly framed
    page.locator('text="10.0.4.10"').first.scroll_into_view_if_needed()
    page.wait_for_timeout(500)

    filtered_shot_path = os.path.join(artifact_dir, "01b_Attack_Graph_Filtered.png")
    page.screenshot(path=filtered_shot_path, full_page=False)
    print(f"Captured clean filtered screenshot: {filtered_shot_path}")

    browser.close()
