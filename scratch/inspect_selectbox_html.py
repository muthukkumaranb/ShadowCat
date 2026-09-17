import os
import sys
from playwright.sync_api import sync_playwright

executable_path = r"C:\Users\NEHA\AppData\Local\ms-playwright\chromium-1200\chrome-win64\chrome.exe"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=executable_path)
    page = browser.new_page(viewport={"width": 1440, "height": 950})

    page.goto("http://localhost:8501/AttackGraph", wait_until="networkidle")
    page.wait_for_timeout(3000)

    sel = page.locator('div[data-testid="stSelectbox"]').first
    print("Outer HTML:", sel.evaluate("el => el.outerHTML"))

    browser.close()
