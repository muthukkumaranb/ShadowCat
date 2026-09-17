import os
import sys
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

executable_path = r"C:\Users\NEHA\AppData\Local\ms-playwright\chromium-1200\chrome-win64\chrome.exe"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=executable_path)
    page = browser.new_page(viewport={"width": 1440, "height": 950})

    page.goto("http://localhost:8501/AttackGraph", wait_until="networkidle")
    page.wait_for_timeout(3000)

    # Find the element containing "#1 10.0."
    host_elem = page.locator('text="#1 10.0."').first
    parent = host_elem.locator('xpath=ancestor::div[contains(@style, "border-left")]').first
    if parent.count() > 0:
        print("Parent style:", parent.get_attribute("style"))
    else:
        print("Could not find ancestor with border-left!")
        print("Host elem parent HTML:", host_elem.locator('xpath=..').evaluate('el => el.outerHTML'))

    browser.close()
