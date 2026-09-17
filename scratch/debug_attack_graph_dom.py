import os
import sys
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

executable_path = r"C:\Users\NEHA\AppData\Local\ms-playwright\chromium-1200\chrome-win64\chrome.exe"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=executable_path)
    page = browser.new_page(viewport={"width": 1440, "height": 950})

    errors = []
    page.on("pageerror", lambda err: errors.append(str(err)))

    page.goto("http://localhost:8501/AttackGraph", wait_until="networkidle")
    page.wait_for_timeout(3000)

    print("Page Title:", page.title())
    print("Page Errors:", errors)

    # Check for Streamlit exception
    st_exception = page.locator('[data-testid="stException"]')
    if st_exception.count() > 0:
        print("STREAMLIT EXCEPTION DETECTED:")
        print(st_exception.first.text_content())

    # Check if Host Threat Distribution is visible
    htd = page.locator('text="Host Threat Distribution"')
    print("Host Threat Distribution header count:", htd.count())

    # Check search input
    search_input = page.locator('input')
    print("Inputs count:", search_input.count())
    for i in range(search_input.count()):
        print(f"Input {i}: placeholder='{search_input.nth(i).get_attribute('placeholder')}'")

    # Check text around hosts
    for ip in ["10.0.2.15", "10.0.4.10", "10.0.4.21", "10.0.5.1", "10.0.3.50"]:
        cnt = page.locator(f'text="{ip}"').count()
        print(f"IP {ip} count in DOM: {cnt}")

    browser.close()
