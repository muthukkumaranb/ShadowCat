import os
import sys
from playwright.sync_api import sync_playwright

executable_path = r"C:\Users\NEHA\AppData\Local\ms-playwright\chromium-1200\chrome-win64\chrome.exe"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=executable_path)
    page = browser.new_page(viewport={"width": 1440, "height": 950})

    page.goto("http://localhost:8501/AttackGraph", wait_until="networkidle")
    page.wait_for_timeout(3000)

    # Scroll down to selectbox
    risk_box = page.locator('div[data-testid="stSelectbox"]').first
    risk_box.scroll_into_view_if_needed()
    page.wait_for_timeout(500)

    inp = risk_box.locator('input')
    print("Input in risk box count:", inp.count())
    if inp.count() > 0:
        print("Clicking input...")
        inp.click()
        page.wait_for_timeout(1000)
        print("Options role count after input click:", page.locator('[role="option"]').count())
        for i in range(page.locator('[role="option"]').count()):
            print(f"Option {i}: '{page.locator('[role=\"option\"]').nth(i).inner_text()}'")

        if page.locator('[role="option"]').count() == 0:
            print("Typing 'Critical' and pressing Enter...")
            inp.fill("Critical")
            inp.press("Enter")
            page.wait_for_timeout(2000)
            print("Value in selectbox now:", risk_box.inner_text())

    browser.close()
