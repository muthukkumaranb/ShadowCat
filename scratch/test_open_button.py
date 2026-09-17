import os
import sys
from playwright.sync_api import sync_playwright

executable_path = r"C:\Users\NEHA\AppData\Local\ms-playwright\chromium-1200\chrome-win64\chrome.exe"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=executable_path)
    page = browser.new_page(viewport={"width": 1440, "height": 950})

    page.goto("http://localhost:8501/AttackGraph", wait_until="networkidle")
    page.wait_for_timeout(3000)

    # Scroll to selectbox
    risk_box = page.locator('div[data-testid="stSelectbox"]').first
    risk_box.scroll_into_view_if_needed()
    page.wait_for_timeout(500)

    open_btn = risk_box.locator('button[aria-label="Open"]')
    print("Open button count:", open_btn.count())
    open_btn.click()
    page.wait_for_timeout(1000)

    options = page.locator('[role="option"]')
    print("Options count:", options.count())
    for i in range(options.count()):
        print(f"Option {i}: '{options.nth(i).inner_text()}'")

    # Click "Critical"
    crit_opt = page.locator('[role="option"]:has-text("Critical")')
    if crit_opt.count() > 0:
        print("Clicking Critical option...")
        crit_opt.click()
        page.wait_for_timeout(2000)
        print("Selected value is now:", risk_box.locator('input').get_attribute('value'))

    browser.close()
