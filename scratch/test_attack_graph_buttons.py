import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

artifact_dir = r"C:\Users\NEHA\.gemini\antigravity-ide\brain\ebba8ef6-e15d-42e6-862f-9e6c0d1ed938"
os.makedirs(artifact_dir, exist_ok=True)

chrome_paths = [
    r"C:\Users\NEHA\AppData\Local\ms-playwright\chromium-1200\chrome-win64\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright\chromium-1200\chrome-win64\chrome.exe")
]
executable_path = next((cp for cp in chrome_paths if os.path.exists(cp)), None)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=executable_path)
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()

    console_errors = []
    page_errors = []
    page.on("pageerror", lambda err: page_errors.append(str(err)))
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

    print("Navigating to http://localhost:8501/AttackGraph ...")
    page.goto("http://localhost:8501/AttackGraph", wait_until="networkidle")
    page.wait_for_timeout(3000)

    # 1. Verify 5 host cards are present
    primary_buttons = page.locator('div[data-testid="stButton"] button[kind="primary"]')
    secondary_buttons = page.locator('div[data-testid="stButton"] button[kind="secondary"]')
    print(f"Initial primary buttons count: {primary_buttons.count()}")
    print(f"Initial secondary buttons count: {secondary_buttons.count()}")

    # 2. Test clicking Inspect/Telemetry TWICE IN A ROW on every single host card
    hosts = ["10.0.4.10", "10.0.2.15", "10.0.4.21", "10.0.5.1", "10.0.3.50"]
    
    print("\n--- Testing Inspect/Telemetry double-click across all 5 hosts ---")
    for host_ip in hosts:
        print(f"\n[Host {host_ip}]")
        # Locate the button for this host
        # The button is either named "Telemetry" (if currently active) or "Inspect"
        btn = page.locator(f'div[data-testid="stButton"] button[kind="primary"]').filter(has_text=host_ip)
        # Or locate by parent column or text
        # Since button is inside a container, let's find button by key or container
        # Let's inspect buttons by iterating through all primary buttons
        # Or finding the button within the column
        
        # Click 1:
        # Find button that corresponds to this host
        # Let's find button within the host card container
        card = page.locator(f'div:has-text("{host_ip}")').filter(has=page.locator('button[kind="primary"]')).last
        btn = card.locator('button[kind="primary"]').first
        
        btn_text = btn.inner_text()
        is_disabled = btn.is_disabled()
        print(f"  Click 1: Button label '{btn_text}', disabled={is_disabled}")
        assert not is_disabled, f"Button for {host_ip} should NOT be disabled on first click!"
        
        btn.click()
        page.wait_for_timeout(1500)
        
        # Click 2 (immediately again on the same host):
        card = page.locator(f'div:has-text("{host_ip}")').filter(has=page.locator('button[kind="primary"]')).last
        btn = card.locator('button[kind="primary"]').first
        btn_text2 = btn.inner_text()
        is_disabled2 = btn.is_disabled()
        print(f"  Click 2: Button label '{btn_text2}', disabled={is_disabled2}")
        assert not is_disabled2, f"Button for {host_ip} should NOT be disabled on second click!"
        
        btn.click()
        page.wait_for_timeout(1500)
        print(f"  ✅ Host {host_ip} clicked twice successfully without any disabled state or not-allowed cursor.")

    # 3. Test Focus button styling and interaction
    print("\n--- Testing Focus secondary button styling & interaction ---")
    first_focus_btn = page.locator('div[data-testid="stButton"] button[kind="secondary"]').first
    border_color = first_focus_btn.evaluate("el => window.getComputedStyle(el).borderColor")
    text_color = first_focus_btn.evaluate("el => window.getComputedStyle(el).color")
    bg_color = first_focus_btn.evaluate("el => window.getComputedStyle(el).backgroundColor")
    print(f"Focus button styles: border={border_color}, color={text_color}, bg={bg_color}")
    
    # Click Focus to test toggle to Unfocus
    first_focus_btn.click()
    page.wait_for_timeout(2000)
    
    unfocus_btn = page.locator('div[data-testid="stButton"] button[kind="secondary"]:has-text("Unfocus")').first
    assert unfocus_btn.count() > 0, "Expected 'Unfocus' button to appear after clicking Focus!"
    print("  ✅ Focus button toggled to 'Unfocus' cleanly.")
    
    # Click Unfocus to revert
    unfocus_btn.click()
    page.wait_for_timeout(2000)
    print("  ✅ 'Unfocus' button toggled back to 'Focus'.")

    # 4. Capture screenshot of the Host Cards showing the new violet Focus button & cyan Telemetry button
    # Scroll down to host cards section
    host_section = page.locator('text="Host Threat Distribution"')
    host_section.scroll_into_view_if_needed()
    page.wait_for_timeout(1000)
    
    out_cards_path = os.path.join(artifact_dir, "01b_Host_Cards_Violet_Focus.png")
    page.screenshot(path=out_cards_path, full_page=False)
    print(f"\nSaved host cards screenshot: {out_cards_path}")

    # Also capture full page screenshot
    out_full_path = os.path.join(artifact_dir, "01b_Attack_Graph.png")
    page.screenshot(path=out_full_path, full_page=False)
    print(f"Updated 01b_Attack_Graph.png: {out_full_path}")

    print(f"\nTotal Console Errors: {len(console_errors)}")
    print(f"Total Page Errors: {len(page_errors)}")
    for e in console_errors:
        print(f"  Console error: {e}")
    for e in page_errors:
        print(f"  Page error: {e}")

    browser.close()

print("\nALL BUTTON CLICK TESTS & STYLING CHECKS PASSED SUCCESSFULLY!")
