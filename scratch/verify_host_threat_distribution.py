import os
import sys
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
    context = browser.new_context(viewport={"width": 1440, "height": 950})
    page = context.new_page()

    print("Navigating to Lateral Movement Attack Graph page...")
    page.goto("http://localhost:8501/AttackGraph", wait_until="networkidle")
    page.wait_for_timeout(3000)

    # Scroll to Host Threat Distribution
    header = page.locator('text="Host Threat Distribution"')
    header.scroll_into_view_if_needed()
    page.wait_for_timeout(1000)

    # Find the container of the paired cards
    # The grid container has style containing "minmax(220px, 1fr) minmax(380px, 2.5fr)"
    grid_container = page.locator('div[style*="minmax(220px, 1fr)"]').first
    assert grid_container.count() > 0, "Could not find grid container for paired cards!"

    # Get the two cards inside the grid
    cards = grid_container.locator('> div.glass-card')
    card_count = cards.count()
    print(f"Found {card_count} glass-cards in the grid.")
    assert card_count == 2, f"Expected 2 cards in paired grid, got {card_count}"

    card1 = cards.nth(0)
    card2 = cards.nth(1)

    box1 = card1.bounding_box()
    box2 = card2.bounding_box()
    print(f"\nCard 1 (Risk Proportion): x={box1['x']}, y={box1['y']}, width={box1['width']:.1f}, height={box1['height']:.1f}")
    print(f"Card 2 (Posture Summary): x={box2['x']}, y={box2['y']}, width={box2['width']:.1f}, height={box2['height']:.1f}")

    height_diff = abs(box1['height'] - box2['height'])
    print(f"Height difference: {height_diff:.2f}px")
    assert height_diff < 1.0, f"Card heights do not match! Card 1: {box1['height']}, Card 2: {box2['height']}"
    print("SUCCESS: Card heights match perfectly!")

    # Check donut chart containment inside Card 1
    donut_svg = card1.locator('svg[viewBox="0 0 120 120"]')
    assert donut_svg.count() > 0, "Could not find donut SVG inside Card 1!"
    donut_box = donut_svg.bounding_box()
    print(f"\nDonut SVG: x={donut_box['x']:.1f}, y={donut_box['y']:.1f}, width={donut_box['width']:.1f}, height={donut_box['height']:.1f}")

    # Verify containment
    assert donut_box['y'] >= box1['y'], f"Donut top ({donut_box['y']}) is above card top ({box1['y']})"
    assert donut_box['y'] + donut_box['height'] <= box1['y'] + box1['height'], (
        f"Donut bottom ({donut_box['y'] + donut_box['height']}) overflows card bottom ({box1['y'] + box1['height']})"
    )
    assert donut_box['x'] >= box1['x'], f"Donut left ({donut_box['x']}) overflows card left ({box1['x']})"
    assert donut_box['x'] + donut_box['width'] <= box1['x'] + box1['width'], (
        f"Donut right ({donut_box['x'] + donut_box['width']}) overflows card right ({box1['x'] + box1['width']})"
    )
    print("SUCCESS: Donut chart is completely contained inside Card 1 boundaries!")

    # Check the icon next to Risk Proportion
    icon = card1.locator('svg.soc-icon')
    assert icon.count() > 0, "Could not find soc-icon SVG in Risk Proportion card!"
    icon_box = icon.bounding_box()
    print(f"\nRisk Proportion Icon: width={icon_box['width']}, height={icon_box['height']}, visible={icon.is_visible()}")
    assert icon.is_visible(), "Risk Proportion icon is not visible!"
    print("SUCCESS: Feather pie-chart icon renders correctly and is visible!")

    # Take screenshot of the paired cards element specifically
    paired_shot_path = os.path.join(artifact_dir, "01b_Host_Threat_Paired_Cards.png")
    grid_container.screenshot(path=paired_shot_path)
    print(f"\nSaved paired cards screenshot: {paired_shot_path}")

    # Take screenshot of the entire Host Threat Distribution section
    section_shot_path = os.path.join(artifact_dir, "01b_Host_Threat_Distribution_Section.png")
    # Take screenshot with some surrounding context
    page.screenshot(path=section_shot_path, full_page=False)
    print(f"Saved section viewport screenshot: {section_shot_path}")

    browser.close()

print("\nALL VERIFICATIONS PASSED!")
