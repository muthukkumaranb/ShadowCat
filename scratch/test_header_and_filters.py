import os
import sys
import time
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

artifact_dir = r"C:\Users\NEHA\.gemini\antigravity-ide\brain\ebba8ef6-e15d-42e6-862f-9e6c0d1ed938"
os.makedirs(artifact_dir, exist_ok=True)

executable_path = r"C:\Users\NEHA\AppData\Local\ms-playwright\chromium-1200\chrome-win64\chrome.exe"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=executable_path)
    context = browser.new_context(viewport={"width": 1440, "height": 950})
    page = context.new_page()

    print("======================================================================")
    print("TEST SUITE 1: Nav Bar 1440px Overflow, Sizing & Alignment Verification")
    print("======================================================================")

    page.goto("http://localhost:8501/Forecast", wait_until="networkidle")
    page.wait_for_timeout(2500)

    # 1. Assert all 6 tabs exist
    tabs = page.locator('.top-nav-tab')
    tab_count = tabs.count()
    print(f"Top nav tabs count: {tab_count}")
    assert tab_count == 6, f"Expected 6 tabs, found {tab_count}"

    tab_labels = [tabs.nth(i).inner_text().strip() for i in range(6)]
    print(f"Tab labels found: {tab_labels}")
    assert "Platform Specs" in tab_labels[5], f"Expected Tab 6 to be 'Platform Specs', got '{tab_labels[5]}'"

    # 2. Check no scrolling needed at 1440px
    tabs_wrapper = page.locator('.top-nav-tabs-wrapper')
    scroll_w = tabs_wrapper.evaluate('el => el.scrollWidth')
    client_w = tabs_wrapper.evaluate('el => el.clientWidth')
    print(f"Tabs Wrapper: scrollWidth={scroll_w}, clientWidth={client_w}")
    assert scroll_w <= client_w, (
        f"FAILED: Nav bar requires horizontal scrolling! scrollWidth ({scroll_w}) > clientWidth ({client_w})"
    )
    print("✅ PASS: Nav bar requires zero scrolling at 1440px (all tabs fit natively with buffer room)!")

    # 3. Check Tab 6 clearance from right-side badges
    tab6_box = tabs.nth(5).bounding_box()
    right_box = page.locator('.top-nav-right').bounding_box()
    clearance = right_box['x'] - (tab6_box['x'] + tab6_box['width'])
    print(f"Tab 6 (Platform Specs) right edge: {tab6_box['x'] + tab6_box['width']:.1f}px")
    print(f"Right-side badges left edge:       {right_box['x']:.1f}px")
    print(f"Clearance buffer:                  {clearance:.1f}px")
    assert clearance > 10.0, f"Collision detected! Clearance is only {clearance:.1f}px"
    print(f"✅ PASS: Clean {clearance:.1f}px clearance! Zero collision or overlap.")

    # 4. Check right-side badges separation
    airgap_badge = page.locator('.badge-offline')
    utc_badge = page.locator('.top-nav-utility-item')
    ab_box = airgap_badge.bounding_box()
    ub_box = utc_badge.bounding_box()
    badge_gap = ub_box['x'] - (ab_box['x'] + ab_box['width'])
    print(f"Spacing between 'AIR-GAPPED SYSTEM' and '● UTC': {badge_gap:.1f}px")
    assert badge_gap >= 16.0, f"Badges too cramped! Gap is {badge_gap:.1f}px"
    print(f"✅ PASS: Generous {badge_gap:.1f}px breathing room between right-side badges.")

    # 5. Capture screenshot of top nav bar at 1440px
    nav_bar = page.locator('.top-nav-bar')
    nav_screenshot_path = os.path.join(artifact_dir, "nav_bar_1440px_verified.png")
    nav_bar.screenshot(path=nav_screenshot_path)
    print(f"Saved nav bar screenshot: {nav_screenshot_path}")

    print("\n======================================================================")
    print("TEST SUITE 2: Host Search & Multi-Tier Filtering Verification")
    print("======================================================================")

    page.goto("http://localhost:8501/AttackGraph", wait_until="networkidle")
    page.wait_for_timeout(3500)

    # Scroll down to Host Threat Distribution section
    header = page.locator('text="Host Threat Distribution"')
    header.scroll_into_view_if_needed()
    page.wait_for_timeout(1000)

    # Helper function to count host cards via distinct inspect button containers
    def count_host_cards():
        return page.locator('div[data-testid="stButton"]:has(button:has-text("Inspect")), div[data-testid="stButton"]:has(button:has-text("Telemetry"))').count()

    initial_count = count_host_cards()
    print(f"Initial host card containers count: {initial_count}")
    assert initial_count == 5, f"Expected 5 initial hosts, got {initial_count}"

    # Verify initial ranks: #1, #2, #3, #4, #5
    for rank_i in range(1, 6):
        assert page.locator(f'text=#{rank_i} ').count() > 0, f"Rank #{rank_i} missing!"
    print("✅ PASS: All 5 initial hosts present with ranks #1 through #5.")

    # 1. Test Search by role "SSH"
    print("\n[Test 1: Search by 'SSH']")
    search_input = page.locator('input[placeholder*="Search by IP or host role"]')
    search_input.fill("SSH")
    search_input.press("Enter")
    page.wait_for_timeout(2500)

    ssh_count = count_host_cards()
    print(f"Cards matching 'SSH': {ssh_count}")
    assert ssh_count == 1, f"Expected 1 host for 'SSH', got {ssh_count}"
    assert page.locator('text=#1 ').count() == 1, "Expected host to be dynamically renumbered to #1"
    assert page.locator('text=10.0.4.10').count() > 0, "Expected 10.0.4.10 to be displayed"
    print("✅ PASS: 'SSH' search correctly filtered to 10.0.4.10 and renumbered as #1.")

    # 2. Test Risk Tier Filter: "Critical"
    print("\n[Test 2: Risk Tier Filter 'Critical']")
    search_input.fill("")  # clear search
    search_input.press("Enter")
    page.wait_for_timeout(2000)

    # Select "Critical" in risk filter selectbox
    risk_select = page.locator('div[data-testid="stSelectbox"]').nth(0)
    risk_select.locator('button[aria-label="Open"]').click()
    page.wait_for_timeout(500)
    page.locator('[role="option"]:has-text("Critical")').click()
    page.wait_for_timeout(2000)

    crit_count = count_host_cards()
    print(f"Cards matching Risk='Critical': {crit_count}")
    assert crit_count == 2, f"Expected 2 Critical hosts (94% and 76%), got {crit_count}"
    assert page.locator('text=#1 ').count() == 1, "First card should be #1"
    assert page.locator('text=#2 ').count() == 1, "Second card should be #2"
    assert page.locator('text=10.0.2.15').count() > 0, "10.0.2.15 should be displayed"
    assert page.locator('text=10.0.4.10').count() > 0, "10.0.4.10 should be displayed"
    print("✅ PASS: Risk Tier 'Critical' correctly filtered to 2 hosts (94% & 76%), renumbered #1 and #2.")

    # Scroll to Host Threat Distribution to capture screenshot with active filter
    header.scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    filtered_shot_path = os.path.join(artifact_dir, "01b_Attack_Graph_Filtered.png")
    page.screenshot(path=filtered_shot_path, full_page=False)
    print(f"Saved active filter screenshot: {filtered_shot_path}")

    # 3. Test Combined Filter: Risk="Critical" AND Asset="Tier 2"
    print("\n[Test 3: Combined Risk='Critical' AND Asset='Tier 2']")
    asset_select = page.locator('div[data-testid="stSelectbox"]').nth(1)
    asset_select.locator('button[aria-label="Open"]').click()
    page.wait_for_timeout(500)
    page.locator('[role="option"]:has-text("Tier 2")').click()
    page.wait_for_timeout(2000)

    combined_count = count_host_cards()
    print(f"Cards matching Risk='Critical' + Asset='Tier 2': {combined_count}")
    assert combined_count == 1, f"Expected 1 host for Critical + Tier 2, got {combined_count}"
    assert page.locator('text=#1 ').count() == 1, "Should be renumbered #1"
    assert page.locator('text=10.0.4.10').count() > 0, "10.0.4.10 should be displayed"
    print("✅ PASS: Combined filter correctly matched only SSH Jump Host (10.0.4.10) as #1.")

    # 4. Test Empty State: Search for "nonexistent"
    print("\n[Test 4: Empty State with zero matches]")
    search_input.fill("nonexistent")
    search_input.press("Enter")
    page.wait_for_timeout(2000)

    empty_count = count_host_cards()
    print(f"Cards matching 'nonexistent': {empty_count}")
    assert empty_count == 0, f"Expected 0 hosts, got {empty_count}"

    empty_msg = page.locator('text="No hosts match the current filters"')
    print(f"Empty state message visible: {empty_msg.is_visible()}")
    assert empty_msg.is_visible(), "Expected empty state card to be visible!"
    print("✅ PASS: Clean empty state card displayed when zero hosts match filters.")

    header.scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    empty_shot_path = os.path.join(artifact_dir, "01b_Attack_Graph_Empty_State.png")
    page.screenshot(path=empty_shot_path, full_page=False)
    print(f"Saved empty state screenshot: {empty_shot_path}")

    # 5. Reset all filters back to default
    print("\n[Test 5: Reset filters to All]")
    search_input.fill("")
    search_input.press("Enter")
    page.wait_for_timeout(1000)
    risk_select.locator('button[aria-label="Open"]').click()
    page.wait_for_timeout(500)
    page.locator('[role="option"]:has-text("All Risks")').click()
    page.wait_for_timeout(1000)
    asset_select.locator('button[aria-label="Open"]').click()
    page.wait_for_timeout(500)
    page.locator('[role="option"]:has-text("All Tiers")').click()
    page.wait_for_timeout(2000)

    reset_count = count_host_cards()
    print(f"Cards after reset: {reset_count}")
    assert reset_count == 5, f"Expected 5 hosts after reset, got {reset_count}"
    print("✅ PASS: All 5 hosts restored cleanly on filter reset.")

    browser.close()

print("\n======================================================================")
print("ALL SUITES & VERIFICATIONS COMPLETED SUCCESSFULLY!")
print("======================================================================")
