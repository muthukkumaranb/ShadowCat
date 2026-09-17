"""Verify nav bar at 1440px: all 6 tabs visible, no scroll, no overlap."""
import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ARTIFACT = r"C:\Users\NEHA\.gemini\antigravity-ide\brain\ebba8ef6-e15d-42e6-862f-9e6c0d1ed938"

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto("http://localhost:8501", timeout=20000)
    page.wait_for_timeout(4000)

    # Take nav bar screenshot (top 56px)
    page.screenshot(path=os.path.join(ARTIFACT, "nav_final_1440px.png"), clip={"x": 0, "y": 0, "width": 1440, "height": 56})

    # Measure
    result = page.evaluate("""() => {
        const bar = document.querySelector('.top-nav-bar');
        const tabs = document.querySelectorAll('.top-nav-tab');
        const right = document.querySelector('.top-nav-right');
        const wrapper = document.querySelector('.top-nav-tabs-wrapper');
        return {
            barWidth: bar ? bar.getBoundingClientRect().width : null,
            tabCount: tabs.length,
            tabs: Array.from(tabs).map(t => ({
                text: t.textContent.trim().slice(0, 30),
                left: t.getBoundingClientRect().left,
                right: t.getBoundingClientRect().right,
                width: t.getBoundingClientRect().width
            })),
            rightLeft: right ? right.getBoundingClientRect().left : null,
            rightRight: right ? right.getBoundingClientRect().right : null,
            wrapperScrollWidth: wrapper ? wrapper.scrollWidth : null,
            wrapperClientWidth: wrapper ? wrapper.clientWidth : null,
            needsScroll: wrapper ? wrapper.scrollWidth > wrapper.clientWidth : null
        };
    }""")

    print("=== NAV BAR VERIFICATION @ 1440px ===")
    print(f"Bar width: {result['barWidth']}px")
    print(f"Tab count: {result['tabCount']}")
    for i, t in enumerate(result['tabs']):
        print(f"  Tab {i+1}: '{t['text']}' | left={t['left']:.1f} right={t['right']:.1f} w={t['width']:.1f}px")
    print(f"Right cluster: left={result['rightLeft']:.1f} right={result['rightRight']:.1f}")
    print(f"Wrapper scrollWidth={result['wrapperScrollWidth']} clientWidth={result['wrapperClientWidth']}")
    print(f"Needs scroll: {result['needsScroll']}")
    if result['tabs'] and result['rightLeft']:
        clearance = result['rightLeft'] - result['tabs'][-1]['right']
        print(f"Clearance (Tab6 right -> Right cluster left): {clearance:.1f}px")
        if clearance > 0 and not result['needsScroll']:
            print("PASS: All tabs fit, no overlap, no scrolling needed")
        elif clearance <= 0:
            print(f"FAIL: Tab6 overlaps right cluster by {-clearance:.1f}px")
        else:
            print("FAIL: Scrolling required")

    browser.close()
