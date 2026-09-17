import os
from playwright.sync_api import sync_playwright

PAGES = [
    ("01_Threat_Forecast", "http://localhost:8501/Forecast"),
    ("01b_Attack_Graph", "http://localhost:8501/AttackGraph"),
    ("02_Evidence_Attribution", "http://localhost:8501/Evidence"),
    ("01a_Telemetry_Ingestion", "http://localhost:8501/Input"),
    ("03_Validation_Benchmarks", "http://localhost:8501/Validation"),
    ("05_Platform_Specifications", "http://localhost:8501/About"),
]

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

    for name, url in PAGES:
        print(f"Capturing {name} from {url}...")
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(2500)
        out_path = os.path.join(artifact_dir, f"{name}.png")
        page.screenshot(path=out_path, full_page=False)
        print(f"Saved: {out_path}")
        
        # Also capture full page for validation table inspection
        if "Validation" in name:
            full_out_path = os.path.join(artifact_dir, f"{name}_full.png")
            page.screenshot(path=full_out_path, full_page=True)
            print(f"Saved: {full_out_path}")

    browser.close()
print("All screenshots captured successfully!")
