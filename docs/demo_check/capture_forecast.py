"""s2-fix6: capture Forecast-page screenshots for two real demo slices.

Expects the dashboard running locally (streamlit run frontend/app.py --server.port 8510).
For each slice: open the Ingestion page, click the slice's button, wait until the inference
result is shown, switch to the Forecast page in the same session and save a full-page PNG to
docs/demo_check/forecast_<slice>.png. Uses Playwright with the locally installed Chrome.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8510"
OUT = Path(__file__).resolve().parent
SLICES = {"benign": "Benign traffic", "ssh": "SSH-Bruteforce episode"}


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        for key, label in SLICES.items():
            page = browser.new_page(viewport={"width": 1500, "height": 1000})
            page.goto(f"{URL}/ingestion", wait_until="networkidle")
            page.get_by_role("button", name=label).first.click(timeout=600_000)
            page.get_by_text("Last inference").first.wait_for(timeout=900_000)
            page.get_by_role("button", name="Forecast", exact=True).first.click()
            page.get_by_text("Onset Forecast").first.wait_for(timeout=600_000)
            page.wait_for_timeout(3000)
            page.screenshot(path=str(OUT / f"forecast_{key}.png"), full_page=True)
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
