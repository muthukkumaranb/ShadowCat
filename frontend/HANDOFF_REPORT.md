# SHADOWCAT Frontend Engineering Hand-off & Architecture Report

**To:** Engineering Team Lead & Backend Integration Engineer  
**From:** Frontend / UI Engineering  
**Project:** SHADOWCAT — Autonomous Pre-Emptive Cyber Threat Forecasting Platform  
**Target:** NTRO PS 6153 (Air-Gapped Enterprise Network Defense)  
**Status:** **100% Complete & Verified** · Ready for Live Backend Integration  
**Commit Reference:** `8e31866` on `main` (`https://github.com/nehasatheeshann-hash/Shadowcat.git`)

---

## 1. Executive Summary

Over this cycle, the SHADOWCAT frontend underwent a complete visual refinement, information architecture streamlining, attack graph rendering upgrade, and regression audit.

The primary objectives achieved:
1. **Zero-Friction Default Experience:** Eliminated redundant landing pages and briefing boxes so evaluators land immediately in the high-density **Threat Forecast** cockpit.
2. **Attack Graph Integrity & Advanced Visualization:** Resolved duplicate node identities, implemented dynamic topology layouts for arbitrary host counts, and added high-contrast asset-criticality sizing, laser-glow lateral movement paths, an in-canvas legend, and interactive blast-radius isolation.
3. **Copy & Visual De-Cluttering:** Trimmed repetitive paragraphs and marketing text across all 5 views while preserving 100% of underlying data values, scientific claims, and methodology expanders.
4. **Clean Backend Integration Boundary:** Standardized `data_provider.py` and created an exhaustive contract in `INTERFACE.md`, allowing backend ML engineers to plug in real PyTorch/GNN weights without modifying a single line of UI code.

---

## 2. Information Architecture & Navigation Overhaul

* **Streamlined 2-Tier Navigation Structure:**
  * **Operations (Cockpit / Front Door):**
    1. `Threat Forecast` (`views/01_Forecast.py`) — **Default / Root Route**
    2. `Evidence & Attribution` (`views/02_Evidence.py`)
    3. `Telemetry Ingestion` (`views/01a_Input.py`)
    4. `Validation & Benchmarks` (`views/03_Validation.py`)
  * **Platform Specs & Audit:**
    5. `Platform Specifications` (`views/05_About.py`)
* **Decommissioned Pages:** Removed deprecated `00_Home.py` (Executive Overview) and `00b_Architecture.py` (redundant architecture views). Removed the collapsible 3-column "Operational Briefing for Evaluators" from the top of the Forecast view.
* **Sidebar Visual Primacy (CSS):** Updated `styles.py` to give `Threat Forecast` distinct visual prominence (sky-blue left border `#38BDF8`, subtle background glow, bold weight), while secondary operations pages render as clean list items.

---

## 3. Threat Forecast & Attack Graph Upgrades

* **Attack Graph Data Integrity:**
  * Fixed duplicate node collisions where single IPs were rendered multiple times with conflicting labels.
  * Replaced static coordinate iterations with a dynamic circular layout generator capable of handling dynamic host counts without runtime `KeyError` exceptions.
* **SVG Canvas & Visual Storytelling:**
  * **Predicted Attack Path:** Distinct marching-ants animation with laser-glow filter and scaled crimson arrowheads clearly contrasts against static grey enterprise links.
  * **Asset Criticality Encoding:** Node diameters strictly encode operational tier (Domain Controller: 38px + radar ring; Gateway/Jump Host: 30px; Endpoint: 24px).
  * **Integrated In-Canvas Legend:** Embedded glassmorphism overlay displaying risk colors, sizing rules, and path line styles.
  * **Telemetry Inspector & Interactivity:** Fixed narrow-column button truncation on `Inspect` and `Focus`. Clicking `Inspect` updates the Host Telemetry Inspector with socket metadata and multi-step trajectories; clicking `Focus` isolates the 1-hop lateral blast radius.
* **Top Threat Alert Banner:**
  * Integrated a full-width status strip positioned above the KPI row displaying the current predicted ATT&CK stage, horizon, probability, and lead time.
  * Formatted status pill to a single status word (`CRITICAL`, `ELEVATED`, `NORMAL`). Applied an ambient, non-blinking glow to reflect snapshot evaluation without implying uncalibrated live streaming.

---

## 4. Visual & Copy-Only Polish Pass (Zero Logic Touched)

| Page | Refinement Applied | Purpose / Impact |
| :--- | :--- | :--- |
| **Header** | Removed `SENSOR: TAP-DMZ-01` and condensed `FEED: BENCHMARK: CIC-IDS2018 (Infiltration)` to `FEED: CIC-IDS2018`. | Eliminates visual clutter; focuses analyst attention on the active stream. |
| **Threat Forecast** | Trimmed chart subtitle to `"Click any marker to inspect."` Condensed attack graph subtitle and trimmed baseline comparison cards to 3–4 word fragments. | Increases information density; eliminates repetitive phrasing. |
| **Telemetry Ingestion** | Removed the "Dual-Level Telemetry Fusion Architecture" block and Scapy implementation line. | Telemetry ingestion is now purely dedicated to dataset selection and PCAP/CSV upload. |
| **Platform Specs (About)** | Relocated the Dual-Level Architecture block into the "Validation methodology & protocol details" expander. Kept minimal top: `SHADOWCAT v1.0`, 1-line description, 3 badges (`AIR-GAPPED`, `MITRE ATT&CK ALIGNED`, `LOEO 37-FOLD EVALUATED`), and the plain $n=1$ disclaimer. | Clean platform specifications without losing PS Section 1 compliance details. |
| **Validation** | Removed metric card subtext (showing only value + label). Moved `[Protocol: LOEO 37-Fold Cross-Validation]` tag inside the methodology expander and eliminated duplicate metric calibration grids. | Prevents redundant metric repetition while keeping mathematical rigor intact in the expander. |
| **Evidence** | Condensed subtitle to `"Correlated flow evidence and feature attribution rankings."` and moved deletion-tested protocol tag into a clean hover tooltip. | Replaces intrusive header badges with tooltip metadata. |

---

## 5. Compliance & Air-Gapped Verification

* **Zero External Network Dependencies:** Removed all `@import` Google Fonts rules from `styles.py` in favor of a clean, robust system font stack (`-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `Roboto`).
* **Telemetry Privacy:** Added `gatherUsageStats = false` under `[browser]` in `.streamlit/config.toml` ensuring strict air-gapped compliance with zero outbound telemetry pings.
* **Claims & Metric Discipline:** Enforced strict guardrails prohibiting uncalibrated claims, stale fold numbers (`5-fold` or `38-fold`), prescriptive `Action:` directives, and unhedged recommendations.

---

## 6. Automated Test Suite Results (100% Passing)

Prior to push, the entire codebase was compiled and validated against 4 independent test suites:

```text
[1] python -m py_compile:
    ==> ALL PYTHON FILES COMPILED CLEANLY (0 syntax errors).

[2] verify_attack_graph.py:
    ==> ALL ATTACK GRAPH INTEGRITY & INTERACTIVITY TESTS PASSED SUCCESSFULLY!
    • get_host_risk_graph data integrity verified across all 5 rollout steps (k=0..4)
    • Mandatory n=1 caption & MOCK provenance badges confirmed
    • Dynamic SVG laser glow, markers, criticality sizing, and legend verified
    • Inspect & Focus button interactivity verified

[3] verify_claims.py:
    ==> ALL CLAIM DISCIPLINE, FOLD COUNT, AND GUIDANCE CHECKS PASSED.
    • 0 stale fold labels ('5-Fold', '38-fold')
    • 0 prescriptive 'Action:' directives
    • 0 unhedged 'Recommended:' statements
    • Authoritative 'LOEO 37-Fold' confirmed in validation views

[4] verify_all_pages.py:
    ==> ALL 5 PAGES PASSED ALL REGRESSION AND GOVERNANCE CHECKS.
    • Threat Forecast (views/01_Forecast.py):     PASS (0 Exceptions, 0 Debug, 0 Emojis)
    • Evidence & Attribution (views/02_Evidence.py): PASS (0 Exceptions, 0 Debug, 0 Emojis)
    • Telemetry Ingestion (views/01a_Input.py):       PASS (0 Exceptions, 0 Debug, 0 Emojis)
    • Validation & Benchmarks (views/03_Validation.py): PASS (0 Exceptions, 0 Debug, 0 Emojis)
    • Platform Specifications (views/05_About.py):      PASS (0 Exceptions, 0 Debug, 0 Emojis)

[5] interactive_audit.py:
    ==> ALL INTERACTIVE WIDGET AUDIT CHECKS PASSED WITH ZERO RUNTIME EXCEPTIONS.
```

---

## 7. Instructions for Backend Engineer

Your entire integration scope is isolated to the data provider layer:

1. **Do NOT touch any files in `views/` or `components/`** — the UI layout, routing, animations, and CSS are finalized.
2. **Review `INTERFACE.md`**: Contains the complete input/output schema contract for every function the UI consumes.
3. **Implement functions in `data_provider.py`**:
   * Replace the mock return blocks with calls to your live feature extraction (`pcap`/`flow`) and model inference heads (`predict()`).
4. **Checkpoint Auto-Detection**:
   * If saving checkpoint files to disk, place them in `models/`:
     * `models/world_model.pt` -> Sequence dynamics model
     * `models/gnn_topology.pt` -> Host risk graph network
     * `models/baseline_benchmarks.json` -> Empirical evaluation results
     * `models/loeo_37fold_results.json` -> Cross-validation results
     * `models/novelty_detector.pt` -> Mahalanobis / Latent novelty head
     * `models/integrated_gradients.npz` -> Feature attributions
   * The UI will automatically detect these files and toggle the status from `[MOCK]` to live production mode.
   * If running as an in-memory service instead of checkpoint files, simply toggle `MOCK_STATUS[key] = False` in `data_provider.py` once live data is wired.
