`feature/dashboard-risk-ui` is **not merged** into main. Test merge on a local scratch worktree (never pushed):
`origin/main` (04eb82e) + `git merge origin/feature/dashboard-risk-ui` (head 0a60864) -> local merge commit f6e62b5,
clean merge, 2 files changed (`frontend/components/attack_graph.py`, `frontend/components/layered_explanation.py`).

| Test | Command | Result |
|---|---|---|
| `test_dashboard_risk_ui.py` | `python -m pytest test_dashboard_risk_ui.py` | **PASS** — 5 passed (test_rba_entity_risk_logic, test_conformal_credibility_contract, test_conformal_forecast_intervals, test_layered_explanation_contracts, test_apptest_views); no failing tests |
| `test_gui_apptest.py` | `python test_gui_apptest.py` (a script: pytest collects 0 tests from it) | **PASS** — exit 0, "ALL GUI-LEVEL INGESTION VERIFICATIONS PASSED" |

These tests exercise main's dashboard code. `fix/demo-integrity` deletes or rewrites several of the components they
check (risk accumulator, conformal credibility stub, synthetic ingestion stream), so after that branch merges these two
test files will need updating.
