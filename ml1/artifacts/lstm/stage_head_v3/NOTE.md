# Stage Head v3 Artifact Status and Historical Context

## Status: Deprecated Evaluation Run (Zero-Sample Bug)

The files `stage_head_report.md` and `stage_head_metrics.json` in this directory reflect an earlier historical run where the 37-fold Leave-One-Entity-Out (LOEO) evaluation silently produced zero samples (0 samples evaluated across all folds) due to a manifest-path bug in the training script.

The bug has since been fixed in `ml1/scripts/train_stage_head.py` (explicit world model checkpoint resolution and correct entity manifest path mapping).

---

## Canonical Evaluation: Stage Head v2

The canonical, official evaluation cited in `README.md` is:
- **Report**: `ml1/artifacts/lstm/stage_head/stage_head_report.md` (v2)
- **Checkpoint**: `ml1/artifacts/lstm/stage_head/stage_head_best.pt`
- **LOEO Accuracy**: 0.8368 (386 total evaluation samples)
- **Credential Access Precision / F1**: 0.9524 / 0.6452 (40 detected out of 82 true windows, 2 false positives on background)

---

## Why These Broken v3 Artifacts Remain in Place

These files are preserved without modification because **Block 6 of the cryptographic audit chain** (`backend/audit_chain.json`) pins the exact SHA-256 hash of `stage_head_report.md`:
```
fa538ac7159f0ec3acca401ada6604dc401dff94720cddcf2eed85d1b44c470b
```
Overwriting or deleting `stage_head_report.md` would invalidate Block 6 and require recomputing 18 downstream chained blocks in the cryptographic ledger - which is out of scope for this documentation fix.

---

## Verified v3 Re-Evaluation (Unofficial / Working Re-Run)

A real, working re-run of the v3 world model (`gaussian_next_state_best_v3.pt`) through the fixed script was executed across all 37 LOEO folds and produced valid results (0.8109 accuracy, 386 samples). The complete verified artifacts are committed directly to the repository at:
- **Path**: `ml1/artifacts/lstm/stage_head_v3/reeval_unofficial/`
  - `stage_head_report.md`
  - `stage_head_metrics.json`
  - `stage_head_best.pt`
- **LOEO Overall Accuracy**: **0.8109** (386 total evaluation samples)
- **Credential Access**: Precision = 1.0000, Recall = 0.3415, F1 = 0.5091 (28 detected out of 82 true windows, 0 false positives on background)

### ATT&CK Stage Detection Scope and Impact-Stage Limitation
- **Credential Access / Brute Force**: Successfully detected with 0 false positives on 285 background windows (Precision = 1.0000).
- **Command & Control / Discovery / Initial Access**: 0 test support across the 37 LOEO folds.
- **Impact Class Detection Failure**: Impact-stage attacks (19 test windows) are not detected by the current Stage Head - 0% recall (0 of 19 correctly identified) in both v2 and v3 evaluations; the model defaults all 19 windows to Unknown/Other.

---

## Newest Verified Re-Evaluation: Full Real Packet-Level Coverage (`reeval_packetcov/`)

Following the `reeval_unofficial/` re-run above, real PCAP packet-level telemetry was extracted for the remaining two headline attack days (Botnet, 02-03-2018, and DDOS-LOIC-UDP, 21-02-2018), joining SSH-Bruteforce (14-02-2018) as genuinely packet-backed rather than presence-masked. The world model and Stage Head were retrained on the resulting feature matrix and re-evaluated across all 37 LOEO folds. The complete verified artifacts are committed at:
- **Path**: `ml1/artifacts/lstm/stage_head_v3/reeval_packetcov/`
  - `stage_head_report.md`
  - `stage_head_metrics.json`
  - `stage_head_best.pt`
- **Paired world model checkpoint**: `ml1/artifacts/lstm/gaussian_next_state_best_v3_packetcov.pt` (a new file — the original `gaussian_next_state_best_v3.pt` at this directory's parent is left untouched for the same audit-chain reason given above, and is no longer the checkpoint `backend/predict.py` loads by default; see its updated `wm_path`/`st_path` defaults).
- **LOEO Overall Accuracy**: **0.8627** (386 total evaluation samples)
- **Credential Access**: Precision = 1.0000, Recall = 0.5854, F1 = 0.7385 (48 detected out of 82 true windows, 0 false positives on background)
- **Impact Class**: still 0% recall (0 of 19), same known limitation as every prior evaluation on this project.

This is now the canonical, currently-deployed Stage Head result — it supersedes both the `v2` citation above and the `reeval_unofficial/` re-run, and is real, extracted-telemetry-backed across all three headline attacks rather than two. `backend/audit_chain.json` documents this as a newly **appended** entry (not a modification of any existing entry); no historical block was recomputed or resealed to produce it.
