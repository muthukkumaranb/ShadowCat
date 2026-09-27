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

A real, working re-run of the v3 world model (`gaussian_next_state_best_v3.pt`) through the fixed script was executed across all 37 LOEO folds and produced valid results (0.8109 accuracy, 386 samples). Originally output at `scratch/stage_head_v3_eval/`, the complete verified artifacts are committed directly to the repository at:
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
