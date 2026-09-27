# Stage Head v3 Artifact Status and Historical Context

## Status: Canonical Evaluation (v3 World Model & Full Telemetry)

The files `stage_head_report.md` and `stage_head_metrics.json` in this directory reflect the canonical and verified 37-fold Leave-One-Episode-Out (LOEO) evaluation of the Stage Head across all three headline attack types (SSH-Bruteforce, Botnet, and DDOS-LOIC-UDP).

The previous manifest-path bug (which historically produced zero samples) has been resolved. Furthermore, all 3 headline attack types are now fully backed by real, extracted packet telemetry (Option A / Option B) from the original PCAP files.

---

## Canonical Evaluation: Stage Head v3

The canonical, official evaluation cited in the master documentation is:
- **Report**: `ml1/artifacts/lstm/stage_head_v3/stage_head_report.md`
- **Checkpoint**: `ml1/artifacts/lstm/stage_head_v3/stage_head_best.pt`
- **LOEO Overall Accuracy**: **0.8627** (386 total evaluation samples)
- **Credential Access Precision / Recall**: 1.0000 / 0.5854 (48 detected out of 82 true windows, 0 false positives on background)

### ATT&CK Stage Detection Scope and Impact-Stage Limitation
- **Credential Access / Brute Force**: Successfully detected with 0 false positives on 285 background windows (Precision = 1.0000).
- **Command & Control / Discovery / Initial Access**: 0 test support across the 37 LOEO folds.
- **Impact Class Detection Failure**: Impact-stage attacks (19 test windows) are not detected by the current Stage Head - 0% recall (0 of 19 correctly identified); the model defaults all 19 windows to Unknown/Other.
