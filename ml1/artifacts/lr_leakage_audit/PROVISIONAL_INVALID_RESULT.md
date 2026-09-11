# PROVISIONAL — INVALID/PENDING LEAKAGE AND COMPOSITION AUDIT

**Result:** Logistic Regression baseline on canonical 37-fold LOEO protocol.
**Performance:**
- **Detection:** F1 = 1.0, Precision = 1.0, Recall = 1.0, FPR = 0.0
- **Onset:** F1 = 1.0, Precision = 1.0, Recall = 1.0, FPR = 0.0
- **Onset pooled F1:** 1.000000

## Configuration
- **Script:** `examples/run_lr_loeo.py`
- **Command:** `python examples/run_lr_loeo.py`
- **Dataset:** `data/ucs/ucs_windows.parquet`
- **Feature Manifest:** `artifacts/lr_set_a/set_a_features.json`
- **Feature Count:** 400
- **LOEO Protocol:** Canonical 37-fold LOEO
- **Preprocessing:** `StandardScaler` fitted independently on each training fold.
- **Model Parameters:** `LogisticRegression(solver='liblinear', max_iter=2000, random_state=42)`
- **Threshold:** 0.5

## Status
This result is **SUSPICIOUS AND INVALID**. It must not be treated as final.
An investigation is being conducted in the `artifacts/lr_leakage_audit/` directory to identify whether this is caused by feature leakage or fold-composition shortcut.
