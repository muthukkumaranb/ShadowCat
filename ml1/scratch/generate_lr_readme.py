import pandas as pd
import json

df = pd.read_csv("artifacts/lr_set_a_corrected/lr_detection_loeo_per_fold.csv")

readme_content = """# Logistic Regression Baseline (Set A Features) - Corrected 37-Fold LOEO Protocol Summary

> [!CAUTION]
> **CRITICAL FINDING - FOLD 16 OUTRIGHT FAILURE**:
> While 36 of 37 folds score a perfect 1.0 across Accuracy, Precision, Recall, and F1, the project's reported aggregate mean F1 (**0.973**) is carried almost entirely by very small folds (1-33 test windows, many with only 1-4 attack samples).
> **Fold 16** (`14-02-2018_SSH-Bruteforce_6`, 66 test windows - the single largest fold in the manifest) scores **Recall = 0.0000, Precision = 0.0000, F1 = 0.0000**. All 33 positive (attack) windows in Fold 16 were misclassified as benign (FN = 33, TP = 0). The aggregate metric completely hides this failure.

---

## Executive Summary & Performance Overview

- **37-Fold LOEO Mean Accuracy**: 0.9865
- **37-Fold LOEO Mean Precision**: 0.9730
- **37-Fold LOEO Mean Recall**: 0.9730
- **37-Fold LOEO Mean F1-Score**: **0.9730**
- **Perfect Folds (F1 = 1.0)**: 36 out of 37 folds (97.3%)
- **Outright Failed Folds (F1 = 0.0)**: 1 out of 37 folds (Fold 16, 66 test windows)

---

## Diagnostic Investigation: Root Cause of Fold 16 Failure

### 1. Empirical Cause: Temporal Purge Enfolding & Training Set Collapse
All 9 SSH-Bruteforce attack episodes in the CIC-IDS-2018 dataset occurred in a tight 1.5-hour continuous time window on 2018-02-14 (between 02:02:00 UTC and 03:31:00 UTC).

Episode `14-02-2018_SSH-Bruteforce_6` is the largest SSH-Bruteforce episode (33 attack windows, 33 benign context windows, spanning 02:44:00 UTC to 03:16:00 UTC). When holding out Episode 6 in Fold 16, the 35-minute generalized temporal purge window before (02:44 - 35 min = 02:09 UTC) and after (03:16 + 35 min = 03:51 UTC) creates a test/embargo envelope spanning **01:55:00 UTC to 04:04:00 UTC**.

Because all other SSH-Bruteforce episodes (Episodes 0–5 and 7–8) fall between 02:02 UTC and 03:31 UTC, **every single other SSH-Bruteforce episode in the dataset falls inside Fold 16's purge envelope and is purged from the training set**.

**As a direct result, Fold 16's training set contains EXACTLY ZERO (0) SSH-Bruteforce attack examples** (0 out of 82 total SSH attack windows in the dataset).

### 2. Model Mechanics: Zero-Shot Transfer & Decision Boundary Misalignment
With 0 SSH training examples, Fold 16's model is forced to perform zero-shot cross-attack generalization using only Botnet, Infiltration, and DDOS training samples.

- **Predicted Probabilities for Benign Test Windows (n=33)**: Mean = `0.0050`, Median = `0.0007`, Min = `0.0001`, Max = `0.0790`.
- **Predicted Probabilities for Attack Test Windows (n=33)**: Mean = `0.0347`, Median = `0.0206`, Min = `0.0011`, Max = `0.2241`.

**Key Observation**: The model successfully ranks attack windows higher than benign windows (yielding a strong **ROC-AUC = 0.9385** and **PR-AUC = 0.9306**). However, because the model received no SSH training signal to calibrate feature weights, **all 33 attack probabilities fall below the standard 0.5 classification decision threshold** ($\text{max } p = 0.2241 < 0.5$). Every single attack window is classified as negative (0), resulting in $\text{FN} = 33, \text{TP} = 0, \text{Recall} = 0.0$.

### 3. Small-Fold Bias & Aggregate Metric Distortion
In smaller SSH folds (e.g. Fold 10 holding out Episode 0 with 1 attack window), the held-out episode span is short, leaving Episode 6 (33 SSH attack windows) in the training set. Thus, Folds 0–15 and 17–36 easily achieve F1 = 1.0 on 1–12 test windows.

Reporting the simple average F1 (0.973) creates a misleading impression of model robustness, as it is carried almost entirely by small folds with minimal sample sizes while failing completely on the one large, informative fold.

---

## Per-Fold LOEO Evaluation Table (37 Folds)

| Fold ID | Held-Out Episode ID | Attack Type | Test Windows (Att/Ben) | Accuracy | Precision | Recall | F1-Score | FPR | ROC-AUC | PR-AUC | TP | FP | TN | FN | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""

table_rows = []
for idx, row in df.iterrows():
    fold_id = int(row['fold_id'])
    is_f16 = (fold_id == 16)
    fold_str = f"**Fold {fold_id}**" if not is_f16 else f"**Fold {fold_id} (FAILURE)**"
    ep_id = f"`{row['held_out_episode_id']}`"
    att_type = row['attack_type']
    n_att_ben = f"{int(row['test_attack_windows'])} / {int(row['test_benign_windows'])}"
    acc = f"{row['accuracy']:.4f}"
    prec = f"{row['precision']:.4f}"
    rec = f"**{row['recall']:.4f}**" if not is_f16 else f"**{row['recall']:.4f}** ⚠️"
    f1 = f"**{row['f1']:.4f}**" if not is_f16 else f"**{row['f1']:.4f}** ⚠️"
    fpr = f"{row['fpr']:.4f}"
    roc = f"{row['roc_auc']:.4f}" if pd.notnull(row['roc_auc']) else "N/A"
    pr = f"{row['pr_auc']:.4f}" if pd.notnull(row['pr_auc']) else "N/A"
    tp = int(row['tp'])
    fp = int(row['fp'])
    tn = int(row['tn'])
    fn = int(row['fn'])
    notes = "Outright Failure (Purged SSH Train Data)" if is_f16 else ("Small Fold (n=1 att)" if int(row['test_attack_windows']) == 1 else "Perfect Split")
    
    table_rows.append(f"| {fold_str} | {ep_id} | {att_type} | {n_att_ben} | {acc} | {prec} | {rec} | {f1} | {fpr} | {roc} | {pr} | {tp} | {fp} | {tn} | {fn} | {notes} |")

readme_content += "\n".join(table_rows) + "\n"

with open("artifacts/lr_set_a_corrected/README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("Generated artifacts/lr_set_a_corrected/README.md successfully.")
