# Logistic Regression Baseline (Set A Features) - Corrected 37-Fold LOEO Protocol Summary

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

**Key Observation**: The model successfully ranks attack windows higher than benign windows (yielding a strong **ROC-AUC = 0.9385** and **PR-AUC = 0.9306**). However, because the model received no SSH training signal to calibrate feature weights, **all 33 attack probabilities fall below the standard 0.5 classification decision threshold** ($	ext{max } p = 0.2241 < 0.5$). Every single attack window is classified as negative (0), resulting in $	ext{FN} = 33, 	ext{TP} = 0, 	ext{Recall} = 0.0$.

### 3. Small-Fold Bias & Aggregate Metric Distortion
In smaller SSH folds (e.g. Fold 10 holding out Episode 0 with 1 attack window), the held-out episode span is short, leaving Episode 6 (33 SSH attack windows) in the training set. Thus, Folds 0–15 and 17–36 easily achieve F1 = 1.0 on 1–12 test windows.

Reporting the simple average F1 (0.973) creates a misleading impression of model robustness, as it is carried almost entirely by small folds with minimal sample sizes while failing completely on the one large, informative fold.

---

## Per-Fold LOEO Evaluation Table (37 Folds)

| Fold ID | Held-Out Episode ID | Attack Type | Test Windows (Att/Ben) | Accuracy | Precision | Recall | F1-Score | FPR | ROC-AUC | PR-AUC | TP | FP | TN | FN | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Fold 0** | `02-03-2018_Botnet_0` | Botnet | 10 / 10 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 10 | 0 | 10 | 0 | Perfect Split |
| **Fold 1** | `02-03-2018_Botnet_1` | Botnet | 12 / 12 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 12 | 0 | 12 | 0 | Perfect Split |
| **Fold 2** | `02-03-2018_Botnet_2` | Botnet | 1 / 11 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 11 | 0 | Small Fold (n=1 att) |
| **Fold 3** | `02-03-2018_Botnet_3` | Botnet | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 4** | `02-03-2018_Botnet_4` | Botnet | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 5** | `02-03-2018_Botnet_5` | Botnet | 4 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 4 | 0 | 5 | 0 | Perfect Split |
| **Fold 6** | `02-03-2018_Botnet_6` | Botnet | 4 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 4 | 0 | 5 | 0 | Perfect Split |
| **Fold 7** | `02-03-2018_Botnet_7` | Botnet | 3 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 3 | 0 | 5 | 0 | Perfect Split |
| **Fold 8** | `02-03-2018_Botnet_8` | Botnet | 7 / 7 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 7 | 0 | 7 | 0 | Perfect Split |
| **Fold 9** | `02-03-2018_Botnet_9` | Botnet | 5 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 5 | 0 | 5 | 0 | Perfect Split |
| **Fold 10** | `14-02-2018_SSH-Bruteforce_0` | SSH-Bruteforce | 1 / 11 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 11 | 0 | Small Fold (n=1 att) |
| **Fold 11** | `14-02-2018_SSH-Bruteforce_1` | SSH-Bruteforce | 4 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 4 | 0 | 5 | 0 | Perfect Split |
| **Fold 12** | `14-02-2018_SSH-Bruteforce_2` | SSH-Bruteforce | 10 / 10 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 10 | 0 | 10 | 0 | Perfect Split |
| **Fold 13** | `14-02-2018_SSH-Bruteforce_3` | SSH-Bruteforce | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 14** | `14-02-2018_SSH-Bruteforce_4` | SSH-Bruteforce | 8 / 8 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 8 | 0 | 8 | 0 | Perfect Split |
| **Fold 15** | `14-02-2018_SSH-Bruteforce_5` | SSH-Bruteforce | 12 / 12 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 12 | 0 | 12 | 0 | Perfect Split |
| **Fold 16 (FAILURE)** | `14-02-2018_SSH-Bruteforce_6` | SSH-Bruteforce | 33 / 33 | 0.5000 | 0.0000 | **0.0000** ⚠️ | **0.0000** ⚠️ | 0.0000 | 0.9385 | 0.9306 | 0 | 0 | 33 | 33 | Outright Failure (Purged SSH Train Data) |
| **Fold 17** | `14-02-2018_SSH-Bruteforce_7` | SSH-Bruteforce | 6 / 6 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 6 | 0 | 6 | 0 | Perfect Split |
| **Fold 18** | `14-02-2018_SSH-Bruteforce_8` | SSH-Bruteforce | 7 / 7 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 7 | 0 | 7 | 0 | Perfect Split |
| **Fold 19** | `21-02-2018_DDOS-LOIC-UDP_0` | DDOS-LOIC-UDP | 1 / 11 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 11 | 0 | Small Fold (n=1 att) |
| **Fold 20** | `21-02-2018_DDOS-LOIC-UDP_1` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 21** | `21-02-2018_DDOS-LOIC-UDP_10` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 22** | `21-02-2018_DDOS-LOIC-UDP_11` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 23** | `21-02-2018_DDOS-LOIC-UDP_12` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 24** | `21-02-2018_DDOS-LOIC-UDP_13` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 25** | `21-02-2018_DDOS-LOIC-UDP_14` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 26** | `21-02-2018_DDOS-LOIC-UDP_15` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 27** | `21-02-2018_DDOS-LOIC-UDP_16` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 28** | `21-02-2018_DDOS-LOIC-UDP_17` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 29** | `21-02-2018_DDOS-LOIC-UDP_2` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 30** | `21-02-2018_DDOS-LOIC-UDP_3` | DDOS-LOIC-UDP | 2 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 2 | 0 | 5 | 0 | Perfect Split |
| **Fold 31** | `21-02-2018_DDOS-LOIC-UDP_4` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 32** | `21-02-2018_DDOS-LOIC-UDP_5` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 33** | `21-02-2018_DDOS-LOIC-UDP_6` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 34** | `21-02-2018_DDOS-LOIC-UDP_7` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 35** | `21-02-2018_DDOS-LOIC-UDP_8` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
| **Fold 36** | `21-02-2018_DDOS-LOIC-UDP_9` | DDOS-LOIC-UDP | 1 / 5 | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1 | 0 | 5 | 0 | Small Fold (n=1 att) |
