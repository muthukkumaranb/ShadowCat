# Hazard Head Validation-Only Threshold Calibration Report (Leakage-Free Protocol)

## 1. Executive Summary & Calibration Protocol

In previous iterations (cf999c9), threshold sweeps were evaluated directly against held-out LOEO test folds (`X_test`/`y_test`), resulting in an overfitted operational threshold recommendation ($\tau=0.45$ global, $\tau=0.40$ for SSH/DDOS).

This report documents the **honest, leakage-free calibration protocol**:
1. Threshold sweeps are computed **strictly on the chronological validation split** ($X_{val} / y_{val}$) computed during LOEO fold preparation (`val_start = int(len(train_df_sorted) * 0.8)`).
2. Per-fold validation-optimal thresholds are identified, and the deployable threshold per group is selected via the **median across folds** (consistent with the Botnet detection/onset calibration in `9fae9ab`).
3. Held-out test performance is reported strictly out-of-sample on $X_{test} / y_{test}$ using the fixed, validation-derived threshold.

## 2. Validation-Derived Thresholds & Held-Out Test Evaluation Matrix

| Attack Group | Horizon | Validation Optimal $\tau$ | Per-Fold Median $\tau$ | Per-Fold Spread [Min..Max] | Deployed $\tau$ | Held-Out Test F1 | Test Precision | Test Recall | Test FPR | Test ROC-AUC | Default $\tau=0.50$ Test F1 |
|---|:---:|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SSH-Bruteforce** | **H=1** | 0.02 | 0.35 | `[0.28..0.45] (IQR=0.05)` | **0.35** | **0.7302** | 0.6481 | 1.0000 | 0.5556 | 0.9778 | 0.2963 |
| **SSH-Bruteforce** | **H=2** | 0.25 | 0.30 | `[0.25..0.35] (IQR=0.07)` | **0.30** | **0.6980** | 0.5692 | 1.0000 | 0.7778 | 0.9398 | 0.4174 |
| **SSH-Bruteforce** | **H=5** | 0.25 | 0.35 | `[0.25..0.40] (IQR=0.05)` | **0.35** | **0.7635** | 0.6446 | 1.0000 | 0.6852 | 0.8302 | 0.1429 |
| **DDOS-LOIC-UDP** | **H=1** | 0.12 | 0.12 | `[0.10..0.18] (IQR=0.02)` | **0.12** | **0.2787** | 0.1640 | 0.9444 | 1.0000 | 0.9882 | 0.4259 |
| **DDOS-LOIC-UDP** | **H=2** | 0.08 | 0.12 | `[0.08..0.18] (IQR=0.04)` | **0.12** | **0.2850** | 0.1687 | 0.9444 | 1.0000 | 0.9847 | 0.4444 |
| **DDOS-LOIC-UDP** | **H=5** | 0.12 | 0.15 | `[0.10..0.18] (IQR=0.03)` | **0.15** | **0.2998** | 0.1825 | 0.9444 | 1.0000 | 0.9699 | 0.5000 |
| **Botnet** | **H=1** | 0.22 | 0.29 | `[0.22..0.40] (IQR=0.11)` | **0.30** | **0.4123** | 0.3937 | 0.7083 | 0.6800 | 0.6310 | 0.0000 |
| **Botnet** | **H=2** | 0.22 | 0.30 | `[0.20..0.40] (IQR=0.13)` | **0.30** | **0.5126** | 0.3939 | 0.8467 | 0.7978 | 0.6261 | 0.0000 |
| **Botnet** | **H=5** | 0.25 | 0.38 | `[0.25..0.45] (IQR=0.05)` | **0.35** | **0.4450** | 0.3208 | 0.8000 | 0.9400 | 0.2832 | 0.0000 |
| **Blended Total** | **H=1** | 0.12 | 0.22 | `[0.10..0.45] (IQR=0.23)` | **0.22** | **0.4855** | 0.3532 | 0.9730 | 0.8703 | 0.8864 | 0.2793 |
| **Blended Total** | **H=2** | 0.08 | 0.20 | `[0.08..0.40] (IQR=0.18)` | **0.20** | **0.4501** | 0.3165 | 0.9572 | 0.9162 | 0.8739 | 0.3177 |
| **Blended Total** | **H=5** | 0.12 | 0.25 | `[0.10..0.45] (IQR=0.20)` | **0.25** | **0.4681** | 0.3279 | 0.9730 | 0.9203 | 0.7442 | 0.2780 |

---

## 3. Recommended Deployed Thresholds for Backend Inference

Summarizing across horizons using the median validation threshold:

- **SSH-Bruteforce**: $\tau = 0.35$ (Median across validated horizons H=1, H=2, H=5)
- **DDOS-LOIC-UDP**: $\tau = 0.12$ (Median across validated horizons H=1, H=2, H=5)
- **Botnet**: $\tau = 0.30$ (Median across validated horizons H=1, H=2, H=5)
- **Blended Total**: $\tau = 0.22$ (Median across validated horizons H=1, H=2, H=5)
