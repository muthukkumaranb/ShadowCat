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
| **SSH-Bruteforce** | **H=1** | 0.12 | 0.15 | `[0.08..0.20] (IQR=0.06)` | **0.15** | **0.6283** | 0.4926 | 1.0000 | 0.8519 | 0.9500 | 0.8750 |
| **SSH-Bruteforce** | **H=2** | 0.15 | 0.15 | `[0.10..0.22] (IQR=0.06)` | **0.15** | **0.5816** | 0.4211 | 1.0000 | 1.0000 | 0.9342 | 0.7470 |
| **SSH-Bruteforce** | **H=5** | 0.15 | 0.15 | `[0.15..0.22] (IQR=0.03)` | **0.15** | **0.6186** | 0.4568 | 1.0000 | 1.0000 | 0.9475 | 0.8528 |
| **DDOS-LOIC-UDP** | **H=1** | 0.18 | 0.18 | `[0.08..0.22] (IQR=0.03)` | **0.18** | **0.2787** | 0.1640 | 0.9444 | 1.0000 | 0.9912 | 0.5556 |
| **DDOS-LOIC-UDP** | **H=2** | 0.12 | 0.15 | `[0.08..0.22] (IQR=0.06)` | **0.15** | **0.2850** | 0.1687 | 0.9444 | 1.0000 | 0.9913 | 0.7222 |
| **DDOS-LOIC-UDP** | **H=5** | 0.15 | 0.15 | `[0.10..0.28] (IQR=0.10)` | **0.15** | **0.2998** | 0.1825 | 0.9444 | 1.0000 | 0.9804 | 0.5556 |
| **Botnet** | **H=1** | 0.15 | 0.15 | `[0.12..0.25] (IQR=0.04)` | **0.15** | **0.5461** | 0.3970 | 1.0000 | 0.9417 | 0.9550 | 0.0571 |
| **Botnet** | **H=2** | 0.18 | 0.18 | `[0.08..0.22] (IQR=0.07)` | **0.18** | **0.6885** | 0.6551 | 0.9200 | 0.5889 | 0.9688 | 0.0000 |
| **Botnet** | **H=5** | 0.15 | 0.18 | `[0.10..0.22] (IQR=0.05)` | **0.18** | **0.6297** | 0.4963 | 1.0000 | 0.7986 | 0.9889 | 0.0889 |
| **Blended Total** | **H=1** | 0.15 | 0.15 | `[0.08..0.25] (IQR=0.03)` | **0.15** | **0.4360** | 0.3069 | 0.9730 | 0.9482 | 0.9708 | 0.4986 |
| **Blended Total** | **H=2** | 0.15 | 0.15 | `[0.08..0.22] (IQR=0.06)` | **0.15** | **0.4561** | 0.3285 | 0.9622 | 0.9270 | 0.9708 | 0.5331 |
| **Blended Total** | **H=5** | 0.15 | 0.15 | `[0.10..0.28] (IQR=0.05)` | **0.15** | **0.4578** | 0.3210 | 0.9730 | 0.9720 | 0.9745 | 0.5017 |

---

## 3. Recommended Deployed Thresholds for Backend Inference

Summarizing across horizons using the median validation threshold:

- **SSH-Bruteforce**: $\tau = 0.15$ (Median across validated horizons H=1, H=2, H=5)
- **DDOS-LOIC-UDP**: $\tau = 0.15$ (Median across validated horizons H=1, H=2, H=5)
- **Botnet**: $\tau = 0.18$ (Median across validated horizons H=1, H=2, H=5)
- **Blended Total**: $\tau = 0.15$ (Median across validated horizons H=1, H=2, H=5)
