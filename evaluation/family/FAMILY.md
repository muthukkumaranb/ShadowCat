# ATT&CK Stage & Family Classifier Evaluation Report (Phase B)

## 1. Overview & Protocol
- **Classifier**: Multinomial Logistic Regression (`lbfgs`, `max_iter=2000`, `seed=42`).
- **Feature Set**: `ml1/artifacts/lr_set_a/set_a_features.json` (406 features).
- **Dataset**: `data-engineering/data/ucs/ucs_windows_models_v1.parquet` (2,787 windows, 6 days).
- **Cross-Validation**: 37-fold Leave-One-Episode-Out (LOEO) from `corrected_37fold_manifest.json`.
- **Training Invariant**: Scaler fitted on each training fold independently; trained ONLY on `label_binary == 1` and non-Benign windows.
- **Production Inference (`predict()`)**: Averages probabilities from all 37 fold models. Therefore, on dataset windows, the confidence is partly in-sample. The true out-of-sample skill is the LOEO macro-F1 only.
- **Scope**: LOEO covers 3 of 6 families (`Botnet`, `DDOS-LOIC-UDP`, `SSH-Bruteforce`). Note that `DDOS-LOIC-UDP` family F1 = 0.

---

## 2. LOEO Evaluation Results (Held-Out Episodes)
- **Total Evaluable Test Attack Windows**: 149
- **LOEO Macro-F1 (Family)**: **0.4608**
- **LOEO Macro-F1 (ATT&CK Tactic)**: **0.6402**

### Per-Family Performance (LOEO)
| Family | ATT&CK Tactic | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Botnet** | Command and Control | 48 | 0.5541 | 0.8542 | **0.6721** |
| **DDOS-LOIC-UDP** | Impact | 19 | 0.0000 | 0.0000 | **0.0000** |
| **SSH-Bruteforce** | Credential Access | 82 | 0.8750 | 0.5976 | **0.7101** |

### Confusion Matrix (Family)
Rows = True (`['Botnet', 'DDOS-LOIC-UDP', 'SSH-Bruteforce']`), Columns = Predicted (`['Botnet', 'DDOS-HOIC', 'DDOS-LOIC-UDP', 'Infiltration-Compromise', 'Infiltration-Portscan', 'SSH-Bruteforce']`):
```text
True \ Pred              | Botnet             | DDOS-HOIC          | DDOS-LOIC-UDP      | Infiltration-Compromise | Infiltration-Portscan | SSH-Bruteforce    
--------------------------------------------------------------------------------------------------------------------------------------------------------------
Botnet                   |                 41 |                  0 |                  0 |                  7 |                  0 |                  0
DDOS-LOIC-UDP            |                  0 |                  7 |                  0 |                  0 |                  5 |                  7
SSH-Bruteforce           |                 33 |                  0 |                  0 |                  0 |                  0 |                 49
```

### Per-Tactic Performance (LOEO)
| ATT&CK Tactic | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Command and Control** | 48 | 0.5541 | 0.8542 | **0.6721** |
| **Credential Access** | 82 | 0.8750 | 0.5976 | **0.7101** |
| **Impact** | 19 | 1.0000 | 0.3684 | **0.5385** |

### Confusion Matrix (ATT&CK Tactic)
Rows = True (`['Command and Control', 'Credential Access', 'Impact']`), Columns = Predicted (`['Command and Control', 'Credential Access', 'Discovery', 'Execution', 'Impact']`):
```text
True \ Pred              | Command and Control  | Credential Access    | Discovery            | Execution            | Impact              
-------------------------------------------------------------------------------------------------------------------------------------------
Command and Control      |                   41 |                    0 |                    0 |                    7 |                    0
Credential Access        |                   33 |                   49 |                    0 |                    0 |                    0
Impact                   |                    0 |                    7 |                    5 |                    0 |                    7
```

---

## 3. Single-Episode Families Evaluation
> [!NOTE]
> Families with only 1 episode (`DDOS-HOIC`, `Infiltration-Compromise`, `Infiltration-Portscan`) cannot be evaluated under LOEO because holding out the episode removes 100% of its training examples. Consequently, the LOEO metrics in Section 2 measure generalizability using ONLY the multi-episode families (`Botnet`, `DDOS-LOIC-UDP`, `SSH-Bruteforce`).

### Seen-Family Evaluation (Within-Episode Chronological Split 70% / 30% with 30-Window Purge)
| Family | Status | Windows (Total / Train / Purge / Test) | Precision | Recall | F1-Score | Mean Conf |
|---|---|---|---|---|---|---|
| **DDOS-HOIC** | Not Evaluable | 8 (too few windows for 30-win purge) | - | - | - | - |
| **Infiltration-Compromise** | Evaluated | 97 / 37 / 30 / 30 | 1.0000 | 0.5667 | **0.7234** | 0.8996 |
| **Infiltration-Portscan** | Evaluated | 58 / 10 / 30 / 18 | 0.0000 | 0.0000 | **0.0000** | 0.9989 |

---

## 4. Analysis of 614 Windows (`label_binary = 1` and `label_attack_type = Benign`)
- **Total Isolated Windows**: 614
- **Training Status**: Excluded from family classifier training per rule B-t2.
- **Mean Classifier Confidence**: 0.8856 (Min: 0.3685, Max: 1.0000)

### Predicted Family Distribution on these 614 Windows
| Predicted Family | Count | Share (%) |
|---|---|---|
| Botnet | 339 | 55.2% |
| Infiltration-Compromise | 167 | 27.2% |
| Infiltration-Portscan | 66 | 10.7% |
| DDOS-LOIC-UDP | 18 | 2.9% |
| DDOS-HOIC | 14 | 2.3% |
| SSH-Bruteforce | 10 | 1.6% |

### Predicted ATT&CK Tactic Distribution
| Predicted Tactic | Count | Share (%) |
|---|---|---|
| Command and Control | 339 | 55.2% |
| Execution | 167 | 27.2% |
| Discovery | 66 | 10.7% |
| Impact | 32 | 5.2% |
| Credential Access | 10 | 1.6% |
