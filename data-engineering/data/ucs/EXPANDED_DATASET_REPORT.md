# Expanded Dataset and Forecasting Analysis Report

## 1. Dataset Expansion and Class Distribution
The dataset has been expanded from the original 3 days to a total of 10 days to resolve data scarcity constraints. The total number of windows extracted increased from approximately 900 to 4,545. 

Class distribution before expansion (3 days):
- Benign (Unknown/Other): ~850
- SSH-Bruteforce: 76
- FTP-BruteForce: 91
- Botnet: 342
- All other tactics: 0

Class distribution after expansion (10 days, 4,545 total windows):
- Unknown/Other: 3,611 (79.45%)
- Command and Control: 342 (7.52%)
- Credential Access: 318 (7.00%)
- Initial Access: 205 (4.51%)
- Impact: 50 (1.10%)
- Discovery: 19 (0.42%)

## 2. Retraction of Earlier Claims
An earlier evaluation utilizing `run_loeo_corrected.py` was misinterpreted as showing the world model eliminated the H*=0 ceiling. This claim is retracted. The `run_loeo_corrected.py` script does not evaluate the world model or multi-step forecasting skill. Instead, it compares Set A (traffic and packet features) versus Set B (hour-of-day and day-of-week only) using a shallow decision tree classifier. The near-perfect scores achieved on Set B were a day-memorization shortcut (identifying which calendar day an attack occurred on), not genuine forecasting skill.

## 3. World Model Horizon Forecasting Result
The MDN world model was re-evaluated on the expanded 10-day dataset using the `horizon_analysis.py` script. The evaluation confirms that the predictability horizon remains at H* = 0. Across K=1-5, the model achieved an F1 score between 0.30 and 0.36, which is heavily outperformed by the persistence baseline (F1 ~0.93-0.98). 

This serves as the third and final independent line of evidence (after the hyperparameter sweep and the MDN architecture change) that the forecasting ceiling is not caused by data scarcity, model capacity, or output-distribution shape.

## 4. Disclosed Data-Quality Gap
The 5 newly added days (Feb 15, 16, 20, Feb 28, Mar 1) do not have genuine Scapy packet-level extraction (`pkt_*` features), unlike the original 3 verified days. These days rely exclusively on CSV features and should not be described as "Option B" verified data.
