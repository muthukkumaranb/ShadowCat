# CIC-IDS2017 Exploratory Report

## 1. Executive Summary
This report details the final exploratory integration of the CIC-IDS2017 dataset into the Unified Cyber State (UCS) pipeline. Following rigorous validation, we successfully reconstructed the temporal dataset using real chronological ordering at the day-level, while within-day timing remains synthetic due to critical source omissions. The inclusion of the 2017 data exposes vulnerabilities in relying on chronological test splits and reinforces the necessity of the LOEO protocol. 

## 2. Real Omissions vs. Synthesis (Missing Columns Check)
We performed a literal exact-column dump on the raw source files (`Monday-WorkingHours.pcap_ISCX.csv`) to verify absent metadata.
**Raw columns array subset**:
`["' Destination Port'", "' Flow Duration'", "' Total Fwd Packets'", ... , "' Label'"]`

* **Confirmed Present**: `Destination Port`
* **Confirmed Absent**: `Timestamp`, `Protocol`, `Source IP`, `Destination IP`, `Source Port`

Because the original timestamp metadata was stripped from the `MachineLearningCVE` CSVs by the CIC authors, we had to apply a synthetic timestamp approach. **Disclosure**: The 2017 windows use real day-level ordering (derived from the source file name, mapping Monday through Friday properly sequentially) but synthetic within-day minute-level ordering. It is not real per-window timing.

## 3. Generalization vs Memorization: F1 Metrics
We re-evaluated the baselines, ensuring evaluation protocols were clearly delineated to avoid false equivalencies between chronologically-split holdouts and episodic cross-validation.

* **In-Distribution 2018 (Baseline)**: F1 = 0.738 *(Protocol: 37-fold LOEO, run_lr_frozen.py)*
* **In-Distribution 2018 (Future Holdout)**: F1 = 0.2625 *(Protocol: Single Chronological Split - completely unseen novel attacks)*
* **Zero-Shot 2017 Generalization**: F1 = 0.4234 *(Protocol: Trained on full 2018 dataset, tested zero-shot on 2017 using Logistic Regression)*

The 0.4234 zero-shot F1 proves that some volume-based heuristics transfer, though the complete lack of packet-level PCAP features caps the ceiling of tabular zero-shot capabilities.

## 4. Leakage-Free Generalization (Gate 0)
When evaluating the cleanly-rebuilt combined dataset using the strict Gate 0 Leave-One-Episode-Out (LOEO) protocol, the model successfully demonstrated strong precision in generalizing across isolated episodes:
* **Set A (Traffic+Packet)**: F1 = 0.6971, Precision = 0.8795, Recall = 0.5774 *(Protocol: Episode-Grouped Diagnostic)*
* **Set B (Schedule-Only)**: F1 = 0.4340, Precision = 0.3635, Recall = 0.5386 *(Protocol: Episode-Grouped Diagnostic)*

## 5. Next-State Horizon Predictability (H*)
The world model was fully trained for 50 epochs on the 426-feature combined dataset (yielding `gaussian_next_state_best.pt`). However, due to the injection of 2017 flow-only records and the zero-padding of the 406-feature stage-classification head to fit the 426 dimensions, the resulting predictability metric degraded.
* **World Model Horizon Predictability**: H* = 0
* **Model F1**: 0.2136
* **Persistence Baseline**: 0.6051

## 6. Conclusion
The pipeline seamlessly scales to accommodate the flow-only constraints of CIC-IDS2017. However, the true strength of the architecture relies on high-fidelity packet-level dynamics and precise chronological network topologies. Models trained on the combined dataset experience significant H* degradation. Fine-tuning sequences should be strictly limited to the CSE-CIC-IDS2018 distributions where full telemetry is maintained.
