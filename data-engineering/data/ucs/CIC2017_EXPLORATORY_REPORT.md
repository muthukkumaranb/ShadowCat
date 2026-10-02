# CIC-IDS2017 Exploratory Report

## 1. Usability of CIC-IDS2017 Data
Based on the documented ground-truth labeling errors noted by Engelen et al. (2021) and Lanvin et al. (2023), specifically regarding the mislabeling of benign flows as attacks (and vice versa) on Thursday (Infiltration) and Friday (DDoS/PortScan), we have strictly excluded the Thursday and Friday PCAP/CSV files from our training splits. These days were flagged as unreliable. Consequently, only Monday through Wednesday (covering Normal, Brute Force, and DoS attacks) were retained for the combined dataset, representing roughly 60% of the total dataset volume.

## 2. Schema-Mapping Approach
A new mapping layer, `cicids2017_mapper.py`, was implemented to normalize CIC-IDS2017's specific nomenclature (e.g., `Total Length of Fwd Packets`) into the 2018 canonical schema (`byte_count_fwd`) expected by the existing `ucs_extractor.py`. Any 2017 column that could not be confidently mapped to a 2018 counterpart was explicitly dropped and logged during ingestion, avoiding manual hacks in the core normalizer logic.

## 3. Leakage Checks
Gate 0 leakage checks (`gate0_leakage_test.py` and `gate0_protocol4_loeo.py`) were successfully run against the newly combined (2018+2017) Unified Cyber State dataset. The strict purge/embargo boundaries successfully adapted to the new multi-dataset chronology without leaking information across splits.

## 4. Combined-Training H*/F1 Result
After retraining the LSTM and stage classification head on the combined data volume (CSE-CIC-IDS2018 + CIC-IDS2017), we re-ran `horizon_analysis.py`.
**Result**: The forecasting horizon ceiling remains firmly at **H* = 0** (i.e. zero reliable early warning capability, exactly as observed in the main branch). F1 scores drop sharply for $H > 0$. This serves as a fourth independent line of evidence (following hyperparameter sweeps, architecture changes, and the 2018 10-day expansion) confirming that the forecasting ceiling is an inherent limitation of the flow-based features and attack characteristics, NOT a symptom of data scarcity. Adding more flow-based volume from a second dataset did not break the forecasting ceiling.

## 5. Cross-Dataset Generalization Test (Zero-Shot)
The most critical test for the problem statement's "generalizes to unseen attack patterns" requirement:
* **Train**: CSE-CIC-IDS2018 (Main 2018 pipeline)
* **Test (Zero-Shot)**: CIC-IDS2017 (Held-out windows)

**Result**: Generalization dropped significantly when evaluating the 2018-trained model directly on 2017 traffic. F1 scores for attack detection plummeted from ~0.92 (within-dataset LOEO) to ~0.35 (cross-dataset zero-shot). This massive degradation highlights severe distribution shifts between the two datasets, even for nominally identical attack categories (like Brute Force), demonstrating that the model was largely memorizing dataset-specific signatures and network topologies rather than fundamental attack behaviors. 

## 6. Conclusion
This branch answers the core question: adding a second independently collected dataset does not resolve the H*=0 limit, and true zero-shot cross-dataset generalization fails significantly. The model memorizes dataset-specific artifacts. 

**This branch will remain purely exploratory and will NOT be merged into main.**
