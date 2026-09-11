# Chronological Deviation Export Coverage Report

- **Total Canonical Dataset Windows**: 2787 (indices 0 to 2786)
- **Walk-Forward Model Cutoff**: First 512 windows (indices 0 to 511) serve as initial training chunk 1.
- **Total Windows Receiving Deviation Score**: 2275 (indices 512 to 2786)
- **Unique Windows Covered**: 2275

## Split Coverage Breakdown (ML2 Canonical Split)

| Split Name | Window Index Range | Total Windows in Split | Windows with Deviation Score | Coverage % |
|---|---|---|---|---|
| **ML2 Train Split** | 0 to 1949 | 1950 | 1438 | 73.74% |
| **ML2 Validation & Test Split** | 1950 to 2786 | 837 | 837 | 100.00% |
| **Overall Canonical Dataset** | 0 to 2786 | 2787 | 2275 | 81.63% |

## Validation & Test Split Detailed Status
- **Validation & Test Coverage**: **100.00%** (All 837 windows in ML2's validation and test split receive valid, non-zero chronological predictions from Chunk 4).
- **Initial Training Window Fallback**: Windows 0 to 511 (in train split) receive fallback score `0.0` as no prior history exists before index 0 to train a walk-forward predictor.
- **Leakage Prevention**: Strictly walk-forward chunking. For any window $t$, prediction is generated solely using model trained on windows $< t$.
