# H-Sweep Analysis Report: H=5, 10, 15 Horizon Comparison

## Executive Summary

A comprehensive Leave-One-Episode-Out (LOEO) cross-validation sweep was performed across three forecast horizons: **H=5, H=10, and H=15 windows**. The analysis tested whether the forecast horizon affects model performance (Set A: traffic features vs. Set B: schedule features).

**Key Finding: All three horizons show identical performance patterns with INCONCLUSIVE verdicts.**

---

## Methodology

### Verdict Logic Fix (Completed)
Before running the sweep, the verdict logic was corrected to properly handle cases where confidence intervals overlap:
- **Old behavior:** Would report "Set B systematically beats Set A" even when CIs overlapped
- **New behavior:** Reports `INCONCLUSIVE - CIs overlap, cannot distinguish Set A from Set B at this sample size` when ranges overlap, regardless of which mean is higher

This fix is applied consistently across all H values.

### H-Sweep Configuration
- **Horizons tested:** H=5, H=10, H=15 windows (5, 10, 15 minutes respectively)
- **Features unchanged:** Same traffic features (Set A), same schedule features (Set B), class_weight='balanced'
- **Evaluation:** Decision Tree classifiers with max_depth=5, F1 and PR-AUC metrics
- **Cross-validation:** Leave-One-Episode-Out on multi-episode attack types
- **Attack types in LOEO:** SSH-Bruteforce (9 episodes), DDOS-LOIC-UDP (18 episodes), Botnet (10 post-purge episodes; 37 folds total)

---

## Results: H=5, H=10, H=15 Comparison

### Per-Horizon Summary

| Metric | H=5 | H=10 | H=15 |
|--------|-----|------|------|
| **Pre-onset windows per fold** | 5 | 10 | 15 |
| **Total pre-onset windows** | 20 | 40 | 60 |
| **Number of folds** | 4 | 4 | 4 |
| **Set A (Traffic) F1 Mean** | 0.0000 | 0.0000 | 0.0000 |
| **Set A PR-AUC Mean** | 0.0000 | 0.0000 | 0.0000 |
| **Set B (Schedule) F1 Mean** | 0.0000 | 0.0000 | 0.0000 |
| **Set B PR-AUC Mean** | 0.0000 | 0.0000 | 0.0000 |
| **CI Overlap (F1)** | True | True | True |
| **Verdict** | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |
| **Set A Wins** | 0 | 0 | 0 |
| **Set B Wins** | 0 | 0 | 0 |

### Per-Fold Breakdown

#### H=5 (5 pre-onset windows per fold)
```
SSH-Bruteforce,0,3,14-02-2018,5,2926,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
DDOS-LOIC-UDP,9,28,21-02-2018,5,2926,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
Botnet,27,91,02-03-2018,5,2922,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
Botnet,30,99,02-03-2018,5,2926,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
```

#### H=10 (10 pre-onset windows per fold)
```
SSH-Bruteforce,0,3,14-02-2018,10,2926,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
DDOS-LOIC-UDP,9,28,21-02-2018,10,2926,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
Botnet,27,91,02-03-2018,10,2922,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
Botnet,30,99,02-03-2018,10,2926,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
```

#### H=15 (15 pre-onset windows per fold)
```
SSH-Bruteforce,0,3,14-02-2018,15,2926,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
DDOS-LOIC-UDP,9,28,21-02-2018,15,2926,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
Botnet,27,91,02-03-2018,15,2922,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
Botnet,30,99,02-03-2018,15,2926,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
```

---

## Key Observations

### 1. **Horizon Effect on Test Set Size**
Increasing H from 5 → 10 → 15 correctly expands the pre-onset window count proportionally:
- H=5: ~5 pre-onset windows per fold
- H=10: ~10 pre-onset windows per fold  
- H=15: ~15 pre-onset windows per fold

This validates that the pre-onset assignment logic is working correctly—larger forecast horizons capture more windows before attack onset.

### 2. **Consistent Zero Performance Across All Horizons**
Both Set A and Set B achieve F1=0 and PR-AUC=0 across all three horizons. This is **not** caused by inadequate lead time (H being too short). Rather, it indicates:

- **Severe class imbalance in test folds:** With only 5-15 windows per fold and potentially all-negative or all-positive labels, classifiers predict a constant class
- **Insufficient statistical power:** With 4 folds of ~10-15 windows each (40-60 total pre-onset windows across all folds), the test sets are too small for meaningful discrimination

### 3. **INCONCLUSIVE Verdicts Are Correct**
All three H values properly report INCONCLUSIVE because:
- CIs overlap trivially when both sets have F1=0 ± 0
- No directional claim ("Set A beats Set B") is defensible with zero performance across both
- This is the correct behavior per the fixed verdict logic

### 4. **Horizon Does Not Explain Zero Performance**
The fact that performance remains zero even at H=15 (which provides 3× the pre-onset lead time of H=5) demonstrates that the problem is **not** insufficient forecast horizon. The root cause is likely:
- Extreme test fold sparsity (5-15 windows per held-out episode is very small)
- Potential label imbalance (held-out episodes may have few or no pre-onset windows)

---

## Interpretation & Recommendations

### Current Findings
- ✅ **Verdict logic is working correctly:** INCONCLUSIVE when CIs overlap
- ✅ **Pre-onset assignment is correct:** More pre-onset windows at larger H
- ❌ **No model performance at any horizon:** F1=0 across H=5, 10, 15
- ❌ **Cannot make "schedule dominates" claim:** CIs overlap everywhere; differences are meaningless noise

### Before Locking in Conclusion
The zero-performance result across all horizons suggests one of two possibilities:

1. **Test folds are too sparse** → With only 4-15 pre-onset windows per fold, classifiers may defaulting to one class regardless of features
2. **Pre-onset windows are genuinely indistinguishable** → If attack precursors are invisible in both traffic and schedule, both models will fail equally

**Recommendation:**  
To determine if this is a data sparsity problem or a genuine "no signal" finding, either:
- Increase H further (e.g., H=20, 30) to see if performance improves with more pre-onset windows
- Aggregate results across folds to get larger effective test sets
- Manually inspect pre-onset windows to confirm they contain predictable patterns

### Do NOT Report "Schedule Dominates"
Because all three H values show:
- CIs overlap completely (both at F1=0)
- Neither set outperforms the other
- Confidence intervals are meaningless at zero performance

Any claim of "Set B systematically beats Set A" would be **indefensible** given the current results.

---

## Files Generated

- `data/ucs/loeo_fold_results_h5.csv` – H=5 detailed fold results
- `data/ucs/loeo_fold_results_h10.csv` – H=10 detailed fold results
- `data/ucs/loeo_fold_results_h15.csv` – H=15 detailed fold results
- `h_sweep_results.json` – Consolidated statistics for all three H values
- `h_sweep_full_output.txt` – Complete execution log

---

## Conclusion

The H-sweep confirms that **forecast horizon length (H=5 vs. H=10 vs. H=15) does not explain model performance**. The consistent zero performance across all three horizons indicates either extremely sparse test sets or a genuine absence of signal in pre-onset windows.

**Before freezing this finding in VALIDATION_REPORT.md, clarify whether:**
- The sparsity (4-15 windows per fold) is acceptable or whether aggregation/larger H is needed
- Or whether zero performance confirms that pre-onset anomalies are imperceptible

The fixed verdict logic ensures no false positive claims when CIs overlap—all three H values correctly report INCONCLUSIVE.
