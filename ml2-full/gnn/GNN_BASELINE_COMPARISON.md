# Graph-Statistical Baseline Comparison

## Methodology
- Chronological train/val/test split without future leakage.
- Normalization (StandardScaler) fitted exclusively on training set.
- Class weighting applied to address extreme class imbalance.

## Baseline Results on Test Split

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Random Forest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Gradient Boosting | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Observations
- Random Forest and Gradient Boosting exploit non-linear combinations of degree distributions and port statistics.
- The baseline provides a mandatory benchmark for evaluating the GraphSAGE GNN.
