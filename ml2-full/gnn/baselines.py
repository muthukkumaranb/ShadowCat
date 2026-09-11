"""
Graph statistical baselines for cyber attack risk forecasting.
Extracts graph-level statistics (node count, edge count, degree statistics, density,
flow volume, etc.) and trains scikit-learn baseline models (Logistic Regression, Random Forest, Gradient Boosting).
Evaluates using chronological splits without future leakage.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc, roc_auc_score, confusion_matrix
)
import torch
from torch_geometric.data import Data

from gnn.graph_builder import extract_graph_statistics
from gnn.graph_dataset import create_chronological_splits


def build_feature_matrix(graphs: List[Data]) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Extract graph-level statistical feature matrix and labels from a list of graphs.
    """
    records = []
    labels = []
    timestamps = []

    for g in graphs:
        stats = extract_graph_statistics(g)
        records.append(stats)
        labels.append(int(g.y.item()))
        timestamps.append(str(g.timestamp) if hasattr(g, "timestamp") else "")

    df_feats = pd.DataFrame(records).fillna(0.0)
    y = np.array(labels, dtype=np.int64)
    return df_feats, y, timestamps


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Precision, Recall, F1
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        else:
            roc_auc = 0.0
    except Exception:
        roc_auc = 0.0

    # PR-AUC
    try:
        if len(np.unique(y_true)) > 1:
            p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(r_curve, p_curve))
        else:
            pr_auc = 0.0
    except Exception:
        pr_auc = 0.0

    # FPR
    if len(np.unique(y_true)) > 1:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "fpr": fpr,
    }


def evaluate_baselines(
    graphs: List[Data],
    splits: Dict[str, Tuple[int, int]],
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train and evaluate Logistic Regression, Random Forest, and Gradient Boosting baselines.
    """
    X_df, y_all, timestamps = build_feature_matrix(graphs)
    feature_names = list(X_df.columns)

    train_start, train_end = splits["train"]
    val_start, val_end = splits["val"]
    test_start, test_end = splits["test"]

    train_idx = list(range(train_start, train_end + 1))
    val_idx = list(range(val_start, val_end + 1))
    test_idx = list(range(test_start, test_end + 1))

    X_train_raw = X_df.iloc[train_idx].values
    y_train = y_all[train_idx]

    X_val_raw = X_df.iloc[val_idx].values
    y_val = y_all[val_idx]

    X_test_raw = X_df.iloc[test_idx].values
    y_test = y_all[test_idx]

    # Chronological normalization: fit scaler on training split ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Check class counts
    n_pos_train = int(np.sum(y_train))
    n_neg_train = len(y_train) - n_pos_train
    pos_weight = n_neg_train / max(1, n_pos_train)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=random_state
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)

        # Predict on test
        if hasattr(model, "predict_proba"):
            y_prob_test = model.predict_proba(X_test)[:, 1]
            y_prob_val = model.predict_proba(X_val)[:, 1]
            y_prob_train = model.predict_proba(X_train)[:, 1]
        else:
            y_prob_test = model.decision_function(X_test)
            y_prob_val = model.decision_function(X_val)
            y_prob_train = model.decision_function(X_train)

        test_metrics = compute_metrics(y_test, y_prob_test)
        val_metrics = compute_metrics(y_val, y_prob_val)
        train_metrics = compute_metrics(y_train, y_prob_train)

        # Check for lead time in test/full dataset
        # If attack exists at index t_atk, find first alert before t_atk
        results[name] = {
            "model": model,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "y_prob_test": y_prob_test,
        }

    return {
        "results": results,
        "feature_names": feature_names,
        "scaler": scaler,
        "splits": {
            "train_count": len(train_idx),
            "train_pos": int(np.sum(y_train)),
            "val_count": len(val_idx),
            "val_pos": int(np.sum(y_val)),
            "test_count": len(test_idx),
            "test_pos": int(np.sum(y_test)),
        }
    }
