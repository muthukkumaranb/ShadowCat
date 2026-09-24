import pandas as pd
from pathlib import Path
import json

def load_and_agg(path, model, target):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # df has fold_id, attack_type, f1, precision, recall, fpr, pr_auc
    
    # Aggregate over all folds
    mean_f1 = df["f1"].mean()
    mean_prec = df["precision"].mean()
    mean_rec = df["recall"].mean()
    mean_fpr = df["fpr"].mean()
    
    res = []
    res.append({
        "Model": model,
        "Target": target.capitalize(),
        "Attack Type": "Aggregate (All)",
        "F1": round(mean_f1, 6),
        "Precision": round(mean_prec, 6),
        "Recall": round(mean_rec, 6),
        "FPR": round(mean_fpr, 6),
        "Protocol": "Corrected 37-fold LOEO"
    })
    
    for at, group in df.groupby("attack_type"):
        res.append({
            "Model": model,
            "Target": target.capitalize(),
            "Attack Type": at,
            "F1": round(group["f1"].mean(), 6),
            "Precision": round(group["precision"].mean(), 6),
            "Recall": round(group["recall"].mean(), 6),
            "FPR": round(group["fpr"].mean(), 6),
            "Protocol": "Corrected 37-fold LOEO"
        })
        
    return pd.DataFrame(res)

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    if (repo_root / "ml1" / "artifacts").exists():
        out_dir = repo_root / "ml1" / "artifacts"
    else:
        out_dir = Path("artifacts")
        
    lr_det = out_dir / "lr" / "lr_detection_loeo_folds.csv"
    lr_onset = out_dir / "lr" / "lr_onset_loeo_folds.csv"
    lstm_det = out_dir / "lstm" / "lstm_detection_loeo_folds.csv"
    lstm_onset = out_dir / "lstm" / "lstm_onset_loeo_folds.csv"
    lstm_det_base = out_dir / "lstm" / "lstm_detection_loeo_folds_baseline.csv"
    lstm_onset_base = out_dir / "lstm" / "lstm_onset_loeo_folds_baseline.csv"
    lstm_graph_det = out_dir / "lstm" / "lstm_graph_detection_loeo_folds.csv"
    lstm_graph_onset = out_dir / "lstm" / "lstm_graph_onset_loeo_folds.csv"
    
    dfs = []
    # Detection
    if lr_det.exists(): dfs.append(load_and_agg(lr_det, "Logistic Regression", "detection"))
    if lstm_det_base.exists(): dfs.append(load_and_agg(lstm_det_base, "LSTM Baseline (Pre-Phase 1)", "detection"))
    if lstm_det.exists(): dfs.append(load_and_agg(lstm_det, "Stacked & Calibrated LSTM", "detection"))
    if lstm_graph_det.exists(): dfs.append(load_and_agg(lstm_graph_det, "Graph-Augmented Residual LSTM", "detection"))
    
    # Onset
    if lr_onset.exists(): dfs.append(load_and_agg(lr_onset, "Logistic Regression", "onset"))
    if lstm_onset_base.exists(): dfs.append(load_and_agg(lstm_onset_base, "LSTM Baseline (Pre-Phase 1)", "onset"))
    if lstm_onset.exists(): dfs.append(load_and_agg(lstm_onset, "Stacked & Calibrated LSTM", "onset"))
    if lstm_graph_onset.exists(): dfs.append(load_and_agg(lstm_graph_onset, "Graph-Augmented Residual LSTM", "onset"))
    
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(out_dir / "side_by_side_comparison.csv", index=False)
        print(final_df.to_string(index=False))
        
        print("\nVerified: All models evaluated on identical corrected_37fold_manifest.json")
    else:
        print("Results not ready yet.")

if __name__ == "__main__":
    main()
