import json
import pandas as pd

# Inspect stage head report and data to get exact training counts for Impact and other classes
df = pd.read_parquet("data/ucs/ucs_windows.parquet")
print("Dataset attack stage breakdown (raw_label_dominant or MITRE stage if present):")
cols = [c for c in df.columns if "stage" in c or "label" in c or "attack" in c]
print("Relevant columns:", cols)

for c in ["label_stage", "stage_label", "raw_label_dominant", "label_attack_type"]:
    if c in df.columns:
        print(f"\nBreakdown for {c}:")
        print(df[c].value_counts(dropna=False))

with open("artifacts/loeo/corrected_37fold_manifest.json", "r") as f:
    manifest = json.load(f)

# Check per-fold training counts for Impact stage
# If label_stage or stage column exists:
stage_col = "label_stage" if "label_stage" in df.columns else ("stage_label" if "stage_label" in df.columns else None)
if stage_col is None:
    for c in df.columns:
        if "stage" in c:
            stage_col = c
            break

print(f"\nUsing stage column: {stage_col}")
if stage_col:
    impact_folds = []
    for f in manifest["folds"]:
        tr_df = df.iloc[f["train_indices"]]
        te_df = df.iloc[f["test_indices"]]
        tr_impact = (tr_df[stage_col] == "Impact").sum() if "Impact" in tr_df[stage_col].values else 0
        te_impact = (te_df[stage_col] == "Impact").sum() if "Impact" in te_df[stage_col].values else 0
        impact_folds.append({"fold_id": f["fold_id"], "held_out": f["held_out_episode_id"], "train_impact": tr_impact, "test_impact": te_impact})
    
    impact_df = pd.DataFrame(impact_folds)
    print("\nImpact counts per fold summary:")
    print(impact_df.describe())
    print("\nFolds with test_impact > 0:")
    print(impact_df[impact_df["test_impact"] > 0])
