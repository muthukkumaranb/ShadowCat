import pandas as pd

labels = pd.read_parquet("data/processed/ucs_v1/labels.parquet")
positive = labels[labels["future_attack"] == True]
print(positive[["timestamp", "future_attack", "attack_type", "attack_onset"]])