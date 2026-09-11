import json
import os

lr_repo = r"C:\Users\Vicky\Documents\SIH_2026\SIH2026---UCS-INGESTION-PIPELINE-for-logistic-regression"
features_path = os.path.join(lr_repo, "artifacts", "lr_set_a", "set_a_features.json")

with open(features_path, 'r') as f:
    features_meta = json.load(f)

feature_list = features_meta.get("ordered_feature_names", [])
print(f"Loaded {len(feature_list)} features from ordered_feature_names")

# Print first 50 features to see what they look like
for f in feature_list[:50]:
    print(f)
