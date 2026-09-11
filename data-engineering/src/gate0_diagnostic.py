import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, average_precision_score

# 1. Load data
df = pd.read_parquet('data/ucs/ucs_windows.parquet')
df = df.sort_values('window_start_utc').reset_index(drop=True)

metadata_cols = {
    'window_id', 'window_start_utc', 'window_end_utc', 'source_day',
    'split', 'label_binary', 'label_attack_type', 'future_attack_label',
    'raw_label_dominant', 'has_malicious_flows', 'episode_id',
    'mask_has_traffic_volume_features', 'mask_has_flow_timing_features',
    'mask_has_packet_level_features', 'mask_has_tcp_flags',
    'mask_has_graph_topology', 'mask_has_identity_auth'
}
feature_cols_a = [c for c in df.select_dtypes(include=[np.number]).columns if c not in metadata_cols]

X_full = df[feature_cols_a].fillna(0.0).values
y_full = df['future_attack_label'].values

print("=== Diagnostic 1: In-Sample Sanity Check ===")
clf_in_sample = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
clf_in_sample.fit(X_full, y_full)
y_pred_in = clf_in_sample.predict(X_full)
y_prob_in = clf_in_sample.predict_proba(X_full)

print(f"clf.classes_: {clf_in_sample.classes_}")

# Find index of class 1
class_1_idx = np.where(clf_in_sample.classes_ == 1)[0][0] if 1 in clf_in_sample.classes_ else 1

y_prob_in_class1 = y_prob_in[:, class_1_idx]
print(f"Index used for class 1 probability: {class_1_idx}")

f1_in = f1_score(y_full, y_pred_in, zero_division=0)
prauc_in = average_precision_score(y_full, y_prob_in_class1)
print(f"In-Sample F1: {f1_in:.4f}")
print(f"In-Sample PR-AUC: {prauc_in:.4f}")
print("In-Sample Confusion Matrix:")
print(confusion_matrix(y_full, y_pred_in))
print()

print("=== Diagnostic 2 & 3: Single Fold Check ===")
# Just grab the first fold manually
atk_type = 'Botnet'
day_shift = df['source_day'] != df['source_day'].shift(1)
bin_shift = df['label_binary'] != df['label_binary'].shift(1)
atk_shift = df['label_attack_type'] != df['label_attack_type'].shift(1)
df['episode_id'] = (day_shift | bin_shift | atk_shift).cumsum()

ep_meta = df.groupby('episode_id').agg(
    source_day=('source_day', 'first'),
    label_binary=('label_binary', 'first'),
    label_attack_type=('label_attack_type', 'first'),
    window_count=('window_id', 'count')
).reset_index()

atk_episodes = ep_meta[ep_meta['label_attack_type'] == atk_type]['episode_id'].tolist()
held_out_ep = sorted(atk_episodes)[0]

test_attack = df[df['episode_id'] == held_out_ep].copy()
benign_df = df[df['label_attack_type'] == 'Benign'].copy()
benign_test = benign_df.sample(n=len(test_attack), random_state=42)

test_fold = pd.concat([test_attack, benign_test]).sort_values('window_start_utc')
test_fold_filtered = test_fold[test_fold['label_binary'] == 0].copy()
test_fold = test_fold_filtered

train_attack = df[(df['label_attack_type'] != 'Benign') & (df['episode_id'] != held_out_ep)].copy()
benign_train = benign_df[~benign_df.index.isin(benign_test.index)].copy()
train_fold = pd.concat([train_attack, benign_train]).sort_values('window_start_utc')

X_tr = train_fold[feature_cols_a].fillna(0.0).values
y_tr = train_fold['future_attack_label'].values
X_te = test_fold[feature_cols_a].fillna(0.0).values
y_te = test_fold['future_attack_label'].values

clf_fold = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
clf_fold.fit(X_tr, y_tr)
y_pr = clf_fold.predict(X_te)
y_prob = clf_fold.predict_proba(X_te)

print(f"clf.classes_: {clf_fold.classes_}")
class_1_idx_fold = np.where(clf_fold.classes_ == 1)[0][0]
y_prob_class1 = y_prob[:, class_1_idx_fold]

print(f"Index used for class 1 probability: {class_1_idx_fold}")

print("Fold Probabilities (Class 1):")
if len(y_prob_class1) > 0:
    print(f"  Min : {y_prob_class1.min():.4f}")
    print(f"  Max : {y_prob_class1.max():.4f}")
    print(f"  Mean: {y_prob_class1.mean():.4f}")
else:
    print("  No windows in test fold after filter!")

print("Fold Confusion Matrix:")
print(confusion_matrix(y_te, y_pr, labels=[0, 1]))
