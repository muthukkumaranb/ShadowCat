"""
Gate 0 Protocol 4: Leave-One-Episode-Out (LOEO) Cross-Validation
For multi-episode attack types: SSH-Bruteforce (9 eps), DDOS-LOIC-UDP (18 eps), Botnet (10 post-purge eps; 37 folds total)
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, average_precision_score

# Load and assign episodes
df = pd.read_parquet('data/ucs/ucs_windows.parquet')
df = df.sort_values('window_start_utc').reset_index(drop=True)

print("=== future_attack_label Class Balance in Full Dataset ===")
print(df['future_attack_label'].value_counts(normalize=True))
print(df['future_attack_label'].value_counts())
print()

day_shift = df['source_day'] != df['source_day'].shift(1)
bin_shift = df['label_binary'] != df['label_binary'].shift(1)
atk_shift = df['label_attack_type'] != df['label_attack_type'].shift(1)
df['episode_id'] = (day_shift | bin_shift | atk_shift).cumsum()

# Assign forecasting windows to the attack episode they forecast.
df['forecast_episode_id'] = np.nan
attack_mask = df['label_binary'] == 1
df.loc[attack_mask, 'forecast_episode_id'] = df.loc[attack_mask, 'episode_id']
typed_attack_episode = df['episode_id'].where(df['label_attack_type'] != 'Benign')
next_typed_attack_episode = typed_attack_episode.bfill()
pre_onset_mask = (df['future_attack_label'] == 1) & (df['label_binary'] == 0)
df.loc[pre_onset_mask, 'forecast_episode_id'] = next_typed_attack_episode[pre_onset_mask]
not_relevant = (df['future_attack_label'] == 0) & (df['label_binary'] == 0)
df.loc[not_relevant, 'forecast_episode_id'] = np.nan

# Metadata cols to exclude from Set A
metadata_cols = {
    'window_id', 'window_start_utc', 'window_end_utc', 'source_day',
    'split', 'label_binary', 'label_attack_type', 'future_attack_label',
    'raw_label_dominant', 'has_malicious_flows', 'episode_id',
    'forecast_episode_id',
    'mask_has_traffic_volume_features', 'mask_has_flow_timing_features',
    'mask_has_packet_level_features', 'mask_has_tcp_flags',
    'mask_has_graph_topology', 'mask_has_identity_auth'
}
feature_cols_a = [c for c in df.select_dtypes(include=[np.number]).columns if c not in metadata_cols]

# Episode metadata
ep_meta = df.groupby('episode_id').agg(
    source_day=('source_day', 'first'),
    label_binary=('label_binary', 'first'),
    label_attack_type=('label_attack_type', 'first'),
    window_count=('window_id', 'count')
).reset_index()

# Identify multi-episode attack types
attack_ep_counts = ep_meta[ep_meta['label_attack_type'] != 'Benign'].groupby('label_attack_type')['episode_id'].count()
print("=== Episode Counts by Attack Type ===")
print(attack_ep_counts.to_string())
print()

multi_ep_types = attack_ep_counts[attack_ep_counts > 1].index.tolist()
singleton_types = attack_ep_counts[attack_ep_counts == 1].index.tolist()
print("Multi-episode types (LOEO eligible):", multi_ep_types)
print("Singleton types (excluded from LOEO):", singleton_types)
print()

print("=== Pre-onset Sanity Check (first 3 held-out episodes) ===")
eligible_episode_ids = sorted(ep_meta[ep_meta['label_attack_type'].isin(multi_ep_types)]['episode_id'].unique())
sample_episode_ids = [
    sample_ep for sample_ep in eligible_episode_ids
    if ((df['forecast_episode_id'] == sample_ep) & (df['label_binary'] == 0) &
        (df['future_attack_label'] == 1)).any()
][:3]
for sample_ep in sample_episode_ids:
    sample_test_fold = df[df['forecast_episode_id'] == sample_ep].copy()
    sample_test_fold = sample_test_fold[sample_test_fold['label_binary'] == 0]
    print("  Episode {}: {} windows; future_attack_label counts: {}".format(
        sample_ep, len(sample_test_fold), sample_test_fold['future_attack_label'].value_counts().to_dict()))
print()

# All benign windows, grouped by source_day for proportional sampling
benign_df = df[df['label_attack_type'] == 'Benign'].copy()
benign_by_day = benign_df.groupby('source_day').size()
total_benign = len(benign_df)

def evaluate_fold(train_data, test_data):
    """Evaluate Set A and Set B on a single LOEO fold."""
    y_tr = train_data['future_attack_label'].values
    y_te = test_data['future_attack_label'].values

    # Set A: traffic + packet features
    X_tr_a = train_data[feature_cols_a].fillna(0.0).values
    X_te_a = test_data[feature_cols_a].fillna(0.0).values

    # Set B: hour + day
    tr_hour = train_data['window_start_utc'].dt.hour
    te_hour = test_data['window_start_utc'].dt.hour
    tr_days = pd.get_dummies(train_data['source_day'], prefix='day')
    te_days = pd.get_dummies(test_data['source_day'], prefix='day')
    all_d = sorted(list(set(tr_days.columns).union(set(te_days.columns))))
    tr_days = tr_days.reindex(columns=all_d, fill_value=0)
    te_days = te_days.reindex(columns=all_d, fill_value=0)
    X_tr_b = np.column_stack([tr_hour.values, tr_days.values])
    X_te_b = np.column_stack([te_hour.values, te_days.values])

    clf_a = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
    clf_a.fit(X_tr_a, y_tr)
    y_pr_a = clf_a.predict(X_te_a)
    y_prob_a = clf_a.predict_proba(X_te_a)[:, 1] if clf_a.classes_.shape[0] > 1 else np.zeros(len(y_te))

    clf_b = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
    clf_b.fit(X_tr_b, y_tr)
    y_pr_b = clf_b.predict(X_te_b)
    y_prob_b = clf_b.predict_proba(X_te_b)[:, 1] if clf_b.classes_.shape[0] > 1 else np.zeros(len(y_te))

    res_a = {
        'f1': float(f1_score(y_te, y_pr_a, labels=[0,1], zero_division=0)),
        'precision': float(precision_score(y_te, y_pr_a, labels=[0,1], zero_division=0)),
        'recall': float(recall_score(y_te, y_pr_a, labels=[0,1], zero_division=0)),
        'prauc': float(average_precision_score(y_te, y_prob_a)) if len(np.unique(y_te)) > 1 else 0.0,
    }
    res_b = {
        'f1': float(f1_score(y_te, y_pr_b, labels=[0,1], zero_division=0)),
        'precision': float(precision_score(y_te, y_pr_b, labels=[0,1], zero_division=0)),
        'recall': float(recall_score(y_te, y_pr_b, labels=[0,1], zero_division=0)),
        'prauc': float(average_precision_score(y_te, y_prob_b)) if len(np.unique(y_te)) > 1 else 0.0,
    }
    return res_a, res_b

# Run LOEO
all_fold_results = []
np.random.seed(42)

for atk_type in multi_ep_types:
    atk_episodes = ep_meta[ep_meta['label_attack_type'] == atk_type]['episode_id'].tolist()
    print("=" * 80)
    print("LOEO for: {} ({} episodes)".format(atk_type, len(atk_episodes)))
    print("=" * 80)

    for fold_idx, held_out_ep in enumerate(sorted(atk_episodes)):
        # Test: the held-out episode windows
        test_attack = df[df['forecast_episode_id'] == held_out_ep].copy()
        held_out_day = test_attack['source_day'].iloc[0]
        held_out_size = len(test_attack)

        # Sample benign windows proportionally across all 6 days for test
        # Use ~same number as held-out attack windows, proportional across days
        benign_test_target = max(held_out_size, 10)  # at least 10 benign windows
        benign_test_frames = []
        for day, day_count in benign_by_day.items():
            day_benign = benign_df[benign_df['source_day'] == day]
            day_share = int(round(benign_test_target * day_count / total_benign))
            day_share = min(day_share, len(day_benign))
            if day_share > 0:
                sampled = day_benign.sample(n=day_share, random_state=42 + fold_idx)
                benign_test_frames.append(sampled)

        benign_test = pd.concat(benign_test_frames) if benign_test_frames else pd.DataFrame()

        test_fold = pd.concat([test_attack, benign_test]).sort_values('window_start_utc')

        # NEW - filter test fold to pre-onset windows only
        test_fold_filtered = test_fold[test_fold['label_binary'] == 0].copy()
        num_excluded = len(test_fold) - len(test_fold_filtered)
        test_fold = test_fold_filtered

        # Train: all other attack episodes + remaining benign
        train_attack = df[(df['label_attack_type'] != 'Benign') & (df['episode_id'] != held_out_ep)].copy()
        test_excluded_indices = test_attack.index.union(benign_test.index)
        benign_train = benign_df[~benign_df.index.isin(test_excluded_indices)].copy()
        train_fold = pd.concat([train_attack, benign_train]).sort_values('window_start_utc')

        # Evaluate
        res_a, res_b = evaluate_fold(train_fold, test_fold)

        # Benign day breakdown in test
        benign_day_breakdown = benign_test.groupby('source_day').size().to_dict() if len(benign_test) > 0 else {}

        fold_record = {
            'attack_type': atk_type,
            'fold': fold_idx,
            'held_out_ep': held_out_ep,
            'held_out_day': held_out_day,
            'test_attack_windows': held_out_size,
            'test_benign_windows': len(benign_test),
            'test_total': len(test_fold),
            'excluded_windows': num_excluded,
            'train_total': len(train_fold),
            'benign_day_breakdown': str(benign_day_breakdown),
            'a_f1': res_a['f1'], 'a_prec': res_a['precision'], 'a_rec': res_a['recall'], 'a_prauc': res_a['prauc'],
            'b_f1': res_b['f1'], 'b_prec': res_b['precision'], 'b_rec': res_b['recall'], 'b_prauc': res_b['prauc'],
        }
        all_fold_results.append(fold_record)

        print("  Fold {:2d} | Ep {:3d} | Day {} | Test Original: {} atk + {} ben = {} | Excl: {} | Res Test: {} | A: F1={:.4f} PR-AUC={:.4f} | B: F1={:.4f} PR-AUC={:.4f}".format(
            fold_idx, held_out_ep, held_out_day,
            held_out_size, len(benign_test), held_out_size + len(benign_test), num_excluded, len(test_fold),
            res_a['f1'], res_a['prauc'],
            res_b['f1'], res_b['prauc']))

    print()

# Aggregate results
results_df = pd.DataFrame(all_fold_results)

print("\n" + "=" * 80)
print("LOEO AGGREGATE RESULTS")
print("=" * 80)

# Per-type summary
print("\n--- Per-Type Summary ---")
for atk_type in multi_ep_types:
    type_df = results_df[results_df['attack_type'] == atk_type]
    n_folds = len(type_df)
    a_f1_mean = type_df['a_f1'].mean()
    a_f1_std = type_df['a_f1'].std()
    a_prec_mean = type_df['a_prec'].mean()
    a_prec_std = type_df['a_prec'].std()
    a_rec_mean = type_df['a_rec'].mean()
    a_rec_std = type_df['a_rec'].std()
    a_prauc_mean = type_df['a_prauc'].mean()
    a_prauc_std = type_df['a_prauc'].std()
    b_f1_mean = type_df['b_f1'].mean()
    b_f1_std = type_df['b_f1'].std()
    b_prec_mean = type_df['b_prec'].mean()
    b_prec_std = type_df['b_prec'].std()
    b_rec_mean = type_df['b_rec'].mean()
    b_rec_std = type_df['b_rec'].std()
    b_prauc_mean = type_df['b_prauc'].mean()
    b_prauc_std = type_df['b_prauc'].std()

    a_wins = (type_df['a_f1'] > type_df['b_f1']).sum()
    b_wins = (type_df['b_f1'] > type_df['a_f1']).sum()
    ties = (type_df['a_f1'] == type_df['b_f1']).sum()

    print("\n  {} ({} folds):".format(atk_type, n_folds))
    print("    Set A: F1 = {:.4f} +/- {:.4f}  PR-AUC = {:.4f} +/- {:.4f}".format(
        a_f1_mean, a_f1_std, a_prauc_mean, a_prauc_std))
    print("    Set B: F1 = {:.4f} +/- {:.4f}  PR-AUC = {:.4f} +/- {:.4f}".format(
        b_f1_mean, b_f1_std, b_prauc_mean, b_prauc_std))
    print("    Head-to-head: A wins {} / B wins {} / Ties {}".format(a_wins, b_wins, ties))

# Overall aggregate
print("\n--- Overall LOEO Aggregate (all {} folds across all multi-episode types) ---".format(len(results_df)))
a_f1_all_mean = results_df['a_f1'].mean()
a_f1_all_std = results_df['a_f1'].std()
b_f1_all_mean = results_df['b_f1'].mean()
b_f1_all_std = results_df['b_f1'].std()
a_prec_all_mean = results_df['a_prec'].mean()
a_prec_all_std = results_df['a_prec'].std()
b_prec_all_mean = results_df['b_prec'].mean()
b_prec_all_std = results_df['b_prec'].std()
a_rec_all_mean = results_df['a_rec'].mean()
a_rec_all_std = results_df['a_rec'].std()
a_prauc_all_mean = results_df['a_prauc'].mean()
a_prauc_all_std = results_df['a_prauc'].std()
b_rec_all_mean = results_df['b_rec'].mean()
b_rec_all_std = results_df['b_rec'].std()
b_prauc_all_mean = results_df['b_prauc'].mean()
b_prauc_all_std = results_df['b_prauc'].std()

print("  Set A: F1 = {:.4f} +/- {:.4f}  PR-AUC = {:.4f} +/- {:.4f}".format(
    a_f1_all_mean, a_f1_all_std, a_prauc_all_mean, a_prauc_all_std))
print("  Set B: F1 = {:.4f} +/- {:.4f}  PR-AUC = {:.4f} +/- {:.4f}".format(
    b_f1_all_mean, b_f1_all_std, b_prauc_all_mean, b_prauc_all_std))

a_wins_total = (results_df['a_f1'] > results_df['b_f1']).sum()
b_wins_total = (results_df['b_f1'] > results_df['a_f1']).sum()
ties_total = (results_df['a_f1'] == results_df['b_f1']).sum()
print("  Head-to-head: A wins {} / B wins {} / Ties {} out of {} folds".format(
    a_wins_total, b_wins_total, ties_total, len(results_df)))

# Overlap check
a_lower = a_f1_all_mean - a_f1_all_std
a_upper = a_f1_all_mean + a_f1_all_std
b_lower = b_f1_all_mean - b_f1_all_std
b_upper = b_f1_all_mean + b_f1_all_std
overlap = a_lower <= b_upper and b_lower <= a_upper
print("\n  Set A range: [{:.4f}, {:.4f}]".format(a_lower, a_upper))
print("  Set B range: [{:.4f}, {:.4f}]".format(b_lower, b_upper))
print("  Ranges overlap: {}".format(overlap))

if overlap:
    print("\n  VERDICT: INCONCLUSIVE - CIs overlap, cannot distinguish Set A from Set B at this sample size")
elif a_f1_all_mean > b_f1_all_mean:
    print("\n  VERDICT: PASS - Set A mean +/- std strictly above Set B")
else:
    print("\n  VERDICT: FAIL - Set B systematically beats Set A (non-overlapping ranges)")

# Save raw fold data
results_df.to_csv('data/ucs/loeo_fold_results.csv', index=False)
print("\nRaw fold results saved to data/ucs/loeo_fold_results.csv")
