"""
LOEO Cross-Validation Harness using episode_id as held-out unit
Evaluates BOTH targets:
1. Detection (label_binary)
2. Onset Forecasting (future_attack_label, H=5, pre-onset test filter label_binary == 0)

Evaluates Set A (Traffic + Packet Features) vs Set B (Hour + Day Schedule Features)
using DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced').
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

def run_loeo(parquet_path: str = "data/ucs/ucs_windows.parquet", output_csv: str = "data/ucs/loeo_fold_results.csv"):
    df = pd.read_parquet(parquet_path)
    df = df.sort_values('window_start_utc').reset_index(drop=True)

    print(f"Loaded dataset from {parquet_path}: {len(df)} rows, {len(df.columns)} columns")
    assert 'episode_id' in df.columns, "episode_id column missing from dataset"

    # Pre-onset forecast episode mapping (day-safe bfill)
    df['forecast_episode_id'] = pd.Series(None, index=df.index, dtype='object')
    attack_mask = df['label_attack_type'] != 'Benign'
    df.loc[attack_mask, 'forecast_episode_id'] = df.loc[attack_mask, 'episode_id']

    # Bfill within day for pre-onset windows (future_attack_label == 1 & label_binary == 0)
    for day in df['source_day'].unique():
        day_idx = df[df['source_day'] == day].index
        day_eps = df.loc[day_idx, 'episode_id'].where(df.loc[day_idx, 'label_attack_type'] != 'Benign')
        next_attack_ep = day_eps.bfill()
        pre_onset_day = (df.loc[day_idx, 'future_attack_label'] == 1) & (df.loc[day_idx, 'label_binary'] == 0)
        df.loc[day_idx[pre_onset_day], 'forecast_episode_id'] = next_attack_ep[pre_onset_day]

    # Metadata columns to exclude from Set A
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
    print(f"Set A feature count: {len(feature_cols_a)}")

    # Episode metadata
    ep_meta = df.groupby('episode_id', sort=False).agg(
        source_day=('source_day', 'first'),
        label_binary=('label_binary', 'first'),
        label_attack_type=('label_attack_type', 'first'),
        window_count=('window_id', 'count')
    ).reset_index()

    attack_ep_meta = ep_meta[ep_meta['label_attack_type'] != 'Benign'].copy()
    attack_counts_by_type = attack_ep_meta.groupby('label_attack_type')['episode_id'].count()

    print("\n=== Contiguous Attack Episode Counts by Type ===")
    print(attack_counts_by_type.to_string())
    total_attack_episodes = len(attack_ep_meta)
    print(f"Total Attack Episodes: {total_attack_episodes}")

    multi_ep_types = attack_counts_by_type[attack_counts_by_type > 1].index.tolist()
    singleton_types = attack_counts_by_type[attack_counts_by_type == 1].index.tolist()
    print(f"Multi-episode types (>1 ep): {multi_ep_types} -> {sum(attack_counts_by_type[multi_ep_types])} folds")
    print(f"Singleton types (==1 ep): {singleton_types} -> {sum(attack_counts_by_type[singleton_types])} folds")

    benign_df = df[df['label_attack_type'] == 'Benign'].copy()
    benign_by_day = benign_df.groupby('source_day').size()
    total_benign = len(benign_df)

    def evaluate_model(X_tr, y_tr, X_te, y_te):
        if len(np.unique(y_tr)) < 2:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'prauc': 0.0}
        clf = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1] if clf.classes_.shape[0] > 1 else np.zeros(len(y_te))
        return {
            'f1': float(f1_score(y_te, y_pred, labels=[0, 1], zero_division=0)),
            'precision': float(precision_score(y_te, y_pred, labels=[0, 1], zero_division=0)),
            'recall': float(recall_score(y_te, y_pred, labels=[0, 1], zero_division=0)),
            'prauc': float(average_precision_score(y_te, y_prob)) if len(np.unique(y_te)) > 1 else 0.0
        }

    # Run for both tasks: Detection and Forecasting
    all_results = []
    all_attack_episodes = attack_ep_meta['episode_id'].tolist()

    for target_task in ['detection', 'forecasting']:
        print(f"\n=======================================================")
        print(f"RUNNING LOEO FOR TARGET TASK: {target_task.upper()}")
        print(f"=======================================================")

        for fold_idx, held_out_ep in enumerate(all_attack_episodes):
            ep_row = attack_ep_meta[attack_ep_meta['episode_id'] == held_out_ep].iloc[0]
            atk_type = ep_row['label_attack_type']
            held_out_day = ep_row['source_day']

            if target_task == 'detection':
                # Detection: held-out episode attack windows (label_binary == 1)
                test_attack = df[df['episode_id'] == held_out_ep].copy()
                held_out_size = len(test_attack)

                benign_test_target = max(held_out_size, 10)
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

                # Train: all other attack episodes + remaining benign
                train_attack = df[(df['label_attack_type'] != 'Benign') & (df['episode_id'] != held_out_ep)].copy()
                test_excluded_indices = test_attack.index.union(benign_test.index)
                benign_train = benign_df[~benign_df.index.isin(test_excluded_indices)].copy()
                train_fold = pd.concat([train_attack, benign_train]).sort_values('window_start_utc')

                y_tr = train_fold['label_binary'].values
                y_te = test_fold['label_binary'].values
                num_excluded = 0

            else: # forecasting
                # Forecasting: held-out episode pre-onset windows
                test_attack = df[df['forecast_episode_id'] == held_out_ep].copy()
                held_out_size = len(test_attack)

                benign_test_target = max(held_out_size, 10)
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

                # Pre-onset filter: test fold must only contain label_binary == 0
                test_fold_filtered = test_fold[test_fold['label_binary'] == 0].copy()
                num_excluded = len(test_fold) - len(test_fold_filtered)
                test_fold = test_fold_filtered

                train_attack = df[(df['label_attack_type'] != 'Benign') & (df['episode_id'] != held_out_ep)].copy()
                test_excluded_indices = test_attack.index.union(benign_test.index)
                benign_train = benign_df[~benign_df.index.isin(test_excluded_indices)].copy()
                train_fold = pd.concat([train_attack, benign_train]).sort_values('window_start_utc')

                y_tr = train_fold['future_attack_label'].values
                y_te = test_fold['future_attack_label'].values

            # Features Set A: traffic + packet
            X_tr_a = train_fold[feature_cols_a].fillna(0.0).values
            X_te_a = test_fold[feature_cols_a].fillna(0.0).values

            # Features Set B: hour + day one-hot
            tr_hour = train_fold['window_start_utc'].dt.hour
            te_hour = test_fold['window_start_utc'].dt.hour
            tr_days = pd.get_dummies(train_fold['source_day'], prefix='day')
            te_days = pd.get_dummies(test_fold['source_day'], prefix='day')
            all_d = sorted(list(set(tr_days.columns).union(set(te_days.columns))))
            tr_days = tr_days.reindex(columns=all_d, fill_value=0)
            te_days = te_days.reindex(columns=all_d, fill_value=0)
            X_tr_b = np.column_stack([tr_hour.values, tr_days.values])
            X_te_b = np.column_stack([te_hour.values, te_days.values])

            res_a = evaluate_model(X_tr_a, y_tr, X_te_a, y_te)
            res_b = evaluate_model(X_tr_b, y_tr, X_te_b, y_te)

            rec = {
                'task': target_task,
                'attack_type': atk_type,
                'is_multi_episode': atk_type in multi_ep_types,
                'fold': fold_idx,
                'held_out_ep': held_out_ep,
                'held_out_day': held_out_day,
                'test_attack_windows': held_out_size,
                'test_benign_windows': len(benign_test),
                'test_total': len(test_fold),
                'excluded_windows': num_excluded,
                'train_total': len(train_fold),
                'a_f1': res_a['f1'], 'a_prec': res_a['precision'], 'a_rec': res_a['recall'], 'a_prauc': res_a['prauc'],
                'b_f1': res_b['f1'], 'b_prec': res_b['precision'], 'b_rec': res_b['recall'], 'b_prauc': res_b['prauc'],
            }
            all_results.append(rec)

            print(f"  [{target_task[:3].upper()}] Fold {fold_idx:2d} | Ep: {held_out_ep} ({atk_type}) | Test: {len(test_fold)} w | A: F1={res_a['f1']:.4f} PR-AUC={res_a['prauc']:.4f} | B: F1={res_b['f1']:.4f} PR-AUC={res_b['prauc']:.4f}")

    results_df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved raw fold results to {output_csv}")
    return results_df

if __name__ == '__main__':
    run_loeo()
