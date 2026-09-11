"""
H-Sweep: Direct LOEO execution for H=5, 10, 15
Regenerates future_attack_label for each H and runs LOEO logic directly.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, confusion_matrix
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from labeler_and_splits import generate_future_attack_labels

def run_loeo_for_horizon(h_value):
    """Run full LOEO analysis for a given horizon value."""
    print(f"\n{'='*80}")
    print(f"RUNNING LOEO FOR H={h_value}")
    print(f"{'='*80}\n")
    
    # Load base data
    df = pd.read_parquet('data/ucs/ucs_windows.parquet')
    df = df.sort_values('window_start_utc').reset_index(drop=True)
    
    # Drop old future_attack_label and regenerate with new H
    if 'future_attack_label' in df.columns:
        df = df.drop(columns=['future_attack_label'])
    
    print(f"[*] Regenerating future_attack_label with H={h_value}...")
    df = generate_future_attack_labels(
        df,
        horizon_windows=h_value,
        target_col="label_binary",
        future_col="future_attack_label"
    )
    
    pos_count = (df['future_attack_label'] == 1).sum()
    print(f"[+] future_attack_label=1: {pos_count:,} windows\n")
    
    # ===== LOEO LOGIC FROM gate0_protocol4_loeo.py =====
    
    print("=== future_attack_label Class Balance ===")
    print(df['future_attack_label'].value_counts(normalize=True))
    print(df['future_attack_label'].value_counts())
    print()
    
    # Episode assignment
    day_shift = df['source_day'] != df['source_day'].shift(1)
    bin_shift = df['label_binary'] != df['label_binary'].shift(1)
    atk_shift = df['label_attack_type'] != df['label_attack_type'].shift(1)
    df['episode_id'] = (day_shift | bin_shift | atk_shift).cumsum()
    
    # Assign forecasting windows to attack episodes
    df['forecast_episode_id'] = np.nan
    attack_mask = df['label_binary'] == 1
    df.loc[attack_mask, 'forecast_episode_id'] = df.loc[attack_mask, 'episode_id']
    typed_attack_episode = df['episode_id'].where(df['label_attack_type'] != 'Benign')
    next_typed_attack_episode = typed_attack_episode.bfill()
    pre_onset_mask = (df['future_attack_label'] == 1) & (df['label_binary'] == 0)
    df.loc[pre_onset_mask, 'forecast_episode_id'] = next_typed_attack_episode[pre_onset_mask]
    not_relevant = (df['future_attack_label'] == 0) & (df['label_binary'] == 0)
    df.loc[not_relevant, 'forecast_episode_id'] = np.nan
    
    # Metadata and features
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
    print("Multi-episode types:", multi_ep_types)
    print()
    
    # Benign sampling
    benign_df = df[df['label_attack_type'] == 'Benign'].copy()
    benign_by_day = benign_df.groupby('source_day').size()
    total_benign = len(benign_df)
    
    def evaluate_fold(train_data, test_data):
        """Evaluate Set A and Set B."""
        y_tr = train_data['future_attack_label'].values
        y_te = test_data['future_attack_label'].values
        
        # Set A: traffic + packet features
        X_tr_a = train_data[feature_cols_a].fillna(0.0).values
        X_te_a = test_data[feature_cols_a].fillna(0.0).values
        
        # Set B: hour + day-of-week
        tr_hour = train_data['window_start_utc'].dt.hour
        te_hour = test_data['window_start_utc'].dt.hour
        tr_days = pd.get_dummies(train_data['source_day'], prefix='day')
        te_days = pd.get_dummies(test_data['source_day'], prefix='day')
        all_d = sorted(list(set(tr_days.columns).union(set(te_days.columns))))
        for d in all_d:
            if d not in tr_days.columns:
                tr_days[d] = 0
            if d not in te_days.columns:
                te_days[d] = 0
        tr_days = tr_days[all_d]
        te_days = te_days[all_d]
        
        # Combine hour and day
        X_tr_b = np.column_stack([tr_hour, tr_days.values])
        X_te_b = np.column_stack([te_hour, te_days.values])
        
        metrics = {}
        
        # Train classifiers
        clf_a = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
        clf_b = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
        
        clf_a.fit(X_tr_a, y_tr)
        clf_b.fit(X_tr_b, y_tr)
        
        # Predictions
        y_pred_a = clf_a.predict(X_te_a)
        y_proba_a = clf_a.predict_proba(X_te_a)[:, 1]
        y_pred_b = clf_b.predict(X_te_b)
        y_proba_b = clf_b.predict_proba(X_te_b)[:, 1]
        
        # Metrics for Set A
        if len(np.unique(y_te)) > 1:
            metrics['a_f1'] = f1_score(y_te, y_pred_a, zero_division=0)
            metrics['a_prec'] = precision_score(y_te, y_pred_a, zero_division=0)
            metrics['a_rec'] = recall_score(y_te, y_pred_a, zero_division=0)
            metrics['a_prauc'] = average_precision_score(y_te, y_proba_a)
            
            metrics['b_f1'] = f1_score(y_te, y_pred_b, zero_division=0)
            metrics['b_prec'] = precision_score(y_te, y_pred_b, zero_division=0)
            metrics['b_rec'] = recall_score(y_te, y_pred_b, zero_division=0)
            metrics['b_prauc'] = average_precision_score(y_te, y_proba_b)
        else:
            metrics = {k: 0.0 for k in ['a_f1', 'a_prec', 'a_rec', 'a_prauc', 'b_f1', 'b_prec', 'b_rec', 'b_prauc']}
        
        return metrics, len(test_data)
    
    # LOEO loops
    eligible_episode_ids = sorted(ep_meta[ep_meta['label_attack_type'].isin(multi_ep_types)]['episode_id'].unique())
    
    results = []
    for fold_idx, held_out_ep_id in enumerate(eligible_episode_ids):
        held_out_ep_data = df[df['forecast_episode_id'] == held_out_ep_id]
        
        if len(held_out_ep_data) == 0:
            continue
        
        # Test fold: pre-onset only
        test_fold = held_out_ep_data[(held_out_ep_data['label_binary'] == 0) & (held_out_ep_data['future_attack_label'] == 1)].copy()
        
        if len(test_fold) == 0:
            continue
        
        # Training fold: all other episodes
        train_episodes = set(df['episode_id'].unique()) - {held_out_ep_id}
        train_fold = df[df['episode_id'].isin(train_episodes)].copy()
        
        # Sample benign in proportion to training set
        n_benign_in_train = (train_fold['label_attack_type'] == 'Benign').sum()
        
        try:
            metrics, test_total = evaluate_fold(train_fold, test_fold)
            
            atk_type = ep_meta[ep_meta['episode_id'] == held_out_ep_id]['label_attack_type'].iloc[0]
            held_out_day = ep_meta[ep_meta['episode_id'] == held_out_ep_id]['source_day'].iloc[0]
            
            result = {
                'attack_type': atk_type,
                'fold': fold_idx,
                'held_out_ep': held_out_ep_id,
                'held_out_day': held_out_day,
                'test_total': test_total,
                'train_total': len(train_fold),
            }
            result.update(metrics)
            results.append(result)
            
            print(f"  [{fold_idx}] {atk_type} (ep {held_out_ep_id}): "
                  f"Set A F1={metrics['a_f1']:.4f}, Set B F1={metrics['b_f1']:.4f}")
        
        except Exception as e:
            print(f"  [!] Error on fold {fold_idx}: {e}")
            continue
    
    # Results dataframe
    if len(results) > 0:
        results_df = pd.DataFrame(results)
    else:
        print("[!] No results generated")
        return {}
    
    # Save results
    output_file = f'data/ucs/loeo_fold_results_h{h_value}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n[+] Saved fold results to: {output_file}")
    
    # Aggregate stats
    print("\n=== AGGREGATE RESULTS ===")
    print(f"Total folds: {len(results_df)}")
    print(f"\nSet A (Traffic Features):")
    print(f"  F1:    {results_df['a_f1'].mean():.4f} ± {results_df['a_f1'].std():.4f}")
    print(f"  PR-AUC: {results_df['a_prauc'].mean():.4f} ± {results_df['a_prauc'].std():.4f}")
    print(f"\nSet B (Schedule Features):")
    print(f"  F1:    {results_df['b_f1'].mean():.4f} ± {results_df['b_f1'].std():.4f}")
    print(f"  PR-AUC: {results_df['b_prauc'].mean():.4f} ± {results_df['b_prauc'].std():.4f}")
    
    # CI check
    a_f1_mean = results_df['a_f1'].mean()
    a_f1_std = results_df['a_f1'].std()
    b_f1_mean = results_df['b_f1'].mean()
    b_f1_std = results_df['b_f1'].std()
    
    ci_overlap = (a_f1_mean - a_f1_std <= b_f1_mean + b_f1_std) and (b_f1_mean - b_f1_std <= a_f1_mean + a_f1_std)
    print(f"\nCI Ranges overlap: {ci_overlap}")
    
    if ci_overlap:
        print("VERDICT: INCONCLUSIVE - CIs overlap, cannot distinguish Set A from Set B at this sample size")
    else:
        if a_f1_mean > b_f1_mean:
            print("VERDICT: Set A (Traffic) statistically outperforms Set B (Schedule)")
        else:
            print("VERDICT: Set B (Schedule) statistically outperforms Set A (Traffic)")
    
    # Return stats for comparison table
    return {
        'h': h_value,
        'n_folds': len(results_df),
        'a_f1_mean': a_f1_mean,
        'a_f1_std': a_f1_std,
        'a_prauc_mean': results_df['a_prauc'].mean(),
        'a_prauc_std': results_df['a_prauc'].std(),
        'b_f1_mean': b_f1_mean,
        'b_f1_std': b_f1_std,
        'b_prauc_mean': results_df['b_prauc'].mean(),
        'b_prauc_std': results_df['b_prauc'].std(),
        'ci_overlap': ci_overlap,
        'a_wins': (results_df['a_f1'] > results_df['b_f1']).sum(),
        'b_wins': (results_df['b_f1'] > results_df['a_f1']).sum(),
    }

def print_h_comparison(all_h_results):
    """Print consolidated H-sweep comparison table."""
    print("\n\n" + "="*150)
    print("H-SWEEP CONSOLIDATED COMPARISON: H=5, H=10, H=15")
    print("="*150 + "\n")
    
    print("| Metric | H=5 Set A | H=5 Set B | H=10 Set A | H=10 Set B | H=15 Set A | H=15 Set B |")
    print("|--------|-----------|-----------|------------|------------|------------|------------|")
    
    # F1
    row = "| F1 Mean ± Std |"
    for stats in all_h_results:
        row += f" {stats['a_f1_mean']:.4f}±{stats['a_f1_std']:.4f} / {stats['b_f1_mean']:.4f}±{stats['b_f1_std']:.4f} |"
    print(row)
    
    # PR-AUC
    row = "| PR-AUC Mean ± Std |"
    for stats in all_h_results:
        row += f" {stats['a_prauc_mean']:.4f}±{stats['a_prauc_std']:.4f} / {stats['b_prauc_mean']:.4f}±{stats['b_prauc_std']:.4f} |"
    print(row)
    
    # CI Overlap
    row = "| CI Overlap |"
    for stats in all_h_results:
        row += f" {str(stats['ci_overlap']):<5} |"
    print(row)
    
    # Head-to-head
    row = "| A wins / B wins |"
    for stats in all_h_results:
        row += f" {stats['a_wins']:>2} / {stats['b_wins']:<2} |"
    print(row)
    
    # Analysis
    print("\n### ANALYSIS ###")
    h_values = [r['h'] for r in all_h_results]
    a_f1s = [r['a_f1_mean'] for r in all_h_results]
    b_f1s = [r['b_f1_mean'] for r in all_h_results]
    
    print(f"\nSet A (Traffic) F1 trend across H: {' -> '.join([f'{v:.4f}' for v in a_f1s])}")
    print(f"Set B (Schedule) F1 trend across H: {' -> '.join([f'{v:.4f}' for v in b_f1s])}")
    
    # Interpretation
    a_improvement = a_f1s[-1] - a_f1s[0]  # H=15 vs H=5
    b_improvement = b_f1s[-1] - b_f1s[0]  # H=15 vs H=5
    
    if abs(a_improvement) > abs(b_improvement) and a_improvement > 0:
        print(f"\n✓ Set A shows stronger improvement with larger H (+{a_improvement:.4f})")
        print(f"  Interpretation: H=5 may be too short; Set A benefits from longer forecast horizon")
    elif abs(b_improvement) > abs(a_improvement) and b_improvement > 0:
        print(f"\n✓ Set B shows stronger improvement with larger H (+{b_improvement:.4f})")
        print(f"  Interpretation: Schedule seasonality becomes more predictive at longer horizons")
    else:
        print(f"\n- Both sets show similar trends or improvement patterns")

if __name__ == "__main__":
    print("[*] H-Sweep: Running LOEO for H=5, 10, 15")
    print("[*] This directly regenerates future_attack_label and runs LOEO inline\n")
    
    all_h_results = []
    
    for h in [5, 10, 15]:
        try:
            stats = run_loeo_for_horizon(h)
            if stats:
                all_h_results.append(stats)
        except Exception as e:
            print(f"[!] Error for H={h}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_h_results:
        print_h_comparison(all_h_results)
        
        # Convert to JSON-serializable format
        json_results = []
        for r in all_h_results:
            json_r = {}
            for k, v in r.items():
                if isinstance(v, (np.bool_, bool)):
                    json_r[k] = bool(v)
                elif isinstance(v, (np.integer, np.floating)):
                    json_r[k] = float(v) if isinstance(v, np.floating) else int(v)
                else:
                    json_r[k] = v
            json_results.append(json_r)
        
        with open('h_sweep_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n[+] Results saved to h_sweep_results.json")
    else:
        print("[!] No results generated")
    
    print("\n[+] H-Sweep complete!")
