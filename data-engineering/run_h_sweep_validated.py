"""
H-Sweep for Gate 0 Protocol 4: Run full 37-fold LOEO at H=5, 10, 15
Uses the validated gate0_protocol4_loeo.py logic (no rewrites), 
only changing H parameter and adding explicit error handling.
"""
import pandas as pd
import numpy as np
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from labeler_and_splits import generate_future_attack_labels
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

def run_loeo_for_h(h_value):
    """
    Run full 37-fold LOEO (all 3 multi-episode attack types) at horizon H=h_value.
    Uses validated gate0_protocol4_loeo.py logic with explicit error handling.
    """
    print(f"\n\n{'='*100}")
    print(f"RUNNING FULL 37-FOLD LOEO FOR H={h_value}")
    print(f"{'='*100}\n")
    
    # Load base data
    df = pd.read_parquet('data/ucs/ucs_windows.parquet')
    df = df.sort_values('window_start_utc').reset_index(drop=True)
    
    # Regenerate future_attack_label with new H
    print(f"[*] Regenerating future_attack_label with H={h_value}...")
    if 'future_attack_label' in df.columns:
        df = df.drop(columns=['future_attack_label'])
    
    df = generate_future_attack_labels(
        df,
        horizon_windows=h_value,
        target_col="label_binary",
        future_col="future_attack_label"
    )
    
    print(f"[+] future_attack_label distribution:")
    print(df['future_attack_label'].value_counts(normalize=True))
    print(df['future_attack_label'].value_counts())
    print()
    
    # ===== VALIDATED GATE0_PROTOCOL4_LOEO.PY LOGIC STARTS HERE =====
    
    # Episode assignment
    day_shift = df['source_day'] != df['source_day'].shift(1)
    bin_shift = df['label_binary'] != df['label_binary'].shift(1)
    atk_shift = df['label_attack_type'] != df['label_attack_type'].shift(1)
    df['episode_id'] = (day_shift | bin_shift | atk_shift).cumsum()
    
    # Assign forecasting windows to attack episodes (VALIDATED FIXED LOGIC)
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
    singleton_types = attack_ep_counts[attack_ep_counts == 1].index.tolist()
    print(f"Multi-episode types (LOEO eligible): {multi_ep_types}")
    print(f"Singleton types (excluded): {singleton_types}")
    print()
    
    # Benign windows
    benign_df = df[df['label_attack_type'] == 'Benign'].copy()
    benign_by_day = benign_df.groupby('source_day').size()
    total_benign = len(benign_df)
    
    def evaluate_fold(train_data, test_data, fold_id, atk_type):
        """
        Evaluate Set A and Set B on a single fold.
        EXPLICIT ERROR HANDLING: print full traceback and re-raise on error.
        """
        try:
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
            
            # Train Set A
            clf_a = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
            clf_a.fit(X_tr_a, y_tr)
            y_pr_a = clf_a.predict(X_te_a)
            y_prob_a = clf_a.predict_proba(X_te_a)[:, 1] if clf_a.classes_.shape[0] > 1 else np.zeros(len(y_te))
            
            # Train Set B
            clf_b = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
            clf_b.fit(X_tr_b, y_tr)
            y_pr_b = clf_b.predict(X_te_b)
            y_prob_b = clf_b.predict_proba(X_te_b)[:, 1] if clf_b.classes_.shape[0] > 1 else np.zeros(len(y_te))
            
            # Score Set A
            res_a = {
                'f1': float(f1_score(y_te, y_pr_a, labels=[0,1], zero_division=0)),
                'precision': float(precision_score(y_te, y_pr_a, labels=[0,1], zero_division=0)),
                'recall': float(recall_score(y_te, y_pr_a, labels=[0,1], zero_division=0)),
                'prauc': float(average_precision_score(y_te, y_prob_a)) if len(np.unique(y_te)) > 1 else 0.0,
            }
            
            # Score Set B
            res_b = {
                'f1': float(f1_score(y_te, y_pr_b, labels=[0,1], zero_division=0)),
                'precision': float(precision_score(y_te, y_pr_b, labels=[0,1], zero_division=0)),
                'recall': float(recall_score(y_te, y_pr_b, labels=[0,1], zero_division=0)),
                'prauc': float(average_precision_score(y_te, y_prob_b)) if len(np.unique(y_te)) > 1 else 0.0,
            }
            
            return res_a, res_b
        
        except Exception as e:
            print(f"\n[ERROR] Fold {fold_id} ({atk_type}) failed during evaluation:")
            print(traceback.format_exc())
            raise
    
    # Run LOEO (full 37 folds across 3 multi-episode types)
    all_fold_results = []
    np.random.seed(42)
    fold_counter = 0
    
    for atk_type in multi_ep_types:
        atk_episodes = ep_meta[ep_meta['label_attack_type'] == atk_type]['episode_id'].tolist()
        print("=" * 100)
        print(f"LOEO for: {atk_type} ({len(atk_episodes)} episodes)")
        print("=" * 100)
        
        for fold_idx, held_out_ep in enumerate(sorted(atk_episodes)):
            try:
                # Test: held-out episode windows
                test_attack = df[df['forecast_episode_id'] == held_out_ep].copy()
                held_out_day = test_attack['source_day'].iloc[0]
                held_out_size = len(test_attack)
                
                # Sample benign windows proportionally
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
                
                # Filter test fold to pre-onset windows only (VALIDATED)
                test_fold_filtered = test_fold[test_fold['label_binary'] == 0].copy()
                num_excluded = len(test_fold) - len(test_fold_filtered)
                test_fold = test_fold_filtered
                
                # Train: all other episodes + benign
                train_attack = df[(df['label_attack_type'] != 'Benign') & (df['episode_id'] != held_out_ep)].copy()
                test_excluded_indices = test_attack.index.union(benign_test.index)
                benign_train = benign_df[~benign_df.index.isin(test_excluded_indices)].copy()
                train_fold = pd.concat([train_attack, benign_train]).sort_values('window_start_utc')
                
                # Evaluate
                res_a, res_b = evaluate_fold(train_fold, test_fold, fold_counter, atk_type)
                
                # Benign day breakdown
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
                
                print(f"  Fold {fold_idx:2d} | Ep {held_out_ep:3d} | {held_out_day} | Test: {held_out_size:2d} atk + {len(benign_test):2d} ben = {held_out_size + len(benign_test):2d} (excl {num_excluded:2d}) | Final: {len(test_fold):2d} | A: F1={res_a['f1']:.4f} PR-AUC={res_a['prauc']:.4f} | B: F1={res_b['f1']:.4f} PR-AUC={res_b['prauc']:.4f}")
                
                fold_counter += 1
            
            except Exception as e:
                print(f"\n[CRITICAL] Fold {fold_idx} ({atk_type}, ep {held_out_ep}) failed - STOPPING")
                print(traceback.format_exc())
                return None
        
        print()
    
    # Aggregate and analyze
    results_df = pd.DataFrame(all_fold_results)
    
    print("\n" + "=" * 100)
    print(f"LOEO AGGREGATE RESULTS (H={h_value})")
    print("=" * 100)
    
    # Per-type summary
    print("\n--- Per-Type Summary ---")
    for atk_type in multi_ep_types:
        type_df = results_df[results_df['attack_type'] == atk_type]
        n_folds = len(type_df)
        a_f1_mean = type_df['a_f1'].mean()
        a_f1_std = type_df['a_f1'].std()
        a_prauc_mean = type_df['a_prauc'].mean()
        a_prauc_std = type_df['a_prauc'].std()
        b_f1_mean = type_df['b_f1'].mean()
        b_f1_std = type_df['b_f1'].std()
        b_prauc_mean = type_df['b_prauc'].mean()
        b_prauc_std = type_df['b_prauc'].std()
        
        a_wins = (type_df['a_f1'] > type_df['b_f1']).sum()
        b_wins = (type_df['b_f1'] > type_df['a_f1']).sum()
        
        print(f"\n  {atk_type} ({n_folds} folds):")
        print(f"    Set A: F1 = {a_f1_mean:.4f} +/- {a_f1_std:.4f}  PR-AUC = {a_prauc_mean:.4f} +/- {a_prauc_std:.4f}")
        print(f"    Set B: F1 = {b_f1_mean:.4f} +/- {b_f1_std:.4f}  PR-AUC = {b_prauc_mean:.4f} +/- {b_prauc_std:.4f}")
        print(f"    Head-to-head: A wins {a_wins} / B wins {b_wins}")
    
    # Overall aggregate
    print(f"\n--- Overall LOEO Aggregate (all {len(results_df)} folds across all multi-episode types) ---")
    a_f1_mean = results_df['a_f1'].mean()
    a_f1_std = results_df['a_f1'].std()
    b_f1_mean = results_df['b_f1'].mean()
    b_f1_std = results_df['b_f1'].std()
    a_prauc_mean = results_df['a_prauc'].mean()
    a_prauc_std = results_df['a_prauc'].std()
    b_prauc_mean = results_df['b_prauc'].mean()
    b_prauc_std = results_df['b_prauc'].std()
    
    print(f"  Set A: F1 = {a_f1_mean:.4f} +/- {a_f1_std:.4f}  PR-AUC = {a_prauc_mean:.4f} +/- {a_prauc_std:.4f}")
    print(f"  Set B: F1 = {b_f1_mean:.4f} +/- {b_f1_std:.4f}  PR-AUC = {b_prauc_mean:.4f} +/- {b_prauc_std:.4f}")
    
    a_wins_total = (results_df['a_f1'] > results_df['b_f1']).sum()
    b_wins_total = (results_df['b_f1'] > results_df['a_f1']).sum()
    ties_total = (results_df['a_f1'] == results_df['b_f1']).sum()
    print(f"  Head-to-head: A wins {a_wins_total} / B wins {b_wins_total} / Ties {ties_total} out of {len(results_df)} folds")
    
    # CI overlap check
    a_lower = a_f1_mean - a_f1_std
    a_upper = a_f1_mean + a_f1_std
    b_lower = b_f1_mean - b_f1_std
    b_upper = b_f1_mean + b_f1_std
    overlap = a_lower <= b_upper and b_lower <= a_upper
    
    print(f"\n  Set A F1 range: [{a_lower:.4f}, {a_upper:.4f}]")
    print(f"  Set B F1 range: [{b_lower:.4f}, {b_upper:.4f}]")
    print(f"  Ranges overlap: {overlap}")
    
    if overlap:
        print("\n  VERDICT: INCONCLUSIVE - CIs overlap, cannot distinguish Set A from Set B at this sample size")
    elif a_f1_mean > b_f1_mean:
        print("\n  VERDICT: PASS - Set A mean +/- std strictly above Set B")
    else:
        print("\n  VERDICT: FAIL - Set B systematically beats Set A (non-overlapping ranges)")
    
    # Save results
    output_file = f'data/ucs/loeo_fold_results_h{h_value}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n[+] Raw fold results saved to {output_file}")
    
    # Return aggregate stats
    return {
        'h': h_value,
        'n_folds': len(results_df),
        'a_f1_mean': a_f1_mean,
        'a_f1_std': a_f1_std,
        'a_prauc_mean': a_prauc_mean,
        'a_prauc_std': a_prauc_std,
        'b_f1_mean': b_f1_mean,
        'b_f1_std': b_f1_std,
        'b_prauc_mean': b_prauc_mean,
        'b_prauc_std': b_prauc_std,
        'ci_overlap': overlap,
        'verdict': ('INCONCLUSIVE' if overlap else ('PASS' if a_f1_mean > b_f1_mean else 'FAIL')),
        'a_wins': a_wins_total,
        'b_wins': b_wins_total,
    }

def print_h_comparison(results_h5, results_h10, results_h15):
    """Print side-by-side H=5, 10, 15 comparison."""
    print("\n\n" + "=" * 160)
    print("H-SWEEP COMPARISON: FULL 37-FOLD LOEO AT H=5, 10, 15")
    print("=" * 160 + "\n")
    
    results = [results_h5, results_h10, results_h15]
    h_values = [r['h'] for r in results]
    
    print("| Metric | H=5 Set A | H=5 Set B | H=10 Set A | H=10 Set B | H=15 Set A | H=15 Set B |")
    print("|--------|-----------|-----------|------------|------------|------------|------------|")
    
    # F1 Mean +/- Std
    row = "| F1 Mean ± Std |"
    for r in results:
        row += f" {r['a_f1_mean']:.4f}±{r['a_f1_std']:.4f} / {r['b_f1_mean']:.4f}±{r['b_f1_std']:.4f} |"
    print(row)
    
    # PR-AUC Mean +/- Std
    row = "| PR-AUC Mean ± Std |"
    for r in results:
        row += f" {r['a_prauc_mean']:.4f}±{r['a_prauc_std']:.4f} / {r['b_prauc_mean']:.4f}±{r['b_prauc_std']:.4f} |"
    print(row)
    
    # CI Overlap
    row = "| CI Overlap (F1) |"
    for r in results:
        row += f" {str(r['ci_overlap']):<5} |"
    print(row)
    
    # Head-to-head
    row = "| A wins / B wins |"
    for r in results:
        row += f" {r['a_wins']:>2} / {r['b_wins']:<2} |"
    print(row)
    
    # Number of folds
    row = "| Total Folds |"
    for r in results:
        row += f" {r['n_folds']:<17} |"
    print(row)

if __name__ == "__main__":
    print("[*] H-Sweep: Full 37-Fold LOEO at H=5, 10, 15")
    print("[*] Using validated gate0_protocol4_loeo.py logic with explicit error handling\n")
    
    # Check if H=5 results already exist
    h5_file = 'data/ucs/loeo_fold_results.csv'
    if os.path.exists(h5_file):
        print(f"[+] H=5 results already exist at {h5_file}")
        print("[*] Loading H=5 results to include in comparison...")
        h5_df = pd.read_csv(h5_file)
        results_h5 = {
            'h': 5,
            'n_folds': len(h5_df),
            'a_f1_mean': h5_df['a_f1'].mean(),
            'a_f1_std': h5_df['a_f1'].std(),
            'a_prauc_mean': h5_df['a_prauc'].mean(),
            'a_prauc_std': h5_df['a_prauc'].std(),
            'b_f1_mean': h5_df['b_f1'].mean(),
            'b_f1_std': h5_df['b_f1'].std(),
            'b_prauc_mean': h5_df['b_prauc'].mean(),
            'b_prauc_std': h5_df['b_prauc'].std(),
        }
        a_lower = results_h5['a_f1_mean'] - results_h5['a_f1_std']
        a_upper = results_h5['a_f1_mean'] + results_h5['a_f1_std']
        b_lower = results_h5['b_f1_mean'] - results_h5['b_f1_std']
        b_upper = results_h5['b_f1_mean'] + results_h5['b_f1_std']
        results_h5['ci_overlap'] = a_lower <= b_upper and b_lower <= a_upper
        results_h5['a_wins'] = (h5_df['a_f1'] > h5_df['b_f1']).sum()
        results_h5['b_wins'] = (h5_df['b_f1'] > h5_df['a_f1']).sum()
    else:
        print(f"[!] H=5 results not found. Must run full H=5 LOEO first.")
        print(f"[!] File {h5_file} does not exist")
        sys.exit(1)
    
    # Run H=10 and H=15
    try:
        print("\n" + "="*100)
        print("STARTING H=10 RUN")
        print("="*100)
        results_h10 = run_loeo_for_h(10)
        
        if results_h10 is None:
            print("[!] H=10 run failed - see error above. Stopping sweep.")
            sys.exit(1)
        
        print("\n" + "="*100)
        print("STARTING H=15 RUN")
        print("="*100)
        results_h15 = run_loeo_for_h(15)
        
        if results_h15 is None:
            print("[!] H=15 run failed - see error above. Stopping sweep.")
            sys.exit(1)
        
        # Print side-by-side comparison
        print_h_comparison(results_h5, results_h10, results_h15)
        
        print(f"\n[+] All H-sweep runs complete!")
        print(f"[+] Results saved to:")
        print(f"    - data/ucs/loeo_fold_results_h5.csv (existing, {results_h5['n_folds']} folds)")
        print(f"    - data/ucs/loeo_fold_results_h10.csv ({results_h10['n_folds']} folds)")
        print(f"    - data/ucs/loeo_fold_results_h15.csv ({results_h15['n_folds']} folds)")
    
    except Exception as e:
        print(f"\n[CRITICAL] Unexpected error in H-sweep:")
        print(traceback.format_exc())
        sys.exit(1)
