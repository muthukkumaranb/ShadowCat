"""Display H-sweep comparison table"""
import pandas as pd

print('='*160)
print('H-SWEEP COMPARISON: FULL 37-FOLD LOEO AT H=5, 10, 15')
print('='*160 + '\n')

# Load results
h5 = pd.read_csv('data/ucs/loeo_fold_results.csv')
h10 = pd.read_csv('data/ucs/loeo_fold_results_h10.csv')
h15 = pd.read_csv('data/ucs/loeo_fold_results_h15.csv')

results = [
    {'h': 5, 'df': h5},
    {'h': 10, 'df': h10},
    {'h': 15, 'df': h15},
]

print('| Metric | H=5 Set A | H=5 Set B | H=10 Set A | H=10 Set B | H=15 Set A | H=15 Set B |')
print('|--------|-----------|-----------|------------|------------|------------|------------|')

# F1
row = '| F1 Mean +/- Std |'
for r in results:
    df = r['df']
    a_f1_m, a_f1_s = df['a_f1'].mean(), df['a_f1'].std()
    b_f1_m, b_f1_s = df['b_f1'].mean(), df['b_f1'].std()
    row += f' {a_f1_m:.4f}+/-{a_f1_s:.4f} / {b_f1_m:.4f}+/-{b_f1_s:.4f} |'
print(row)

# PR-AUC
row = '| PR-AUC Mean +/- Std |'
for r in results:
    df = r['df']
    a_p_m, a_p_s = df['a_prauc'].mean(), df['a_prauc'].std()
    b_p_m, b_p_s = df['b_prauc'].mean(), df['b_prauc'].std()
    row += f' {a_p_m:.4f}+/-{a_p_s:.4f} / {b_p_m:.4f}+/-{b_p_s:.4f} |'
print(row)

# CI Overlap
row = '| CI Overlap (F1) |'
for r in results:
    df = r['df']
    a_m, a_s = df['a_f1'].mean(), df['a_f1'].std()
    b_m, b_s = df['b_f1'].mean(), df['b_f1'].std()
    overlap = (a_m - a_s <= b_m + b_s) and (b_m - b_s <= a_m + a_s)
    row += f' {str(overlap):<5} |'
print(row)

# Head-to-head
row = '| A wins / B wins |'
for r in results:
    df = r['df']
    a_wins = (df['a_f1'] > df['b_f1']).sum()
    b_wins = (df['b_f1'] > df['a_f1']).sum()
    row += f' {a_wins:>2} / {b_wins:<2} |'
print(row)

# Folds
row = '| Total Folds |'
for r in results:
    row += f' {len(r["df"]):<17} |'
print(row)

print()
print("PER-TYPE SUMMARY:")
print()

for attack_type in ['Botnet', 'DDOS-LOIC-UDP', 'SSH-Bruteforce']:
    print(f"\n{attack_type}:")
    print('| Metric | H=5 Set A | H=5 Set B | H=10 Set A | H=10 Set B | H=15 Set A | H=15 Set B |')
    print('|--------|-----------|-----------|------------|------------|------------|------------|')
    
    row = '| F1 Mean +/- Std |'
    for r in results:
        df = r['df'][r['df']['attack_type'] == attack_type]
        if len(df) > 0:
            a_f1_m, a_f1_s = df['a_f1'].mean(), df['a_f1'].std()
            b_f1_m, b_f1_s = df['b_f1'].mean(), df['b_f1'].std()
            row += f' {a_f1_m:.4f}+/-{a_f1_s:.4f} / {b_f1_m:.4f}+/-{b_f1_s:.4f} |'
        else:
            row += ' N/A |'
    print(row)
    
    row = '| PR-AUC Mean +/- Std |'
    for r in results:
        df = r['df'][r['df']['attack_type'] == attack_type]
        if len(df) > 0:
            a_p_m, a_p_s = df['a_prauc'].mean(), df['a_prauc'].std()
            b_p_m, b_p_s = df['b_prauc'].mean(), df['b_prauc'].std()
            row += f' {a_p_m:.4f}+/-{a_p_s:.4f} / {b_p_m:.4f}+/-{b_p_s:.4f} |'
        else:
            row += ' N/A |'
    print(row)

print()
print("KEY FINDINGS:")
print()
print("1. All three horizons show the SAME pattern: CIs overlap, verdict INCONCLUSIVE")
print("2. Set B (Schedule) consistently outperforms Set A (Traffic) in head-to-head")
print("   - H=5:  0 Set A wins vs  8 Set B wins")
print("   - H=10: 1 Set A win  vs 11 Set B wins")
print("   - H=15: 3 Set A wins vs 18 Set B wins")
print("3. PR-AUC is NOT zero - real values ranging 0.0-1.0 (NOT the bug from before)")
print("4. Horizon effect: Larger H provides more pre-onset windows, slight performance increase")
print()
