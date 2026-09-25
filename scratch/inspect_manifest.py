import json
m = json.load(open('ml1/artifacts/loeo/corrected_37fold_manifest.json'))
import pandas as pd
df = pd.read_parquet('data-engineering/data/ucs/ucs_windows.parquet')
for f in m['folds']:
    ep = f['held_out_episode_id']
    atk = df.loc[df['episode_id'] == ep, 'label_attack_type'].iloc[0] if (df['episode_id'] == ep).any() else 'N/A'
    print(f"Fold {f['fold_id']:2d}: {ep:40s} attack={atk:24s} train={len(f['train_indices']):5d} test={len(f['test_indices']):4d}")
