import json
from pathlib import Path

results_file = Path("scratch/endtoend_graphsage_results.json")
if results_file.exists():
    with open(results_file, "r") as f:
        data = json.load(f)
    rows = data.get("detection", [])
    attacks = {}
    for r in rows:
        attacks.setdefault(r["attack_type"], []).append(r)
    
    print(f"Total Folds: {len(rows)}")
    for atk, flist in attacks.items():
        n = len(flist)
        plain_f1 = sum(x["plain_f1"] for x in flist) / n
        scalar_f1 = sum(x["scalar_f1"] for x in flist) / n
        sage_f1 = sum(x["graphsage_f1"] for x in flist) / n
        sage_prec = sum(x["graphsage_prec"] for x in flist) / n
        sage_rec = sum(x["graphsage_rec"] for x in flist) / n
        print(f"{atk:<18} | Folds: {n:>2} | Plain F1: {plain_f1:.4f} | Scalar F1: {scalar_f1:.4f} | GraphSAGE F1: {sage_f1:.4f} (P: {sage_prec:.4f}, R: {sage_rec:.4f})")
    
    print("\n--- DDOS-LOIC-UDP Detailed Folds ---")
    for r in attacks.get("DDOS-LOIC-UDP", []):
        print(f"Fold {r['fold_id']:>2} ({r['held_out_episode']}): Plain F1={r['plain_f1']:.4f}, Sage F1={r['graphsage_f1']:.4f}, Prec={r['graphsage_prec']:.4f}, Rec={r['graphsage_rec']:.4f}")
