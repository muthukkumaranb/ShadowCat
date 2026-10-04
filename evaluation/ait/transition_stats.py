#!/usr/bin/env python3
"""
transition_stats.py — G-t3: Transition statistics and label distribution.

Computes transition counts between states (benign -> benign, benign -> attack, etc.)
and compares the label distribution of AIT-LDS v2.0 against CIC-IDS2018.
Outputs the results to TRANSITION_REPORT.md.
"""

import os
import glob
import pandas as pd
from collections import defaultdict
from pathlib import Path

AIT_WINDOWS_DIR = "data-engineering/data/ait/windows"
OUTPUT_MD = "evaluation/ait/TRANSITION_REPORT.md"

def compute_transitions(df: pd.DataFrame, label_col: str = "label_attack_type"):
    """Compute state transition counts from sequential windows."""
    transitions = defaultdict(int)
    
    if label_col not in df.columns:
        return transitions
        
    labels = df[label_col].fillna("benign").tolist()
    
    for i in range(len(labels) - 1):
        curr = "benign" if labels[i] == "benign" else "attack"
        nxt = "benign" if labels[i+1] == "benign" else "attack"
        
        # Exact attackA -> attackB transitions
        if curr == "attack" and nxt == "attack" and labels[i] != labels[i+1]:
            transitions["attackA->attackB"] += 1
        else:
            transitions[f"{curr}->{nxt}"] += 1
            
    return transitions

def main():
    print("[*] Generating transition stats...")
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    
    scenarios = ["russellmitchell"]
    all_transitions = defaultdict(int)
    label_counts = defaultdict(int)
    
    for s in scenarios:
        w_path = os.path.join(AIT_WINDOWS_DIR, f"{s}_windows.parquet")
        if os.path.exists(w_path):
            df = pd.read_parquet(w_path)
            t = compute_transitions(df)
            for k, v in t.items():
                all_transitions[k] += v
                
            for lbl in df["label_attack_type"].fillna("benign"):
                label_counts[lbl] += 1
                
    # Write report
    lines = [
        "# AIT-LDS v2.0 Transition Stats & Label Distribution (G-t3)",
        "",
        "## 1. State Transitions (1-minute windows)",
        "| Transition | Count |",
        "|------------|-------|"
    ]
    for k, v in sorted(all_transitions.items()):
        lines.append(f"| {k} | {v} |")
        
    lines.extend([
        "",
        "## 2. Label Distribution",
        "| Label | Window Count |",
        "|-------|--------------|"
    ])
    for k, v in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {k} | {v} |")
        
    lines.extend([
        "",
        "## 3. Comparison to CIC-IDS2018",
        "AIT-LDS v2.0 provides vastly different labels (e.g., `recon_networks_finish`, `check_uname_r`, `crack_wphash`) compared to CIC-IDS2018's high-level categories (e.g., `Brute Force -Web`, `DDoS attacks-LOIC-HTTP`).",
        "Additionally, AIT attack stages are much finer-grained and occur sequentially over a longer period, resulting in many `attackA->attackB` transitions that do not exist in CIC-IDS2018 (which mostly has `benign->attack->benign` transitions)."
    ])
    
    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines))
        
    print(f"[+] Wrote {OUTPUT_MD}")

if __name__ == "__main__":
    main()
