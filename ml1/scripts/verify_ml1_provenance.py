"""
ML1 Live Artifact Provenance Verification Script
"""
import os
import sys
import hashlib
import subprocess
import json
import pandas as pd
from pathlib import Path

def get_sha256(filepath):
    p = Path(filepath)
    if not p.exists():
        return "FILE NOT FOUND"
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== 1. GIT CONFIGURATION ===")
    for cmd in ["git remote -v", "git branch -vv", "git status --short", "git rev-parse HEAD"]:
        print(f">>> {cmd}")
        try:
            out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
            print(out)
        except Exception as e:
            print(f"Error: {e}")
        print()

    print("=== 2. DEVIATION ARTIFACTS ===")
    dev_paths = [
        "artifacts/lstm/loeo_world_model/deviation_predictions.csv",
        "artifacts/experiments/world_model_20260904/probabilistic/deviation_predictions.csv"
    ]
    for p in dev_paths:
        path = Path(p)
        print(f"Path: {path.absolute()}")
        print(f"Exists: {path.exists()}")
        if path.exists():
            size = path.stat().st_size
            sha = get_sha256(path)
            df = pd.read_csv(path)
            print(f"Size: {size} bytes")
            print(f"SHA-256: {sha}")
            print(f"Rows: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            if "fold_id" in df.columns:
                folds = sorted(df["fold_id"].unique())
                print(f"Folds count: {len(folds)}, Folds list: {folds}")
        print("-" * 40)

    print("\n=== 3. EDGELIST ARTIFACTS ===")
    edge_candidates = [
        "data/ucs/ucs_graph_edgelists.parquet",
        "data/ucs/ucs_edges.parquet",
        "data/ucs/ucs_edges.csv"
    ]
    for p in edge_candidates:
        path = Path(p)
        print(f"Path: {path.absolute()}")
        print(f"Exists: {path.exists()}")
        if path.exists():
            size = path.stat().st_size
            sha = get_sha256(path)
            print(f"Size: {size} bytes")
            print(f"SHA-256: {sha}")
            if p.endswith(".parquet"):
                df = pd.read_parquet(path)
                print(f"Rows: {len(df)}")
                print(f"Columns: {list(df.columns)}")
        print("-" * 40)

    print("\n=== 4. 37-FOLD LOEO MANIFEST ===")
    manifest_paths = [
        "artifacts/loeo/corrected_37fold_manifest.json",
        "artifacts/loeo_protocol_audit/loeo_protocol.json"
    ]
    for p in manifest_paths:
        path = Path(p)
        print(f"Path: {path.absolute()}")
        print(f"Exists: {path.exists()}")
        if path.exists():
            size = path.stat().st_size
            sha = get_sha256(path)
            print(f"Size: {size} bytes")
            print(f"SHA-256: {sha}")
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and "folds" in data:
                print(f"Fold count: {len(data['folds'])}")
                f0 = data['folds'][0]
                print(f"Sample fold keys: {list(f0.keys())}")
                has_ep = 'held_out_episode_id' in f0 or 'episode_id' in f0
                print(f"Episode grouping key present: {has_ep}")
        print("-" * 40)

if __name__ == "__main__":
    main()
