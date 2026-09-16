
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from backend.predict import ShadowcatPipeline

def test_graph_fusion():
    print("Loading datasets...")
    df = pd.read_parquet("data-engineering/data/ucs/ucs_windows.parquet").sort_values("window_start_utc")
    
    ben_seq = df[df["has_malicious_flows"] == 0].head(30)
    mal_seq = df[df["has_malicious_flows"] == 1].head(30)
    
    print("Initializing pipeline...")
    pipeline = ShadowcatPipeline()
    
    print("\n--- Running Benign Sequence ---")
    res_ben = pipeline.predict(ben_seq, source_type="flows")
    z_prime_ben = res_ben.get("fusion_experimental_result", {}).get("z_prime_t", [[0]*64])[0]
    note_ben = res_ben.get("fusion_experimental_result", {}).get("note", "")
    print(f"Status: {res_ben.get(\"fusion_experimental_result\", {}).get(\"status\")}")
    print(f"Note: {note_ben}")
    print(f"z_prime_t (first 5 dims): {z_prime_ben[:5]}")
    
    print("\n--- Running Malicious Sequence ---")
    res_mal = pipeline.predict(mal_seq, source_type="flows")
    z_prime_mal = res_mal.get("fusion_experimental_result", {}).get("z_prime_t", [[0]*64])[0]
    note_mal = res_mal.get("fusion_experimental_result", {}).get("note", "")
    print(f"Status: {res_mal.get(\"fusion_experimental_result\", {}).get(\"status\")}")
    print(f"Note: {note_mal}")
    print(f"z_prime_t (first 5 dims): {z_prime_mal[:5]}")
    
    # Calculate difference
    diff = sum(abs(a - b) for a, b in zip(z_prime_ben, z_prime_mal))
    print(f"\nTotal Absolute Difference between Z_prime outputs: {diff:.4f}")

if __name__ == "__main__":
    test_graph_fusion()

