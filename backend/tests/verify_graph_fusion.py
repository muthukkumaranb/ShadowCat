
import os, sys, pandas as pd, logging
logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("."))
from backend.predict import ShadowcatPipeline

def test_graph_fusion():
    print("Loading datasets...")
    df = pd.read_parquet("data-engineering/data/ucs/ucs_windows.parquet").sort_values("window_start_utc")
    ben_seq = df[df["has_malicious_flows"] == 0].head(30)
    mal_seq = df[df["has_malicious_flows"] == 1].head(30)
    
    pipeline = ShadowcatPipeline()
    print("--- Benign Sequence ---")
    r1 = pipeline.predict(ben_seq, source_type="flows")
    f1 = r1.get("fusion_experimental", {})
    print(f"Status: {f1.get('status')}")
    print(f"Note: {f1.get('note')}")
    z1 = f1.get("z_prime_t", [[0]*64])[0]
    print(f"z_prime_t[:5]: {z1[:5]}")
    
    print("--- Malicious Sequence ---")
    r2 = pipeline.predict(mal_seq, source_type="flows")
    f2 = r2.get("fusion_experimental", {})
    print(f"Status: {f2.get('status')}")
    print(f"Note: {f2.get('note')}")
    z2 = f2.get("z_prime_t", [[0]*64])[0]
    print(f"z_prime_t[:5]: {z2[:5]}")
    
    diff = sum(abs(a - b) for a, b in zip(z1, z2))
    print(f"Total Diff: {diff:.4f}")

if __name__ == "__main__":
    test_graph_fusion()

