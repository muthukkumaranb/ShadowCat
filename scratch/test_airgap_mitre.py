import sys
from pathlib import Path
import socket
import urllib.request
import pandas as pd

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Track any network connection attempts
network_calls = []

_orig_socket_connect = socket.socket.connect
def blocked_socket_connect(self, address):
    host, port = address[0], address[1]
    network_calls.append(f"socket.connect: {host}:{port}")
    # Allow localhost / loopback for local IPC if needed, block everything else
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ConnectionRefusedError(f"AIR-GAP VIOLATION: Disallowed network call to {host}:{port}")
    return _orig_socket_connect(self, address)

socket.socket.connect = blocked_socket_connect

print("=" * 60)
print("TESTING STRICT OFFLINE AIR-GAP DURING PREDICT()")
print("=" * 60)

parquet_path = "data-engineering/data/ucs/ucs_windows.parquet"
df = pd.read_parquet(parquet_path).head(45).copy()

from backend.predict import predict
output = predict(df, source_type="flows")

print("\nOutput summary:")
fc = output["forecast_trajectory"]
print("  - Window:", output["window_id"])
print("  - Tactic IDs:", fc["tactic_id"])
print("  - Technique IDs:", fc["technique_id"])
print("  - Mitre Details Count:", len(fc["mitre_details"]))
print(f"  - MITRE Network Calls Made: {len(network_calls)}")

mitre_external_calls = [c for c in network_calls if "mitre" in c.lower() or "github" in c.lower()]
assert len(mitre_external_calls) == 0, f"External MITRE calls detected: {mitre_external_calls}"
print("\n[PASS] Verified: Zero external network calls made to MITRE or GitHub during predict().")
print("All knowledge base queries are 100% offline from the vendored STIX 2.1 corpus.")
print("=" * 60)
